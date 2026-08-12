# SPDX-License-Identifier: Apache-2.0
"""The world abstraction the runtime drives.

:class:`WorldAdapter` is the only seam between :class:`~
carla_driver_interface.runtime.carla_runtime.CarlaRuntime` and the simulator.
It exists for two reasons:

1. CARLA 0.9.x and 0.10.x differ in small but load-bearing ways (weather fields,
   wheel-position units, blueprint availability).  Keeping the differences here
   means the closed loop above is version-agnostic.
2. ``carla`` is an optional dependency and cannot be installed in CI, so tests
   substitute :class:`~carla_driver_interface.fakes.fake_world.FakeWorld`.

Adapters speak alpasim conventions: everything crossing this interface is
already right-handed, in metres, seconds-free (microseconds) and rig-anchored.
"""

from __future__ import annotations

import logging
import math
import queue
import random
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from carla_driver_interface.geometry import Pose
from carla_driver_interface.grpc_api import (
    AvailableCamera,
    CarlaActorState,
    CarlaWeather,
    TrafficLightState,
)
from carla_driver_interface.runtime.config import CameraConfig, RuntimeConfig, ScenarioSpec
from carla_driver_interface.runtime.control import VehicleCommand
from carla_driver_interface.runtime.conversions import (
    available_camera,
    carla_transform_to_pose,
    carla_vector_to_local,
    rig_pose_from_actor_transform,
    seconds_to_us,
    vector_local_to_rig,
)
from carla_driver_interface.runtime.images import bgra_to_rgb, encode_rgb

logger = logging.getLogger(__name__)

__all__ = [
    "CameraCapture",
    "CarlaWorldAdapter",
    "EgoState",
    "EnvironmentInfo",
    "RolloutEvents",
    "WorldAdapter",
    "WorldSetup",
    "WorldSnapshot",
    "load_carla_module",
]


# ---------------------------------------------------------------------------
# Data crossing the seam
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EgoState:
    """Ego kinematics at one instant, in alpasim conventions."""

    timestamp_us: int
    #: Active transform ``local -> rig``.
    pose_local_to_rig: Pose
    #: Resolved in the rig frame, as ``common.DynamicState`` requires.
    linear_velocity_in_rig: np.ndarray
    angular_velocity_in_rig: np.ndarray
    linear_acceleration_in_rig: np.ndarray

    @property
    def speed_mps(self) -> float:
        return float(np.linalg.norm(self.linear_velocity_in_rig))


@dataclass(frozen=True)
class CameraCapture:
    """An encoded frame ready for ``submit_image_observation``."""

    logical_id: str
    frame_start_us: int
    frame_end_us: int
    image_bytes: bytes


@dataclass(frozen=True)
class WorldSnapshot:
    """The result of one simulator tick."""

    frame_id: int
    timestamp_us: int
    ego: EgoState
    captures: list[CameraCapture] = field(default_factory=list)


@dataclass(frozen=True)
class EnvironmentInfo:
    """Ground truth for the ``CarlaRendererData`` extension payload."""

    map_name: str
    weather: CarlaWeather
    ego_traffic_light: TrafficLightState = TrafficLightState.TRAFFIC_LIGHT_STATE_UNKNOWN
    ego_traffic_light_distance_m: float = -1.0
    speed_limit_mps: float = 0.0
    actors: list[CarlaActorState] = field(default_factory=list)


@dataclass
class RolloutEvents:
    """Counters the metrics collector turns into rollout scores."""

    collisions: int = 0
    lane_invasions: int = 0
    #: Set when the world itself decided the rollout is over.
    finished_reason: str = ""


@dataclass(frozen=True)
class WorldSetup:
    """What the adapter learned while building the scenario."""

    map_name: str
    cameras: list[AvailableCamera]
    #: Signed offset from the actor origin to the rig origin, in metres.
    rear_axle_offset_m: float
    #: The full route, in the ``local`` frame, as ``(N, 3)``.
    route_in_local: np.ndarray


@runtime_checkable
class WorldAdapter(Protocol):
    """What :class:`CarlaRuntime` needs from a simulator."""

    def setup(self) -> WorldSetup:
        """Build the scenario and return its description. Called once."""

    def tick(self) -> WorldSnapshot:
        """Advance by one ``fixed_delta_s`` and collect sensor output."""

    def apply_control(self, command: VehicleCommand) -> None:
        """Latch actuation, applied on the next :meth:`tick`."""

    def environment(self, ego: EgoState) -> EnvironmentInfo:
        """Ground truth for the extension payload at the current instant."""

    def events(self) -> RolloutEvents:
        """Cumulative collision / lane-invasion counters."""

    def close(self) -> None:
        """Destroy actors and restore the simulator's settings."""


# ---------------------------------------------------------------------------
# CARLA implementation
# ---------------------------------------------------------------------------


def load_carla_module(python_path: str | None = None) -> Any:
    """Import ``carla``, optionally from an out-of-tree PythonAPI.

    CARLA 0.10.x is not published on PyPI; it ships its own PythonAPI directory.
    ``python_path`` prepends that directory to ``sys.path`` before importing.
    """
    if python_path:
        import sys

        if python_path not in sys.path:
            sys.path.insert(0, python_path)
    try:
        import carla
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the `carla` module is not importable. Install the extra "
            "(`uv sync --extra carla`, CARLA 0.9.x) or point --carla-python-path "
            "at a 0.10.x PythonAPI directory."
        ) from exc
    return carla


class CarlaWorldAdapter:
    """Drives a real CARLA server.

    Written against the API surface shared by 0.9.x and 0.10.x; the few places
    where they diverge are marked and probed defensively rather than branched on
    a version string, because forks report versions inconsistently.
    """

    def __init__(
        self,
        config: RuntimeConfig,
        scenario: ScenarioSpec,
        carla_python_path: str | None = None,
    ) -> None:
        self.config = config
        self.scenario = scenario
        self._carla = load_carla_module(carla_python_path)
        self._rng = random.Random(config.seed)

        self._client: Any = None
        self._world: Any = None
        self._map: Any = None
        self._traffic_manager: Any = None
        self._original_settings: Any = None

        self._ego: Any = None
        self._sensors: list[Any] = []
        self._background: list[Any] = []
        self._frame_queues: dict[str, queue.Queue] = {}
        self._events = RolloutEvents()
        self._rear_axle_offset_m = 0.0
        self._pending_control: VehicleCommand | None = None

    # -- setup -------------------------------------------------------------

    def setup(self) -> WorldSetup:
        carla = self._carla
        self._client = carla.Client(self.config.carla_host, self.config.carla_port)
        self._client.set_timeout(self.config.carla_timeout_s)

        self._world = self._client.load_world(self.scenario.map_name)
        self._map = self._world.get_map()

        self._original_settings = self._world.get_settings()
        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.config.fixed_delta_s
        self._world.apply_settings(settings)

        self._traffic_manager = self._client.get_trafficmanager(self.config.traffic_manager_port)
        self._traffic_manager.set_synchronous_mode(True)
        self._traffic_manager.set_random_device_seed(self.config.seed)

        if self.scenario.weather_preset:
            self._apply_weather(self.scenario.weather_preset)

        spawn_transform = self._spawn_ego()
        self._rear_axle_offset_m = self._resolve_rear_axle_offset()
        cameras = self._spawn_cameras()
        self._spawn_collision_sensors()
        self._spawn_background_traffic()

        route = self._build_route(spawn_transform)

        return WorldSetup(
            map_name=self.scenario.map_name,
            cameras=cameras,
            rear_axle_offset_m=self._rear_axle_offset_m,
            route_in_local=route,
        )

    def _apply_weather(self, preset: str) -> None:
        weather = getattr(self._carla.WeatherParameters, preset, None)
        if weather is None:
            raise ValueError(f"unknown CARLA weather preset {preset!r}")
        self._world.set_weather(weather)

    def _spawn_ego(self) -> Any:
        blueprints = self._world.get_blueprint_library()
        matches = blueprints.filter(self.config.ego_blueprint)
        if not matches:
            raise ValueError(
                f"no blueprint matches {self.config.ego_blueprint!r} on this CARLA build"
            )
        blueprint = matches[0]
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "hero")

        spawn_points = self._map.get_spawn_points()
        if not spawn_points:
            raise RuntimeError(f"map {self.scenario.map_name!r} has no spawn points")
        index = self.scenario.spawn_point_index
        if index is None:
            index = self._rng.randrange(len(spawn_points))
        transform = spawn_points[index % len(spawn_points)]

        self._ego = self._world.spawn_actor(blueprint, transform)
        # Let the actor settle before anything reads its transform.
        self._world.tick()
        return transform

    def _resolve_rear_axle_offset(self) -> float:
        if self.config.rear_axle_offset_m is not None:
            logger.info("rear axle offset: %.3f m (from config)", self.config.rear_axle_offset_m)
            return self.config.rear_axle_offset_m

        offset = self._derive_rear_axle_offset()
        logger.info(
            "rear axle offset: %.3f m (derived from wheel physics; override with "
            "RuntimeConfig.rear_axle_offset_m if the geometry looks wrong)",
            offset,
        )
        return offset

    def _derive_rear_axle_offset(self) -> float:
        """Longitudinal distance from the actor origin back to the rear axle.

        CARLA 0.9.x reports ``WheelPhysicsControl.position`` in **world
        centimetres**, not vehicle-local metres. We therefore convert the wheel
        positions into the actor frame explicitly rather than trusting them to
        already be relative.
        """
        try:
            physics = self._ego.get_physics_control()
            wheels = list(physics.wheels)
        except (AttributeError, RuntimeError):  # pragma: no cover - build dependent
            wheels = []

        if len(wheels) < 4:
            fallback = -0.5 * self._ego.bounding_box.extent.x
            logger.warning(
                "wheel physics unavailable; falling back to half the bounding box "
                "(%.3f m). Set RuntimeConfig.rear_axle_offset_m for accuracy.",
                fallback,
            )
            return float(fallback)

        actor_pose = self._actor_pose()
        world_to_actor = actor_pose.inverse()
        rear_positions = []
        for wheel in wheels[2:4]:  # CARLA orders wheels FL, FR, RL, RR
            position_cm = wheel.position
            world_point = carla_vector_to_local(
                position_cm.x / 100.0, position_cm.y / 100.0, position_cm.z / 100.0
            )
            rear_positions.append(world_to_actor.transform_points(world_point)[0])

        return float(np.mean([p[0] for p in rear_positions]))

    def _spawn_cameras(self) -> list[AvailableCamera]:
        carla = self._carla
        blueprints = self._world.get_blueprint_library()
        cameras: list[AvailableCamera] = []

        for cam in self.config.cameras:
            blueprint = blueprints.find("sensor.camera.rgb")
            blueprint.set_attribute("image_size_x", str(cam.width))
            blueprint.set_attribute("image_size_y", str(cam.height))
            blueprint.set_attribute("fov", str(cam.fov_deg))
            # One capture per simulator tick; the runtime decides which to send.
            blueprint.set_attribute("sensor_tick", str(self.config.fixed_delta_s))

            transform = carla.Transform(
                carla.Location(x=cam.x, y=cam.y, z=cam.z),
                carla.Rotation(pitch=cam.pitch_deg, yaw=cam.yaw_deg, roll=cam.roll_deg),
            )
            sensor = self._world.spawn_actor(blueprint, transform, attach_to=self._ego)

            frames: queue.Queue = queue.Queue()
            self._frame_queues[cam.logical_id] = frames
            sensor.listen(self._make_camera_callback(cam, frames))
            self._sensors.append(sensor)

            cameras.append(
                available_camera(
                    logical_id=cam.logical_id,
                    width=cam.width,
                    height=cam.height,
                    horizontal_fov_deg=cam.fov_deg,
                    camera_pose_in_rig=self._camera_pose_in_rig(cam),
                )
            )
        return cameras

    def _camera_pose_in_rig(self, cam: CameraConfig) -> Pose:
        """Mount transform relative to the rig origin, not the actor origin."""
        pose_actor_to_camera = carla_transform_to_pose(
            (cam.x, cam.y, cam.z), (cam.pitch_deg, cam.yaw_deg, cam.roll_deg)
        )
        pose_actor_to_rig = Pose(
            np.array([self._rear_axle_offset_m, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0])
        )
        return pose_actor_to_rig.inverse() @ pose_actor_to_camera

    def _make_camera_callback(self, cam: CameraConfig, frames: queue.Queue):
        image_format = self.config.image_format
        quality = self.config.image_quality
        epoch = self.config.epoch_offset_us

        # A CARLA RGB capture is instantaneous, so start == end. alpasim's
        # rolling-shutter drivers accept that; see docs/COMPATIBILITY.md.
        def callback(image: Any) -> None:
            try:
                rgb = bgra_to_rgb(bytes(image.raw_data), image.width, image.height)
                timestamp_us = seconds_to_us(image.timestamp, epoch)
                frames.put(
                    CameraCapture(
                        logical_id=cam.logical_id,
                        frame_start_us=timestamp_us,
                        frame_end_us=timestamp_us,
                        image_bytes=encode_rgb(rgb, image_format, quality),
                    )
                )
            except Exception:  # pragma: no cover - sensor thread must not die
                logger.exception("failed to encode a frame from %s", cam.logical_id)

        return callback

    def _spawn_collision_sensors(self) -> None:
        blueprints = self._world.get_blueprint_library()
        carla = self._carla
        origin = carla.Transform()

        collision = self._world.spawn_actor(
            blueprints.find("sensor.other.collision"), origin, attach_to=self._ego
        )
        collision.listen(lambda _event: self._record_event("collision"))
        self._sensors.append(collision)

        lane = self._world.spawn_actor(
            blueprints.find("sensor.other.lane_invasion"), origin, attach_to=self._ego
        )
        lane.listen(lambda _event: self._record_event("lane_invasion"))
        self._sensors.append(lane)

    def _record_event(self, kind: str) -> None:
        if kind == "collision":
            self._events.collisions += 1
        else:
            self._events.lane_invasions += 1

    def _spawn_background_traffic(self) -> None:
        if self.scenario.num_background_vehicles <= 0:
            return
        blueprints = self._world.get_blueprint_library().filter("vehicle.*")
        spawn_points = list(self._map.get_spawn_points())
        self._rng.shuffle(spawn_points)

        spawned = 0
        for transform in spawn_points:
            if spawned >= self.scenario.num_background_vehicles:
                break
            blueprint = blueprints[self._rng.randrange(len(blueprints))]
            actor = self._world.try_spawn_actor(blueprint, transform)
            if actor is None:
                continue
            actor.set_autopilot(True, self.config.traffic_manager_port)
            self._background.append(actor)
            spawned += 1
        logger.info("spawned %d background vehicles", spawned)

    def _build_route(self, spawn_transform: Any) -> np.ndarray:
        """Follow lanes forward from the spawn point to make a driveable route.

        alpasim gets its route from the recording; there is no recording here, so
        the route is generated by walking the lane graph with seeded choices at
        junctions. Deterministic for a given ``RuntimeConfig.seed``.
        """
        step = max(0.5, self.config.route_resolution_m)
        # Long enough that a full rollout at a plausible speed stays on it.
        target_length_m = max(200.0, self.config.max_steps * self.config.policy_timestep_s * 20.0)

        waypoint = self._map.get_waypoint(spawn_transform.location, project_to_road=True)
        points = [self._waypoint_to_local(waypoint)]
        travelled = 0.0
        while travelled < target_length_m:
            options = waypoint.next(step)
            if not options:
                break
            waypoint = (
                options[self._rng.randrange(len(options))] if len(options) > 1 else options[0]
            )
            points.append(self._waypoint_to_local(waypoint))
            travelled += step

        if len(points) < 2:
            raise RuntimeError(
                "could not build a route from the spawn point; the map may lack "
                "connected lanes at that location"
            )
        return np.stack(points)

    def _waypoint_to_local(self, waypoint: Any) -> np.ndarray:
        location = waypoint.transform.location
        return carla_vector_to_local(location.x, location.y, location.z)

    # -- stepping ----------------------------------------------------------

    def tick(self) -> WorldSnapshot:
        if self._pending_control is not None:
            self._ego.apply_control(self._to_carla_control(self._pending_control))
            self._pending_control = None

        frame_id = self._world.tick()
        snapshot = self._world.get_snapshot()
        timestamp_us = seconds_to_us(
            snapshot.timestamp.elapsed_seconds, self.config.epoch_offset_us
        )
        return WorldSnapshot(
            frame_id=int(frame_id),
            timestamp_us=timestamp_us,
            ego=self._ego_state(timestamp_us),
            captures=self._drain_captures(),
        )

    def apply_control(self, command: VehicleCommand) -> None:
        self._pending_control = command

    def _to_carla_control(self, command: VehicleCommand) -> Any:
        return self._carla.VehicleControl(
            throttle=float(np.clip(command.throttle, 0.0, 1.0)),
            steer=float(np.clip(command.steer, -1.0, 1.0)),
            brake=float(np.clip(command.brake, 0.0, 1.0)),
            hand_brake=command.hand_brake,
            reverse=command.reverse,
        )

    def _actor_pose(self) -> Pose:
        transform = self._ego.get_transform()
        location, rotation = transform.location, transform.rotation
        return carla_transform_to_pose(
            (location.x, location.y, location.z),
            (rotation.pitch, rotation.yaw, rotation.roll),
        )

    def _ego_state(self, timestamp_us: int) -> EgoState:
        transform = self._ego.get_transform()
        location, rotation = transform.location, transform.rotation
        pose = rig_pose_from_actor_transform(
            (location.x, location.y, location.z),
            (rotation.pitch, rotation.yaw, rotation.roll),
            self._rear_axle_offset_m,
        )

        velocity = self._ego.get_velocity()
        acceleration = self._ego.get_acceleration()
        angular = self._ego.get_angular_velocity()  # degrees/s in CARLA

        velocity_local = carla_vector_to_local(velocity.x, velocity.y, velocity.z)
        acceleration_local = carla_vector_to_local(acceleration.x, acceleration.y, acceleration.z)
        angular_local = carla_vector_to_local(
            math.radians(angular.x), math.radians(angular.y), math.radians(angular.z)
        )

        return EgoState(
            timestamp_us=timestamp_us,
            pose_local_to_rig=pose,
            linear_velocity_in_rig=vector_local_to_rig(velocity_local, pose),
            angular_velocity_in_rig=vector_local_to_rig(angular_local, pose),
            linear_acceleration_in_rig=vector_local_to_rig(acceleration_local, pose),
        )

    def _drain_captures(self) -> list[CameraCapture]:
        """Take the newest frame per camera, discarding any backlog.

        A backlog means the encoder fell behind the simulator; sending stale
        frames would silently violate the sensor-freshness invariant alpasim
        asserts (``assert_sensors_up_to_date``).
        """
        captures = []
        for logical_id, frames in self._frame_queues.items():
            newest: CameraCapture | None = None
            dropped = -1
            while True:
                try:
                    newest = frames.get_nowait()
                    dropped += 1
                except queue.Empty:
                    break
            if newest is not None:
                if dropped > 0:
                    logger.debug("dropped %d stale frames from %s", dropped, logical_id)
                captures.append(newest)
        return captures

    # -- ground truth ------------------------------------------------------

    def environment(self, ego: EgoState) -> EnvironmentInfo:
        return EnvironmentInfo(
            map_name=self.scenario.map_name,
            weather=self._weather(),
            ego_traffic_light=self._traffic_light_state(),
            ego_traffic_light_distance_m=self._traffic_light_distance(ego),
            speed_limit_mps=self._speed_limit_mps(),
            actors=self._actor_states(ego) if self.config.send_actor_ground_truth else [],
        )

    def _weather(self) -> CarlaWeather:
        weather = self._world.get_weather()

        # 0.10.x dropped some 0.9.x weather fields; read defensively.
        def value(name: str) -> float:
            return float(getattr(weather, name, 0.0))

        return CarlaWeather(
            cloudiness=value("cloudiness"),
            precipitation=value("precipitation"),
            precipitation_deposits=value("precipitation_deposits"),
            wind_intensity=value("wind_intensity"),
            sun_azimuth_angle=value("sun_azimuth_angle"),
            sun_altitude_angle=value("sun_altitude_angle"),
            fog_density=value("fog_density"),
            wetness=value("wetness"),
        )

    def _traffic_light_state(self) -> TrafficLightState:
        if not self._ego.is_at_traffic_light():
            return TrafficLightState.TRAFFIC_LIGHT_STATE_NONE
        light = self._ego.get_traffic_light()
        if light is None:
            return TrafficLightState.TRAFFIC_LIGHT_STATE_NONE
        mapping = {
            "Red": TrafficLightState.TRAFFIC_LIGHT_STATE_RED,
            "Yellow": TrafficLightState.TRAFFIC_LIGHT_STATE_YELLOW,
            "Green": TrafficLightState.TRAFFIC_LIGHT_STATE_GREEN,
            "Off": TrafficLightState.TRAFFIC_LIGHT_STATE_OFF,
        }
        return mapping.get(str(light.get_state()), TrafficLightState.TRAFFIC_LIGHT_STATE_UNKNOWN)

    def _traffic_light_distance(self, ego: EgoState) -> float:
        if not self._ego.is_at_traffic_light():
            return -1.0
        light = self._ego.get_traffic_light()
        if light is None:
            return -1.0
        waypoints = light.get_stop_waypoints()
        if not waypoints:
            location = light.get_transform().location
            stop_points = [carla_vector_to_local(location.x, location.y, location.z)]
        else:
            stop_points = [self._waypoint_to_local(wp) for wp in waypoints]
        ego_position = ego.pose_local_to_rig.position
        return float(min(np.linalg.norm(np.asarray(p) - ego_position) for p in stop_points))

    def _speed_limit_mps(self) -> float:
        return float(self._ego.get_speed_limit() or 0.0) / 3.6  # CARLA reports km/h

    def _actor_states(self, ego: EgoState) -> list[CarlaActorState]:
        from carla_driver_interface.geometry import dynamic_state_proto
        from carla_driver_interface.grpc_api import AABB

        states = []
        ego_position = ego.pose_local_to_rig.position
        for actor in self._world.get_actors().filter("*vehicle*"):
            if actor.id == self._ego.id:
                continue
            transform = actor.get_transform()
            location, rotation = transform.location, transform.rotation
            pose = carla_transform_to_pose(
                (location.x, location.y, location.z),
                (rotation.pitch, rotation.yaw, rotation.roll),
            )
            if float(np.linalg.norm(pose.position - ego_position)) > 150.0:
                continue

            velocity = actor.get_velocity()
            extent = actor.bounding_box.extent
            states.append(
                CarlaActorState(
                    track_id=str(actor.id),
                    type_id=actor.type_id,
                    pose_local_to_aabb=pose.to_proto(),
                    aabb=AABB(size_x=2.0 * extent.x, size_y=2.0 * extent.y, size_z=2.0 * extent.z),
                    dynamic_state=dynamic_state_proto(
                        linear_velocity=carla_vector_to_local(velocity.x, velocity.y, velocity.z),
                        angular_velocity=np.zeros(3),
                        linear_acceleration=np.zeros(3),
                    ),
                )
            )
        return states

    def events(self) -> RolloutEvents:
        return self._events

    # -- teardown ----------------------------------------------------------

    def close(self) -> None:
        for sensor in self._sensors:
            try:
                sensor.stop()
                sensor.destroy()
            except RuntimeError:  # pragma: no cover - actor may already be gone
                logger.debug("sensor teardown failed", exc_info=True)
        self._sensors.clear()

        for actor in [*self._background, self._ego]:
            if actor is None:
                continue
            try:
                actor.destroy()
            except RuntimeError:  # pragma: no cover
                logger.debug("actor teardown failed", exc_info=True)
        self._background.clear()
        self._ego = None

        if self._traffic_manager is not None:
            try:
                self._traffic_manager.set_synchronous_mode(False)
            except RuntimeError:  # pragma: no cover
                logger.debug("traffic manager teardown failed", exc_info=True)

        # Leaving the server in synchronous mode would hang the next client.
        if self._world is not None and self._original_settings is not None:
            self._world.apply_settings(self._original_settings)
