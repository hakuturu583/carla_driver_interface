# SPDX-License-Identifier: Apache-2.0
"""The :class:`~carla_driver_interface.runtime.world.WorldAdapter` backed by CARLA.

Split out from the contract module so that a process without CARLA -- CI running
against the fake, or a driver-only install -- never imports this file, and so
that the seam stays readable next to its one large implementation.

Written against the API surface shared by CARLA 0.9.x and 0.10.x; the few places
where they diverge are marked and probed defensively rather than branched on a
version string, because forks report versions inconsistently.
"""

from __future__ import annotations

import logging
import math
import queue
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from carla_driver_interface.geometry import Pose, dynamic_state_proto
from carla_driver_interface.grpc_api import (
    AABB,
    AvailableCamera,
    CarlaActorState,
    CarlaRendererData,
    CarlaWeather,
    TrafficLightState,
)
from carla_driver_interface.runtime.config import CameraConfig, RuntimeConfig, ScenarioSpec
from carla_driver_interface.runtime.control import VehicleCommand
from carla_driver_interface.runtime.conversions import (
    available_camera,
    camera_pose_in_rig,
    carla_transform_to_pose,
    carla_vector_to_local,
    rig_pose_from_actor_transform,
    seconds_to_us,
    vector_local_to_rig,
)
from carla_driver_interface.runtime.images import encode_bgra
from carla_driver_interface.runtime.world import (
    CameraCapture,
    EgoState,
    RolloutEvents,
    WorldSetup,
    WorldSnapshot,
)

logger = logging.getLogger(__name__)

__all__ = ["CarlaWorldAdapter", "load_carla_module"]


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


@dataclass(frozen=True)
class _RawFrame:
    """An un-encoded CARLA capture, waiting to be picked up by the runtime."""

    timestamp_us: int
    width: int
    height: int
    bgra: bytes


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
                    pose_in_rig=camera_pose_in_rig(
                        cam.x,
                        cam.y,
                        cam.z,
                        cam.pitch_deg,
                        cam.yaw_deg,
                        cam.roll_deg,
                        self._rear_axle_offset_m,
                    ),
                )
            )
        return cameras

    def _make_camera_callback(self, cam: CameraConfig, frames: queue.Queue):
        """Queue the raw frame; encoding happens in :meth:`_drain_captures`.

        With ``sensor_tick == fixed_delta_s`` the sensor fires on every tick but
        only the last tick of a policy step is ever submitted, so encoding here
        would compress frames nobody sees -- measured at roughly a fifth of the
        rollout. Copying the buffer is unavoidable (it is only valid for the
        duration of the callback); the JPEG is not.
        """
        epoch = self.config.epoch_offset_us

        def callback(image: Any) -> None:
            try:
                frames.put(
                    _RawFrame(
                        timestamp_us=seconds_to_us(image.timestamp, epoch),
                        width=image.width,
                        height=image.height,
                        bgra=bytes(image.raw_data),
                    )
                )
            except Exception:  # pragma: no cover - sensor thread must not die
                logger.exception("failed to receive a frame from %s", cam.logical_id)
                self._events.encode_failures += 1

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
        """Take the newest frame per camera, discarding any backlog, and encode it.

        A backlog means the sensor produced more frames than this policy step
        consumes -- expected when ``sensor_tick`` is finer than the policy step,
        and also what happens if the loop falls behind. Either way, submitting a
        stale frame would silently violate the sensor-freshness invariant alpasim
        asserts (``assert_sensors_up_to_date``).
        """
        captures = []
        for logical_id, frames in self._frame_queues.items():
            newest: _RawFrame | None = None
            dropped = -1
            while True:
                try:
                    newest = frames.get_nowait()
                    dropped += 1
                except queue.Empty:
                    break
            if newest is None:
                continue
            if dropped > 0:
                logger.debug("dropped %d stale frames from %s", dropped, logical_id)
            try:
                image_bytes = encode_bgra(
                    newest.bgra,
                    newest.width,
                    newest.height,
                    self.config.image_format,
                    self.config.image_quality,
                )
            except Exception:  # pragma: no cover - must not kill the rollout
                logger.exception("failed to encode a frame from %s", logical_id)
                self._events.encode_failures += 1
                continue
            captures.append(
                CameraCapture(
                    logical_id=logical_id,
                    # A CARLA RGB capture is instantaneous, so start == end.
                    # alpasim's rolling-shutter drivers accept that; see
                    # docs/COMPATIBILITY.md.
                    frame_start_us=newest.timestamp_us,
                    frame_end_us=newest.timestamp_us,
                    image_bytes=image_bytes,
                )
            )
        return captures

    # -- ground truth ------------------------------------------------------

    def environment(self, snapshot: WorldSnapshot) -> CarlaRendererData:
        light = self._ego.get_traffic_light() if self._ego.is_at_traffic_light() else None
        return CarlaRendererData(
            snapshot_timestamp_us=snapshot.timestamp_us,
            frame_id=snapshot.frame_id,
            map_name=self.scenario.map_name,
            weather=self._weather(),
            ego_traffic_light=self._traffic_light_state(light),
            ego_traffic_light_distance_m=self._traffic_light_distance(light, snapshot.ego),
            speed_limit_mps=self._speed_limit_mps(),
            actors=self._actor_states(snapshot.ego) if self.config.send_actor_ground_truth else [],
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

    def _traffic_light_state(self, light: Any | None) -> TrafficLightState:
        if light is None:
            return TrafficLightState.TRAFFIC_LIGHT_STATE_NONE
        mapping = {
            "Red": TrafficLightState.TRAFFIC_LIGHT_STATE_RED,
            "Yellow": TrafficLightState.TRAFFIC_LIGHT_STATE_YELLOW,
            "Green": TrafficLightState.TRAFFIC_LIGHT_STATE_GREEN,
            "Off": TrafficLightState.TRAFFIC_LIGHT_STATE_OFF,
        }
        return mapping.get(str(light.get_state()), TrafficLightState.TRAFFIC_LIGHT_STATE_UNKNOWN)

    def _traffic_light_distance(self, light: Any | None, ego: EgoState) -> float:
        """Distance the ego still has to travel before the stop line.

        Measured **along the ego's heading**, not as a straight-line distance,
        and negative once the line is behind.  The difference matters: a
        Euclidean distance starts growing again the moment the ego crosses the
        line, so a policy reading it cannot tell "1 m to go" from "1 m past",
        and ``is_at_traffic_light`` stays true throughout the trigger volume.

        A policy that stops inside that volume therefore never leaves it, and
        is told forever that there is a stop line ahead which it has in fact
        already crossed -- a deadlock, and one that only appears when the
        policy does the right thing and stops.

        Returning a negative value here matches what the field already means
        elsewhere: ``CarlaRendererData.ego_traffic_light_distance_m`` is
        documented as negative when no stop line applies to the ego, and once
        the line is behind, none does.
        """
        if light is None:
            return -1.0
        waypoints = light.get_stop_waypoints()
        if not waypoints:
            location = light.get_transform().location
            stop_points = [carla_vector_to_local(location.x, location.y, location.z)]
        else:
            stop_points = [self._waypoint_to_local(wp) for wp in waypoints]

        # Resolve into the rig frame, whose +x is straight ahead.
        to_rig = ego.pose_local_to_rig.inverse()
        longitudinal = to_rig.transform_points(np.asarray(stop_points, dtype=np.float64))[:, 0]

        # The nearest line still ahead governs us; if they are all behind, the
        # nearest of those does, so the value stays continuous as we cross.
        ahead = longitudinal[longitudinal >= 0.0]
        return float(ahead.min()) if len(ahead) else float(longitudinal.max())

    def _speed_limit_mps(self) -> float:
        return float(self._ego.get_speed_limit() or 0.0) / 3.6  # CARLA reports km/h

    def _actor_states(self, ego: EgoState) -> list[CarlaActorState]:
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
        # Order matters, and not merely for tidiness. The traffic manager runs
        # its own thread inside the CARLA client, and while it is in
        # synchronous mode it keeps issuing commands for every vehicle
        # registered to it. Destroying those vehicles first leaves it
        # operating on actors that no longer exist, and the resulting C++
        # exception is thrown on *its* thread, where no Python `except` can
        # reach it -- the process dies with SIGABRT and
        #
        #     terminate called after throwing an instance of 'std::runtime_error'
        #       what(): trying to operate on a destroyed actor
        #
        # after a rollout that had already completed successfully. Standing
        # the traffic manager down first makes the whole teardown ordinary.
        if self._traffic_manager is not None:
            try:
                self._traffic_manager.set_synchronous_mode(False)
            except RuntimeError:  # pragma: no cover
                logger.debug("traffic manager teardown failed", exc_info=True)

        for sensor in self._sensors:
            try:
                sensor.stop()
                sensor.destroy()
            except RuntimeError:  # pragma: no cover - actor may already be gone
                logger.debug("sensor teardown failed", exc_info=True)
        self._sensors.clear()

        # Destroy in one batch where the client supports it: a single
        # round trip closes the window in which the server holds a
        # partially torn-down scene.
        actors = [actor for actor in [*self._background, self._ego] if actor is not None]
        if actors and self._client is not None:
            try:
                import carla

                self._client.apply_batch_sync(
                    [carla.command.DestroyActor(actor) for actor in actors], True
                )
                actors = []
            except (RuntimeError, ImportError, AttributeError):  # pragma: no cover
                logger.debug("batch actor teardown failed; falling back", exc_info=True)

        for actor in actors:
            try:
                actor.destroy()
            except RuntimeError:  # pragma: no cover
                logger.debug("actor teardown failed", exc_info=True)
        self._background.clear()
        self._ego = None

        # Give the port back, or the next run waits four seconds for it.
        #
        # The server registers a traffic manager against a port and keeps that
        # registration after the client that made it exits. A later process
        # asking for the same port therefore tries to reach a manager that is
        # no longer there, waits for the attempt to time out, and only then
        # creates a new one. Measured against this server: 0.05 s for the
        # first process to claim a port, 4.05 s for every process after it,
        # and 0.05 s throughout once `shut_down` is called -- on a 200-step
        # rollout that is a ninth of the whole run, paid every time because
        # the default port never changes.
        #
        # After the vehicles are gone rather than before, so that standing the
        # manager down keeps the ordering that stops it operating on destroyed
        # actors.
        if self._traffic_manager is not None:
            try:
                self._traffic_manager.shut_down()
            except (RuntimeError, AttributeError):  # pragma: no cover
                logger.debug("traffic manager shutdown failed", exc_info=True)
            self._traffic_manager = None

        # Leaving the server in synchronous mode would hang the next client.
        if self._world is not None and self._original_settings is not None:
            self._world.apply_settings(self._original_settings)
