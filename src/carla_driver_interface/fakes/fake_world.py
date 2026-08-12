# SPDX-License-Identifier: Apache-2.0
"""A CARLA-free :class:`WorldAdapter` for tests and demos.

The `carla` module cannot be installed in CI, so the closed loop would otherwise
be untestable.  :class:`FakeWorld` implements the same protocol on top of a
kinematic bicycle model, which is enough to exercise every step of the
runtime--driver exchange: session setup, image submission, egomotion, route,
the ``drive`` call, control application and metrics.

It is deliberately *not* a physics model.  Anything relying on real vehicle
dynamics belongs in a test against a real CARLA server.
"""

from __future__ import annotations

import math

import numpy as np

from carla_driver_interface.geometry import Pose
from carla_driver_interface.grpc_api import CarlaRendererData, CarlaWeather, TrafficLightState
from carla_driver_interface.runtime.config import RuntimeConfig, ScenarioSpec
from carla_driver_interface.runtime.control import VehicleCommand
from carla_driver_interface.runtime.conversions import (
    available_camera,
    camera_pose_in_rig,
    seconds_to_us,
)
from carla_driver_interface.runtime.images import encode_rgb
from carla_driver_interface.runtime.world import (
    CameraCapture,
    EgoState,
    RolloutEvents,
    WorldSetup,
    WorldSnapshot,
)

__all__ = ["FakeWorld", "straight_then_curve_route"]


def straight_then_curve_route(
    straight_m: float = 40.0,
    radius_m: float = 30.0,
    turn_rad: float = math.pi / 2,
    resolution_m: float = 1.0,
) -> np.ndarray:
    """A route that goes straight, then turns left at constant radius.

    Exercises both the lateral controller and the policy's curvature slowdown.
    """
    points = [
        np.array([distance, 0.0, 0.0]) for distance in np.arange(0.0, straight_m, resolution_m)
    ]
    centre = np.array([straight_m, radius_m, 0.0])
    n_arc = max(2, int(round(radius_m * turn_rad / resolution_m)))
    for i in range(n_arc + 1):
        theta = turn_rad * i / n_arc
        points.append(centre + radius_m * np.array([math.sin(theta), -math.cos(theta), 0.0]))
    return np.stack(points)


class FakeWorld:
    """Kinematic bicycle model standing in for a CARLA server."""

    def __init__(
        self,
        config: RuntimeConfig,
        scenario: ScenarioSpec | None = None,
        route_in_local: np.ndarray | None = None,
        max_speed_mps: float = 20.0,
        max_accel_mps2: float = 4.0,
        max_brake_mps2: float = 8.0,
        off_route_tolerance_m: float = 6.0,
    ) -> None:
        self.config = config
        self.scenario = scenario or ScenarioSpec(map_name="FakeTown", name="straight_then_curve")
        self._route = (
            straight_then_curve_route()
            if route_in_local is None
            else np.asarray(route_in_local, dtype=np.float64).reshape(-1, 3)
        )
        self.max_speed_mps = max_speed_mps
        self.max_accel_mps2 = max_accel_mps2
        self.max_brake_mps2 = max_brake_mps2
        self.off_route_tolerance_m = off_route_tolerance_m

        self._x = float(self._route[0][0])
        self._y = float(self._route[0][1])
        self._yaw = math.atan2(
            float(self._route[1][1] - self._route[0][1]),
            float(self._route[1][0] - self._route[0][0]),
        )
        self._speed = 0.0
        self._yaw_rate = 0.0
        self._acceleration = 0.0

        self._frame_id = 0
        self._time_s = 0.0
        self._command = VehicleCommand()
        self._events = RolloutEvents()
        #: Every ego state produced so far; tests assert on the driven path.
        self.history: list[EgoState] = []

    # -- WorldAdapter ------------------------------------------------------

    def setup(self) -> WorldSetup:
        cameras = [
            available_camera(
                logical_id=cam.logical_id,
                width=cam.width,
                height=cam.height,
                horizontal_fov_deg=cam.fov_deg,
                # Through the same conversion the real adapter uses, so a
                # mount rotation cannot be honoured in CARLA and dropped in CI.
                # The fake rig has no actor/rig offset to reconcile.
                pose_in_rig=camera_pose_in_rig(
                    cam.x, cam.y, cam.z, cam.pitch_deg, cam.yaw_deg, cam.roll_deg, 0.0
                ),
            )
            for cam in self.config.cameras
        ]
        return WorldSetup(
            map_name=self.scenario.map_name,
            cameras=cameras,
            rear_axle_offset_m=0.0,
            route_in_local=self._route,
        )

    def apply_control(self, command: VehicleCommand) -> None:
        self._command = command

    def tick(self) -> WorldSnapshot:
        dt = self.config.fixed_delta_s
        self._integrate(dt)

        self._frame_id += 1
        self._time_s += dt
        timestamp_us = seconds_to_us(self._time_s, self.config.epoch_offset_us)

        ego = self._ego_state(timestamp_us)
        self.history.append(ego)
        self._update_events(ego)

        return WorldSnapshot(
            frame_id=self._frame_id,
            timestamp_us=timestamp_us,
            ego=ego,
            captures=self._captures(timestamp_us),
        )

    def environment(self, snapshot: WorldSnapshot) -> CarlaRendererData:
        return CarlaRendererData(
            snapshot_timestamp_us=snapshot.timestamp_us,
            frame_id=snapshot.frame_id,
            map_name=self.scenario.map_name,
            weather=CarlaWeather(sun_altitude_angle=45.0),
            ego_traffic_light=TrafficLightState.TRAFFIC_LIGHT_STATE_NONE,
            ego_traffic_light_distance_m=-1.0,
            speed_limit_mps=self.max_speed_mps,
        )

    def events(self) -> RolloutEvents:
        return self._events

    def close(self) -> None:
        return None

    # -- model -------------------------------------------------------------

    def _integrate(self, dt: float) -> None:
        """Advance the bicycle model by ``dt`` under the latched command."""
        command = self._command
        accel = command.throttle * self.max_accel_mps2 - command.brake * self.max_brake_mps2
        # Rolling resistance, so coasting decays rather than holding forever.
        accel -= 0.05 * self._speed

        previous_speed = self._speed
        self._speed = float(np.clip(self._speed + accel * dt, 0.0, self.max_speed_mps))
        self._acceleration = (self._speed - previous_speed) / dt

        # CARLA steers positive to the right; the local frame is positive to the
        # left, hence the sign flip -- the same convention the controller uses.
        steer_angle = -command.steer * self.config.control.max_steer_angle_rad
        self._yaw_rate = self._speed * math.tan(steer_angle) / self.config.control.wheelbase_m

        self._yaw += self._yaw_rate * dt
        self._x += self._speed * math.cos(self._yaw) * dt
        self._y += self._speed * math.sin(self._yaw) * dt

    def _ego_state(self, timestamp_us: int) -> EgoState:
        return EgoState(
            timestamp_us=timestamp_us,
            pose_local_to_rig=Pose.from_xyz_yaw(self._x, self._y, 0.0, self._yaw),
            # A bicycle model has no lateral slip, so rig-frame velocity is
            # purely longitudinal.
            linear_velocity_in_rig=np.array([self._speed, 0.0, 0.0]),
            angular_velocity_in_rig=np.array([0.0, 0.0, self._yaw_rate]),
            linear_acceleration_in_rig=np.array([self._acceleration, 0.0, 0.0]),
        )

    def _update_events(self, ego: EgoState) -> None:
        """Treat leaving the route corridor as a lane invasion."""
        position = ego.pose_local_to_rig.position[:2]
        distance = float(np.min(np.linalg.norm(self._route[:, :2] - position, axis=1)))
        if distance > self.off_route_tolerance_m:
            self._events.lane_invasions += 1

    def _captures(self, timestamp_us: int) -> list[CameraCapture]:
        """Synthesise a frame per camera: a horizon gradient, so it is not blank."""
        captures = []
        for cam in self.config.cameras:
            rgb = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
            horizon = cam.height // 2
            rgb[:horizon] = (110, 160, 220)  # sky
            rgb[horizon:] = (70, 70, 70)  # road
            # A moving marker keeps successive frames distinguishable.
            column = int((self._x * 4) % cam.width)
            rgb[:, column] = (255, 0, 0)
            captures.append(
                CameraCapture(
                    logical_id=cam.logical_id,
                    frame_start_us=timestamp_us,
                    frame_end_us=timestamp_us,
                    image_bytes=encode_rgb(
                        rgb, self.config.image_format, self.config.image_quality
                    ),
                )
            )
        return captures
