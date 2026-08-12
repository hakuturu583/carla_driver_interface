# SPDX-License-Identifier: Apache-2.0
"""Reference policy: follow the submitted route.

This exists so the closed loop is demonstrable end to end without a neural
network.  It uses only observations the alpasim contract guarantees (the route
from ``submit_route`` and the ego state from ``submit_egomotion_observation``),
and *optionally* sharpens its behaviour with the CARLA extension payload
(speed limit, traffic light) when the runtime is this project's.  Under an
upstream alpasim runtime the extension is simply absent and the policy still
drives.
"""

from __future__ import annotations

import math

import numpy as np

from carla_driver_interface import polyline
from carla_driver_interface.driver.base import BaseDriver, DriveContext, DriveResult
from carla_driver_interface.geometry import Pose, Trajectory
from carla_driver_interface.grpc_api import CarlaRendererData, TrafficLightState

__all__ = ["RouteFollowerPolicy"]

#: Used when there is no route to follow: the plan runs straight ahead in the
#: rig frame, which is what "hold this heading" means to the controller.
_STRAIGHT_AHEAD = np.array([[0.0, 0.0, 0.0], [1.0e4, 0.0, 0.0]])


class RouteFollowerPolicy(BaseDriver):
    """Track the route centreline, easing off for curves and red lights."""

    name = "route_follower"

    def __init__(
        self,
        cruise_speed_mps: float = 8.0,
        horizon_s: float = 4.0,
        step_s: float = 0.1,
        max_lateral_accel: float = 2.5,
        max_accel_mps2: float = 2.0,
        max_decel_mps2: float = 4.0,
        stop_line_margin_m: float = 4.0,
    ) -> None:
        self.cruise_speed_mps = cruise_speed_mps
        self.horizon_s = horizon_s
        self.step_s = step_s
        self.max_lateral_accel = max_lateral_accel
        self.max_accel_mps2 = max_accel_mps2
        self.max_decel_mps2 = max_decel_mps2
        self.stop_line_margin_m = stop_line_margin_m

    # -- main --------------------------------------------------------------

    def drive(self, ctx: DriveContext) -> DriveResult:
        session = ctx.session
        with session.lock:
            route = np.asarray(session.route_waypoints_in_rig, dtype=np.float64)
        current_speed = session.speed_mps()

        if len(route) < 2:
            # No route yet: hold position rather than guess a direction.
            return DriveResult(
                trajectory_in_rig=self._plan(ctx, _STRAIGHT_AHEAD, current_speed, 0.0),
                debug_scalars={"reason_no_route": 1.0, "current_speed_mps": current_speed},
            )

        arc = polyline.arc_lengths(route)
        target_speed = self._target_speed(ctx, route, arc)
        return DriveResult(
            trajectory_in_rig=self._plan(ctx, route, current_speed, target_speed, arc),
            debug_scalars={
                "current_speed_mps": current_speed,
                "target_speed_mps": target_speed,
                "route_length_m": float(arc[-1]),
            },
        )

    # -- speed -------------------------------------------------------------

    def _target_speed(self, ctx: DriveContext, route: np.ndarray, arc: np.ndarray) -> float:
        target = self.cruise_speed_mps

        data = ctx.renderer_data
        if data is not None:
            if data.speed_limit_mps > 0.0:
                target = min(target, float(data.speed_limit_mps))
            target = min(target, self._traffic_light_speed(data))

        target = min(target, self._curvature_speed(route, arc))
        target = min(target, self._end_of_route_speed(arc))
        return max(0.0, target)

    def _end_of_route_speed(self, arc: np.ndarray) -> float:
        """Come to a stop at the end of the route rather than extrapolating past it.

        The runtime only publishes the route ahead, so a short polyline means
        either the horizon is short or the route is ending. Braking for it is
        right either way, and it stops the plan from running off the end of the
        polyline into an extrapolated heading.
        """
        remaining = float(arc[-1])
        if remaining >= self.cruise_speed_mps * self.horizon_s:
            return math.inf
        return math.sqrt(2.0 * self.max_decel_mps2 * max(0.0, remaining))

    def _traffic_light_speed(self, data: CarlaRendererData) -> float:
        """Speed cap implied by the light governing the ego lane."""
        stopping = data.ego_traffic_light in (
            TrafficLightState.TRAFFIC_LIGHT_STATE_RED,
            TrafficLightState.TRAFFIC_LIGHT_STATE_YELLOW,
        )
        if not stopping or data.ego_traffic_light_distance_m < 0.0:
            return math.inf

        distance = float(data.ego_traffic_light_distance_m) - self.stop_line_margin_m
        if distance <= 0.0:
            return 0.0
        # Fastest speed from which we can still stop in `distance` at max decel.
        return math.sqrt(2.0 * self.max_decel_mps2 * distance)

    def _curvature_speed(self, route: np.ndarray, arc: np.ndarray) -> float:
        """Cap speed so lateral acceleration stays within budget on the route ahead.

        Only the stretch we could actually reach this horizon matters, so
        curvature far down the road does not slow us down now.
        """
        lookahead_m = max(5.0, self.cruise_speed_mps * self.horizon_s)
        window = route[arc <= lookahead_m]
        curvature = polyline.max_curvature(window)
        if curvature <= 1e-6:
            return math.inf
        return math.sqrt(self.max_lateral_accel / curvature)

    # -- plan --------------------------------------------------------------

    def _plan(
        self,
        ctx: DriveContext,
        route: np.ndarray,
        current_speed: float,
        target_speed: float,
        arc: np.ndarray | None = None,
    ) -> Trajectory:
        """Integrate a speed profile along ``route`` and stamp it as a plan.

        The no-route case is the same integration over a straight line, so it
        shares this method rather than duplicating the loop.
        """
        arc = polyline.arc_lengths(route) if arc is None else arc
        n_steps = max(1, int(round(self.horizon_s / self.step_s)))

        # Integrate the speed profile first, so the polyline can be sampled for
        # every waypoint in one vectorised call.
        speed = current_speed
        travelled = 0.0
        distances = np.empty(n_steps, dtype=np.float64)
        for i in range(n_steps):
            speed = _approach(
                speed, target_speed, self.max_accel_mps2, self.max_decel_mps2, self.step_s
            )
            travelled += speed * self.step_s
            distances[i] = travelled

        positions = polyline.sample(route, arc, distances, extrapolate=True)
        step_us = int(round(self.step_s * 1e6))

        plan = Trajectory.empty()
        for i, position in enumerate(positions, start=1):
            heading = _heading_at(route, arc, distances[i - 1])
            plan.append(
                ctx.time_now_us + i * step_us,
                Pose.from_xyz_yaw(position[0], position[1], position[2], heading),
            )
        return plan


def _heading_at(route: np.ndarray, arc: np.ndarray, distance: float) -> float:
    """Heading of the route segment containing ``distance``."""
    idx = int(np.clip(np.searchsorted(arc, distance), 1, len(route) - 1))
    return polyline.segment_heading(route[idx - 1], route[idx])


def _approach(
    speed: float, target: float, max_accel: float, max_decel: float, dt_s: float
) -> float:
    """Move ``speed`` towards ``target`` respecting acceleration limits."""
    return float(np.clip(target, speed - max_decel * dt_s, speed + max_accel * dt_s))
