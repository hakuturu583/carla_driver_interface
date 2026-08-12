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

from carla_driver_interface.driver.base import BaseDriver, DriveContext, DriveResult
from carla_driver_interface.geometry import Pose, Trajectory
from carla_driver_interface.grpc_api import TrafficLightState

__all__ = ["RouteFollowerPolicy"]


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
                trajectory_in_rig=self._straight_plan(ctx, 0.0, current_speed),
                debug_scalars={"reason_no_route": 1.0, "current_speed_mps": current_speed},
            )

        target_speed = self._target_speed(ctx, route, current_speed)
        plan = self._plan_along_route(ctx, route, current_speed, target_speed)
        return DriveResult(
            trajectory_in_rig=plan,
            debug_scalars={
                "current_speed_mps": current_speed,
                "target_speed_mps": target_speed,
                "route_length_m": float(_arc_lengths(route)[-1]),
            },
        )

    # -- speed -------------------------------------------------------------

    def _target_speed(self, ctx: DriveContext, route: np.ndarray, current_speed: float) -> float:
        target = self.cruise_speed_mps

        data = ctx.renderer_data
        if data is not None:
            if data.speed_limit_mps > 0.0:
                target = min(target, float(data.speed_limit_mps))
            target = min(target, self._traffic_light_speed(data, current_speed))

        target = min(target, self._curvature_speed(route))
        target = min(target, self._end_of_route_speed(route))
        return max(0.0, target)

    def _end_of_route_speed(self, route: np.ndarray) -> float:
        """Come to a stop at the end of the route rather than extrapolating past it.

        The runtime only publishes the route ahead, so a short polyline means
        either the horizon is short or the route is ending. Braking for it is
        right either way, and it stops the plan from running off the end of the
        polyline into an extrapolated heading.
        """
        remaining = float(_arc_lengths(route)[-1])
        if remaining >= self.cruise_speed_mps * self.horizon_s:
            return math.inf
        return math.sqrt(2.0 * self.max_decel_mps2 * max(0.0, remaining))

    def _traffic_light_speed(self, data, current_speed: float) -> float:
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

    def _curvature_speed(self, route: np.ndarray) -> float:
        """Cap speed so lateral acceleration stays within budget on the route ahead.

        Only the stretch we could actually reach this horizon matters, so
        curvature far down the road does not slow us down now.
        """
        lookahead_m = max(5.0, self.cruise_speed_mps * self.horizon_s)
        arc = _arc_lengths(route)
        window = route[arc <= lookahead_m]
        if len(window) < 3:
            return math.inf

        curvature = _max_curvature(window)
        if curvature <= 1e-6:
            return math.inf
        return math.sqrt(self.max_lateral_accel / curvature)

    # -- plan --------------------------------------------------------------

    def _plan_along_route(
        self,
        ctx: DriveContext,
        route: np.ndarray,
        current_speed: float,
        target_speed: float,
    ) -> Trajectory:
        arc = _arc_lengths(route)
        plan = Trajectory.empty()

        speed = current_speed
        travelled = 0.0
        n_steps = max(1, int(round(self.horizon_s / self.step_s)))
        for i in range(1, n_steps + 1):
            speed = _approach(
                speed, target_speed, self.max_accel_mps2, self.max_decel_mps2, self.step_s
            )
            travelled += speed * self.step_s
            position, heading = _sample_polyline(route, arc, travelled)
            plan.append(
                ctx.time_now_us + int(round(i * self.step_s * 1e6)),
                Pose.from_xyz_yaw(position[0], position[1], position[2], heading),
            )
        return plan

    def _straight_plan(
        self, ctx: DriveContext, target_speed: float, current_speed: float
    ) -> Trajectory:
        plan = Trajectory.empty()
        speed = current_speed
        travelled = 0.0
        n_steps = max(1, int(round(self.horizon_s / self.step_s)))
        for i in range(1, n_steps + 1):
            speed = _approach(
                speed, target_speed, self.max_accel_mps2, self.max_decel_mps2, self.step_s
            )
            travelled += speed * self.step_s
            plan.append(
                ctx.time_now_us + int(round(i * self.step_s * 1e6)),
                Pose.from_xyz_yaw(travelled, 0.0, 0.0, 0.0),
            )
        return plan


# ---------------------------------------------------------------------------
# Polyline helpers
# ---------------------------------------------------------------------------


def _arc_lengths(points: np.ndarray) -> np.ndarray:
    """Cumulative arc length along ``points``, starting at 0."""
    if len(points) == 0:
        return np.zeros(0, dtype=np.float64)
    deltas = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(deltas)])


def _sample_polyline(
    points: np.ndarray, arc: np.ndarray, distance: float
) -> tuple[np.ndarray, float]:
    """Point and heading at ``distance`` along the polyline.

    Past the end, extrapolate along the final segment so a short route does not
    make the plan collapse to a stop.
    """
    total = float(arc[-1])
    if distance <= 0.0:
        return points[0], _segment_heading(points[0], points[1])
    if distance >= total:
        heading = _segment_heading(points[-2], points[-1])
        overshoot = distance - total
        direction = np.array([math.cos(heading), math.sin(heading), 0.0])
        return points[-1] + overshoot * direction, heading

    idx = int(np.searchsorted(arc, distance))
    p0, p1 = points[idx - 1], points[idx]
    span = arc[idx] - arc[idx - 1]
    alpha = 0.0 if span <= 0.0 else (distance - arc[idx - 1]) / span
    return p0 + alpha * (p1 - p0), _segment_heading(p0, p1)


def _segment_heading(p0: np.ndarray, p1: np.ndarray) -> float:
    return math.atan2(float(p1[1] - p0[1]), float(p1[0] - p0[0]))


def _max_curvature(points: np.ndarray) -> float:
    """Largest Menger curvature over consecutive point triples."""
    worst = 0.0
    for a, b, c in zip(points[:-2], points[1:-1], points[2:], strict=True):
        ab = float(np.linalg.norm(b[:2] - a[:2]))
        bc = float(np.linalg.norm(c[:2] - b[:2]))
        ca = float(np.linalg.norm(a[:2] - c[:2]))
        if ab < 1e-6 or bc < 1e-6 or ca < 1e-6:
            continue
        # Twice the triangle area via the 2-D cross product.
        cross = float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        worst = max(worst, abs(2.0 * cross) / (ab * bc * ca))
    return worst


def _approach(
    speed: float, target: float, max_accel: float, max_decel: float, dt_s: float
) -> float:
    """Move ``speed`` towards ``target`` respecting acceleration limits."""
    if target > speed:
        return min(target, speed + max_accel * dt_s)
    return max(target, speed - max_decel * dt_s)
