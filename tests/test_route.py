# SPDX-License-Identifier: Apache-2.0
"""Route slicing and the progress marker.

``waypoints_in_rig`` is the only method allowed to advance progress. That is not
a style rule: the forward search is bounded, so it does not converge in one
call, and a second caller on the same pose can move the marker again -- which
shows up as a rollout that thinks it finished the route early.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from carla_driver_interface.geometry import Pose
from carla_driver_interface.runtime.route import RouteProvider

STRAIGHT = np.stack([np.arange(0.0, 201.0, 1.0), np.zeros(201), np.zeros(201)], axis=1)


def at(x: float, y: float = 0.0, yaw: float = 0.0) -> Pose:
    return Pose.from_xyz_yaw(x, y, 0.0, yaw)


def test_total_length():
    assert RouteProvider(STRAIGHT).total_length_m == pytest.approx(200.0)


def test_a_route_needs_two_points():
    with pytest.raises(ValueError, match="at least two points"):
        RouteProvider(np.zeros((1, 3)))


def test_waypoints_are_expressed_in_the_rig_frame():
    """Facing +y, a route running +x must come back as running -y in the rig."""
    route = RouteProvider(STRAIGHT, horizon_m=10.0, resolution_m=2.0)
    waypoints = route.waypoints_in_rig(at(0.0, yaw=math.pi / 2))
    # Rig x is forward (which is local +y here), rig y is left (local -x).
    assert np.allclose(waypoints[:, 0], 0.0, atol=1e-9)
    assert waypoints[-1][1] < -5.0


def test_waypoints_end_on_the_exact_route_endpoint():
    """Without this the last stride is truncated and completion never reaches 1."""
    route = RouteProvider(STRAIGHT, horizon_m=500.0, resolution_m=3.0)
    waypoints = route.waypoints_in_rig(at(0.0))
    assert waypoints[-1][0] == pytest.approx(200.0)


def test_completion_reaches_one_at_the_end_of_the_route():
    route = RouteProvider(STRAIGHT, horizon_m=80.0, resolution_m=2.0)
    for x in range(0, 201, 5):
        route.waypoints_in_rig(at(float(x)))
    assert route.completion == pytest.approx(1.0)


def test_completion_is_monotonic_and_bounded():
    route = RouteProvider(STRAIGHT)
    seen = []
    for x in range(0, 201, 10):
        route.waypoints_in_rig(at(float(x)))
        seen.append(route.completion)
    assert seen == sorted(seen)
    assert 0.0 <= seen[0] and seen[-1] <= 1.0


def test_lateral_error_does_not_advance_progress():
    """The regression this method was rewritten for."""
    route = RouteProvider(STRAIGHT, horizon_m=80.0, resolution_m=2.0)
    route.waypoints_in_rig(at(100.0))
    before = route.completion

    for _ in range(5):
        route.lateral_error_m(at(100.0))

    assert route.completion == before


def test_repeated_waypoint_calls_are_what_would_advance_the_marker():
    """Documents *why* only one caller may advance it.

    The forward search window is bounded, so from a standing start far along the
    route the marker needs several calls to catch up. Any second consumer of
    ``waypoints_in_rig`` in the same step would therefore double-step it.
    """
    route = RouteProvider(STRAIGHT, horizon_m=80.0, resolution_m=2.0)
    pose = at(180.0)
    route.waypoints_in_rig(pose)
    first = route.completion
    route.waypoints_in_rig(pose)
    assert route.completion > first


def test_lateral_error_sign_is_positive_to_the_left():
    route = RouteProvider(STRAIGHT)
    # Ego 2 m to the right of the route (local -y), facing along it.
    assert route.lateral_error_m(at(50.0, y=-2.0)) == pytest.approx(2.0, abs=0.2)
    # Ego 2 m to the left.
    assert route.lateral_error_m(at(50.0, y=2.0)) == pytest.approx(-2.0, abs=0.2)


def test_progress_does_not_jump_backwards_onto_an_earlier_lap():
    """A loop route must not re-match the ego onto where it started."""
    thetas = np.linspace(0.0, 2 * math.pi, 400)
    radius = 30.0
    loop = np.stack(
        [radius * np.cos(thetas), radius * np.sin(thetas), np.zeros_like(thetas)], axis=1
    )
    route = RouteProvider(loop, horizon_m=40.0, resolution_m=2.0)

    seen = []
    for theta in np.linspace(0.0, 2 * math.pi, 120):
        route.waypoints_in_rig(at(radius * math.cos(theta), radius * math.sin(theta)))
        seen.append(route.completion)

    assert seen == sorted(seen)
    assert seen[-1] > 0.9


def test_a_short_remaining_route_still_publishes_two_points():
    """The policy needs a direction even on the last metre."""
    route = RouteProvider(STRAIGHT, horizon_m=80.0, resolution_m=2.0)
    for x in range(0, 201, 5):
        waypoints = route.waypoints_in_rig(at(float(x)))
    assert len(waypoints) >= 2
