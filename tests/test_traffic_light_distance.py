# SPDX-License-Identifier: Apache-2.0
"""The traffic-light distance reported to the policy.

Exercised without a CARLA server by calling the method against stub objects,
because the property under test is geometric and the simulator contributes
nothing to it.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from carla_driver_interface.geometry import Pose
from carla_driver_interface.runtime.carla_world import CarlaWorldAdapter
from carla_driver_interface.runtime.world import EgoState


def ego_at(x: float, y: float, yaw_rad: float = 0.0) -> EgoState:
    return EgoState(
        timestamp_us=0,
        pose_local_to_rig=Pose.from_xyz_yaw(x, y, 0.0, yaw_rad),
        linear_velocity_in_rig=np.zeros(3),
        angular_velocity_in_rig=np.zeros(3),
        linear_acceleration_in_rig=np.zeros(3),
    )


def light_with_stop_lines(*points_in_local: tuple[float, float]) -> SimpleNamespace:
    """A stub traffic light whose stop waypoints sit at the given points.

    ``carla_vector_to_local`` flips the y axis (CARLA is left-handed), so the
    stub hands back CARLA-frame coordinates that map to the points requested.
    """
    waypoints = [
        SimpleNamespace(transform=SimpleNamespace(location=SimpleNamespace(x=x, y=-y, z=0.0)))
        for x, y in points_in_local
    ]
    return SimpleNamespace(get_stop_waypoints=lambda: waypoints)


class _Geometry:
    """Just the two methods under test, borrowed off the adapter.

    Neither touches the simulator, so there is no reason to stand up a real
    :class:`CarlaWorldAdapter` -- which would need a CARLA server -- to
    exercise a purely geometric property.
    """

    _waypoint_to_local = CarlaWorldAdapter._waypoint_to_local
    _traffic_light_distance = CarlaWorldAdapter._traffic_light_distance


def distance(light, ego: EgoState) -> float:
    return _Geometry()._traffic_light_distance(light, ego)


def test_no_light_reports_a_negative_distance():
    assert distance(None, ego_at(0.0, 0.0)) == -1.0


def test_a_line_ahead_reports_its_longitudinal_distance():
    assert distance(light_with_stop_lines((12.0, 0.0)), ego_at(0.0, 0.0)) == pytest.approx(12.0)


def test_a_line_behind_reports_a_negative_distance():
    """The property the whole change exists for.

    A Euclidean distance cannot express this: it reports the same +3 m whether
    the line is three metres ahead or three metres behind. A policy stopped
    just past the line is then told forever that it still has a line to stop
    for, and never moves again -- a deadlock reachable only by stopping
    correctly.
    """
    assert distance(light_with_stop_lines((-3.0, 0.0)), ego_at(0.0, 0.0)) == pytest.approx(-3.0)


def test_the_distance_decreases_monotonically_through_the_line():
    """It must cross zero once, rather than rebounding."""
    light = light_with_stop_lines((10.0, 0.0))
    values = [distance(light, ego_at(x, 0.0)) for x in np.arange(0.0, 20.0, 1.0)]
    assert all(later < earlier for earlier, later in zip(values, values[1:], strict=False))
    assert values[0] > 0.0
    assert values[-1] < 0.0


def test_it_is_measured_along_the_heading_not_in_a_straight_line():
    """A line off to the side is not as close as its straight-line distance."""
    # 10 m ahead and 6 m to the left; straight-line 11.66 m, longitudinal 10 m.
    light = light_with_stop_lines((10.0, 6.0))
    assert distance(light, ego_at(0.0, 0.0)) == pytest.approx(10.0)


def test_it_respects_the_ego_heading():
    light = light_with_stop_lines((0.0, 8.0))
    # Facing north, the line 8 m to the north is 8 m ahead.
    assert distance(light, ego_at(0.0, 0.0, math.pi / 2)) == pytest.approx(8.0)
    # Facing south, the same line is 8 m behind.
    assert distance(light, ego_at(0.0, 0.0, -math.pi / 2)) == pytest.approx(-8.0)


def test_the_nearest_line_still_ahead_governs():
    """With several stop lines, the one we have yet to cross is the one that counts."""
    light = light_with_stop_lines((-4.0, 0.0), (6.0, 0.0), (14.0, 0.0))
    assert distance(light, ego_at(0.0, 0.0)) == pytest.approx(6.0)


def test_once_every_line_is_behind_the_nearest_of_them_is_reported():
    """Keeps the value continuous as the last line is crossed."""
    light = light_with_stop_lines((-4.0, 0.0), (-11.0, 0.0))
    assert distance(light, ego_at(0.0, 0.0)) == pytest.approx(-4.0)


def test_a_light_without_stop_waypoints_falls_back_to_its_own_position():
    """Some maps have lights with no stop waypoints registered."""
    light = SimpleNamespace(
        get_stop_waypoints=lambda: [],
        get_transform=lambda: SimpleNamespace(location=SimpleNamespace(x=9.0, y=0.0, z=0.0)),
    )
    assert distance(light, ego_at(0.0, 0.0)) == pytest.approx(9.0)
