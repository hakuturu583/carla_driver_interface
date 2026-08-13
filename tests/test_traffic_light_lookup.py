# SPDX-License-Identifier: Apache-2.0
"""Which traffic light is reported as governing the ego.

CARLA's own answer -- ``is_at_traffic_light()`` -- is true only inside the
light's trigger volume, and those are about a metre thick along the road:
measured across the fifteen lights of Town10HD_Opt they reach a median of
0.55 m back from the stop line. A policy told about a red light at the moment
it arrives at the line cannot stop for it from any ordinary speed, and the
overrun that follows is this interface's, not the policy's.

So the lane graph is walked forward instead. Exercised against stubs, like the
distance tests next door: the property is about lane bookkeeping and the
simulator contributes nothing to it.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from carla_driver_interface.runtime.carla_world import CarlaWorldAdapter


def waypoint(
    road_id: int,
    lane_id: int,
    following: list[Any] | None = None,
    is_junction: bool = False,
) -> SimpleNamespace:
    """A lane waypoint that knows what comes after it."""
    node = SimpleNamespace(road_id=road_id, lane_id=lane_id, is_junction=is_junction)
    node.next = lambda _step, node=node: getattr(node, "_following", []) or []
    node._following = following or []
    return node


def lane(
    road_id: int, lane_id: int, length: int, is_junction: bool = False
) -> list[SimpleNamespace]:
    """A run of waypoints along one lane, chained front to back."""
    nodes = [waypoint(road_id, lane_id, is_junction=is_junction) for _ in range(length)]
    for earlier, later in zip(nodes, nodes[1:], strict=False):
        earlier._following = [later]
    return nodes


def light_on(*lanes: tuple[int, int]) -> SimpleNamespace:
    return SimpleNamespace(
        get_stop_waypoints=lambda: [waypoint(road, lane_id) for road, lane_id in lanes]
    )


class _Lookup:
    """The lookup methods, borrowed off the adapter and given stubs to walk."""

    _governing_traffic_light = CarlaWorldAdapter._governing_traffic_light
    _lanes_ahead = CarlaWorldAdapter._lanes_ahead
    _lights_by_lane_ahead = CarlaWorldAdapter._lights_by_lane_ahead
    _stop_lines_by_lane = CarlaWorldAdapter._stop_lines_by_lane

    def __init__(
        self,
        path: list[SimpleNamespace],
        lights: list[Any],
        sight_distance_m: float = 60.0,
        at_light: Any = None,
    ) -> None:
        self.config = SimpleNamespace(
            traffic_light_sight_distance_m=sight_distance_m,
            route_resolution_m=2.0,
        )
        self._stop_lines = None
        self._map = SimpleNamespace(get_waypoint=lambda *_a, **_k: path[0] if path else None)
        self._world = SimpleNamespace(
            get_actors=lambda: SimpleNamespace(filter=lambda _pattern: lights)
        )
        self._ego = SimpleNamespace(
            get_transform=lambda: SimpleNamespace(location=SimpleNamespace(x=0.0, y=0.0, z=0.0)),
            is_at_traffic_light=lambda: at_light is not None,
            get_traffic_light=lambda: at_light,
        )


def test_a_light_down_the_road_is_reported_before_its_trigger_volume():
    """The whole point: seen from where a driver would have seen it.

    The ego is on road 1 and the stop line is on road 2, several lanes'
    walking away -- far outside any trigger volume, and the only distance from
    which stopping is possible at all.
    """
    path = lane(1, -1, 20)
    path[-1]._following = lane(2, -1, 20)
    governing = light_on((2, -1))
    assert _Lookup(path, [governing])._governing_traffic_light() is governing


def test_a_light_on_another_lane_does_not_govern_us():
    """Cross traffic has stop lines too, and they are not ours."""
    path = lane(1, -1, 20)
    assert _Lookup(path, [light_on((7, 3))])._governing_traffic_light() is None


def test_the_first_light_along_the_lane_governs():
    """Two junctions ahead, the near one is the one to stop for."""
    path = lane(1, -1, 5)
    path[-1]._following = lane(2, -1, 5)
    path[-1]._following[-1]._following = lane(3, -1, 5)
    near, far = light_on((2, -1)), light_on((3, -1))
    assert _Lookup(path, [far, near])._governing_traffic_light() is near


def test_the_walk_stops_at_the_sight_distance():
    """A light beyond it is not yet our business, and reporting it would make
    the policy brake for something two junctions away."""
    path = lane(1, -1, 3)
    path[-1]._following = lane(2, -1, 3)
    lookup = _Lookup(path, [light_on((2, -1))], sight_distance_m=2.0)
    assert lookup._governing_traffic_light() is None


def test_zero_sight_distance_restores_carla_s_own_answer():
    """The escape hatch, so a caller can have the previous behaviour exactly."""
    at_light = light_on((9, 9))
    lookup = _Lookup(lane(1, -1, 5), [light_on((1, -1))], sight_distance_m=0.0, at_light=at_light)
    assert lookup._governing_traffic_light() is at_light


def test_standing_in_a_trigger_volume_still_counts_when_the_walk_finds_nothing():
    """A light on a lane the graph does not reach still governs us.

    The walk takes one branch at a junction, so it can miss a stop line the
    ego is nonetheless sitting at -- and being *in* the volume is the one
    unambiguous piece of evidence CARLA offers.
    """
    at_light = light_on((9, 9))
    lookup = _Lookup(lane(1, -1, 5), [light_on((4, 2))], at_light=at_light)
    assert lookup._governing_traffic_light() is at_light


def test_nothing_governs_a_vehicle_already_inside_the_junction():
    """Having crossed the line, the thing to do is clear the box.

    That is the law, and it is also what stops a policy from finding a reason
    to wait where waiting is worst. Reported from inside a junction, the next
    junction's light -- 70-odd metres away, nothing to brake for -- was enough
    for a policy gating "am I free to move" on whether a light applies to
    stand still in the middle of the box for seventeen seconds.
    """
    inside = lane(2, 1, 5, is_junction=True)
    inside[-1]._following = lane(3, -1, 20)
    lookup = _Lookup(inside, [light_on((3, -1))])
    assert lookup._governing_traffic_light() is None


def test_the_walk_does_not_reach_past_the_junction_it_arrives_at():
    """The light beyond belongs to a junction we have yet to arrive at.

    Walking through was also what made the answer flicker: the lane the ego
    projects onto inside a junction is ambiguous, so consecutive steps took
    different branches and reported different lights -- 76 m away one step,
    3.9 m the next.
    """
    approach = lane(1, -1, 4)
    junction = lane(2, 1, 3, is_junction=True)
    approach[-1]._following = junction
    junction[-1]._following = lane(3, -1, 20)

    near, beyond = light_on((1, -1)), light_on((3, -1))
    assert _Lookup(approach, [beyond, near])._governing_traffic_light() is near
    # And with only the far one to find, nothing is reported at all.
    assert _Lookup(approach, [beyond])._governing_traffic_light() is None


def test_a_stop_line_on_the_junction_lane_itself_still_counts():
    """Some maps register the line on the first junction waypoint."""
    approach = lane(1, -1, 4)
    approach[-1]._following = lane(2, 1, 3, is_junction=True)
    governing = light_on((2, 1))
    assert _Lookup(approach, [governing])._governing_traffic_light() is governing


def test_a_map_without_a_lane_under_the_ego_reports_nothing():
    assert _Lookup([], [light_on((1, -1))])._governing_traffic_light() is None
