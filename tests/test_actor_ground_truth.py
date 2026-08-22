# SPDX-License-Identifier: Apache-2.0
"""Which actors reach the driver as ground truth.

``CarlaRendererData.actors`` is the only account a policy gets of anything
moving that is not the ego -- the alpasim contract has no field for it -- so
whatever is left out of here is invisible to every rule written about clearance.
A driver reading vehicles alone satisfies "collision free" while walking through
a pedestrian crossing, and nothing in the payload says the pedestrian was
omitted rather than absent.

Exercised against stubs, like the traffic-light tests next door: the property is
about which blueprints are selected and how their transforms are mirrored, and
the simulator contributes nothing to either.
"""

from __future__ import annotations

import fnmatch
from types import SimpleNamespace
from typing import Any

import numpy as np

from carla_driver_interface.geometry import Pose
from carla_driver_interface.runtime.ground_truth import CarlaGroundTruth


def actor(
    actor_id: int,
    type_id: str,
    x: float = 10.0,
    y: float = 0.0,
    velocity: tuple[float, float] = (0.0, 0.0),
    extent: tuple[float, float, float] = (2.4, 1.0, 0.75),
) -> SimpleNamespace:
    """A CARLA actor with just enough surface for the ground-truth walk."""
    return SimpleNamespace(
        id=actor_id,
        type_id=type_id,
        get_transform=lambda: SimpleNamespace(
            location=SimpleNamespace(x=x, y=y, z=0.0),
            rotation=SimpleNamespace(pitch=0.0, yaw=0.0, roll=0.0),
        ),
        get_velocity=lambda: SimpleNamespace(x=velocity[0], y=velocity[1], z=0.0),
        bounding_box=SimpleNamespace(extent=SimpleNamespace(x=extent[0], y=extent[1], z=extent[2])),
    )


class _ActorList:
    """CARLA's ActorList, which matches blueprint ids by shell pattern."""

    def __init__(self, actors: list[Any]) -> None:
        self._actors = actors

    def filter(self, pattern: str) -> list[Any]:
        return [a for a in self._actors if fnmatch.fnmatch(a.type_id, pattern)]


def ground(actors: list[Any], ego_id: int = 1, horizon_m: float = 150.0) -> CarlaGroundTruth:
    """A reader over a stub world holding *actors*."""
    return CarlaGroundTruth(
        world=SimpleNamespace(get_actors=lambda: _ActorList(actors)),
        ego=SimpleNamespace(id=ego_id),
        carla_map=None,
        config=SimpleNamespace(actor_horizon_m=horizon_m),
        map_name="Stub",
    )


def ego_state(x: float = 0.0, y: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(pose_local_to_rig=Pose.from_xyz_yaw(x, y, 0.0, 0.0))


def states_for(
    actors: list[Any], ego_id: int = 1, ego_x: float = 0.0, horizon_m: float = 150.0
) -> list[Any]:
    return ground(actors, ego_id=ego_id, horizon_m=horizon_m)._actor_states(ego_state(ego_x))


def test_vehicles_are_reported() -> None:
    reported = states_for([actor(1, "vehicle.ego"), actor(2, "vehicle.tesla.model3")])
    assert [s.track_id for s in reported] == ["2"]


def test_pedestrians_are_reported() -> None:
    """A rule about clearance means clearance from people too."""
    reported = states_for([actor(1, "vehicle.ego"), actor(9, "walker.pedestrian.0007")])
    assert [s.track_id for s in reported] == ["9"]
    assert reported[0].type_id == "walker.pedestrian.0007"


def test_vehicles_and_pedestrians_are_reported_together() -> None:
    reported = states_for(
        [
            actor(1, "vehicle.ego"),
            actor(2, "vehicle.tesla.model3"),
            actor(9, "walker.pedestrian.0007"),
        ]
    )
    assert sorted(s.track_id for s in reported) == ["2", "9"]


def test_walker_controllers_are_not_reported() -> None:
    """They steer a pedestrian; they have no body to avoid.

    CARLA parks them at the origin, so reporting one would put a zero-extent
    obstacle wherever that happens to be.
    """
    reported = states_for(
        [actor(1, "vehicle.ego"), actor(5, "controller.ai.walker", extent=(0.0, 0.0, 0.0))]
    )
    assert reported == []


def test_traffic_lights_are_not_reported_as_actors() -> None:
    reported = states_for([actor(1, "vehicle.ego"), actor(3, "traffic.traffic_light")])
    assert reported == []


def test_nothing_is_reported_twice() -> None:
    """The two blueprint namespaces are disjoint; a change that overlaps them
    would silently double every obstacle."""
    reported = states_for(
        [actor(1, "vehicle.ego"), actor(2, "vehicle.audi.tt"), actor(9, "walker.pedestrian.1")]
    )
    assert len(reported) == len({s.track_id for s in reported})


def test_the_ego_is_not_reported_to_itself() -> None:
    reported = states_for([actor(1, "vehicle.ego"), actor(2, "vehicle.audi.tt")], ego_id=1)
    assert [s.track_id for s in reported] == ["2"]


def test_actors_beyond_the_horizon_are_dropped() -> None:
    reported = states_for([actor(1, "vehicle.ego"), actor(2, "vehicle.audi.tt", x=400.0)])
    assert reported == []


def test_the_horizon_is_configurable() -> None:
    """Narrowing it is how a dense map bounds the payload -- and what it costs."""
    far = [actor(1, "vehicle.ego"), actor(2, "vehicle.audi.tt", x=80.0)]
    assert [s.track_id for s in states_for(far, horizon_m=150.0)] == ["2"]
    assert states_for(far, horizon_m=50.0) == []


def test_an_actor_exactly_at_the_horizon_is_kept() -> None:
    """The bound is exclusive, so the edge case does not flicker with rounding."""
    edge = [actor(1, "vehicle.ego"), actor(2, "vehicle.audi.tt", x=50.0)]
    assert [s.track_id for s in states_for(edge, horizon_m=50.0)] == ["2"]


def test_pose_is_mirrored_into_the_local_frame() -> None:
    """CARLA's y runs south; the local frame's runs north."""
    reported = states_for([actor(1, "vehicle.ego"), actor(2, "vehicle.audi.tt", x=20.0, y=4.0)])
    assert reported[0].pose_local_to_aabb.vec.x == 20.0
    assert reported[0].pose_local_to_aabb.vec.y == -4.0


def test_velocity_is_mirrored_too() -> None:
    reported = states_for(
        [actor(1, "vehicle.ego"), actor(2, "vehicle.audi.tt", velocity=(6.0, 3.0))]
    )
    velocity = reported[0].dynamic_state.linear_velocity
    assert velocity.x == 6.0
    assert velocity.y == -3.0


def test_aabb_carries_the_full_size_not_the_half_extent() -> None:
    reported = states_for(
        [actor(1, "vehicle.ego"), actor(2, "vehicle.audi.tt", extent=(2.4, 1.0, 0.75))]
    )
    aabb = reported[0].aabb
    assert np.allclose([aabb.size_x, aabb.size_y, aabb.size_z], [4.8, 2.0, 1.5])
