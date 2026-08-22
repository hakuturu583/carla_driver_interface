# SPDX-License-Identifier: Apache-2.0
"""What a ground-truth reader may ask its caller for.

Reading CARLA ground truth needs four settings, none of which describes a
simulator connection. Taking a whole ``RuntimeConfig`` said otherwise, and said
it in the one signature a runtime that owns its own world has to call: such a
runtime has no client of ours to point at, no driver address to forward, and no
traffic manager whose port would mean anything here.

So the reader asks for a protocol instead. ``RuntimeConfig`` satisfies it
structurally -- the tests below pin that, since nothing else would notice if it
stopped -- and so does anything else carrying the four.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from carla_driver_interface.runtime.config import RuntimeConfig
from carla_driver_interface.runtime.ground_truth import CarlaGroundTruth, GroundTruthConfig


@dataclass(frozen=True)
class _OwnConfig:
    """A runtime's own settings: the four, and nothing about CARLA."""

    traffic_light_sight_distance_m: float = 60.0
    route_resolution_m: float = 2.0
    actor_horizon_m: float = 150.0
    send_actor_ground_truth: bool = True


def _vehicle(actor_id: int, x: float) -> SimpleNamespace:
    return SimpleNamespace(
        id=actor_id,
        type_id="vehicle.audi.tt",
        get_transform=lambda: SimpleNamespace(
            location=SimpleNamespace(x=x, y=0.0, z=0.0),
            rotation=SimpleNamespace(pitch=0.0, yaw=0.0, roll=0.0),
        ),
        get_velocity=lambda: SimpleNamespace(x=0.0, y=0.0, z=0.0),
        bounding_box=SimpleNamespace(extent=SimpleNamespace(x=2.4, y=1.0, z=0.75)),
    )


class _ActorList:
    """CARLA's ActorList: matches blueprint ids by pattern, once per pattern."""

    def __init__(self, actors: list[Any]) -> None:
        self._actors = actors

    def filter(self, pattern: str) -> list[Any]:
        return [a for a in self._actors if fnmatch.fnmatch(a.type_id, pattern)]


def _reader(config: Any) -> CarlaGroundTruth:
    others = [_vehicle(1, 0.0), _vehicle(2, 12.0)]
    return CarlaGroundTruth(
        world=SimpleNamespace(get_actors=lambda: _ActorList(others)),
        ego=SimpleNamespace(id=1),
        carla_map=None,
        config=config,
        map_name="Stub",
    )


def _ego_state() -> Any:
    from carla_driver_interface.geometry import Pose

    return SimpleNamespace(pose_local_to_rig=Pose.from_xyz_yaw(0.0, 0.0, 0.0, 0.0))


def test_a_runtime_with_no_carla_connection_to_describe_can_still_read() -> None:
    """The reason the protocol exists, exercised rather than asserted."""
    reported = _reader(_OwnConfig())._actor_states(_ego_state())
    assert [state.track_id for state in reported] == ["2"]


def test_runtime_config_still_satisfies_it() -> None:
    """The existing caller must keep working; nothing else would catch it."""
    reported = _reader(RuntimeConfig())._actor_states(_ego_state())
    assert [state.track_id for state in reported] == ["2"]


def test_the_protocol_is_read_only() -> None:
    """``RuntimeConfig`` is frozen, so a settable member would exclude it.

    Declared as properties for that reason. If someone rewrites them as plain
    annotations, mypy fails on the one existing call site -- but only mypy, so
    this states the constraint where a reader will find it.
    """
    for name in (
        "traffic_light_sight_distance_m",
        "route_resolution_m",
        "actor_horizon_m",
        "send_actor_ground_truth",
    ):
        assert isinstance(getattr(GroundTruthConfig, name), property)


def test_the_protocol_asks_for_nothing_about_a_simulator() -> None:
    """The point is what it leaves out; pin that so it stays left out."""
    asked = {name for name in vars(GroundTruthConfig) if not name.startswith("_")}
    assert asked == {
        "traffic_light_sight_distance_m",
        "route_resolution_m",
        "actor_horizon_m",
        "send_actor_ground_truth",
    }
    assert not any(
        word in name for name in asked for word in ("carla", "traffic_manager", "driver", "port")
    )
