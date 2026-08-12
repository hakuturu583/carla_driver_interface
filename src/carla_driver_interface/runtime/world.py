# SPDX-License-Identifier: Apache-2.0
"""The contract between the runtime and whatever is simulating the world.

:class:`WorldAdapter` is the only seam between
:class:`~carla_driver_interface.runtime.carla_runtime.CarlaRuntime` and the
simulator. It exists for two reasons:

1. CARLA 0.9.x and 0.10.x differ in small but load-bearing ways (weather fields,
   wheel-position units, blueprint availability). Keeping the differences behind
   this seam means the closed loop above is version-agnostic.
2. ``carla`` is an optional dependency and cannot be installed in CI, so tests
   substitute :class:`~carla_driver_interface.fakes.fake_world.FakeWorld`.

**Whose job is the handedness flip.** Adapters speak alpasim conventions:
everything crossing this interface is already right-handed, in metres,
rig-anchored and in microseconds. The conversion from CARLA's left-handed world
belongs to the adapter, and adapters should do it with
:mod:`carla_driver_interface.runtime.conversions` rather than open-coding a sign
flip -- that is how a fake ends up quietly disagreeing with the real thing.

Only the contract lives here. The CARLA implementation is
:mod:`carla_driver_interface.runtime.carla_world`, so that a CARLA-free process
(the fake, or a driver-only install) never imports it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from carla_driver_interface.geometry import Pose
from carla_driver_interface.grpc_api import AvailableCamera, CarlaRendererData
from carla_driver_interface.runtime.control import VehicleCommand

__all__ = [
    "CameraCapture",
    "EgoState",
    "RolloutEvents",
    "WorldAdapter",
    "WorldSetup",
    "WorldSnapshot",
]


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


@dataclass
class RolloutEvents:
    """Counters the metrics collector turns into rollout scores."""

    collisions: int = 0
    lane_invasions: int = 0
    #: Frames that failed to encode. Non-zero means the driver saw fewer images
    #: than the rollout claims, which is otherwise invisible.
    encode_failures: int = 0


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

    def environment(self, snapshot: WorldSnapshot) -> CarlaRendererData:
        """Ground truth for this instant, as the extension payload itself.

        Returning the proto rather than a mirror dataclass keeps one definition
        of what the driver can be told: adding a field to
        ``carla_driver.v0.CarlaRendererData`` is a change here and nowhere else.
        """

    def events(self) -> RolloutEvents:
        """Cumulative collision / lane-invasion counters."""

    def close(self) -> None:
        """Destroy actors and restore the simulator's settings."""
