# SPDX-License-Identifier: Apache-2.0
"""Policy-facing types for the driver half.

:class:`BaseDriver` is what a policy implements.  Everything gRPC -- session
bookkeeping, frame caching, ego history, proto conversion -- lives in
:mod:`carla_driver_interface.driver.service`, so a policy only sees numpy and
plain Python.

The observation set mirrors alpasim's ``EgodriverService`` one-for-one:
images, egomotion, route and recording ground truth arrive as separate
callbacks before every :meth:`BaseDriver.drive` call, exactly as the alpasim
runtime sequences them (``events/policy.py`` submits all observations, waits on
a barrier, then calls ``drive``).
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from carla_driver_interface.geometry import Pose, Trajectory
from carla_driver_interface.grpc_api import (
    AvailableCamera,
    CarlaRendererData,
    DynamicState,
)

__all__ = ["BaseDriver", "CameraFrame", "DriveContext", "DriveResult", "SessionState"]


@dataclass(frozen=True)
class CameraFrame:
    """One encoded camera image, as it arrived over the wire.

    ``image_bytes`` is left encoded (PNG or JPEG) rather than decoded eagerly --
    a policy that only needs the front camera should not pay to decode six.
    Use :meth:`as_array` when you want pixels.
    """

    logical_id: str
    frame_start_us: int
    frame_end_us: int
    image_bytes: bytes

    def as_array(self) -> np.ndarray:
        """Decode to an ``(H, W, 3)`` uint8 RGB array. Requires Pillow."""
        import io

        from PIL import Image

        with Image.open(io.BytesIO(self.image_bytes)) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)


@dataclass
class SessionState:
    """Per-rollout state the service maintains on the policy's behalf."""

    uuid: str
    seed: int
    scene_id: str
    cameras: dict[str, AvailableCamera]

    #: Bounded history per camera, oldest first. Seeded with an empty list for
    #: every declared camera, so a lookup before the first frame gives ``[]``
    #: rather than a KeyError. Use :meth:`latest_frame` for the newest one.
    frame_history: dict[str, list[CameraFrame]] = field(default_factory=dict)
    #: Estimated ego trajectory in the ``local`` frame (``local -> rig_est``).
    ego_trajectory: Trajectory = field(default_factory=Trajectory.empty)
    #: Dynamic states in the rig frame, aligned 1:1 with ``ego_trajectory``.
    dynamic_states: list[DynamicState] = field(default_factory=list)
    #: Latest route waypoints in the rig frame, ``(N, 3)``. Empty when unset.
    route_waypoints_in_rig: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float64)
    )
    route_timestamp_us: int = 0
    #: Latest recording ground truth in the rig frame. Never set under CARLA
    #: unless the runtime is configured to synthesise one.
    ground_truth_in_rig: Trajectory | None = None
    drive_count: int = 0

    #: Guards mutation from concurrent gRPC handler threads.
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def latest_frame(self, logical_id: str) -> CameraFrame | None:
        """Newest frame from one camera, or ``None`` before the first arrives."""
        frames = self.frame_history.get(logical_id)
        return frames[-1] if frames else None

    def latest_pose(self) -> Pose | None:
        """Newest estimated ego pose (``local -> rig_est``), if any."""
        return self.ego_trajectory.last_pose if self.ego_trajectory else None

    def latest_dynamic_state(self) -> DynamicState | None:
        return self.dynamic_states[-1] if self.dynamic_states else None

    def speed_mps(self) -> float:
        """Longitudinal speed from the newest dynamic state, in m/s."""
        state = self.latest_dynamic_state()
        if state is None:
            return 0.0
        v = state.linear_velocity
        return float(np.linalg.norm([v.x, v.y, v.z]))


@dataclass(frozen=True)
class DriveContext:
    """Everything a policy gets for one :meth:`BaseDriver.drive` call."""

    session: SessionState
    #: Planning time; the pose at this instant anchors the returned plan.
    time_now_us: int
    #: The instant the runtime will next step the vehicle to.
    time_query_us: int
    #: CARLA ground truth, when the runtime is this project's.  ``None`` when
    #: driven by upstream alpasim, or when the payload could not be parsed.
    renderer_data: CarlaRendererData | None


@dataclass
class DriveResult:
    """What a policy returns.

    ``trajectory_in_rig`` is the plan in the rig frame at ``time_now_us``, which
    is the natural frame for a policy.  The service converts it into the
    ``local`` frame that ``DriveResponse.trajectory`` requires, by composing
    with the ego pose at ``time_now_us``.
    """

    trajectory_in_rig: Trajectory
    #: Set to end the rollout immediately (``DriveResponse.terminate_session``).
    terminate_session: bool = False
    #: Surfaced through ``CarlaDriveDebugInfo.scalars``.
    debug_scalars: dict[str, float] = field(default_factory=dict)
    #: Alternative plans considered, in the rig frame; forwarded to
    #: ``DriveResponse.DebugInfo.sampled_trajectories`` (converted to local).
    sampled_trajectories_in_rig: list[Trajectory] = field(default_factory=list)


class BaseDriver(ABC):
    """Base class for a driving policy.

    Subclasses must implement :meth:`drive`.  The observation hooks are
    optional; the default implementations do nothing because
    :class:`SessionState` already records everything a simple policy needs.
    """

    #: Reported through ``get_version`` and ``CarlaDriveDebugInfo.policy_name``.
    name: str = "base"

    #: How many frames per camera to retain in ``SessionState.frame_history``.
    #: 1 keeps only the newest; raise it for policies that need temporal context.
    frame_history_length: int = 1

    def on_session_start(self, session: SessionState) -> None:
        """Called once per rollout, before any observation."""

    def on_session_close(self, session: SessionState) -> None:
        """Called once per rollout, after the last :meth:`drive`."""

    def on_image(self, session: SessionState, frame: CameraFrame) -> None:
        """Called for every camera frame, after it lands in ``session``."""

    def on_egomotion(self, session: SessionState, new_poses: Trajectory) -> None:
        """Called for every egomotion batch, after it lands in ``session``."""

    def on_route(self, session: SessionState) -> None:
        """Called whenever a new route lands in ``session``."""

    def on_ground_truth(self, session: SessionState) -> None:
        """Called whenever recording ground truth lands in ``session``."""

    @abstractmethod
    def drive(self, ctx: DriveContext) -> DriveResult:
        """Produce a plan for ``ctx.session`` at ``ctx.time_now_us``."""
