"""Policy-facing types: what a driving policy implements and what it is given.

:class:`BaseDriver` is what a policy implements. Everything gRPC -- session
bookkeeping, frame retention, ego history, rig/local conversion -- lives in
:mod:`carla_driver_interface.service`, so a policy only sees numpy and plain Python.

The observation set mirrors alpasim's ``EgodriverService`` one for one: images,
egomotion, route and recorded ground truth arrive through separate calls before every
:meth:`BaseDriver.drive`, as the alpasim runtime sequences them.

Every sensor reaches a policy the same way, whichever call carried it: as a frame
(:class:`CameraFrame`, :class:`LidarFrame`) recorded in
``SessionState.frame_history`` under its logical id, then announced through
:meth:`BaseDriver.on_frame`, all before :meth:`BaseDriver.drive`.  (The contract has
an RPC for camera images but none for LiDAR, so sweeps travel inside
``DriveRequest.renderer_data`` and the servicer records them just before ``drive`` --
a transport detail a policy never sees.)
"""

from __future__ import annotations

import io
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from .geometry import Pose, Trajectory
from .hdmap import MapFiles, StopLine
from .protocol import (
    AvailableCamera,
    CarlaRendererData,
    DynamicState,
    LidarSweep,
    decode_lidar_points,
)

__all__ = [
    "BaseDriver",
    "CameraFrame",
    "DriveContext",
    "DriveResult",
    "LidarFrame",
    "SensorFrame",
    "SessionState",
]


@dataclass(frozen=True)
class CameraFrame:
    """One encoded camera image, as it arrived over the wire.

    ``image_bytes`` stays encoded (PNG or JPEG): a policy that only needs the front
    camera should not pay to decode the others. :meth:`as_array` decodes.
    """

    logical_id: str
    frame_start_us: int
    frame_end_us: int
    image_bytes: bytes

    @property
    def timestamp_us(self) -> int:
        """When the frame was complete: the end of its exposure."""
        return self.frame_end_us

    def as_array(self) -> NDArray[np.uint8]:
        """Decode to an ``(H, W, 3)`` uint8 RGB array."""
        from PIL import Image

        with Image.open(io.BytesIO(self.image_bytes)) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)


@dataclass(frozen=True)
class LidarFrame:
    """One LiDAR sweep, as it arrived over the wire.

    The points stay packed, as a camera frame stays encoded; :meth:`as_array` unpacks.
    They are already in the rig frame, so the mount is needed only for the sensor's
    origin (ray casting, say).
    """

    logical_id: str
    timestamp_us: int
    #: The sensor's mount in the rig frame.
    rig_to_lidar: Pose
    num_points: int
    #: Little-endian float32 ``[num_points, 4]``: x, y, z (rig frame), intensity.
    points_xyzi: bytes

    @classmethod
    def from_proto(cls, sweep: LidarSweep) -> "LidarFrame":
        return cls(
            logical_id=sweep.logical_id,
            timestamp_us=int(sweep.timestamp_us),
            rig_to_lidar=Pose.from_proto(sweep.rig_to_lidar),
            num_points=int(sweep.num_points),
            points_xyzi=sweep.points_xyzi,
        )

    def as_array(self) -> NDArray[np.float32]:
        """Unpack to a writable ``(N, 4)`` float32 array: x, y, z, intensity."""
        return decode_lidar_points(self.logical_id, self.num_points, self.points_xyzi)


#: Any sensor's frame.  Both kinds carry ``logical_id``, ``timestamp_us`` and
#: ``as_array()``.
SensorFrame = Union[CameraFrame, LidarFrame]


@dataclass
class SessionState:
    """Per-rollout state the servicer maintains on the policy's behalf."""

    uuid: str
    seed: int
    scene_id: str
    #: Declared cameras by logical id.  ``rig_to_camera`` is the camera body's pose
    #: in the rig (x along the optical axis, y left, z up) in either mode.
    cameras: Dict[str, AvailableCamera]

    #: Bounded history per sensor, by logical id, oldest first; ``[]`` for every
    #: declared camera before its first frame (a LiDAR, which the contract has no
    #: way to declare, appears with its first sweep).  :meth:`latest_frame` gives
    #: the newest.
    frame_history: Dict[str, List[SensorFrame]] = field(default_factory=dict)
    #: Estimated ego trajectory in the ``local`` frame (``local -> rig_est``).
    ego_trajectory: Trajectory = field(default_factory=Trajectory.empty)
    #: Dynamic states in the rig frame, aligned 1:1 with ``ego_trajectory``.
    dynamic_states: List[DynamicState] = field(default_factory=list)
    #: Latest route waypoints in the rig frame, ``(N, 3)``; empty when unset.
    route_waypoints_in_rig: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float64)
    )
    route_timestamp_us: int = 0
    #: Latest recorded ground truth in the rig frame, when the runtime sends one.
    ground_truth_in_rig: Optional[Trajectory] = None
    drive_count: int = 0

    #: Guards mutation from concurrent gRPC handler threads.
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def latest_frame(self, logical_id: str) -> Optional[SensorFrame]:
        """Newest frame from one sensor, or ``None`` before the first arrives."""
        frames = self.frame_history.get(logical_id)
        return frames[-1] if frames else None

    def latest_pose(self) -> Optional[Pose]:
        """Newest estimated ego pose (``local -> rig_est``), if any."""
        return self.ego_trajectory.last_pose

    def latest_dynamic_state(self) -> Optional[DynamicState]:
        return self.dynamic_states[-1] if self.dynamic_states else None

    def speed_mps(self) -> float:
        """Speed from the newest dynamic state, in m/s; 0 before the first."""
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
    #: CARLA ground truth (lights, other vehicles, speed limit) when the runtime
    #: sends it; ``None`` under an upstream alpasim runtime.  The LiDAR sweeps it
    #: carries are already in ``session.frame_history``, with the camera frames.
    renderer_data: Optional[CarlaRendererData]
    #: The world's map in the policy's format (:attr:`BaseDriver.map_format`),
    #: opened from :attr:`BaseDriver.map_dir` by ``CarlaRendererData.map_id``.
    #: ``None`` when the policy sets no ``map_dir`` or the runtime wrote no map.
    map: Optional[MapFiles] = None

    def stop_lines(self) -> List[StopLine]:
        """Every traffic light's stop lines, named by :attr:`map`'s own elements.

        Empty without a map or without lights.  The state is this step's; the
        positions are in the ``local`` frame, the map's.
        """
        if self.map is None or self.renderer_data is None:
            return []
        return self.map.stop_lines(self.renderer_data.traffic_lights)


@dataclass
class DriveResult:
    """What a policy returns.

    ``trajectory_in_rig`` is the plan in the rig frame at ``time_now_us``; the
    servicer converts it into the ``local`` frame the response carries.
    """

    trajectory_in_rig: Trajectory
    #: End the rollout now (``DriveResponse.terminate_session``).
    terminate_session: bool = False
    #: Sent back in ``CarlaDriveDebugInfo.scalars``.
    debug_scalars: Dict[str, float] = field(default_factory=dict)
    #: Alternative plans considered, in the rig frame.
    sampled_trajectories_in_rig: List[Trajectory] = field(default_factory=list)


class BaseDriver(ABC):
    """Base class for a driving policy: implement :meth:`drive`.

    The observation hooks are optional; :class:`SessionState` already records
    everything a simple policy needs.
    """

    #: Reported through ``get_version`` and ``CarlaDriveDebugInfo.policy_name``.
    name: str = "base"

    #: Your copy of the runtime's ``driver.map_dir``.  When set, each
    #: :class:`DriveContext` carries the world's map from it (``ctx.map``) and the
    #: lights resolved against it (``ctx.stop_lines()``).
    map_dir: Optional[str] = None
    #: The format to read the map in; the runtime must write it
    #: (``driver.map_formats``).
    map_format: str = "lanelet2"

    #: Frames per sensor kept in ``SessionState.frame_history``; 1 keeps only the
    #: newest.  Raise it for a policy that needs temporal context.
    frame_history_length: int = 1

    def on_session_start(self, session: SessionState) -> None:  # noqa: B027
        """Called once per rollout, before any observation."""

    def on_session_close(self, session: SessionState) -> None:  # noqa: B027
        """Called once per rollout, after the last :meth:`drive`."""

    def on_frame(self, session: SessionState, frame: SensorFrame) -> None:  # noqa: B027
        """Called for every sensor frame, after it lands in ``session``."""

    def on_egomotion(self, session: SessionState, new_poses: Trajectory) -> None:  # noqa: B027
        """Called for every egomotion batch, after it lands in ``session``."""

    def on_route(self, session: SessionState) -> None:  # noqa: B027
        """Called whenever a new route lands in ``session``."""

    def on_ground_truth(self, session: SessionState) -> None:  # noqa: B027
        """Called whenever recorded ground truth lands in ``session``."""

    @abstractmethod
    def drive(self, ctx: DriveContext) -> DriveResult:
        """Produce a plan for ``ctx.session`` at ``ctx.time_now_us``."""
