"""Rigid-body primitives for the ``egodriver`` gRPC contract.

The alpasim wire format expresses every pose as a translation plus a quaternion
(:class:`~carla_driver_interface.protocol.Pose`) and every path as a
list of timestamped poses
(:class:`~carla_driver_interface.protocol.Trajectory`).  This module
provides the Python-side counterparts and the conversions between them.

The rotation maths is scipy's (:class:`scipy.spatial.transform.Rotation` and
:class:`~scipy.spatial.transform.Slerp`); this module only adds what the wire needs on
top -- a pose type pairing a translation with a rotation, timestamped sequences of them,
and the protobuf conversions.

Conventions, matching alpasim:

* Quaternions are in scipy's ``(x, y, z, w)`` order and converted to the protobuf's
  ``(w, x, y, z)`` field order on the wire.
* Composition is the standard rigid transform: ``(a @ b).position ==
  a.rotation_matrix @ b.position + a.position``.  So ``ego_pose @ pose_in_rig`` lifts a
  rig-frame pose into the local frame, and ``ego_pose.inverse() @ pose_in_local`` does
  the reverse.
* The frame itself is **right-handed** (x forward, y left, z up), unlike CARLA's
  left-handed world.  The CARLA runtime (autoware_carla_scenario's
  ``driver.observation``) performs that conversion; nothing in this module knows about
  CARLA.

:class:`Pose` and :class:`Trajectory` keep the class and method names they had in
carla-driver-interface 0.1.0, which this release succeeds, so code using them ports
unchanged; 0.1.0's free quaternion helpers (``quat_mul``, ``yaw_to_quat_xyzw``, ...) gave
way to scipy's :class:`~scipy.spatial.transform.Rotation`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation, Slerp

from ._proto import common_pb2

__all__ = ["BODY_TO_OPTICAL", "Pose", "Trajectory", "waypoints_to_proto"]

#: Quaternion norms below this are treated as degenerate and replaced by the identity.
_QUAT_NORM_EPSILON: float = 1e-12

_AXES = "xyz"

#: A camera body (x along the optical axis, y left, z up) to its optical frame (x
#: right, y down, z along the optical axis): the optical axes as columns, in body
#: coordinates.  ``body_rotation * Rotation.from_matrix(BODY_TO_OPTICAL)`` is the
#: optical frame's rotation.
BODY_TO_OPTICAL: NDArray[np.float64] = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
)


def _rotation(quat_xyzw: ArrayLike) -> Rotation:
    """*quat_xyzw* as a rotation; a zero quaternion is the identity.

    scipy refuses a zero-norm quaternion, but a default-constructed protobuf
    message (all fields zero) is one, and must round-trip to the identity rather
    than fail.
    """
    quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(-1)
    if quat.size != 4:
        raise ValueError(f"expected 4 quaternion components, got {quat.size}")
    if float(np.linalg.norm(quat)) < _QUAT_NORM_EPSILON:
        return Rotation.identity()
    return Rotation.from_quat(quat)


@dataclass(frozen=True, eq=False)
class Pose:
    """A rigid transform: a translation and a rotation.

    Args:
        position: ``(3,)`` translation in metres.
        quat_xyzw: ``(4,)`` quaternion in ``(x, y, z, w)`` order.  Normalised on
            construction; all zeros means the identity.
    """

    position: NDArray[np.float64]
    quat_xyzw: NDArray[np.float64]
    rotation: Rotation = field(init=False, repr=False)

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=np.float64).reshape(-1)
        if position.size != 3:
            raise ValueError(f"expected 3 components, got {position.size}")
        rotation = _rotation(self.quat_xyzw)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "quat_xyzw", rotation.as_quat())

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_rotation(cls, position: ArrayLike, rotation: Rotation) -> "Pose":
        """Return the pose translating by *position* and rotating by *rotation*."""
        return cls(np.asarray(position, dtype=np.float64), rotation.as_quat())

    @classmethod
    def identity(cls) -> "Pose":
        """Return the identity transform."""
        return cls(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))

    @classmethod
    def from_xyz_yaw(cls, x: float, y: float, z: float, yaw: float) -> "Pose":
        """Return a pose with a yaw-only rotation (radians, counter-clockwise)."""
        return cls.from_rotation([x, y, z], Rotation.from_euler("z", yaw))

    @classmethod
    def from_proto(cls, message: common_pb2.Pose) -> "Pose":
        """Return the pose described by a protobuf message."""
        return cls(
            np.array([message.vec.x, message.vec.y, message.vec.z], dtype=np.float64),
            np.array(
                [message.quat.x, message.quat.y, message.quat.z, message.quat.w],
                dtype=np.float64,
            ),
        )

    @classmethod
    def from_axis_angle(cls, axis: int, angle_rad: float) -> "Pose":
        """Return a rotation-only pose about principal axis ``0``, ``1`` or ``2`` (x, y, z)."""
        return cls.from_rotation(
            np.zeros(3), Rotation.from_euler(_AXES[axis], angle_rad)
        )

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def rotation_matrix(self) -> NDArray[np.float64]:
        """Return the ``(3, 3)`` rotation matrix for this pose."""
        return self.rotation.as_matrix()

    @property
    def yaw(self) -> float:
        """Return the rotation about the z axis in radians."""
        return float(self.rotation.as_euler("ZYX")[0])

    def as_matrix(self) -> NDArray[np.float64]:
        """Return the ``(4, 4)`` homogeneous transform for this pose."""
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation_matrix
        matrix[:3, 3] = self.position
        return matrix

    # ------------------------------------------------------------------
    # Algebra
    # ------------------------------------------------------------------

    def __matmul__(self, other: "Pose") -> "Pose":
        """Compose two transforms: ``self`` applied after *other*."""
        return Pose.from_rotation(
            self.rotation.apply(other.position) + self.position,
            self.rotation * other.rotation,
        )

    def inverse(self) -> "Pose":
        """Return the inverse transform."""
        inverse = self.rotation.inv()
        return Pose.from_rotation(-inverse.apply(self.position), inverse)

    def transform_points(self, points: ArrayLike) -> NDArray[np.float64]:
        """Apply this transform to an ``(N, 3)`` array of points."""
        array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if array.size == 0:
            return array
        return self.rotation.apply(array) + self.position

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_proto(self) -> common_pb2.Pose:
        """Return the protobuf representation of this pose."""
        x, y, z, w = self.quat_xyzw
        return common_pb2.Pose(
            vec=common_pb2.Vec3(
                x=self.position[0], y=self.position[1], z=self.position[2]
            ),
            quat=common_pb2.Quat(w=w, x=x, y=y, z=z),
        )

    def to_proto_at_time(self, timestamp_us: int) -> common_pb2.PoseAtTime:
        """Return this pose stamped with *timestamp_us*."""
        return common_pb2.PoseAtTime(pose=self.to_proto(), timestamp_us=timestamp_us)


@dataclass
class Trajectory:
    """A time-ordered sequence of poses.

    Args:
        timestamps_us: Microsecond timestamps, one per pose.
        poses: Poses, all expressed in the same frame.

    Raises:
        ValueError: If the two sequences differ in length.
    """

    timestamps_us: List[int]
    poses: List[Pose]

    def __post_init__(self) -> None:
        if len(self.timestamps_us) != len(self.poses):
            raise ValueError(
                "timestamps_us and poses must have the same length, got "
                f"{len(self.timestamps_us)} and {len(self.poses)}"
            )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def empty(cls) -> "Trajectory":
        """Return a trajectory with no poses."""
        return cls([], [])

    @classmethod
    def from_proto(cls, message: common_pb2.Trajectory) -> "Trajectory":
        """Return the trajectory described by a protobuf message."""
        timestamps = [int(entry.timestamp_us) for entry in message.poses]
        poses = [Pose.from_proto(entry.pose) for entry in message.poses]
        return cls(timestamps, poses)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.poses)

    def __bool__(self) -> bool:
        return bool(self.poses)

    @property
    def positions(self) -> NDArray[np.float64]:
        """Return the positions as an ``(N, 3)`` array."""
        if not self.poses:
            return np.zeros((0, 3))
        return np.stack([pose.position for pose in self.poses])

    @property
    def last_pose(self) -> Optional[Pose]:
        """Return the final pose, or ``None`` when the trajectory is empty."""
        return self.poses[-1] if self.poses else None

    # ------------------------------------------------------------------
    # Mutation and transforms
    # ------------------------------------------------------------------

    def append(self, timestamp_us: int, pose: Pose) -> None:
        """Append *pose* stamped at *timestamp_us*."""
        self.timestamps_us.append(int(timestamp_us))
        self.poses.append(pose)

    def transform(self, delta: Pose) -> "Trajectory":
        """Return a copy with *delta* applied to every pose.

        ``trajectory.transform(ego_pose)`` lifts a rig-frame trajectory into the local
        frame; ``trajectory.transform(ego_pose.inverse())`` does the reverse.
        """
        return Trajectory(
            list(self.timestamps_us), [delta @ pose for pose in self.poses]
        )

    def interpolate(self, timestamp_us: int) -> Optional[Pose]:
        """Return the pose at *timestamp_us*, interpolating between samples.

        Positions are interpolated linearly and rotations by slerp.  Timestamps
        outside the trajectory's span clamp to the nearest endpoint.

        Returns:
            The interpolated pose, or ``None`` when the trajectory is empty.
        """
        if not self.poses:
            return None
        if len(self.poses) == 1 or timestamp_us <= self.timestamps_us[0]:
            return self.poses[0]
        if timestamp_us >= self.timestamps_us[-1]:
            return self.poses[-1]

        index = int(np.searchsorted(self.timestamps_us, timestamp_us))
        previous, following = self.poses[index - 1], self.poses[index]
        start, end = self.timestamps_us[index - 1], self.timestamps_us[index]
        if end <= start:
            return following
        alpha = (timestamp_us - start) / (end - start)
        slerp = Slerp(
            [0.0, 1.0], Rotation.concatenate([previous.rotation, following.rotation])
        )
        return Pose.from_rotation(
            previous.position + alpha * (following.position - previous.position),
            slerp([alpha])[0],
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_proto(self) -> common_pb2.Trajectory:
        """Return the protobuf representation of this trajectory."""
        return common_pb2.Trajectory(
            poses=[
                pose.to_proto_at_time(timestamp)
                for timestamp, pose in zip(self.timestamps_us, self.poses)
            ]
        )


def waypoints_to_proto(points: NDArray[np.float64]) -> List[common_pb2.Vec3]:
    """Return *points* as protobuf ``Vec3`` messages.

    Args:
        points: ``(N, 3)`` array of waypoints.
    """
    return [
        common_pb2.Vec3(x=float(point[0]), y=float(point[1]), z=float(point[2]))
        for point in points
    ]
