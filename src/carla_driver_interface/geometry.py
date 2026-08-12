# SPDX-License-Identifier: Apache-2.0
"""Poses and trajectories in the alpasim convention, backed by numpy only.

alpasim's own geometry types live in a Rust extension (``utils_rs``) that is not
worth pulling in here, so this module reimplements the small subset the driver
contract needs, matching upstream's conventions exactly:

* Right-handed frames.  ``local`` is ENU-like and inertial; ``rig`` is
  body-fixed with x forward, y left, z up, origin at the rear axle centre
  projected onto the ground.
* Poses are **active** transforms: ``pose_local_to_rig`` maps a point given in
  the rig frame into the local frame, and equally *is* the rig's pose in local.
* Composition is ``a_to_c = a_to_b @ b_to_c``.
* Quaternions are stored ``(x, y, z, w)`` (scipy order) but the proto carries
  ``(w, x, y, z)``; the conversions here are the only place that matters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from carla_driver_interface.grpc_api import (
    DynamicState,
    PoseAtTime,
    Quat,
    Vec3,
)
from carla_driver_interface.grpc_api import Pose as PoseProto
from carla_driver_interface.grpc_api import Trajectory as TrajectoryProto

__all__ = [
    "Pose",
    "Trajectory",
    "dynamic_state_proto",
    "quat_to_yaw",
    "yaw_to_quat_xyzw",
]


def yaw_to_quat_xyzw(yaw: float) -> np.ndarray:
    """Quaternion ``(x, y, z, w)`` for a rotation of ``yaw`` radians about +z."""
    half = 0.5 * yaw
    return np.array([0.0, 0.0, math.sin(half), math.cos(half)], dtype=np.float64)


def quat_to_yaw(quat_xyzw: np.ndarray) -> float:
    """Yaw in radians extracted from a ``(x, y, z, w)`` quaternion."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _quat_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat(rot: np.ndarray) -> np.ndarray:
    """Rotation matrix -> ``(x, y, z, w)``, via the numerically stable branch."""
    trace = float(rot[0, 0] + rot[1, 1] + rot[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
    return _normalize_quat(np.array([x, y, z, w], dtype=np.float64))


def _normalize_quat(quat_xyzw: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quat_xyzw))
    if norm == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat_xyzw / norm


@dataclass(frozen=True)
class Pose:
    """An active rigid transform: rotate by ``quat``, then translate by ``position``."""

    position: np.ndarray
    quat_xyzw: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", np.asarray(self.position, dtype=np.float64).reshape(3))
        object.__setattr__(
            self,
            "quat_xyzw",
            _normalize_quat(np.asarray(self.quat_xyzw, dtype=np.float64).reshape(4)),
        )

    # -- constructors ------------------------------------------------------

    @staticmethod
    def identity() -> Pose:
        return Pose(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))

    @staticmethod
    def from_xyz_yaw(x: float, y: float, z: float, yaw: float) -> Pose:
        return Pose(np.array([x, y, z], dtype=np.float64), yaw_to_quat_xyzw(yaw))

    @staticmethod
    def from_matrix(matrix: np.ndarray) -> Pose:
        matrix = np.asarray(matrix, dtype=np.float64)
        return Pose(matrix[:3, 3], _matrix_to_quat(matrix[:3, :3]))

    @staticmethod
    def from_proto(proto: PoseProto) -> Pose:
        return Pose(
            np.array([proto.vec.x, proto.vec.y, proto.vec.z], dtype=np.float64),
            # proto order is (w, x, y, z)
            np.array([proto.quat.x, proto.quat.y, proto.quat.z, proto.quat.w], dtype=np.float64),
        )

    # -- accessors ---------------------------------------------------------

    @property
    def rotation_matrix(self) -> np.ndarray:
        return _quat_to_matrix(self.quat_xyzw)

    @property
    def yaw(self) -> float:
        return quat_to_yaw(self.quat_xyzw)

    def as_matrix(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.rotation_matrix
        matrix[:3, 3] = self.position
        return matrix

    # -- algebra -----------------------------------------------------------

    def __matmul__(self, other: Pose) -> Pose:
        """``self @ other`` composes ``a_to_b @ b_to_c -> a_to_c``."""
        return Pose.from_matrix(self.as_matrix() @ other.as_matrix())

    def inverse(self) -> Pose:
        rot_t = self.rotation_matrix.T
        return Pose(-rot_t @ self.position, _matrix_to_quat(rot_t))

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Apply this pose to an ``(N, 3)`` array of points."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return points @ self.rotation_matrix.T + self.position

    # -- proto -------------------------------------------------------------

    def to_proto(self) -> PoseProto:
        x, y, z, w = (float(v) for v in self.quat_xyzw)
        return PoseProto(
            vec=Vec3(
                x=float(self.position[0]),
                y=float(self.position[1]),
                z=float(self.position[2]),
            ),
            quat=Quat(w=w, x=x, y=y, z=z),
        )

    def to_proto_at_time(self, timestamp_us: int) -> PoseAtTime:
        return PoseAtTime(pose=self.to_proto(), timestamp_us=int(timestamp_us))


@dataclass
class Trajectory:
    """Time-ordered poses sharing one frame.

    Timestamps are microseconds and must be strictly increasing, matching what
    the alpasim runtime sends and expects back.
    """

    timestamps_us: list[int]
    poses: list[Pose]

    def __post_init__(self) -> None:
        if len(self.timestamps_us) != len(self.poses):
            raise ValueError(
                f"timestamps and poses must match in length, got "
                f"{len(self.timestamps_us)} and {len(self.poses)}"
            )

    def __len__(self) -> int:
        return len(self.poses)

    def __bool__(self) -> bool:
        return bool(self.poses)

    @staticmethod
    def empty() -> Trajectory:
        return Trajectory([], [])

    @staticmethod
    def from_proto(proto: TrajectoryProto) -> Trajectory:
        return Trajectory(
            [int(p.timestamp_us) for p in proto.poses],
            [Pose.from_proto(p.pose) for p in proto.poses],
        )

    def to_proto(self) -> TrajectoryProto:
        return TrajectoryProto(
            poses=[
                pose.to_proto_at_time(ts)
                for ts, pose in zip(self.timestamps_us, self.poses, strict=True)
            ]
        )

    @property
    def positions(self) -> np.ndarray:
        """``(N, 3)`` array of positions; ``(0, 3)`` when empty."""
        if not self.poses:
            return np.zeros((0, 3), dtype=np.float64)
        return np.stack([p.position for p in self.poses])

    @property
    def last_pose(self) -> Pose:
        if not self.poses:
            raise ValueError("empty trajectory has no last pose")
        return self.poses[-1]

    def append(self, timestamp_us: int, pose: Pose) -> None:
        if self.timestamps_us and timestamp_us <= self.timestamps_us[-1]:
            raise ValueError(
                f"timestamps must strictly increase, got {timestamp_us} after "
                f"{self.timestamps_us[-1]}"
            )
        self.timestamps_us.append(int(timestamp_us))
        self.poses.append(pose)

    def transform(self, delta: Pose) -> Trajectory:
        """Left-multiply every pose: ``delta @ pose``.

        Used the way alpasim uses it -- to re-express a trajectory given in
        frame ``B`` into frame ``A``, pass ``pose_a_to_b``.
        """
        return Trajectory(list(self.timestamps_us), [delta @ pose for pose in self.poses])

    def interpolate(self, timestamp_us: int) -> Pose:
        """Pose at ``timestamp_us``; linear in position, slerp in rotation.

        Clamps to the endpoints outside the covered range, which is what the
        controller wants when the driver's plan is shorter than the step.
        """
        if not self.poses:
            raise ValueError("cannot interpolate an empty trajectory")
        if timestamp_us <= self.timestamps_us[0]:
            return self.poses[0]
        if timestamp_us >= self.timestamps_us[-1]:
            return self.poses[-1]

        idx = int(np.searchsorted(np.asarray(self.timestamps_us), timestamp_us))
        t0, t1 = self.timestamps_us[idx - 1], self.timestamps_us[idx]
        p0, p1 = self.poses[idx - 1], self.poses[idx]
        alpha = (timestamp_us - t0) / (t1 - t0)
        return Pose(
            (1.0 - alpha) * p0.position + alpha * p1.position,
            _slerp(p0.quat_xyzw, p1.quat_xyzw, alpha),
        )


def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = _normalize_quat(q0)
    q1 = _normalize_quat(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:  # take the short way round
        q1 = -q1
        dot = -dot
    if dot > 0.9995:  # nearly parallel: lerp is accurate and avoids div-by-zero
        return _normalize_quat(q0 + alpha * (q1 - q0))
    theta_0 = math.acos(dot)
    theta = theta_0 * alpha
    q_perp = _normalize_quat(q1 - q0 * dot)
    return q0 * math.cos(theta) + q_perp * math.sin(theta)


def dynamic_state_proto(
    linear_velocity: np.ndarray,
    angular_velocity: np.ndarray,
    linear_acceleration: np.ndarray,
    angular_acceleration: np.ndarray | None = None,
) -> DynamicState:
    """Build a :class:`DynamicState`; all vectors resolved in the rig frame."""
    zero = np.zeros(3, dtype=np.float64)
    ang_acc = zero if angular_acceleration is None else angular_acceleration

    def vec(values: np.ndarray) -> Vec3:
        arr = np.asarray(values, dtype=np.float64).reshape(3)
        return Vec3(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))

    return DynamicState(
        linear_velocity=vec(linear_velocity),
        angular_velocity=vec(angular_velocity),
        linear_acceleration=vec(linear_acceleration),
        angular_acceleration=vec(ang_acc),
    )
