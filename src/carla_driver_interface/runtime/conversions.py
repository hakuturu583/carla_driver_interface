# SPDX-License-Identifier: Apache-2.0
"""CARLA <-> alpasim frame and unit conversions.

This is the single place that knows CARLA is left handed.  Everything above it
works in the alpasim conventions described in
:mod:`carla_driver_interface.geometry`.

**The handedness flip.**  CARLA's world is x forward, **y right**, z up, with
rotations reported as degrees ``(pitch, yaw, roll)`` in the Unreal convention.
alpasim is right handed with y *left*.  Mirroring the y axis maps one to the
other:

    position:  (x, y, z)          -> (x, -y, z)
    rotation:  (pitch, yaw, roll) -> (-pitch, -yaw, roll)   [degrees -> radians]

Roll keeps its sign because a mirror about the xz plane reverses the two
rotations whose axes lie in that plane (yaw about z, pitch about y) and
preserves the one about the axis normal to it... about x, which is roll.

**The rig origin.**  alpasim puts the rig origin at the rear axle centre
projected onto the ground; a CARLA vehicle actor's origin sits at the vehicle
centre.  :func:`rig_pose_from_actor_transform` shifts by
``rear_axle_offset_m`` along the body x axis to reconcile them.

To avoid depending on the ``carla`` module here (it is an optional extra and
absent in CI), these functions take plain floats and tuples.  The adapter in
:mod:`carla_driver_interface.runtime.world` is what unpacks CARLA objects.
"""

from __future__ import annotations

import math

import numpy as np

from carla_driver_interface.geometry import Pose
from carla_driver_interface.grpc_api import (
    AvailableCamera,
    CameraSpec,
    OpenCVPinholeCameraParam,
    ShutterType,
)

__all__ = [
    "available_camera",
    "carla_rotation_to_quat_xyzw",
    "carla_transform_to_pose",
    "carla_vector_to_local",
    "pinhole_camera_spec",
    "rig_pose_from_actor_transform",
    "seconds_to_us",
    "vector_local_to_rig",
]


def seconds_to_us(seconds: float, epoch_offset_us: int = 0) -> int:
    """CARLA's float seconds -> the microsecond timestamps the protos use.

    ``epoch_offset_us`` lets a deployment place a rollout on an absolute
    timeline; alpasim only ever compares timestamps within a session, so 0 is
    fine by default.
    """
    return epoch_offset_us + int(round(seconds * 1e6))


def carla_vector_to_local(x: float, y: float, z: float) -> np.ndarray:
    """Mirror a CARLA world/relative vector into the right-handed convention."""
    return np.array([x, -y, z], dtype=np.float64)


def carla_rotation_to_quat_xyzw(pitch_deg: float, yaw_deg: float, roll_deg: float) -> np.ndarray:
    """CARLA Euler angles (degrees) -> a right-handed ``(x, y, z, w)`` quaternion."""
    roll = math.radians(roll_deg)
    pitch = math.radians(-pitch_deg)
    yaw = math.radians(-yaw_deg)

    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)

    # Z-Y-X intrinsic (yaw, then pitch, then roll), matching Unreal's order.
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float64,
    )


def carla_transform_to_pose(
    location: tuple[float, float, float],
    rotation_pyr_deg: tuple[float, float, float],
) -> Pose:
    """A ``carla.Transform``'s components -> an alpasim-convention pose."""
    return Pose(
        carla_vector_to_local(*location),
        carla_rotation_to_quat_xyzw(*rotation_pyr_deg),
    )


def rig_pose_from_actor_transform(
    location: tuple[float, float, float],
    rotation_pyr_deg: tuple[float, float, float],
    rear_axle_offset_m: float,
) -> Pose:
    """``local -> rig`` for a vehicle actor.

    ``rear_axle_offset_m`` is the signed distance from the actor origin to the
    rear axle centre along the body's forward axis; it is negative for a normal
    car, whose rear axle sits behind the origin.
    """
    actor_pose = carla_transform_to_pose(location, rotation_pyr_deg)
    shift = Pose(np.array([rear_axle_offset_m, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))
    return actor_pose @ shift


def vector_local_to_rig(vector_local: np.ndarray, pose_local_to_rig: Pose) -> np.ndarray:
    """Resolve a local-frame vector (velocity, acceleration) in the rig frame.

    Only the rotation applies -- these are free vectors, not points.
    """
    return pose_local_to_rig.rotation_matrix.T @ np.asarray(vector_local, dtype=np.float64)


def pinhole_camera_spec(
    logical_id: str,
    width: int,
    height: int,
    horizontal_fov_deg: float,
) -> CameraSpec:
    """Build a ``CameraSpec`` for a CARLA ``sensor.camera.rgb``.

    CARLA renders an ideal pinhole with square pixels and the principal point at
    the image centre, so the OpenCV pinhole model describes it exactly with all
    distortion coefficients left empty.
    """
    focal = width / (2.0 * math.tan(math.radians(horizontal_fov_deg) * 0.5))
    return CameraSpec(
        opencv_pinhole_param=OpenCVPinholeCameraParam(
            principal_point_x=width / 2.0,
            principal_point_y=height / 2.0,
            focal_length_x=focal,
            focal_length_y=focal,
        ),
        logical_id=logical_id,
        resolution_w=width,
        resolution_h=height,
        # CARLA captures the whole frame at one instant.
        shutter_type=ShutterType.GLOBAL,
    )


def available_camera(
    logical_id: str,
    width: int,
    height: int,
    horizontal_fov_deg: float,
    camera_pose_in_rig: Pose,
) -> AvailableCamera:
    """Describe one camera the way ``start_session`` expects it.

    Despite the field name, upstream composes ``pose_local_to_rig @
    rig_to_camera`` to place the sensor (``sensorsim_service.py``), so
    ``rig_to_camera`` holds the camera's pose *in the rig frame*. That is what
    ``camera_pose_in_rig`` is.
    """
    return AvailableCamera(
        intrinsics=pinhole_camera_spec(logical_id, width, height, horizontal_fov_deg),
        rig_to_camera=camera_pose_in_rig.to_proto(),
        logical_id=logical_id,
    )
