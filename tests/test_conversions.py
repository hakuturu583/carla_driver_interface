# SPDX-License-Identifier: Apache-2.0
"""CARLA <-> alpasim frame conversions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from carla_driver_interface.geometry import Pose, Trajectory
from carla_driver_interface.grpc_api import ShutterType
from carla_driver_interface.runtime.conversions import (
    available_camera,
    carla_rotation_to_quat_xyzw,
    carla_transform_to_pose,
    carla_vector_to_local,
    pinhole_camera_spec,
    rig_pose_from_actor_transform,
    seconds_to_us,
    vector_local_to_rig,
)

MIRROR = np.diag([1.0, -1.0, 1.0])


def carla_rotation_matrix(pitch_deg: float, yaw_deg: float, roll_deg: float) -> np.ndarray:
    """CARLA's own ``Transform.get_matrix()`` rotation block, from LibCarla.

    Reproduced here so the test does not depend on the ``carla`` module.
    """
    cy, sy = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    cr, sr = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
    cp, sp = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
    return np.array(
        [
            [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
            [sp, -cp * sr, cp * cr],
        ]
    )


def test_rotation_matches_the_mirrored_carla_matrix():
    """Our quaternion must equal ``M R_carla M`` for the y-mirror ``M``.

    This is the whole handedness conversion, checked against CARLA's own
    rotation convention rather than against our own reasoning about it.
    """
    rng = np.random.default_rng(0)
    for _ in range(200):
        pitch, yaw, roll = rng.uniform(-80, 80), rng.uniform(-180, 180), rng.uniform(-60, 60)
        expected = MIRROR @ carla_rotation_matrix(pitch, yaw, roll) @ MIRROR
        got = Pose(np.zeros(3), carla_rotation_to_quat_xyzw(pitch, yaw, roll)).rotation_matrix
        assert np.allclose(expected, got, atol=1e-9)


def test_yaw_flips_sign():
    """A left turn in CARLA (negative yaw) is a left turn in alpasim (positive)."""
    pose = carla_transform_to_pose((0.0, 0.0, 0.0), (0.0, -90.0, 0.0))
    assert pose.yaw == pytest.approx(math.pi / 2)


def test_position_mirrors_y():
    assert np.allclose(carla_vector_to_local(1.0, 2.0, 3.0), [1.0, -2.0, 3.0])


def test_transform_round_trips_a_point():
    """Transforming a body point must agree with mirroring the CARLA result."""
    location, rotation = (10.0, -4.0, 0.5), (2.0, 35.0, -3.0)
    point_in_body = np.array([1.5, 0.3, 1.6])

    carla_world = carla_rotation_matrix(*rotation) @ point_in_body + np.array(location)
    expected = MIRROR @ carla_world

    pose = carla_transform_to_pose(location, rotation)
    got = pose.transform_points(MIRROR @ point_in_body)[0]
    assert np.allclose(expected, got, atol=1e-9)


def test_rig_origin_sits_behind_the_actor_origin():
    """A negative offset must move the rig backwards along the body x axis."""
    pose = rig_pose_from_actor_transform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), -1.4)
    assert np.allclose(pose.position, [-1.4, 0.0, 0.0])

    # Facing left (CARLA yaw -90), "backwards" is -y in CARLA, i.e. +y... after
    # the mirror it is -y in the alpasim frame.
    turned = rig_pose_from_actor_transform((0.0, 0.0, 0.0), (0.0, -90.0, 0.0), -1.4)
    assert np.allclose(turned.position, [0.0, -1.4, 0.0], atol=1e-9)


def test_velocity_resolves_into_the_rig_frame():
    """A vehicle heading +y at 5 m/s has purely longitudinal rig velocity."""
    pose = Pose.from_xyz_yaw(0.0, 0.0, 0.0, math.pi / 2)
    velocity_local = np.array([0.0, 5.0, 0.0])
    assert np.allclose(vector_local_to_rig(velocity_local, pose), [5.0, 0.0, 0.0], atol=1e-9)


def test_seconds_to_us_applies_the_epoch():
    assert seconds_to_us(1.5) == 1_500_000
    assert seconds_to_us(1.5, epoch_offset_us=1_000) == 1_501_000


def test_pinhole_intrinsics_follow_the_fov():
    spec = pinhole_camera_spec("front", width=800, height=600, horizontal_fov_deg=90.0)
    param = spec.opencv_pinhole_param
    # tan(45 deg) == 1, so fx == width / 2 exactly.
    assert param.focal_length_x == pytest.approx(400.0)
    assert param.focal_length_y == pytest.approx(400.0)
    assert param.principal_point_x == pytest.approx(400.0)
    assert param.principal_point_y == pytest.approx(300.0)
    assert spec.shutter_type == ShutterType.GLOBAL
    # CARLA renders no distortion, so claiming any would be a lie to the driver.
    assert list(param.radial_coeffs) == []
    assert list(param.tangential_coeffs) == []


def test_available_camera_carries_the_mount_pose():
    mount = Pose.from_xyz_yaw(1.5, 0.0, 1.6, 0.0)
    camera = available_camera("front", 640, 480, 100.0, mount)
    assert camera.logical_id == "front"
    assert camera.intrinsics.logical_id == "front"
    assert Pose.from_proto(camera.rig_to_camera).position == pytest.approx([1.5, 0.0, 1.6])


def test_pose_proto_uses_wxyz_quaternion_order():
    """The proto is (w, x, y, z); our internal order is (x, y, z, w)."""
    pose = Pose.from_xyz_yaw(0.0, 0.0, 0.0, math.pi / 2)
    proto = pose.to_proto()
    assert proto.quat.w == pytest.approx(math.cos(math.pi / 4))
    assert proto.quat.z == pytest.approx(math.sin(math.pi / 4))
    assert proto.quat.x == pytest.approx(0.0)
    assert Pose.from_proto(proto).yaw == pytest.approx(math.pi / 2)


def test_trajectory_proto_round_trip():
    original = Trajectory(
        [0, 100_000, 250_000],
        [Pose.identity(), Pose.from_xyz_yaw(1, 0, 0, 0.1), Pose.from_xyz_yaw(3, 1, 0, 0.3)],
    )
    restored = Trajectory.from_proto(original.to_proto())
    assert restored.timestamps_us == original.timestamps_us
    for a, b in zip(original.poses, restored.poses, strict=True):
        assert np.allclose(a.as_matrix(), b.as_matrix(), atol=1e-6)


def test_trajectory_transform_composes_like_alpasim():
    """``traj.transform(a_to_b)`` left-multiplies, matching upstream usage."""
    delta = Pose.from_xyz_yaw(10.0, 0.0, 0.0, math.pi / 2)
    trajectory = Trajectory([0], [Pose.from_xyz_yaw(1.0, 0.0, 0.0, 0.0)])
    moved = trajectory.transform(delta)
    assert np.allclose(moved.poses[0].position, [10.0, 1.0, 0.0], atol=1e-9)


def test_trajectory_rejects_non_increasing_timestamps():
    trajectory = Trajectory([100], [Pose.identity()])
    with pytest.raises(ValueError, match="strictly increase"):
        trajectory.append(100, Pose.identity())
