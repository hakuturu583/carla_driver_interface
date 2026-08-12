# SPDX-License-Identifier: Apache-2.0
"""The trajectory follower that stands in for alpasim's VDCService."""

from __future__ import annotations

import math

import numpy as np
import pytest

from carla_driver_interface.geometry import Pose, Trajectory
from carla_driver_interface.runtime.control import ControlConfig, TrajectoryFollower

from .conftest import arc_plan, straight_plan


def test_straight_plan_steers_straight():
    follower = TrajectoryFollower()
    command = follower.step(straight_plan(), Pose.identity(), current_speed_mps=8.0, dt_s=0.1)
    assert command.steer == pytest.approx(0.0, abs=1e-6)
    assert command.target_speed_mps == pytest.approx(8.0, rel=1e-3)


def test_left_turn_steers_left():
    """CARLA steer is positive to the right, so a left turn must be negative."""
    follower = TrajectoryFollower()
    command = follower.step(arc_plan(radius=25.0), Pose.identity(), 8.0, 0.1)
    assert command.steer < 0.0
    assert command.lookahead_lateral_offset_m > 0.0


def test_right_turn_steers_right():
    command = TrajectoryFollower().step(arc_plan(radius=-25.0), Pose.identity(), 8.0, 0.1)
    assert command.steer > 0.0


def test_pure_pursuit_recovers_the_geometric_steering_angle():
    """On a known arc the command must match the pure-pursuit formula.

    Steering the bicycle model at ``atan(L / R)`` traces radius ``R``; pure
    pursuit approaches that as the lookahead shrinks relative to the radius.
    """
    radius = 60.0
    config = ControlConfig(min_lookahead_m=4.0, max_lookahead_m=4.0, wheelbase_m=2.8)
    command = TrajectoryFollower(config).step(arc_plan(radius), Pose.identity(), 8.0, 0.1)

    steer_angle = -command.steer * config.max_steer_angle_rad
    assert steer_angle == pytest.approx(math.atan(config.wheelbase_m / radius), rel=0.1)


def test_speed_target_is_read_from_the_plan():
    follower = TrajectoryFollower()
    command = follower.step(straight_plan(speed=12.0), Pose.identity(), 12.0, 0.1)
    assert command.target_speed_mps == pytest.approx(12.0, rel=1e-3)


def test_throttle_when_slow_and_brake_when_fast():
    follower = TrajectoryFollower()
    accelerating = follower.step(straight_plan(speed=10.0), Pose.identity(), 2.0, 0.1)
    assert accelerating.throttle > 0.0
    assert accelerating.brake == 0.0

    follower.reset()
    decelerating = follower.step(straight_plan(speed=2.0), Pose.identity(), 15.0, 0.1)
    assert decelerating.brake > 0.0
    assert decelerating.throttle == 0.0


def test_a_stopping_plan_commands_the_brake():
    """A plan that goes nowhere means stop, not coast."""
    plan = Trajectory.empty()
    for i in range(1, 21):
        plan.append(int(i * 0.1 * 1e6), Pose.identity())
    command = TrajectoryFollower().step(plan, Pose.identity(), 3.0, 0.1)
    assert command.brake > 0.0
    assert command.throttle == 0.0


def test_empty_plan_coasts_rather_than_crashing():
    """The driver returns an empty plan before its first observation lands."""
    command = TrajectoryFollower().step(Trajectory.empty(), Pose.identity(), 5.0, 0.1)
    assert command.throttle == 0.0
    assert command.brake > 0.0


def test_steering_is_rate_limited():
    """A hard step in the plan must not translate into an instant full lock."""
    config = ControlConfig(max_steer_rate=1.0)
    follower = TrajectoryFollower(config)
    sharp = arc_plan(radius=5.0)
    command = follower.step(sharp, Pose.identity(), 8.0, dt_s=0.1)
    assert abs(command.steer) <= 1.0 * 0.1 + 1e-9


def test_integrator_does_not_wind_up_while_saturated():
    follower = TrajectoryFollower()
    plan = straight_plan(speed=20.0)
    for _ in range(200):
        follower.step(plan, Pose.identity(), current_speed_mps=0.0, dt_s=0.1)
    assert abs(follower._integral) <= follower.config.integral_limit + 1e-9


def test_plan_expressed_in_local_is_tracked_from_a_rotated_pose():
    """The follower must convert the plan into the rig frame itself.

    A plan running due north while the ego already faces north is straight
    ahead, even though its local coordinates look nothing like the rig ones.
    """
    ego = Pose.from_xyz_yaw(100.0, -50.0, 0.0, math.pi / 2)
    plan = Trajectory.empty()
    for i in range(1, 41):
        dt = i * 0.1
        plan.append(
            int(dt * 1e6),
            Pose.from_xyz_yaw(100.0, -50.0 + 8.0 * dt, 0.0, math.pi / 2),
        )
    command = TrajectoryFollower().step(plan, ego, 8.0, 0.1)
    assert command.steer == pytest.approx(0.0, abs=1e-6)
    assert command.target_speed_mps == pytest.approx(8.0, rel=1e-3)


def test_closed_loop_bicycle_model_converges_onto_an_offset_path():
    """Integrate the controller against a bicycle model and check it converges."""
    config = ControlConfig()
    follower = TrajectoryFollower(config)

    x, y, yaw, speed = 0.0, 3.0, 0.0, 8.0  # start 3 m to the left of the path
    dt = 0.05
    offsets = []
    for step in range(400):
        # The reference path is the x axis; the plan is always the stretch of it
        # ahead of where the ego currently projects.
        plan = Trajectory.empty()
        for i in range(1, 41):
            plan.append(
                int((step * dt + i * 0.1) * 1e6),
                Pose.from_xyz_yaw(max(x, 0.0) + i * 0.1 * speed, 0.0, 0.0, 0.0),
            )
        command = follower.step(plan, Pose.from_xyz_yaw(x, y, 0.0, yaw), speed, dt)

        steer_angle = -command.steer * config.max_steer_angle_rad
        yaw += speed * math.tan(steer_angle) / config.wheelbase_m * dt
        x += speed * math.cos(yaw) * dt
        y += speed * math.sin(yaw) * dt
        offsets.append(abs(y))

    assert offsets[-1] < 0.1, f"did not converge; final offset {offsets[-1]:.3f} m"
    assert max(offsets[100:]) < 1.0, "overshoot after convergence is too large"
    assert np.isfinite(offsets).all()
