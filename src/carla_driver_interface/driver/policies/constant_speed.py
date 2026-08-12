# SPDX-License-Identifier: Apache-2.0
"""The simplest useful policy: drive straight at a fixed speed."""

from __future__ import annotations

from carla_driver_interface.driver.base import BaseDriver, DriveContext, DriveResult
from carla_driver_interface.geometry import Pose, Trajectory

__all__ = ["ConstantSpeedPolicy"]


class ConstantSpeedPolicy(BaseDriver):
    """Plan a straight line ahead in the rig frame.

    Useful as a smoke test for the closed loop and as the minimal example of
    what a policy has to return.
    """

    name = "constant_speed"

    def __init__(
        self,
        target_speed_mps: float = 5.0,
        horizon_s: float = 4.0,
        step_s: float = 0.1,
    ) -> None:
        self.target_speed_mps = target_speed_mps
        self.horizon_s = horizon_s
        self.step_s = step_s

    def drive(self, ctx: DriveContext) -> DriveResult:
        plan = Trajectory.empty()
        n_steps = max(1, int(round(self.horizon_s / self.step_s)))
        for i in range(1, n_steps + 1):
            dt_s = i * self.step_s
            plan.append(
                ctx.time_now_us + int(round(dt_s * 1e6)),
                Pose.from_xyz_yaw(self.target_speed_mps * dt_s, 0.0, 0.0, 0.0),
            )
        return DriveResult(
            trajectory_in_rig=plan,
            debug_scalars={"target_speed_mps": self.target_speed_mps},
        )
