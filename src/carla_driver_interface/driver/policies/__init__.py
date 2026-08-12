# SPDX-License-Identifier: Apache-2.0
"""Reference policies.

These are deliberately dependency-free (numpy only) so the closed loop can be
exercised without a model checkpoint or a GPU.
"""

from __future__ import annotations

from carla_driver_interface.driver.base import BaseDriver
from carla_driver_interface.driver.policies.constant_speed import ConstantSpeedPolicy
from carla_driver_interface.driver.policies.route_follower import RouteFollowerPolicy

#: Name -> factory, used by the CLI's ``--policy`` flag.
POLICY_REGISTRY: dict[str, type[BaseDriver]] = {
    ConstantSpeedPolicy.name: ConstantSpeedPolicy,
    RouteFollowerPolicy.name: RouteFollowerPolicy,
}

__all__ = ["POLICY_REGISTRY", "ConstantSpeedPolicy", "RouteFollowerPolicy"]
