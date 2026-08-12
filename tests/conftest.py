# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and helpers.

The ``driver_stub`` fixture lives here rather than in one test module so every
test that needs a running driver gets the same server setup -- including the
message-size options, which a hand-rolled server would miss.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import grpc
import numpy as np
import pytest

from carla_driver_interface.driver.base import BaseDriver
from carla_driver_interface.driver.policies import ConstantSpeedPolicy
from carla_driver_interface.driver.server import build_server
from carla_driver_interface.geometry import Pose, Trajectory
from carla_driver_interface.grpc_api import EgodriverServiceStub, channel_options


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "policy(instance): serve this policy from the driver_stub fixture"
    )


@pytest.fixture
def driver_stub(request) -> Iterator[EgodriverServiceStub]:
    """A stub talking to a real server running the test's policy.

    Choose the policy with ``@pytest.mark.policy(MyPolicy())``; without the
    marker it serves :class:`ConstantSpeedPolicy`.
    """
    marker = request.node.get_closest_marker("policy")
    policy: BaseDriver = marker.args[0] if marker else ConstantSpeedPolicy()
    server, port = build_server(policy, port=0, host="127.0.0.1")
    server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}", options=channel_options()) as channel:
            yield EgodriverServiceStub(channel)
    finally:
        server.stop(grace=1.0).wait()


# ---------------------------------------------------------------------------
# Plan builders, shared by the controller and service tests
# ---------------------------------------------------------------------------


def straight_plan(
    start_us: int = 0, speed: float = 8.0, horizon_s: float = 4.0, step_s: float = 0.1
) -> Trajectory:
    """A plan running straight along +x at a constant speed."""
    plan = Trajectory.empty()
    for i in range(1, int(horizon_s / step_s) + 1):
        dt = i * step_s
        plan.append(start_us + int(dt * 1e6), Pose.from_xyz_yaw(speed * dt, 0.0, 0.0, 0.0))
    return plan


def arc_plan(
    radius: float, speed: float = 8.0, horizon_s: float = 4.0, step_s: float = 0.1
) -> Trajectory:
    """A constant-radius turn from the origin, heading +x.

    Positive ``radius`` curves left, negative curves right.
    """
    plan = Trajectory.empty()
    for i in range(1, int(horizon_s / step_s) + 1):
        dt = i * step_s
        theta = speed * dt / radius
        plan.append(
            int(dt * 1e6),
            Pose.from_xyz_yaw(
                abs(radius) * math.sin(abs(theta)),
                radius * (1.0 - math.cos(theta)),
                0.0,
                theta,
            ),
        )
    return plan


def max_route_deviation(world, route: np.ndarray) -> float:
    """Furthest the driven path ever strays from ``route``, in metres."""
    path = np.stack([state.pose_local_to_rig.position for state in world.history])
    return max(float(np.min(np.linalg.norm(route[:, :2] - p[:2], axis=1))) for p in path)
