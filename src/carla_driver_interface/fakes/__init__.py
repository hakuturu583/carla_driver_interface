# SPDX-License-Identifier: Apache-2.0
"""Test doubles that let the closed loop run without a CARLA server."""

from __future__ import annotations

from carla_driver_interface.fakes.fake_world import FakeWorld, straight_then_curve_route

__all__ = ["FakeWorld", "straight_then_curve_route"]
