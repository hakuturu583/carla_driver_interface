# SPDX-License-Identifier: Apache-2.0
"""The runtime half: CARLA standing in for alpasim's simulation services."""

from __future__ import annotations

from carla_driver_interface.runtime.carla_runtime import CarlaRuntime, RolloutOutcome
from carla_driver_interface.runtime.carla_world import CarlaWorldAdapter, load_carla_module
from carla_driver_interface.runtime.config import (
    CameraConfig,
    RuntimeConfig,
    ScenarioSpec,
    default_camera_rig,
)
from carla_driver_interface.runtime.control import (
    ControlConfig,
    TrajectoryFollower,
    VehicleCommand,
)
from carla_driver_interface.runtime.route import RouteProvider
from carla_driver_interface.runtime.world import (
    CameraCapture,
    EgoState,
    RolloutEvents,
    WorldAdapter,
    WorldSetup,
    WorldSnapshot,
)

__all__ = [
    "CameraCapture",
    "CameraConfig",
    "CarlaRuntime",
    "CarlaWorldAdapter",
    "ControlConfig",
    "EgoState",
    "RolloutEvents",
    "RolloutOutcome",
    "RouteProvider",
    "RuntimeConfig",
    "ScenarioSpec",
    "TrajectoryFollower",
    "VehicleCommand",
    "WorldAdapter",
    "WorldSetup",
    "WorldSnapshot",
    "default_camera_rig",
    "load_carla_module",
]
