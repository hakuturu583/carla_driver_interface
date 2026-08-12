# SPDX-License-Identifier: Apache-2.0
"""Single import window for the gRPC contract.

Everything the driver and the runtime exchange over the wire comes from
:mod:`alpasim_grpc` unchanged -- this module only re-exports it so call sites
read consistently, and adds the CARLA extension messages that ride inside the
upstream ``bytes`` extension points.

Importing upstream symbols from here (rather than reaching into
``alpasim_grpc``) keeps one place to look when upstream moves.
"""

from __future__ import annotations

from alpasim_grpc import API_VERSION_MESSAGE
from alpasim_grpc.v0.common_pb2 import (
    AABB,
    AvailableScenesReturn,
    DynamicState,
    Empty,
    Pose,
    PoseAtTime,
    Quat,
    SessionRequestStatus,
    StateAtTime,
    Trajectory,
    Vec3,
    VersionId,
)
from alpasim_grpc.v0.egodriver_pb2 import (
    DriveRequest,
    DriveResponse,
    DriveSessionCloseRequest,
    DriveSessionRequest,
    GroundTruth,
    GroundTruthRequest,
    RolloutCameraImage,
    RolloutEgoTrajectory,
    Route,
    RouteRequest,
)
from alpasim_grpc.v0.egodriver_pb2_grpc import (
    EgodriverServiceServicer,
    EgodriverServiceStub,
    add_EgodriverServiceServicer_to_server,
)
from alpasim_grpc.v0.runtime_pb2 import (
    RolloutErrorCode,
    RolloutSpec,
    SimulationReturn,
    TimeAggregation,
)
from alpasim_grpc.v0.sensorsim_pb2 import (
    AvailableCamerasReturn,
    CameraSpec,
    ImageFormat,
    OpenCVPinholeCameraParam,
    ShutterType,
)

from carla_driver_interface.grpc_api.carla_driver.v0.carla_driver_pb2 import (
    CarlaActorState,
    CarlaCompatReport,
    CarlaDriveDebugInfo,
    CarlaDriveSessionInfo,
    CarlaRendererData,
    CarlaWeather,
    CompatEntry,
    CompatLevel,
    TrafficLightState,
)

#: Convenience alias -- the nested camera message is deeply namespaced upstream.
AvailableCamera = AvailableCamerasReturn.AvailableCamera

#: The gRPC method prefix this project speaks. Kept as a constant so tests can
#: assert we never accidentally fork the service name.
EGODRIVER_SERVICE_FULL_NAME = "egodriver.EgodriverService"

__all__ = [
    "AABB",
    "API_VERSION_MESSAGE",
    "EGODRIVER_SERVICE_FULL_NAME",
    "AvailableCamera",
    "AvailableCamerasReturn",
    "AvailableScenesReturn",
    "CameraSpec",
    "CarlaActorState",
    "CarlaCompatReport",
    "CarlaDriveDebugInfo",
    "CarlaDriveSessionInfo",
    "CarlaRendererData",
    "CarlaWeather",
    "CompatEntry",
    "CompatLevel",
    "DriveRequest",
    "DriveResponse",
    "DriveSessionCloseRequest",
    "DriveSessionRequest",
    "DynamicState",
    "EgodriverServiceServicer",
    "EgodriverServiceStub",
    "Empty",
    "GroundTruth",
    "GroundTruthRequest",
    "ImageFormat",
    "OpenCVPinholeCameraParam",
    "Pose",
    "PoseAtTime",
    "Quat",
    "RolloutCameraImage",
    "RolloutEgoTrajectory",
    "RolloutErrorCode",
    "RolloutSpec",
    "Route",
    "RouteRequest",
    "SessionRequestStatus",
    "ShutterType",
    "SimulationReturn",
    "StateAtTime",
    "TimeAggregation",
    "TrafficLightState",
    "Trajectory",
    "Vec3",
    "VersionId",
    "add_EgodriverServiceServicer_to_server",
]
