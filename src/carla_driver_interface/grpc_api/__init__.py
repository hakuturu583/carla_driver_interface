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

#: alpasim submits whole camera frames as single unary messages, so the default
#: 4 MiB limit truncates anything much above 1080p. Both ends of the channel
#: must agree -- a client that can send more than the server accepts fails with
#: RESOURCE_EXHAUSTED, which reads like a driver bug -- so the number lives here
#: rather than being declared once per end.
MAX_MESSAGE_BYTES = 64 * 1024 * 1024


def channel_options() -> list[tuple[str, int]]:
    """gRPC options every channel and server in this package is built with."""
    return [
        ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
        ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
    ]


def describe_api_mismatch(other: VersionId.APIVersion) -> str | None:
    """Compare a peer's alpasim API version to ours; return a message if they differ.

    Lives beside :data:`API_VERSION_MESSAGE` because it is a statement about the
    wire contract. Returns ``None`` when the versions match.
    """
    ours = API_VERSION_MESSAGE
    if (other.major, other.minor, other.patch) == (ours.major, ours.minor, ours.patch):
        return None
    return (
        f"alpasim_grpc API version mismatch: peer reports "
        f"{other.major}.{other.minor}.{other.patch}, we were built against "
        f"{ours.major}.{ours.minor}.{ours.patch}. "
        "Compatible unless the messages in use actually changed between those releases."
    )


__all__ = [
    "AABB",
    "API_VERSION_MESSAGE",
    "EGODRIVER_SERVICE_FULL_NAME",
    "MAX_MESSAGE_BYTES",
    "channel_options",
    "describe_api_mismatch",
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
