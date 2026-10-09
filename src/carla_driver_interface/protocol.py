"""Single import window for the wire contract.

Message and service classes come from the vendored, generated ``_proto`` modules;
this module re-exports the ones either end of the contract uses, together with the
codec for the CARLA payloads that ride in alpasim's two free-form ``bytes`` fields:

* ``DriveRequest.renderer_data`` (runtime to policy): ``CarlaRendererData``
* ``DriveResponse.DebugInfo.unstructured_debug_info`` (policy to runtime):
  ``CarlaDriveDebugInfo``

and for the LiDAR sweeps ``CarlaRendererData.lidar`` carries packed.

Unpacking never raises: those fields are free-form by definition, so a peer may
legitimately put something else there, and an unparseable payload means "no CARLA
data", not "the rollout is broken".
"""

from __future__ import annotations

import logging
from typing import Optional, Type, TypeVar

import numpy as np
from google.protobuf.message import DecodeError, Message
from numpy.typing import NDArray

# The generated modules themselves are public too, for code that works with them
# directly (a runtime building requests field by field, say).
from ._proto import (
    carla_driver_pb2,
    common_pb2,
    egodriver_pb2,
    egodriver_pb2_grpc,
    sensorsim_pb2,
)

logger = logging.getLogger(__name__)

# -- alpasim messages ----------------------------------------------------------
AABB = common_pb2.AABB
DynamicState = common_pb2.DynamicState
Empty = common_pb2.Empty
Pose = common_pb2.Pose
PoseAtTime = common_pb2.PoseAtTime
Quat = common_pb2.Quat
SessionRequestStatus = common_pb2.SessionRequestStatus
Trajectory = common_pb2.Trajectory
Vec3 = common_pb2.Vec3
VersionId = common_pb2.VersionId

DriveRequest = egodriver_pb2.DriveRequest
DriveResponse = egodriver_pb2.DriveResponse
DriveSessionCloseRequest = egodriver_pb2.DriveSessionCloseRequest
DriveSessionRequest = egodriver_pb2.DriveSessionRequest
GroundTruth = egodriver_pb2.GroundTruth
GroundTruthRequest = egodriver_pb2.GroundTruthRequest
RolloutCameraImage = egodriver_pb2.RolloutCameraImage
RolloutEgoTrajectory = egodriver_pb2.RolloutEgoTrajectory
Route = egodriver_pb2.Route
RouteRequest = egodriver_pb2.RouteRequest

AvailableCamerasReturn = sensorsim_pb2.AvailableCamerasReturn
#: The nested camera message is deeply namespaced upstream.
AvailableCamera = sensorsim_pb2.AvailableCamerasReturn.AvailableCamera
CameraSpec = sensorsim_pb2.CameraSpec
ImageFormat = sensorsim_pb2.ImageFormat
OpenCVPinholeCameraParam = sensorsim_pb2.OpenCVPinholeCameraParam
ShutterType = sensorsim_pb2.ShutterType

EgodriverServiceServicer = egodriver_pb2_grpc.EgodriverServiceServicer
EgodriverServiceStub = egodriver_pb2_grpc.EgodriverServiceStub
add_EgodriverServiceServicer_to_server = (
    egodriver_pb2_grpc.add_EgodriverServiceServicer_to_server
)

# -- CARLA extension messages --------------------------------------------------
CarlaActorState = carla_driver_pb2.CarlaActorState
CarlaDriveDebugInfo = carla_driver_pb2.CarlaDriveDebugInfo
CarlaRendererData = carla_driver_pb2.CarlaRendererData
CarlaWeather = carla_driver_pb2.CarlaWeather
LidarSweep = carla_driver_pb2.LidarSweep
StopPoint = carla_driver_pb2.StopPoint
TrafficLight = carla_driver_pb2.TrafficLight
TrafficLightState = carla_driver_pb2.TrafficLightState

#: NVlabs/alpasim revision the contract was vendored from (proto/README.md).
ALPASIM_REV = "68709245a5dc0f2eda4f8cb2c3aa8cbdfa913043"

#: ``alpasim_grpc``'s package version at :data:`ALPASIM_REV`, which upstream reports
#: as its API version (``alpasim_grpc.API_VERSION_MESSAGE``).
ALPASIM_API_VERSION = common_pb2.VersionId.APIVersion(major=0, minor=55, patch=0)

#: The service this package speaks; asserted by the tests so it never forks.
EGODRIVER_SERVICE_FULL_NAME = "egodriver.EgodriverService"

#: alpasim submits whole camera frames as single unary messages, so gRPC's default
#: 4 MiB limit truncates anything much above 1080p. Both ends must agree -- a client
#: that can send more than the server accepts fails with RESOURCE_EXHAUSTED -- so the
#: number lives here, once.
MAX_MESSAGE_BYTES = 64 * 1024 * 1024


def channel_options() -> list[tuple[str, int]]:
    """gRPC options for every channel and server speaking this contract."""
    return [
        ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
        ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
    ]


_M = TypeVar("_M", bound=Message)

#: ``LidarSweep.points_xyzi`` is ``[N, 4]``: x, y, z, intensity.
LIDAR_POINT_COLUMNS = 4

#: The wire byte order of the packed LiDAR points, stated rather than left to the host.
_LIDAR_WIRE_DTYPE = np.dtype("<f4")


def pack_renderer_data(data: carla_driver_pb2.CarlaRendererData) -> bytes:
    """Serialize for ``DriveRequest.renderer_data``."""
    return data.SerializeToString()


def unpack_renderer_data(
    payload: bytes,
) -> Optional[carla_driver_pb2.CarlaRendererData]:
    """Parse ``DriveRequest.renderer_data``; ``None`` if absent or foreign."""
    return _unpack(payload, carla_driver_pb2.CarlaRendererData, "renderer_data")


def pack_debug_info(debug: carla_driver_pb2.CarlaDriveDebugInfo) -> bytes:
    """Serialize for ``DriveResponse.DebugInfo.unstructured_debug_info``."""
    return debug.SerializeToString()


def unpack_debug_info(payload: bytes) -> Optional[carla_driver_pb2.CarlaDriveDebugInfo]:
    """Parse a policy's debug payload; ``None`` if absent or foreign."""
    return _unpack(
        payload, carla_driver_pb2.CarlaDriveDebugInfo, "unstructured_debug_info"
    )


def _unpack(payload: bytes, message_type: Type[_M], field: str) -> Optional[_M]:
    if not payload:
        return None
    message = message_type()
    try:
        message.ParseFromString(payload)
    except (DecodeError, UnicodeDecodeError):
        logger.debug("%s is not a %s; ignoring", field, message_type.__name__)
        return None
    return message


def pack_lidar_sweep(
    logical_id: str,
    timestamp_us: int,
    points_xyzi_in_rig: NDArray[np.floating],
    rig_to_lidar: common_pb2.Pose,
) -> carla_driver_pb2.LidarSweep:
    """Build one ``LidarSweep`` from ``[N, 4]`` rig-frame points."""
    points = np.ascontiguousarray(points_xyzi_in_rig, dtype=_LIDAR_WIRE_DTYPE)
    if points.ndim != 2 or points.shape[1] != LIDAR_POINT_COLUMNS:
        raise ValueError(
            f"LiDAR points must be [N, {LIDAR_POINT_COLUMNS}] (x, y, z, intensity); "
            f"got shape {points.shape}"
        )
    return carla_driver_pb2.LidarSweep(
        logical_id=logical_id,
        timestamp_us=int(timestamp_us),
        rig_to_lidar=rig_to_lidar,
        num_points=int(points.shape[0]),
        points_xyzi=points.tobytes(),
    )


def unpack_lidar_points(sweep: carla_driver_pb2.LidarSweep) -> NDArray[np.float32]:
    """The sweep's points as a writable ``[N, 4]`` float32 array, rig frame."""
    return decode_lidar_points(
        sweep.logical_id, int(sweep.num_points), sweep.points_xyzi
    )


def decode_lidar_points(
    logical_id: str, num_points: int, points_xyzi: bytes
) -> NDArray[np.float32]:
    """Packed ``points_xyzi`` as a writable ``[N, 4]`` float32 array.

    Raises rather than truncating when ``num_points`` and the payload disagree:
    unlike the outer ``renderer_data`` bytes, a sweep that parsed as one is ours,
    and a short buffer means it was corrupted, not that it belongs to a peer.
    """
    expected = num_points * LIDAR_POINT_COLUMNS * _LIDAR_WIRE_DTYPE.itemsize
    if len(points_xyzi) != expected:
        raise ValueError(
            f"LiDAR sweep {logical_id!r} declares {num_points} points "
            f"({expected} bytes) but carries {len(points_xyzi)} bytes"
        )
    flat = np.frombuffer(points_xyzi, dtype=_LIDAR_WIRE_DTYPE)
    return flat.reshape(-1, LIDAR_POINT_COLUMNS).astype(np.float32, copy=True)
