# SPDX-License-Identifier: Apache-2.0
"""Codec for the CARLA payloads that ride inside alpasim's ``bytes`` fields.

Upstream leaves two fields free-form so implementers can carry their own data
without forking the proto:

* ``egodriver.DriveRequest.renderer_data`` -- runtime to driver
* ``egodriver.DriveResponse.DebugInfo.unstructured_debug_info`` -- driver to runtime

Each is written by one process and read by the other, so pack and unpack would
naturally end up in different modules and drift apart -- including in how they
handle a payload they cannot parse. Keeping both halves here makes the pair
round-trippable in a single test and gives the tolerance policy one home.

**Tolerance policy:** unpacking never raises. These fields are free-form by
definition, so a peer -- an upstream alpasim runtime, or a driver from another
project -- may legitimately put something else there. An unparseable payload
means "no CARLA data", not "the rollout is broken".
"""

from __future__ import annotations

import logging

from google.protobuf.message import DecodeError

from carla_driver_interface.grpc_api.carla_driver.v0.carla_driver_pb2 import (
    CarlaDriveDebugInfo,
    CarlaRendererData,
)

logger = logging.getLogger(__name__)

__all__ = [
    "pack_debug_info",
    "pack_renderer_data",
    "unpack_debug_info",
    "unpack_renderer_data",
]


def pack_renderer_data(data: CarlaRendererData) -> bytes:
    """Serialize for ``DriveRequest.renderer_data``."""
    return data.SerializeToString()


def unpack_renderer_data(payload: bytes) -> CarlaRendererData | None:
    """Parse ``DriveRequest.renderer_data``; ``None`` if absent or foreign."""
    return _unpack(payload, CarlaRendererData, "renderer_data")


def pack_debug_info(debug: CarlaDriveDebugInfo) -> bytes:
    """Serialize for ``DriveResponse.DebugInfo.unstructured_debug_info``."""
    return debug.SerializeToString()


def unpack_debug_info(payload: bytes) -> CarlaDriveDebugInfo | None:
    """Parse the driver's debug payload; ``None`` if absent or foreign."""
    return _unpack(payload, CarlaDriveDebugInfo, "unstructured_debug_info")


def _unpack(payload: bytes, message_type: type, field: str):
    if not payload:
        return None
    message = message_type()
    try:
        message.ParseFromString(payload)
    except (DecodeError, UnicodeDecodeError):
        logger.debug("%s is not a %s; ignoring", field, message_type.__name__)
        return None
    return message
