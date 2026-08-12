# SPDX-License-Identifier: Apache-2.0
"""Encoding CARLA camera output into what ``submit_image_observation`` carries.

CARLA hands back a raw BGRA buffer; alpasim's driver contract carries encoded
bytes with the format implied by ``sensorsim.ImageFormat``.  In practice
upstream drivers decode PNG and JPEG, so those are the two we emit.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from carla_driver_interface.grpc_api import ImageFormat

__all__ = ["SUPPORTED_FORMATS", "bgra_to_rgb", "encode_rgb", "parse_image_format"]

#: The formats a CARLA rollout can produce. ``sensorsim.proto`` names more
#: (JPEG2000, AVC, AV1, planar RGB), but alpasim itself only supports PNG and
#: JPEG, so widening this would break the very drivers we target.
SUPPORTED_FORMATS = (ImageFormat.PNG, ImageFormat.JPEG)


def parse_image_format(name: str) -> int:
    """``"png"`` / ``"jpeg"`` -> the ``ImageFormat`` enum value."""
    key = name.strip().upper()
    if key == "JPG":
        key = "JPEG"
    try:
        value = ImageFormat.Value(key)
    except ValueError as exc:
        raise ValueError(
            f"unknown image format {name!r}; supported: "
            + ", ".join(ImageFormat.Name(f).lower() for f in SUPPORTED_FORMATS)
        ) from exc
    if value not in SUPPORTED_FORMATS:
        raise ValueError(
            f"{ImageFormat.Name(value)} is defined in sensorsim.proto but not supported "
            "by alpasim drivers; use png or jpeg"
        )
    return value


def bgra_to_rgb(raw: bytes, width: int, height: int) -> np.ndarray:
    """CARLA's raw BGRA bytes -> an ``(H, W, 3)`` uint8 RGB array."""
    expected = width * height * 4
    if len(raw) != expected:
        raise ValueError(
            f"expected {expected} bytes for a {width}x{height} BGRA image, got {len(raw)}"
        )
    bgra = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
    return bgra[:, :, 2::-1]  # BGRA -> RGB, dropping alpha


def encode_rgb(rgb: np.ndarray, image_format: int, quality: int = 90) -> bytes:
    """Encode an ``(H, W, 3)`` uint8 RGB array."""
    if image_format not in SUPPORTED_FORMATS:
        raise ValueError(f"unsupported image format {image_format}")
    buffer = io.BytesIO()
    image = Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")
    if image_format == ImageFormat.JPEG:
        image.save(buffer, format="JPEG", quality=quality)
    else:
        # compress_level 1: the runtime is in the closed loop, so encode speed
        # matters far more than a few percent of frame size.
        image.save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()
