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

__all__ = [
    "SUPPORTED_FORMATS",
    "bgra_to_rgb",
    "encode_bgra",
    "encode_rgb",
    "parse_image_format",
    "validate_image_format",
]

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
    return validate_image_format(value)


def validate_image_format(value: int) -> int:
    """Return ``value`` if we can encode it, else raise.

    Called from :class:`~carla_driver_interface.runtime.config.RuntimeConfig` so
    an unsupported format fails at construction. Left to the encoder it would
    instead fail inside a CARLA sensor callback thread, where the exception is
    logged and swallowed -- producing a rollout that runs to completion, submits
    zero images, and reports success.
    """
    if value not in SUPPORTED_FORMATS:
        name = ImageFormat.Name(value) if value in ImageFormat.values() else repr(value)
        raise ValueError(
            f"{name} cannot be encoded by this runtime; alpasim drivers decode PNG and JPEG only"
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
    validate_image_format(image_format)
    return _encode(Image.fromarray(np.ascontiguousarray(rgb), mode="RGB"), image_format, quality)


def encode_bgra(raw: bytes, width: int, height: int, image_format: int, quality: int = 90) -> bytes:
    """Encode CARLA's raw BGRA buffer directly, without going through numpy.

    Equivalent to ``encode_rgb(bgra_to_rgb(raw, w, h), ...)`` but skips two full
    frame copies (the ``uint8`` slice is non-contiguous, so the array path has to
    materialise it again before PIL will take it). At 960x604 that is about 4 MB
    of memcpy per frame; ``tests/test_images.py`` pins the two paths to the same
    pixels.
    """
    validate_image_format(image_format)
    expected = width * height * 4
    if len(raw) != expected:
        raise ValueError(
            f"expected {expected} bytes for a {width}x{height} BGRA image, got {len(raw)}"
        )
    # "BGRX" tells PIL to read 4-byte pixels as BGR and ignore the fourth.
    image = Image.frombuffer("RGB", (width, height), raw, "raw", "BGRX", 0, 1)
    return _encode(image, image_format, quality)


def _encode(image: Image.Image, image_format: int, quality: int) -> bytes:
    buffer = io.BytesIO()
    if image_format == ImageFormat.JPEG:
        image.save(buffer, format="JPEG", quality=quality)
    else:
        # compress_level 1: the runtime is in the closed loop, so encode speed
        # matters far more than a few percent of frame size.
        image.save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()
