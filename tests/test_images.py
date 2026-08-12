# SPDX-License-Identifier: Apache-2.0
"""Camera encoding.

The BGRA fast path skips numpy entirely, so it needs pinning against the array
path it replaced -- otherwise a channel-order mistake would only show up as
subtly wrong colours in a real CARLA run, which CI cannot see.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from carla_driver_interface.grpc_api import ImageFormat
from carla_driver_interface.runtime.images import (
    bgra_to_rgb,
    encode_bgra,
    encode_rgb,
    parse_image_format,
    validate_image_format,
)


def random_bgra(width: int, height: int, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 4), dtype=np.uint8).tobytes()


def decode(payload: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(payload)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def test_bgra_to_rgb_reorders_channels_and_drops_alpha():
    # One pixel: B=1, G=2, R=3, A=4 -> RGB (3, 2, 1).
    rgb = bgra_to_rgb(bytes([1, 2, 3, 4]), width=1, height=1)
    assert rgb.shape == (1, 1, 3)
    assert list(rgb[0, 0]) == [3, 2, 1]


def test_bgra_to_rgb_rejects_a_wrong_sized_buffer():
    with pytest.raises(ValueError, match="expected 16 bytes"):
        bgra_to_rgb(b"\x00" * 8, width=2, height=2)


def test_encode_bgra_matches_the_array_path_pixel_for_pixel():
    """The whole point of the fast path is that it changes nothing but the cost."""
    width, height = 37, 23
    raw = random_bgra(width, height)

    via_array = decode(encode_rgb(bgra_to_rgb(raw, width, height), ImageFormat.PNG))
    direct = decode(encode_bgra(raw, width, height, ImageFormat.PNG))

    assert np.array_equal(via_array, direct)
    assert direct.shape == (height, width, 3)


def test_encode_bgra_round_trips_the_original_pixels():
    """PNG is lossless, so the decode must return exactly what went in."""
    width, height = 16, 9
    raw = random_bgra(width, height, seed=7)
    assert np.array_equal(
        decode(encode_bgra(raw, width, height, ImageFormat.PNG)),
        bgra_to_rgb(raw, width, height),
    )


def test_encode_bgra_rejects_a_wrong_sized_buffer():
    with pytest.raises(ValueError, match="expected 16 bytes"):
        encode_bgra(b"\x00" * 8, 2, 2, ImageFormat.PNG)


@pytest.mark.parametrize("image_format", [ImageFormat.PNG, ImageFormat.JPEG])
def test_both_supported_formats_encode(image_format):
    payload = encode_bgra(random_bgra(8, 8), 8, 8, image_format)
    assert decode(payload).shape == (8, 8, 3)


def test_unsupported_formats_are_rejected_by_name():
    """sensorsim.proto names formats alpasim drivers cannot decode."""
    with pytest.raises(ValueError, match="PNG and JPEG only"):
        validate_image_format(ImageFormat.JPEG2000)
    with pytest.raises(ValueError, match="PNG and JPEG only"):
        parse_image_format("jpeg2000")


def test_parse_image_format_accepts_the_usual_spellings():
    assert parse_image_format("png") == ImageFormat.PNG
    assert parse_image_format("JPEG") == ImageFormat.JPEG
    assert parse_image_format("jpg") == ImageFormat.JPEG
    with pytest.raises(ValueError, match="unknown image format"):
        parse_image_format("webp")
