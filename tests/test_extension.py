# SPDX-License-Identifier: Apache-2.0
"""The CARLA payloads carried inside alpasim's ``bytes`` extension points.

Pack and unpack live in one module precisely so they can be round-tripped here;
before that they sat in different processes' modules and handled a bad payload
two different ways.
"""

from __future__ import annotations

import pytest

from carla_driver_interface.grpc_api import (
    CarlaDriveDebugInfo,
    CarlaRendererData,
    CarlaWeather,
    DriveRequest,
    DriveResponse,
    TrafficLightState,
)
from carla_driver_interface.grpc_api.extension import (
    pack_debug_info,
    pack_renderer_data,
    unpack_debug_info,
    unpack_renderer_data,
)


def test_renderer_data_round_trips_through_the_upstream_field():
    original = CarlaRendererData(
        snapshot_timestamp_us=123_456,
        frame_id=42,
        map_name="Town10HD_Opt",
        weather=CarlaWeather(sun_altitude_angle=45.0, precipitation=10.0),
        ego_traffic_light=TrafficLightState.TRAFFIC_LIGHT_STATE_RED,
        ego_traffic_light_distance_m=12.5,
        speed_limit_mps=13.9,
    )
    # Through the actual upstream message, not just bytes-in-bytes-out.
    request = DriveRequest(session_uuid="s", renderer_data=pack_renderer_data(original))
    restored = unpack_renderer_data(request.renderer_data)

    assert restored == original


def test_debug_info_round_trips_through_the_upstream_field():
    original = CarlaDriveDebugInfo(
        policy_name="route_follower",
        inference_seconds=0.0123,
        scalars={"target_speed_mps": 8.0, "current_speed_mps": 7.25},
    )
    response = DriveResponse(
        debug_info=DriveResponse.DebugInfo(unstructured_debug_info=pack_debug_info(original))
    )
    restored = unpack_debug_info(response.debug_info.unstructured_debug_info)

    assert restored is not None
    assert restored.policy_name == "route_follower"
    assert restored.inference_seconds == pytest.approx(0.0123)
    assert dict(restored.scalars) == pytest.approx(
        {"target_speed_mps": 8.0, "current_speed_mps": 7.25}
    )


@pytest.mark.parametrize("unpack", [unpack_renderer_data, unpack_debug_info])
def test_an_empty_payload_is_none(unpack):
    """The field is optional; upstream simply may not set it."""
    assert unpack(b"") is None


@pytest.mark.parametrize("unpack", [unpack_renderer_data, unpack_debug_info])
def test_a_foreign_payload_is_none_rather_than_an_exception(unpack):
    """These fields are free-form, so a peer may legitimately put anything there.

    Both directions must agree on this: an unparseable payload means "no CARLA
    data", not "the rollout is broken".
    """
    assert unpack(b"\xff\xfe not a protobuf \x00\x01\x02\x03") is None


def test_an_all_default_message_is_indistinguishable_from_absent():
    """proto3 serialises an all-default message to zero bytes.

    So "the runtime sent an empty CarlaRendererData" and "the runtime sent
    nothing" arrive identically, and both unpack to ``None``. That is fine --
    a policy reading ``ctx.renderer_data`` has to handle ``None`` regardless --
    but it means the codec cannot be used to signal presence on its own.
    """
    assert pack_renderer_data(CarlaRendererData()) == b""
    assert unpack_renderer_data(pack_renderer_data(CarlaRendererData())) is None


def test_one_set_field_is_enough_to_survive_the_round_trip():
    restored = unpack_renderer_data(pack_renderer_data(CarlaRendererData(frame_id=1)))
    assert restored is not None
    assert restored.frame_id == 1
    # Unset fields come back as proto3 defaults, not as missing attributes.
    assert restored.map_name == ""
    assert restored.ego_traffic_light == TrafficLightState.TRAFFIC_LIGHT_STATE_UNKNOWN
