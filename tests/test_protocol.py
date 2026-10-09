"""The wire contract: service identity and the CARLA payload codec."""

from __future__ import annotations

from carla_driver_interface import protocol
from carla_driver_interface._proto import egodriver_pb2


def test_service_name_is_upstreams() -> None:
    service = egodriver_pb2.DESCRIPTOR.services_by_name["EgodriverService"]
    assert service.full_name == protocol.EGODRIVER_SERVICE_FULL_NAME


def test_payloads_round_trip() -> None:
    data = protocol.CarlaRendererData(speed_limit_mps=13.9, map_name="Town10HD_Opt")
    assert protocol.unpack_renderer_data(protocol.pack_renderer_data(data)) == data
    debug = protocol.CarlaDriveDebugInfo(policy_name="p", scalars={"a": 1.0})
    assert protocol.unpack_debug_info(protocol.pack_debug_info(debug)) == debug


def test_foreign_or_missing_payloads_are_none() -> None:
    assert protocol.unpack_renderer_data(b"") is None
    assert protocol.unpack_debug_info(b"\xff\xff\xff") is None


def test_channel_options_raise_the_message_limit() -> None:
    options = dict(protocol.channel_options())
    assert options["grpc.max_send_message_length"] == protocol.MAX_MESSAGE_BYTES
    assert options["grpc.max_receive_message_length"] == protocol.MAX_MESSAGE_BYTES
