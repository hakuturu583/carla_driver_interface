# SPDX-License-Identifier: Apache-2.0
"""The compatibility claims, checked rather than asserted in prose.

Every entry in :data:`carla_driver_interface.compat.COMPAT_ENTRIES` marked
``COMPAT_LEVEL_EXACT`` should have a test here that would fail if the claim
stopped being true.
"""

from __future__ import annotations

import filecmp
import subprocess
import sys
from pathlib import Path

import pytest
from alpasim_grpc.v0 import egodriver_pb2, egodriver_pb2_grpc

from carla_driver_interface.compat import COMPAT_ENTRIES, build_report, describe_api_mismatch
from carla_driver_interface.driver.service import CarlaEgodriverServicer
from carla_driver_interface.grpc_api import (
    API_VERSION_MESSAGE,
    EGODRIVER_SERVICE_FULL_NAME,
    CarlaDriveSessionInfo,
    CarlaRendererData,
    CompatLevel,
    DriveRequest,
    DriveSessionRequest,
    EgodriverServiceServicer,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The contract, spelled out. Changing this list means changing what an alpasim
#: runtime can call, so it must be a deliberate edit rather than a side effect.
EXPECTED_RPCS = (
    "start_session",
    "close_session",
    "submit_image_observation",
    "submit_egomotion_observation",
    "submit_route",
    "submit_recording_ground_truth",
    "drive",
    "get_version",
)


def test_service_full_name_is_upstream():
    descriptor = egodriver_pb2.DESCRIPTOR.services_by_name["EgodriverService"]
    assert descriptor.full_name == EGODRIVER_SERVICE_FULL_NAME
    # The gRPC method path is derived from the full name; if it ever changed,
    # an alpasim runtime would get UNIMPLEMENTED instead of a plan.
    assert EGODRIVER_SERVICE_FULL_NAME == "egodriver.EgodriverService"


def test_all_upstream_rpcs_exist_and_are_implemented():
    descriptor = egodriver_pb2.DESCRIPTOR.services_by_name["EgodriverService"]
    assert tuple(m.name for m in descriptor.methods) == EXPECTED_RPCS

    for name in EXPECTED_RPCS:
        ours = getattr(CarlaEgodriverServicer, name)
        upstream = getattr(EgodriverServiceServicer, name)
        assert ours is not upstream, f"{name} is not overridden; it would return UNIMPLEMENTED"


def test_servicer_subclasses_the_generated_upstream_servicer():
    assert issubclass(CarlaEgodriverServicer, EgodriverServiceServicer)
    # Inheritance, not duck typing: the generated registration function looks up
    # methods on the instance, so a lookalike class would silently work until
    # upstream added an RPC.
    assert EgodriverServiceServicer in CarlaEgodriverServicer.__mro__


def test_registration_uses_the_upstream_handler_table():
    handlers = {}

    class _RecordingServer:
        def add_generic_rpc_handlers(self, generic_handlers):
            for handler in generic_handlers:
                handlers[handler.service_name()] = handler

    egodriver_pb2_grpc.add_EgodriverServiceServicer_to_server(
        CarlaEgodriverServicer(driver=None), _RecordingServer()
    )
    assert set(handlers) == {EGODRIVER_SERVICE_FULL_NAME}


@pytest.mark.parametrize(
    ("message", "field", "number"),
    [
        # A representative slice of the wire format. Field numbers are the
        # actual compatibility surface; names are not.
        (DriveRequest, "session_uuid", 1),
        (DriveRequest, "time_now_us", 3),
        (DriveRequest, "time_query_us", 4),
        (DriveRequest, "renderer_data", 5),
        (DriveSessionRequest, "session_uuid", 1),
        (DriveSessionRequest, "random_seed", 2),
        (DriveSessionRequest, "debug_info", 4),
        (DriveSessionRequest, "rollout_spec", 5),
    ],
)
def test_upstream_field_numbers(message, field, number):
    assert message.DESCRIPTOR.fields_by_name[field].number == number


def test_extension_messages_compose_upstream_types_rather_than_copying_them():
    info = CarlaDriveSessionInfo()
    # `base` must be the upstream message class itself, so the two can never
    # drift apart the way a hand-copied definition would.
    assert type(info.base) is DriveSessionRequest


def test_extension_payload_round_trips_through_the_upstream_bytes_field():
    data = CarlaRendererData(map_name="Town10HD_Opt", speed_limit_mps=13.9)
    request = DriveRequest(session_uuid="s", renderer_data=data.SerializeToString())

    decoded = CarlaRendererData()
    decoded.ParseFromString(request.renderer_data)
    assert decoded.map_name == "Town10HD_Opt"
    assert decoded.speed_limit_mps == pytest.approx(13.9)


def test_our_proto_declares_no_service():
    """The extension proto must not fork the service surface.

    Adding a service here would create a second, incompatible way to talk to a
    driver. Extensions ride inside the upstream ``bytes`` fields instead.
    """
    from carla_driver_interface.grpc_api.carla_driver.v0 import carla_driver_pb2

    assert not carla_driver_pb2.DESCRIPTOR.services_by_name


def test_api_version_is_forwarded_not_invented():
    report = build_report()
    assert report.alpasim_grpc_api_version == API_VERSION_MESSAGE
    assert describe_api_mismatch(API_VERSION_MESSAGE) is None

    other = type(API_VERSION_MESSAGE)(major=99, minor=0, patch=0)
    assert "mismatch" in (describe_api_mismatch(other) or "")


def test_compat_report_covers_every_level_and_is_populated():
    report = build_report()
    assert report.entries
    levels = {entry.level for entry in report.entries}
    assert CompatLevel.COMPAT_LEVEL_EXACT in levels
    assert CompatLevel.COMPAT_LEVEL_UNIMPLEMENTED in levels
    assert CompatLevel.COMPAT_LEVEL_UNSPECIFIED not in levels

    for entry in COMPAT_ENTRIES:
        assert entry.area and entry.alpasim_behaviour and entry.carla_behaviour


def test_compat_entries_are_mirrored_in_the_docs():
    """The prose and the code must not disagree about what differs."""
    doc = (REPO_ROOT / "docs" / "COMPATIBILITY.md").read_text(encoding="utf-8")
    for entry in COMPAT_ENTRIES:
        assert entry.area in doc, f"{entry.area!r} is missing from docs/COMPATIBILITY.md"


def _load_compile_protos():
    """Load ``scripts/compile_protos.py`` by path.

    Not by package name: the ``alpasim_grpc`` wheel ships its own top-level
    ``scripts`` package, so ``import scripts.compile_protos`` would resolve to
    upstream's copy.
    """
    import importlib.util

    path = REPO_ROOT / "scripts" / "compile_protos.py"
    spec = importlib.util.spec_from_file_location("_cdi_compile_protos", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_protos_are_up_to_date(tmp_path):
    """Regenerating must produce byte-identical output to what is committed."""
    pytest.importorskip("grpc_tools")
    module = _load_compile_protos()
    OUT_ROOT = module.OUT_ROOT

    module.compile_protos(tmp_path)

    relative = Path("carla_driver") / "v0"

    def sources(root: Path) -> list[str]:
        return sorted(p.name for p in (root / relative).iterdir() if p.is_file())

    committed = sources(OUT_ROOT)
    assert committed == sources(tmp_path)

    for name in committed:
        assert filecmp.cmp(OUT_ROOT / relative / name, tmp_path / relative / name, shallow=False), (
            f"{name} is stale; run `uv run python scripts/compile_protos.py`"
        )


def test_compat_report_cli_runs():
    result = subprocess.run(
        [sys.executable, "-m", "carla_driver_interface.cli", "compat-report"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
    )
    assert "EXACT" in result.stdout
    assert "UNIMPLEMENTED" in result.stdout
