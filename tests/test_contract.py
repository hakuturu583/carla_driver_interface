"""Contract revisions: every client release pairs with every server release."""

from __future__ import annotations

from concurrent import futures
from typing import Dict, Iterator, Optional, Tuple

import grpc
import numpy as np
import pytest

from carla_driver_interface.contract import (
    CONTRACT_REVISION,
    LEGACY_REVISION,
    METADATA_KEY,
    ContractError,
    camera_to_revision,
    contract_metadata,
    negotiate,
    read_revision,
)
from carla_driver_interface.driver import BaseDriver, DriveContext, DriveResult
from carla_driver_interface.geometry import (
    Pose,
    Trajectory,
    body_to_optical,
    optical_to_body,
)
from carla_driver_interface.protocol import (
    AvailableCamera,
    DriveSessionRequest,
    EgodriverServiceServicer,
    EgodriverServiceStub,
    Empty,
    VersionId,
    add_EgodriverServiceServicer_to_server,
)
from carla_driver_interface.server import serving
from carla_driver_interface.testing import FakeCamera, FakeLoop

#: Pitched down and yawed left, so a frame mix-up cannot pass by symmetry.
CAMERA = FakeCamera(pitch_down_deg=4.0, yaw_left_deg=7.0)
OPTICAL = body_to_optical(CAMERA.pose_in_rig())


def _same(a: Pose, b: Pose) -> bool:
    # rig_to_camera crosses the wire as float32.
    return bool(
        np.allclose(a.position, b.position, atol=1e-5)
        and (a.rotation * b.rotation.inv()).magnitude() < 1e-6
    )


class _Cameras(BaseDriver):
    """Records the cameras the policy is handed."""

    name = "cameras"

    def __init__(self) -> None:
        self.seen: Dict[str, AvailableCamera] = {}

    def on_session_start(self, session) -> None:  # type: ignore[no-untyped-def]
        self.seen = dict(session.cameras)

    def drive(self, ctx: DriveContext) -> DriveResult:
        return DriveResult(trajectory_in_rig=Trajectory.empty())

    def pose(self) -> Pose:
        return Pose.from_proto(self.seen[CAMERA.logical_id].rig_to_camera)


def _start(
    port: int, camera: AvailableCamera, revision: Optional[int]
) -> grpc.RpcError | None:
    """``start_session`` as a client of *revision* sends it (``None``: declares none)."""
    request = DriveSessionRequest(session_uuid="s")
    request.rollout_spec.vehicle.available_cameras.append(camera)
    metadata = None if revision is None else contract_metadata(revision)
    with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
        try:
            EgodriverServiceStub(channel).start_session(
                request, timeout=10, metadata=metadata
            )
        except grpc.RpcError as error:
            return error
    return None


def test_the_body_and_optical_frames_differ_by_a_fixed_rotation() -> None:
    body = CAMERA.pose_in_rig()
    optical = body_to_optical(body)
    assert _same(optical_to_body(optical), body)
    # The optical axis is the body's x, image right its -y, image down its -z.
    axes = body.rotation.inv() * optical.rotation
    assert np.allclose(axes.apply([0, 0, 1]), [1, 0, 0])
    assert np.allclose(axes.apply([1, 0, 0]), [0, -1, 0])
    assert np.allclose(axes.apply([0, 1, 0]), [0, 0, -1])


def test_this_release_hands_a_policy_the_optical_frame() -> None:
    policy = _Cameras()
    with serving(policy, port=0, host="127.0.0.1") as port:
        with FakeLoop(f"127.0.0.1:{port}", [CAMERA]) as loop:
            loop.run(1)
    assert _same(policy.pose(), OPTICAL)


@pytest.mark.parametrize(
    ("mode", "revision", "declared"),
    [
        # autoware_carla_scenario 4.x: declares nothing, sends the camera body.
        ("carla", None, CAMERA.pose_in_rig()),
        ("carla", LEGACY_REVISION, CAMERA.pose_in_rig()),
        ("carla", CONTRACT_REVISION, OPTICAL),
        # An upstream alpasim runtime: declares nothing, sends the optical frame.
        ("alpasim", None, OPTICAL),
        ("alpasim", LEGACY_REVISION, CAMERA.pose_in_rig()),
    ],
)
def test_every_client_revision_reaches_the_policy_as_the_optical_frame(
    mode: str, revision: Optional[int], declared: Pose
) -> None:
    camera = CAMERA.available_camera()
    camera.rig_to_camera.CopyFrom(declared.to_proto())
    policy = _Cameras()
    with serving(policy, port=0, host="127.0.0.1", mode=mode) as port:  # type: ignore[arg-type]
        assert _start(port, camera, revision) is None
    assert _same(policy.pose(), OPTICAL)


def test_a_revision_this_release_does_not_know_is_refused() -> None:
    with serving(_Cameras(), port=0, host="127.0.0.1") as port:
        error = _start(port, CAMERA.available_camera(), CONTRACT_REVISION + 1)
    assert error is not None
    assert error.code() == grpc.StatusCode.FAILED_PRECONDITION
    assert "revision" in (error.details() or "")


def test_the_server_declares_its_revision() -> None:
    with serving(_Cameras(), port=0, host="127.0.0.1") as port:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            _, revision = negotiate(EgodriverServiceStub(channel), timeout=10)
    assert revision == CONTRACT_REVISION


class _LegacyServer(EgodriverServiceServicer):
    """carla-driver-interface 1.x as far as revisions go: it declares none."""

    def get_version(self, request: Empty, context: grpc.ServicerContext) -> VersionId:
        return VersionId(version_id="legacy-carla-driver-interface-1.1.0")


@pytest.fixture
def legacy_server() -> Iterator[int]:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    add_EgodriverServiceServicer_to_server(_LegacyServer(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    yield port
    server.stop(grace=None)


def test_a_server_that_declares_nothing_gets_the_camera_body(
    legacy_server: int,
) -> None:
    with grpc.insecure_channel(f"127.0.0.1:{legacy_server}") as channel:
        version, revision = negotiate(EgodriverServiceStub(channel), timeout=10)
    assert version.version_id.startswith("legacy")
    assert revision == LEGACY_REVISION
    declared = Pose.from_proto(CAMERA.available_camera(revision).rig_to_camera)
    assert _same(declared, CAMERA.pose_in_rig())


def test_translating_a_camera_leaves_everything_but_its_rotation() -> None:
    camera = CAMERA.available_camera()
    legacy = camera_to_revision(camera, CONTRACT_REVISION, LEGACY_REVISION)
    assert legacy.intrinsics == camera.intrinsics
    assert legacy.logical_id == camera.logical_id
    assert _same(Pose.from_proto(camera.rig_to_camera), OPTICAL)  # input untouched
    back = camera_to_revision(legacy, LEGACY_REVISION, CONTRACT_REVISION)
    assert _same(Pose.from_proto(back.rig_to_camera), OPTICAL)


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ((), 7),
        (((METADATA_KEY, "1"),), 1),
        (((METADATA_KEY, b"2"),), 2),
        ((("other", "3"), (METADATA_KEY, "2")), 2),
    ],
)
def test_revisions_read_from_metadata(
    metadata: Tuple[Tuple[str, object], ...], expected: int
) -> None:
    assert read_revision(metadata, default=7) == expected


@pytest.mark.parametrize("value", ["two", "0", "3"])
def test_unreadable_or_unknown_revisions_are_errors(value: str) -> None:
    with pytest.raises(ContractError):
        read_revision(((METADATA_KEY, value),), default=1)
