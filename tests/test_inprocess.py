"""The in-process channel is the wire without the socket: same bytes, same errors.

Each test runs one script of calls over every transport and compares what the
client saw -- every request's bytes, every response's bytes or status, both
metadata -- with what it saw over TCP.
"""

from __future__ import annotations

import pickle
import time
from typing import Any, Callable, List, Tuple

import grpc
import numpy as np
import pytest
from conftest import TRANSPORTS, Transport

from carla_driver_interface.contract import CONTRACT_REVISION, contract_metadata
from carla_driver_interface.driver import BaseDriver, DriveContext, DriveResult
from carla_driver_interface.geometry import Pose, Trajectory
from carla_driver_interface.inprocess import InProcessChannel, in_process_channel
from carla_driver_interface.protocol import (
    MAX_MESSAGE_BYTES,
    CarlaRendererData,
    CarlaVehicleServiceStub,
    DriveRequest,
    DriveResponse,
    DriveSessionCloseRequest,
    DriveSessionRequest,
    EgodriverServiceServicer,
    EgodriverServiceStub,
    Empty,
    GroundTruthRequest,
    RolloutCameraImage,
    RolloutEgoTrajectory,
    RouteRequest,
    SessionRequestStatus,
    VehicleCommandRequest,
    VersionId,
    add_EgodriverServiceServicer_to_server,
    pack_lidar_sweep,
    pack_renderer_data,
    unpack_debug_info,
)
from carla_driver_interface.testing import FakeCamera

Outcome = Tuple[Any, ...]


class _Recorder(grpc.UnaryUnaryClientInterceptor):
    """Every call as the client saw it: request bytes, then response bytes or status."""

    def __init__(self) -> None:
        self.calls: List[Outcome] = []

    def intercept_unary_unary(self, continuation, details, request):  # type: ignore[no-untyped-def]
        outcome = continuation(details, request)
        error = outcome.exception()
        metadata = (
            tuple(map(tuple, outcome.initial_metadata() or ())),
            tuple(map(tuple, outcome.trailing_metadata() or ())),
        )
        if error is not None:
            result: Outcome = ("error", error.code(), error.details(), *metadata)
        else:
            result = ("ok", _normalized(outcome.result()), *metadata)
        self.calls.append((details.method, request.SerializeToString(), result))
        return outcome


def _normalized(response: Any) -> bytes:
    """The response's bytes, the drive's own wall time (not the wire's) zeroed."""
    if isinstance(response, DriveResponse):
        raw = response.debug_info.unstructured_debug_info
        if raw.startswith(b"\x80"):  # alpasim mode's pickled dict
            extra = pickle.loads(raw)
            extra["inference_seconds"] = 0.0
            response.debug_info.unstructured_debug_info = pickle.dumps(extra)
        elif (debug := unpack_debug_info(raw)) is not None:
            debug.inference_seconds = 0.0
            response.debug_info.unstructured_debug_info = debug.SerializeToString()
    return response.SerializeToString()


def _record(
    transport: Transport,
    serve: Callable[[Transport], Any],
    script: Callable[[grpc.Channel], None],
) -> List[Outcome]:
    recorder = _Recorder()
    with serve(transport) as channel:
        script(grpc.intercept_channel(channel, recorder))
    return recorder.calls


def _same_everywhere(
    serve: Callable[[Transport], Any], script: Callable[[grpc.Channel], None]
):
    recorded = {name: _record(Transport(name), serve, script) for name in TRANSPORTS}
    for name in TRANSPORTS:
        assert recorded[name] == recorded["tcp"], name
    return recorded["tcp"]


def _try(call: Callable[[], Any]) -> None:
    try:
        call()
    except grpc.RpcError:
        pass


# -- a whole session, errors included ----------------------------------------------


class _Planner(BaseDriver):
    """Deterministic: a plan from the route and the sweep's size."""

    name = "planner"

    def drive(self, ctx: DriveContext) -> DriveResult:
        route = ctx.session.route_waypoints_in_rig
        points = len(ctx.renderer_data.lidar[0].points_xyzi) if ctx.renderer_data else 0
        plan = Trajectory.empty()
        for k, (x, y, _) in enumerate(route[:5], start=1):
            plan.append(
                ctx.time_now_us + k * 100_000, Pose.from_xyz_yaw(x, y, 0.0, 0.1 * k)
            )
        return DriveResult(
            trajectory_in_rig=plan, debug_scalars={"points": float(points)}
        )


def _sweep_payload() -> bytes:
    rng = np.random.default_rng(7)
    points = rng.normal(size=(120_000, 4)).astype(
        np.float32
    )  # ~1.9 MB, as a 128-ch sweep
    sweep = pack_lidar_sweep("lidar_top", 50_000, points, Pose.identity().to_proto())
    return pack_renderer_data(CarlaRendererData(map_id="Town10HD_Opt", lidar=[sweep]))


def _session(channel: grpc.Channel) -> None:
    stub, vehicle = EgodriverServiceStub(channel), CarlaVehicleServiceStub(channel)
    stub.get_version.with_call(Empty(), timeout=10)
    start = DriveSessionRequest(session_uuid="s", random_seed=3)
    start.rollout_spec.vehicle.available_cameras.append(FakeCamera().available_camera())
    stub.start_session(start, timeout=10, metadata=contract_metadata(CONTRACT_REVISION))
    _try(lambda: stub.start_session(start, timeout=10))  # ALREADY_EXISTS
    route = RouteRequest(session_uuid="s")
    route.route.timestamp_us = 50_000
    for k in range(8):
        route.route.waypoints.add(x=2.0 * k, y=0.1 * k, z=0.0)
    stub.submit_route(route, timeout=10)
    ego = RolloutEgoTrajectory(session_uuid="s")
    ego.trajectory.CopyFrom(
        Trajectory(
            [0, 50_000], [Pose.identity(), Pose.from_xyz_yaw(0.4, 0, 0, 0)]
        ).to_proto()
    )
    stub.submit_egomotion_observation(ego, timeout=10)
    bad = RolloutEgoTrajectory(session_uuid="s")
    bad.CopyFrom(ego)
    bad.dynamic_states.add()  # one state for two poses: INVALID_ARGUMENT
    _try(lambda: stub.submit_egomotion_observation(bad, timeout=10))
    image = RolloutCameraImage(session_uuid="s")
    image.camera_image.logical_id = "no_such_camera"
    _try(lambda: stub.submit_image_observation(image, timeout=10))  # INVALID_ARGUMENT
    stub.submit_recording_ground_truth(GroundTruthRequest(session_uuid="s"), timeout=10)
    stub.drive(
        DriveRequest(
            session_uuid="s",
            time_now_us=50_000,
            time_query_us=150_000,
            renderer_data=_sweep_payload(),
        ),
        timeout=10,
    )
    vehicle.get_vehicle_command(VehicleCommandRequest(session_uuid="s"), timeout=10)
    _try(
        lambda: stub.drive(DriveRequest(session_uuid="nobody"), timeout=10)
    )  # NOT_FOUND
    huge = DriveRequest(session_uuid="s", renderer_data=b"\0" * MAX_MESSAGE_BYTES)
    _try(lambda: stub.drive(huge, timeout=10))  # RESOURCE_EXHAUSTED, client side
    stub.close_session(DriveSessionCloseRequest(session_uuid="s"), timeout=10)
    stub.close_session(
        DriveSessionCloseRequest(session_uuid="s"), timeout=10
    )  # unknown: Empty
    _try(
        lambda: vehicle.get_vehicle_command(
            VehicleCommandRequest(session_uuid="s"), timeout=10
        )
    )


@pytest.mark.parametrize("mode", ["carla", "alpasim"])
def test_a_session_is_the_same_on_every_transport(mode: str) -> None:
    calls = _same_everywhere(lambda t: t.serve(_Planner(), mode=mode), _session)
    statuses = [result[1] for _, _, result in calls if result[0] == "error"]
    assert statuses == [
        grpc.StatusCode.ALREADY_EXISTS,
        grpc.StatusCode.INVALID_ARGUMENT,
        grpc.StatusCode.INVALID_ARGUMENT,
        grpc.StatusCode.NOT_FOUND,
        grpc.StatusCode.RESOURCE_EXHAUSTED,
        grpc.StatusCode.NOT_FOUND,
    ]
    drive = next(r for m, _, r in calls if m.endswith("/drive") and r[0] == "ok")
    assert len(DriveResponse.FromString(drive[1]).trajectory.poses) == 6


# -- grpc's server semantics, edge by edge -------------------------------------------


class _Edges(EgodriverServiceServicer):
    """A servicer that does every odd thing a servicer may do to its context."""

    def get_version(self, request, context):  # type: ignore[no-untyped-def]
        context.send_initial_metadata((("x-first", "1"),))
        context.send_initial_metadata((("x-second", "2"),))  # ValueError
        return VersionId()

    def start_session(self, request, context):  # type: ignore[no-untyped-def]
        context.abort(grpc.StatusCode.OK, "not ok")  # UNKNOWN, no details

    def close_session(self, request, context):  # type: ignore[no-untyped-def]
        context.set_code(grpc.StatusCode.NOT_FOUND)
        context.set_details("gone")
        return Empty()

    def submit_route(self, request, context):  # type: ignore[no-untyped-def]
        context.send_initial_metadata((("x-initial", "i"),))
        context.set_trailing_metadata((("x-trailing", "t"), ("x-raw-bin", b"\x00\x01")))
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, "bad route")

    def submit_image_observation(self, request, context):  # type: ignore[no-untyped-def]
        raise ValueError("boom")

    def submit_egomotion_observation(self, request, context):  # type: ignore[no-untyped-def]
        return None  # cannot be serialized

    def submit_recording_ground_truth(self, request, context):  # type: ignore[no-untyped-def]
        # What the client declared, back as trailing metadata (grpc adds its own,
        # user-agent and the like, which differ by transport by nature).
        echoed = [
            (k, v) for k, v in context.invocation_metadata() if k.startswith("x-")
        ]
        context.set_trailing_metadata(tuple(echoed))
        assert context.is_active() and context.time_remaining() is not None
        return Empty()

    def drive(self, request, context):  # type: ignore[no-untyped-def]
        if request.session_uuid == "slow":
            time.sleep(0.3)
            return DriveResponse()
        if request.session_uuid == "slow-abort":
            # Aborts after the client's deadline: the client sees the deadline.
            time.sleep(0.3)
            context.abort(grpc.StatusCode.NOT_FOUND, "too late")
        try:
            context.abort(grpc.StatusCode.PERMISSION_DENIED, "swallowed")
        except Exception:  # noqa: BLE001 - the point: the abort stands
            pass
        return DriveResponse()


def _edges(channel: grpc.Channel) -> None:
    stub = EgodriverServiceStub(channel)
    for call in (
        lambda: stub.get_version(Empty(), timeout=10),
        lambda: stub.start_session(DriveSessionRequest(), timeout=10),
        lambda: stub.close_session(DriveSessionCloseRequest(), timeout=10),
        lambda: stub.submit_route(RouteRequest(), timeout=10),
        lambda: stub.submit_image_observation(RolloutCameraImage(), timeout=10),
        lambda: stub.submit_egomotion_observation(RolloutEgoTrajectory(), timeout=10),
        lambda: stub.submit_recording_ground_truth(
            GroundTruthRequest(),
            timeout=10,
            metadata=(("x-str", "v"), ("x-bytes", b"w"), ("x-data-bin", b"\xff\x00")),
        ),
        lambda: stub.drive(DriveRequest(), timeout=10),
        lambda: stub.drive(DriveRequest(session_uuid="slow"), timeout=0.05),
        lambda: stub.get_version(Empty(), timeout=-1),
        lambda: stub.get_version(Empty(), metadata=(("Upper", "1"),)),
        lambda: stub.get_version(Empty(), timeout=0),
        lambda: stub.drive(DriveRequest(session_uuid="slow-abort"), timeout=0.05),
    ):
        _try(call)


def _serve_edges(transport: Transport):  # type: ignore[no-untyped-def]
    return transport.serve_servicers(
        lambda server: add_EgodriverServiceServicer_to_server(_Edges(), server)
    )


def test_servicer_context_semantics_are_grpcs() -> None:
    calls = _same_everywhere(_serve_edges, _edges)
    results = [result for _, _, result in calls]
    assert results[0][:3] == (
        "error",
        grpc.StatusCode.UNKNOWN,
        "Exception calling application: Initial metadata no longer allowed!",
    )
    assert results[1][:3] == ("error", grpc.StatusCode.UNKNOWN, "")
    assert results[2][:3] == ("error", grpc.StatusCode.NOT_FOUND, "gone")
    assert results[3] == (
        "error",
        grpc.StatusCode.INVALID_ARGUMENT,
        "bad route",
        (("x-initial", "i"),),
        (("x-trailing", "t"), ("x-raw-bin", b"\x00\x01")),
    )
    assert results[4][:3] == (
        "error",
        grpc.StatusCode.UNKNOWN,
        "Exception calling application: boom",
    )
    assert results[5][:3] == (
        "error",
        grpc.StatusCode.INTERNAL,
        "Failed to serialize response!",
    )
    assert results[6][0] == "ok"
    assert results[6][3] == (
        ("x-str", "v"),
        ("x-bytes", "w"),
        ("x-data-bin", b"\xff\x00"),
    )
    assert results[7][:3] == ("error", grpc.StatusCode.PERMISSION_DENIED, "swallowed")
    assert results[8][:3] == (
        "error",
        grpc.StatusCode.DEADLINE_EXCEEDED,
        "Deadline Exceeded",
    )
    assert results[9][:3] == (
        "error",
        grpc.StatusCode.DEADLINE_EXCEEDED,
        "Deadline Exceeded",
    )
    assert results[10][:3] == ("error", grpc.StatusCode.INTERNAL, "Invalid metadata")
    # timeout=0 is no deadline to grpc: the call is served like the first one.
    assert results[11][:3] == results[0][:3]
    # A deadline that passes before the servicer's late abort reads as the
    # deadline, not the late status.
    assert results[12][:3] == (
        "error",
        grpc.StatusCode.DEADLINE_EXCEEDED,
        "Deadline Exceeded",
    )


def test_metadata_grpc_refuses_before_sending_is_refused_alike(
    transport: Transport,
) -> None:
    with _serve_edges(transport) as channel:
        stub = EgodriverServiceStub(channel)
        with pytest.raises(TypeError, match="expected bytes"):
            stub.get_version(Empty(), metadata=(("k-bin", "text"),))
        with pytest.raises(TypeError, match="Expected str"):
            stub.get_version(Empty(), metadata=(("k", 1),))


# -- what only the in-process channel needs to show -----------------------------------


class _Holds(BaseDriver):
    """Keeps what the servicer handed it."""

    name = "holds"

    def __init__(self) -> None:
        self.routes: List[np.ndarray] = []

    def on_route(self, session) -> None:  # type: ignore[no-untyped-def]
        self.routes.append(session.route_waypoints_in_rig)

    def drive(self, ctx: DriveContext) -> DriveResult:
        return DriveResult(trajectory_in_rig=Trajectory.empty())


def test_the_servicer_parses_its_own_copy_of_the_bytes() -> None:
    seen: List[Any] = []

    class _Keeps(EgodriverServiceServicer):
        def submit_route(self, request, context):  # type: ignore[no-untyped-def]
            seen.append(request)
            return Empty()

    channel = InProcessChannel()
    add_EgodriverServiceServicer_to_server(_Keeps(), channel)
    request = RouteRequest(session_uuid="s")
    request.route.waypoints.add(x=1.0)
    EgodriverServiceStub(channel).submit_route(request)
    request.route.waypoints[0].x = 99.0
    assert seen[0] is not request
    assert seen[0].route.waypoints[0].x == 1.0
    assert seen[0].SerializeToString() != request.SerializeToString()


def test_future_and_with_call_answer_like_grpcs() -> None:
    with in_process_channel(_Holds()) as channel:
        stub = EgodriverServiceStub(channel)
        future = stub.get_version.future(Empty())
        assert future.done() and future.result().version_id.startswith("holds-")
        assert future.code() == grpc.StatusCode.OK
        failed = stub.drive.future(DriveRequest(session_uuid="nobody"))
        assert isinstance(failed.exception(), grpc.RpcError)
        with pytest.raises(grpc.RpcError):
            failed.result()
        response, call = stub.start_session.with_call(
            DriveSessionRequest(session_uuid="s")
        )
        assert (
            isinstance(response, SessionRequestStatus)
            and call.code() == grpc.StatusCode.OK
        )
        ready: List[grpc.ChannelConnectivity] = []
        channel.subscribe(ready.append)
        assert ready == [grpc.ChannelConnectivity.READY]
        assert grpc.channel_ready_future(channel).result(timeout=1) is None


def test_a_closed_channel_refuses_calls_as_grpc_does() -> None:
    channel = in_process_channel(_Holds())
    channel.close()
    with pytest.raises(ValueError, match="closed channel"):
        EgodriverServiceStub(channel).get_version(Empty())


def test_an_error_is_an_rpc_error_and_a_call() -> None:
    with in_process_channel(_Holds()) as channel:
        with pytest.raises(grpc.RpcError) as error:
            EgodriverServiceStub(channel).drive(DriveRequest(session_uuid="nobody"))
    assert isinstance(error.value, grpc.Call)
    assert error.value.code() == grpc.StatusCode.NOT_FOUND
    assert error.value.details() == "Unknown session nobody"
    assert (
        error.value.initial_metadata() == () and error.value.trailing_metadata() == ()
    )
