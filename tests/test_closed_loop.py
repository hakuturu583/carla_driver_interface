"""Policies served over real gRPC, driven by the CARLA-free FakeLoop."""

from __future__ import annotations

import grpc
import numpy as np
import pytest
from conftest import Transport

from carla_driver_interface.driver import (
    BaseDriver,
    DriveContext,
    DriveResult,
    SessionState,
    TurnIndicator,
)
from carla_driver_interface.geometry import Pose, Trajectory
from carla_driver_interface.policies import ConstantSpeedPolicy, RouteFollowerPolicy
from carla_driver_interface.protocol import (
    CarlaVehicleServiceStub,
    DriveSessionRequest,
    EgodriverServiceStub,
    Empty,
    VehicleCommandRequest,
    add_EgodriverServiceServicer_to_server,
    carla_driver_pb2,
)
from carla_driver_interface.service import EgodriverServicer
from carla_driver_interface.testing import FakeCamera, FakeLoop, render_road


def run(
    transport: Transport, policy: BaseDriver, steps: int = 50, **loop_kwargs: object
):
    with transport.serve(policy) as channel:
        with FakeLoop(channel, **loop_kwargs) as loop:  # type: ignore[arg-type]
            return loop.run(steps)


def test_constant_speed_drives_straight_ahead(transport: Transport) -> None:
    result = run(transport, ConstantSpeedPolicy(cruise_speed_mps=5.0), steps=30)
    assert result.steps == 30 and not result.terminated_by_policy
    # Perfect tracking of a 5 m/s plan: 0.5 m per 0.1 s step.
    np.testing.assert_allclose(np.diff(result.ego.positions[:, 0])[1:], 0.5, atol=1e-6)
    np.testing.assert_allclose(result.lateral_offsets_m, 0.0, atol=1e-9)
    assert all(
        d is not None and d.policy_name == "constant_speed" for d in result.debug
    )


def test_route_follower_returns_to_the_lane_centre(transport: Transport) -> None:
    result = run(
        transport,
        RouteFollowerPolicy(cruise_speed_mps=8.0),
        steps=60,
        initial_lateral_offset_m=1.0,
    )
    assert result.ego.positions[-1, 0] > 20.0
    assert abs(result.lateral_offsets_m[-1]) < 0.2


class _Recorder(BaseDriver):
    name = "recorder"

    def __init__(self) -> None:
        self.sessions: list[SessionState] = []
        self.frames: list[np.ndarray] = []

    def on_session_start(self, session: SessionState) -> None:
        self.sessions.append(session)

    def drive(self, ctx: DriveContext) -> DriveResult:
        frame = ctx.session.latest_frame("camera_front")
        assert frame is not None
        self.frames.append(frame.as_array())
        return DriveResult(
            trajectory_in_rig=Trajectory.empty(),
            terminate_session=len(self.frames) == 3,
        )


def test_observations_reach_the_policy(transport: Transport) -> None:
    recorder = _Recorder()
    camera = FakeCamera(width=160, height=90)
    result = run(transport, recorder, steps=10, cameras=[camera], initial_speed_mps=4.0)
    assert result.terminated_by_policy and result.steps == 3
    (session,) = recorder.sessions
    assert list(session.cameras) == ["camera_front"]
    assert session.speed_mps() == pytest.approx(0.0)  # an empty plan holds the ego
    assert recorder.frames[0].shape == (90, 160, 3)
    # The route is the lane ahead, in the rig frame.
    assert session.route_waypoints_in_rig[1, 0] > session.route_waypoints_in_rig[0, 0]


class _Signals(BaseDriver):
    """Left for two steps, then off, then silent."""

    name = "signals"

    def __init__(self) -> None:
        self.steps = 0

    def drive(self, ctx: DriveContext) -> DriveResult:
        self.steps += 1
        indicator = (
            TurnIndicator.ENABLE_LEFT
            if self.steps <= 2
            else TurnIndicator.DISABLE
            if self.steps == 3
            else TurnIndicator.NO_COMMAND
        )
        return DriveResult(
            trajectory_in_rig=Trajectory.empty(), turn_indicator=indicator
        )


def test_the_turn_indicator_reaches_the_runtime(transport: Transport) -> None:
    result = run(transport, _Signals(), steps=4)
    assert [c.turn_indicator for c in result.commands] == [
        carla_driver_pb2.TURN_INDICATOR_ENABLE_LEFT,
        carla_driver_pb2.TURN_INDICATOR_ENABLE_LEFT,
        carla_driver_pb2.TURN_INDICATOR_DISABLE,
        carla_driver_pb2.TURN_INDICATOR_NO_COMMAND,
    ]
    # Each command names the drive that decided it.
    assert [c.time_now_us for c in result.commands] == [0, 100_000, 200_000, 300_000]


def test_a_policy_that_sets_no_turn_indicator_commands_nothing(
    transport: Transport,
) -> None:
    result = run(transport, ConstantSpeedPolicy(), steps=2)
    assert all(
        c.turn_indicator == carla_driver_pb2.TURN_INDICATOR_NO_COMMAND
        for c in result.commands
    )


def test_the_vehicle_command_is_none_before_the_first_drive(
    transport: Transport,
) -> None:
    with transport.serve(_Signals()) as channel:
        EgodriverServiceStub(channel).start_session(
            DriveSessionRequest(session_uuid="s"), timeout=10
        )
        vehicle = CarlaVehicleServiceStub(channel)
        command = vehicle.get_vehicle_command(
            VehicleCommandRequest(session_uuid="s"), timeout=10
        )
        with pytest.raises(grpc.RpcError) as error:
            vehicle.get_vehicle_command(
                VehicleCommandRequest(session_uuid="unknown"), timeout=10
            )
    assert command.time_now_us == 0
    assert command.turn_indicator == carla_driver_pb2.TURN_INDICATOR_NO_COMMAND
    assert error.value.code() == grpc.StatusCode.NOT_FOUND
    assert error.value.details() == "Unknown session unknown"


class _BadSignal(BaseDriver):
    """Plans, but answers an indicator class that does not exist."""

    name = "bad_signal"

    def drive(self, ctx: DriveContext) -> DriveResult:
        plan = Trajectory.empty()
        plan.append(ctx.time_now_us + 100_000, Pose.from_xyz_yaw(0.5, 0.0, 0.0, 0.0))
        return DriveResult(trajectory_in_rig=plan, turn_indicator=4)  # type: ignore[arg-type]


def test_a_bad_turn_indicator_costs_the_policy_nothing_but_the_command(
    transport: Transport,
) -> None:
    result = run(transport, _BadSignal(), steps=3)
    assert result.steps == 3
    assert result.ego.positions[-1, 0] > 0.0  # the plans still arrived
    assert all(
        c.turn_indicator == carla_driver_pb2.TURN_INDICATOR_NO_COMMAND
        for c in result.commands
    )


def test_fake_loop_reads_a_server_without_the_vehicle_service_as_no_command(
    transport: Transport,
) -> None:
    servicer = EgodriverServicer(_Signals())
    register = lambda server: add_EgodriverServiceServicer_to_server(  # noqa: E731
        servicer, server
    )
    with transport.serve_servicers(register) as channel:
        with pytest.raises(grpc.RpcError) as error:
            CarlaVehicleServiceStub(channel).get_vehicle_command(
                VehicleCommandRequest(session_uuid="s"), timeout=10
            )
        with FakeLoop(channel) as loop:
            result = loop.run(2)
    # What a runtime reads as "carla-driver-interface < 2.1".
    assert error.value.code() == grpc.StatusCode.UNIMPLEMENTED
    assert error.value.details() == "Method not found!"
    assert [c.turn_indicator for c in result.commands] == [
        carla_driver_pb2.TURN_INDICATOR_NO_COMMAND
    ] * 2


def test_turn_indicator_values_are_the_wires() -> None:
    assert {i.name: int(i) for i in TurnIndicator} == {
        name.removeprefix("TURN_INDICATOR_"): value
        for name, value in carla_driver_pb2.TurnIndicator.items()
    }


class _Refuses(BaseDriver):
    name = "refuses"

    def on_session_start(self, session: SessionState) -> None:
        raise ValueError("needs a camera called 'nope'")

    def drive(self, ctx: DriveContext) -> DriveResult:  # pragma: no cover
        raise AssertionError


def test_a_policy_refusing_a_session_is_reported(transport: Transport) -> None:
    with pytest.raises(grpc.RpcError) as error:
        run(transport, _Refuses(), steps=1)
    assert error.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    assert error.value.details() == "refuses: needs a camera called 'nope'"


def test_version_reports_the_policy(transport: Transport) -> None:
    with transport.serve(ConstantSpeedPolicy()) as channel:
        version = EgodriverServiceStub(channel).get_version(Empty())
    assert version.version_id.startswith("constant_speed-carla-driver-interface-")
    assert (version.grpc_api_version.major, version.grpc_api_version.minor) == (0, 55)


def test_rendered_lanes_sit_where_the_geometry_says() -> None:
    """Markings at y = +-1.75 m appear left and right of the image centre, symmetrically."""
    camera = FakeCamera(width=400, height=200, fov_deg=60.0, z=1.5, pitch_down_deg=5.0)
    image = render_road(camera, Pose.identity())
    # A row ~10 m ahead, on a dash: nearer rows see too little road sideways to hold
    # both markings.
    row = image[110]
    white = np.flatnonzero((row == 235).all(axis=1))
    left, right = white[white < 200], white[white >= 200]
    assert len(left) and len(right)
    assert abs((200 - left.mean()) - (right.mean() - 200)) < 2.0
    assert (image[0] == (110, 160, 220)).all()  # sky at the top


def test_policies_load_by_name_or_import_path() -> None:
    from carla_driver_interface.policies import (
        ConstantSpeedPolicy,
        RouteFollowerPolicy,
        load_policy,
    )

    assert isinstance(load_policy("route_follower"), RouteFollowerPolicy)
    loaded = load_policy(
        "carla_driver_interface.policies.constant_speed:ConstantSpeedPolicy"
    )
    assert isinstance(loaded, ConstantSpeedPolicy)
    for bad in ("no_such_policy", "no.such.module:Policy", "math:pi"):
        with pytest.raises(ValueError):
            load_policy(bad)
