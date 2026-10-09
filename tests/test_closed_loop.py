"""Policies served over real gRPC, driven by the CARLA-free FakeLoop."""

from __future__ import annotations

import grpc
import numpy as np
import pytest

from carla_driver_interface.driver import (
    BaseDriver,
    DriveContext,
    DriveResult,
    SessionState,
)
from carla_driver_interface.geometry import Pose, Trajectory
from carla_driver_interface.policies import ConstantSpeedPolicy, RouteFollowerPolicy
from carla_driver_interface.protocol import (
    EgodriverServiceStub,
    Empty,
    channel_options,
)
from carla_driver_interface.server import serving
from carla_driver_interface.testing import FakeCamera, FakeLoop, render_road


def run(policy: BaseDriver, steps: int = 50, **loop_kwargs: object):
    with serving(policy, port=0, host="127.0.0.1") as port:
        with FakeLoop(f"127.0.0.1:{port}", **loop_kwargs) as loop:  # type: ignore[arg-type]
            return loop.run(steps)


def test_constant_speed_drives_straight_ahead() -> None:
    result = run(ConstantSpeedPolicy(cruise_speed_mps=5.0), steps=30)
    assert result.steps == 30 and not result.terminated_by_policy
    # Perfect tracking of a 5 m/s plan: 0.5 m per 0.1 s step.
    np.testing.assert_allclose(np.diff(result.ego.positions[:, 0])[1:], 0.5, atol=1e-6)
    np.testing.assert_allclose(result.lateral_offsets_m, 0.0, atol=1e-9)
    assert all(
        d is not None and d.policy_name == "constant_speed" for d in result.debug
    )


def test_route_follower_returns_to_the_lane_centre() -> None:
    result = run(
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


def test_observations_reach_the_policy() -> None:
    recorder = _Recorder()
    camera = FakeCamera(width=160, height=90)
    result = run(recorder, steps=10, cameras=[camera], initial_speed_mps=4.0)
    assert result.terminated_by_policy and result.steps == 3
    (session,) = recorder.sessions
    assert list(session.cameras) == ["camera_front"]
    assert session.speed_mps() == pytest.approx(0.0)  # an empty plan holds the ego
    assert recorder.frames[0].shape == (90, 160, 3)
    # The route is the lane ahead, in the rig frame.
    assert session.route_waypoints_in_rig[1, 0] > session.route_waypoints_in_rig[0, 0]


class _Refuses(BaseDriver):
    name = "refuses"

    def on_session_start(self, session: SessionState) -> None:
        raise ValueError("needs a camera called 'nope'")

    def drive(self, ctx: DriveContext) -> DriveResult:  # pragma: no cover
        raise AssertionError


def test_a_policy_refusing_a_session_is_reported() -> None:
    with pytest.raises(grpc.RpcError) as error:
        run(_Refuses(), steps=1)
    assert error.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    assert "nope" in error.value.details()


def test_version_reports_the_policy() -> None:
    with serving(ConstantSpeedPolicy(), port=0, host="127.0.0.1") as port:
        with grpc.insecure_channel(
            f"127.0.0.1:{port}", options=channel_options()
        ) as channel:
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
