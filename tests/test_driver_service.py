# SPDX-License-Identifier: Apache-2.0
"""The servicer's side of the contract, exercised over real gRPC."""

from __future__ import annotations

import math
from collections.abc import Iterator

import grpc
import numpy as np
import pytest

from carla_driver_interface.driver.base import BaseDriver, DriveContext, DriveResult
from carla_driver_interface.driver.policies import ConstantSpeedPolicy, RouteFollowerPolicy
from carla_driver_interface.driver.server import build_server
from carla_driver_interface.geometry import Pose, Trajectory, dynamic_state_proto
from carla_driver_interface.grpc_api import (
    CarlaDriveDebugInfo,
    CarlaRendererData,
    DriveRequest,
    DriveSessionCloseRequest,
    DriveSessionRequest,
    EgodriverServiceStub,
    Empty,
    RolloutCameraImage,
    RolloutEgoTrajectory,
    Route,
    RouteRequest,
    TrafficLightState,
    Vec3,
)
from carla_driver_interface.runtime.conversions import available_camera

CAMERA_ID = "camera_front_wide_120fov"


@pytest.fixture
def driver_stub(request) -> Iterator[EgodriverServiceStub]:
    """Serve the policy named by the test's ``policy`` marker, default constant speed."""
    marker = request.node.get_closest_marker("policy")
    policy = marker.args[0] if marker else ConstantSpeedPolicy()
    server, port = build_server(policy, port=0, host="127.0.0.1")
    server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            yield EgodriverServiceStub(channel)
    finally:
        server.stop(grace=1.0).wait()


def start_session(stub: EgodriverServiceStub, uuid: str = "s0") -> None:
    stub.start_session(
        DriveSessionRequest(
            session_uuid=uuid,
            random_seed=7,
            debug_info=DriveSessionRequest.DebugInfo(scene_id="Town10:test"),
            rollout_spec=DriveSessionRequest.RolloutSpec(
                vehicle=DriveSessionRequest.RolloutSpec.VehicleDefinition(
                    available_cameras=[
                        available_camera(CAMERA_ID, 64, 48, 90.0, Pose.from_xyz_yaw(1.5, 0, 1.6, 0))
                    ],
                ),
            ),
        )
    )


def submit_pose(
    stub: EgodriverServiceStub, timestamp_us: int, pose: Pose, uuid: str = "s0"
) -> None:
    stub.submit_egomotion_observation(
        RolloutEgoTrajectory(
            session_uuid=uuid,
            trajectory=Trajectory([timestamp_us], [pose]).to_proto(),
            dynamic_states=[dynamic_state_proto(np.array([5.0, 0, 0]), np.zeros(3), np.zeros(3))],
        )
    )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_get_version_reports_the_policy_and_api_version(driver_stub):
    from carla_driver_interface.grpc_api import API_VERSION_MESSAGE

    version = driver_stub.get_version(Empty())
    assert "constant_speed" in version.version_id
    assert version.grpc_api_version == API_VERSION_MESSAGE


def test_duplicate_session_is_rejected(driver_stub):
    start_session(driver_stub)
    with pytest.raises(grpc.RpcError) as excinfo:
        start_session(driver_stub)
    assert excinfo.value.code() == grpc.StatusCode.ALREADY_EXISTS


def test_unknown_session_is_rejected(driver_stub):
    with pytest.raises(grpc.RpcError) as excinfo:
        driver_stub.drive(DriveRequest(session_uuid="nope", time_now_us=0, time_query_us=1))
    assert excinfo.value.code() == grpc.StatusCode.NOT_FOUND


def test_close_session_is_idempotent(driver_stub):
    start_session(driver_stub)
    driver_stub.close_session(DriveSessionCloseRequest(session_uuid="s0"))
    # A runtime that crashed mid-rollout may close twice; that must not error.
    driver_stub.close_session(DriveSessionCloseRequest(session_uuid="s0"))


def test_undeclared_camera_is_rejected(driver_stub):
    start_session(driver_stub)
    with pytest.raises(grpc.RpcError) as excinfo:
        driver_stub.submit_image_observation(
            RolloutCameraImage(
                session_uuid="s0",
                camera_image=RolloutCameraImage.CameraImage(
                    logical_id="camera_rear", image_bytes=b"x"
                ),
            )
        )
    assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_mismatched_dynamic_state_count_is_rejected(driver_stub):
    start_session(driver_stub)
    with pytest.raises(grpc.RpcError) as excinfo:
        driver_stub.submit_egomotion_observation(
            RolloutEgoTrajectory(
                session_uuid="s0",
                trajectory=Trajectory([0, 1000], [Pose.identity(), Pose.identity()]).to_proto(),
                dynamic_states=[dynamic_state_proto(np.zeros(3), np.zeros(3), np.zeros(3))],
            )
        )
    assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT


# ---------------------------------------------------------------------------
# drive() semantics
# ---------------------------------------------------------------------------


def test_drive_before_any_egomotion_returns_an_empty_plan(driver_stub):
    """There is no frame to answer in yet; upstream's driver also answers empty."""
    start_session(driver_stub)
    response = driver_stub.drive(
        DriveRequest(session_uuid="s0", time_now_us=0, time_query_us=100_000)
    )
    assert len(response.trajectory.poses) == 0


def test_plan_is_anchored_on_the_ego_pose_and_expressed_in_local(driver_stub):
    """``DriveResponse.trajectory`` must be local->rig_est, led by the pose at t0.

    This is the single most important compatibility detail: an alpasim runtime
    feeds this straight into its controller.
    """
    start_session(driver_stub)
    ego = Pose.from_xyz_yaw(100.0, -50.0, 0.0, math.pi / 2)  # facing +y
    submit_pose(driver_stub, 1_000_000, ego)

    response = driver_stub.drive(
        DriveRequest(session_uuid="s0", time_now_us=1_000_000, time_query_us=1_100_000)
    )
    plan = Trajectory.from_proto(response.trajectory)

    assert plan.timestamps_us[0] == 1_000_000
    assert np.allclose(plan.poses[0].position, ego.position, atol=1e-5)
    assert plan.poses[0].yaw == pytest.approx(ego.yaw, abs=1e-5)

    # ConstantSpeedPolicy drives straight ahead in the rig frame, which for an
    # ego facing +y means the plan advances in +y, not +x.
    displacement = plan.poses[-1].position - ego.position
    assert displacement[1] > 5.0
    assert abs(displacement[0]) < 1e-3


def test_plan_timestamps_strictly_increase(driver_stub):
    start_session(driver_stub)
    submit_pose(driver_stub, 500_000, Pose.identity())
    response = driver_stub.drive(
        DriveRequest(session_uuid="s0", time_now_us=500_000, time_query_us=600_000)
    )
    stamps = [p.timestamp_us for p in response.trajectory.poses]
    assert stamps == sorted(stamps)
    assert len(set(stamps)) == len(stamps)


def test_debug_info_carries_the_carla_extension(driver_stub):
    start_session(driver_stub)
    submit_pose(driver_stub, 0, Pose.identity())
    response = driver_stub.drive(
        DriveRequest(session_uuid="s0", time_now_us=0, time_query_us=100_000)
    )
    debug = CarlaDriveDebugInfo()
    debug.ParseFromString(response.debug_info.unstructured_debug_info)
    assert debug.policy_name == "constant_speed"
    assert debug.inference_seconds >= 0.0
    assert debug.scalars["target_speed_mps"] == pytest.approx(5.0)


def test_foreign_renderer_data_is_ignored_not_fatal(driver_stub):
    """An upstream alpasim runtime may put anything in ``renderer_data``."""
    start_session(driver_stub)
    submit_pose(driver_stub, 0, Pose.identity())
    response = driver_stub.drive(
        DriveRequest(
            session_uuid="s0",
            time_now_us=0,
            time_query_us=100_000,
            renderer_data=b"\xff\xfe not a protobuf \x00\x01",
        )
    )
    assert len(response.trajectory.poses) > 0


def test_egomotion_observations_accumulate_and_ignore_replays(driver_stub):
    captured: list[int] = []

    class Recorder(BaseDriver):
        name = "recorder"

        def drive(self, ctx: DriveContext) -> DriveResult:
            captured.append(len(ctx.session.ego_trajectory))
            return DriveResult(trajectory_in_rig=Trajectory.empty())

    server, port = build_server(Recorder(), port=0, host="127.0.0.1")
    server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = EgodriverServiceStub(channel)
            start_session(stub)
            for ts in (0, 50_000, 100_000):
                submit_pose(stub, ts, Pose.from_xyz_yaw(ts / 1e5, 0, 0, 0))
            # The runtime may resend the previous step boundary; it must not
            # break the strictly-increasing invariant.
            submit_pose(stub, 100_000, Pose.from_xyz_yaw(1.0, 0, 0, 0))
            stub.drive(DriveRequest(session_uuid="s0", time_now_us=100_000, time_query_us=200_000))
    finally:
        server.stop(grace=1.0).wait()

    assert captured == [3]


# ---------------------------------------------------------------------------
# Reference policies
# ---------------------------------------------------------------------------


@pytest.mark.policy(RouteFollowerPolicy(cruise_speed_mps=10.0))
def test_route_follower_turns_towards_the_route(driver_stub):
    start_session(driver_stub)
    submit_pose(driver_stub, 0, Pose.identity())
    # A route curving to the left in the rig frame.
    waypoints = [Vec3(x=float(i), y=float(i * i) * 0.02, z=0.0) for i in range(1, 41)]
    driver_stub.submit_route(
        RouteRequest(session_uuid="s0", route=Route(timestamp_us=0, waypoints=waypoints))
    )

    response = driver_stub.drive(
        DriveRequest(session_uuid="s0", time_now_us=0, time_query_us=100_000)
    )
    plan = Trajectory.from_proto(response.trajectory)
    assert plan.poses[-1].position[1] > 0.5, "plan did not follow the route left"


@pytest.mark.policy(RouteFollowerPolicy(cruise_speed_mps=10.0))
def test_route_follower_brakes_for_a_red_light(driver_stub):
    start_session(driver_stub)
    submit_pose(driver_stub, 0, Pose.identity())
    waypoints = [Vec3(x=float(i), y=0.0, z=0.0) for i in range(1, 61)]
    driver_stub.submit_route(
        RouteRequest(session_uuid="s0", route=Route(timestamp_us=0, waypoints=waypoints))
    )

    def plan_length(renderer_data: bytes) -> float:
        response = driver_stub.drive(
            DriveRequest(
                session_uuid="s0",
                time_now_us=0,
                time_query_us=100_000,
                renderer_data=renderer_data,
            )
        )
        return float(Trajectory.from_proto(response.trajectory).poses[-1].position[0])

    green = plan_length(
        CarlaRendererData(
            ego_traffic_light=TrafficLightState.TRAFFIC_LIGHT_STATE_GREEN
        ).SerializeToString()
    )
    red = plan_length(
        CarlaRendererData(
            ego_traffic_light=TrafficLightState.TRAFFIC_LIGHT_STATE_RED,
            ego_traffic_light_distance_m=8.0,
        ).SerializeToString()
    )
    assert red < green, "a red light did not shorten the plan"


@pytest.mark.policy(RouteFollowerPolicy())
def test_route_follower_holds_without_a_route(driver_stub):
    """No route is not a crash; it is a reason to not move."""
    start_session(driver_stub)
    submit_pose(driver_stub, 0, Pose.identity())
    response = driver_stub.drive(
        DriveRequest(session_uuid="s0", time_now_us=0, time_query_us=100_000)
    )
    debug = CarlaDriveDebugInfo()
    debug.ParseFromString(response.debug_info.unstructured_debug_info)
    assert debug.scalars["reason_no_route"] == 1.0


def test_terminate_session_is_propagated():
    class Quitter(BaseDriver):
        name = "quitter"

        def drive(self, ctx: DriveContext) -> DriveResult:
            return DriveResult(trajectory_in_rig=Trajectory.empty(), terminate_session=True)

    server, port = build_server(Quitter(), port=0, host="127.0.0.1")
    server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = EgodriverServiceStub(channel)
            start_session(stub)
            submit_pose(stub, 0, Pose.identity())
            response = stub.drive(
                DriveRequest(session_uuid="s0", time_now_us=0, time_query_us=100_000)
            )
    finally:
        server.stop(grace=1.0).wait()
    assert response.terminate_session is True
