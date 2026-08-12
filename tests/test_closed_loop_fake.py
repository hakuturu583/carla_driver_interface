# SPDX-License-Identifier: Apache-2.0
"""The whole loop, over real gRPC, without a CARLA server.

These are the tests that would catch a regression in the thing the project
exists to do: runtime and driver in separate gRPC endpoints, observations
flowing one way, plans the other, and a vehicle that ends up somewhere sensible.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from carla_driver_interface.driver.base import BaseDriver, DriveContext, DriveResult
from carla_driver_interface.driver.policies import ConstantSpeedPolicy, RouteFollowerPolicy
from carla_driver_interface.driver.server import serving
from carla_driver_interface.fakes import FakeWorld, straight_then_curve_route
from carla_driver_interface.geometry import Pose, Trajectory
from carla_driver_interface.grpc_api import ImageFormat, RolloutErrorCode
from carla_driver_interface.runtime import (
    CarlaRuntime,
    RuntimeConfig,
    ScenarioSpec,
    WorldAdapter,
)

from .conftest import max_route_deviation

SCENARIO = ScenarioSpec(map_name="FakeTown", name="straight_then_curve")


def config(port: int, **overrides) -> RuntimeConfig:
    base = RuntimeConfig(
        driver_address=f"127.0.0.1:{port}",
        max_steps=200,
        # Small frames keep the tests fast; the encoding path is still exercised.
        image_format=ImageFormat.JPEG,
    )
    base = replace(base, cameras=[replace(base.cameras[0], width=64, height=48)])
    return replace(base, **overrides) if overrides else base


def run(policy: BaseDriver, world_factory=None, **overrides):
    """Serve ``policy``, run one rollout, return ``(outcome, world)``."""
    with serving(policy, port=0, host="127.0.0.1") as port:
        cfg = config(port, **overrides)
        world = (world_factory or FakeWorld)(cfg, SCENARIO)
        return CarlaRuntime(world, cfg, SCENARIO).run_rollout(), world


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_route_follower_completes_the_route():
    outcome, world = run(RouteFollowerPolicy())

    assert outcome.success
    assert outcome.rollout_return.error == ""
    assert not outcome.terminated_by_driver
    assert outcome.metrics["route_completion"] == pytest.approx(1.0, abs=1e-3)
    assert outcome.metrics["collisions"] == 0.0
    assert outcome.metrics["lane_invasions"] == 0.0

    route = straight_then_curve_route()
    final = world.history[-1].pose_local_to_rig.position
    assert np.linalg.norm(final[:2] - route[-1][:2]) < 3.0


def test_the_ego_tracks_the_route_closely():
    _, world = run(RouteFollowerPolicy())
    deviation = max_route_deviation(world, straight_then_curve_route())
    assert deviation < 1.5, f"max deviation {deviation:.2f} m"


def test_constant_speed_policy_drives_forward():
    """A policy that ignores the route still moves -- the loop is not route-gated."""
    outcome, world = run(ConstantSpeedPolicy(cruise_speed_mps=6.0), max_steps=60)

    travelled = outcome.metrics["distance_travelled_m"]
    assert travelled > 10.0
    assert outcome.metrics["speed_mps"] > 2.0
    assert world.history[-1].pose_local_to_rig.position[0] > 10.0


def test_result_is_an_alpasim_rollout_return():
    """The result type is upstream's, so alpasim tooling can consume it."""
    outcome, _ = run(RouteFollowerPolicy(), max_steps=30)
    rollout = outcome.rollout_return

    assert rollout.rollout_spec.scenario_id == SCENARIO.scene_id
    assert rollout.rollout_uuid
    assert rollout.error_code == RolloutErrorCode.ROLLOUT_ERROR_CODE_NONE

    names = {metric.name for metric in rollout.timestep_metrics}
    assert {"speed_mps", "route_completion", "route_lateral_error_m"} <= names
    for metric in rollout.timestep_metrics:
        assert len(metric.timestamps_us) == len(metric.values) == len(metric.valid)
        assert list(metric.timestamps_us) == sorted(metric.timestamps_us)


# ---------------------------------------------------------------------------
# What the driver actually receives
# ---------------------------------------------------------------------------


class _Spy(BaseDriver):
    """Records what arrived, then drives straight so the loop keeps going."""

    name = "spy"

    def __init__(self) -> None:
        self.contexts: list[dict] = []

    def drive(self, ctx: DriveContext) -> DriveResult:
        session = ctx.session
        self.contexts.append(
            {
                "time_now_us": ctx.time_now_us,
                "time_query_us": ctx.time_query_us,
                "scene_id": session.scene_id,
                "cameras": sorted(session.cameras),
                "frames": {
                    cid: frames[-1].frame_start_us
                    for cid, frames in session.frame_history.items()
                    if frames
                },
                "ego_poses": len(session.ego_trajectory),
                "route_points": len(session.route_waypoints_in_rig),
                "ground_truth": session.ground_truth_in_rig is not None,
                "renderer_data": ctx.renderer_data,
            }
        )
        plan = Trajectory.empty()
        for i in range(1, 41):
            plan.append(ctx.time_now_us + i * 100_000, Pose.from_xyz_yaw(0.6 * i, 0, 0, 0))
        return DriveResult(trajectory_in_rig=plan)


def test_every_observation_reaches_the_driver_before_drive():
    spy = _Spy()
    outcome, _ = run(spy, max_steps=12)
    assert outcome.rollout_return.error == ""
    assert len(spy.contexts) == 12

    first = spy.contexts[0]
    assert first["scene_id"] == SCENARIO.scene_id
    assert first["cameras"] == ["camera_front_wide_120fov"]
    # Images, egomotion and route must all be present at the first decision.
    assert first["frames"], "no camera frame before the first drive()"
    assert first["ego_poses"] > 0
    assert first["route_points"] >= 2


def test_camera_frames_are_never_stale_at_decision_time():
    """alpasim asserts sensor freshness; violating it silently would be worse."""
    spy = _Spy()
    run(spy, max_steps=15)
    for ctx in spy.contexts:
        for camera, frame_us in ctx["frames"].items():
            assert frame_us <= ctx["time_now_us"], f"{camera} frame is from the future"
            lag_us = ctx["time_now_us"] - frame_us
            assert lag_us <= 100_000, f"{camera} frame lags by {lag_us} us"


def test_query_time_is_one_policy_step_ahead():
    spy = _Spy()
    cfg_step_us = 100_000
    run(spy, max_steps=8)
    for ctx in spy.contexts:
        assert ctx["time_query_us"] - ctx["time_now_us"] == cfg_step_us

    stamps = [ctx["time_now_us"] for ctx in spy.contexts]
    assert stamps == sorted(stamps)
    assert len(set(stamps)) == len(stamps)


def test_intermediate_ticks_are_all_delivered():
    """Two simulator ticks per policy step means two poses per step, not one."""
    spy = _Spy()
    run(spy, max_steps=6, fixed_delta_s=0.05, policy_timestep_s=0.1)
    growth = [
        spy.contexts[i + 1]["ego_poses"] - spy.contexts[i]["ego_poses"]
        for i in range(len(spy.contexts) - 1)
    ]
    assert set(growth) == {2}, f"unexpected pose growth per step: {growth}"


def test_renderer_data_extension_is_populated():
    spy = _Spy()
    run(spy, max_steps=5)
    data = spy.contexts[0]["renderer_data"]
    assert data is not None
    assert data.map_name == "FakeTown"
    assert data.snapshot_timestamp_us == spy.contexts[0]["time_now_us"]
    assert data.frame_id > 0


def test_ground_truth_is_off_by_default_and_opt_in():
    off = _Spy()
    run(off, max_steps=5)
    assert not any(ctx["ground_truth"] for ctx in off.contexts)

    on = _Spy()
    run(on, max_steps=5, send_ground_truth=True)
    assert all(ctx["ground_truth"] for ctx in on.contexts)


# ---------------------------------------------------------------------------
# Frames and noise
# ---------------------------------------------------------------------------


def test_egomotion_noise_is_hidden_from_the_true_trajectory():
    """The driver sees a noised frame; the world must still be driven correctly.

    With noise on, the runtime has to map the plan back out of the estimated
    frame -- the alpasim ``rig_est`` round trip. If it did not, the ego would
    drift by the accumulated noise.
    """
    clean, clean_world = run(RouteFollowerPolicy(), max_steps=120)
    noisy, noisy_world = run(
        RouteFollowerPolicy(),
        max_steps=120,
        egomotion_position_noise_m=0.25,
        egomotion_yaw_noise_rad=0.01,
    )

    assert clean.rollout_return.error == ""
    assert noisy.rollout_return.error == ""

    route = straight_then_curve_route()
    # Noise degrades tracking, but the loop must stay on the road rather than
    # walking off it as the error accumulates.
    assert max_route_deviation(clean_world, route) < 1.5
    assert max_route_deviation(noisy_world, route) < 4.0


def test_driver_termination_stops_the_rollout_immediately():
    class Quitter(BaseDriver):
        name = "quitter"

        def __init__(self) -> None:
            self.calls = 0

        def drive(self, ctx: DriveContext) -> DriveResult:
            self.calls += 1
            return DriveResult(
                trajectory_in_rig=Trajectory.empty(), terminate_session=self.calls >= 3
            )

    quitter = Quitter()
    outcome, _ = run(quitter, max_steps=100)
    assert outcome.terminated_by_driver
    assert outcome.steps == 3
    assert quitter.calls == 3


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


def test_unreachable_driver_fails_the_rollout_without_raising():
    cfg = config(port=1)  # nothing listening
    cfg = replace(cfg, driver_address="127.0.0.1:1", driver_timeout_s=2.0)
    outcome = CarlaRuntime(FakeWorld(cfg, SCENARIO), cfg, SCENARIO).run_rollout()

    assert not outcome.success
    assert "driver RPC failed" in outcome.rollout_return.error
    assert outcome.rollout_return.error_code == RolloutErrorCode.ROLLOUT_ERROR_CODE_UNSPECIFIED


def test_a_driver_that_raises_is_reported_not_swallowed():
    class Broken(BaseDriver):
        name = "broken"

        def drive(self, ctx: DriveContext) -> DriveResult:
            raise RuntimeError("model exploded")

    outcome, _ = run(Broken(), max_steps=5)
    assert not outcome.success
    assert outcome.rollout_return.error


def test_the_world_is_closed_even_when_the_rollout_fails():
    closed: list[bool] = []

    class ClosingWorld(FakeWorld):
        def close(self) -> None:
            closed.append(True)
            super().close()

    cfg = replace(config(port=1), driver_address="127.0.0.1:1", driver_timeout_s=2.0)
    CarlaRuntime(ClosingWorld(cfg, SCENARIO), cfg, SCENARIO).run_rollout()
    assert closed == [True]


# ---------------------------------------------------------------------------
# The seam itself
# ---------------------------------------------------------------------------


def test_fake_world_satisfies_the_adapter_protocol():
    cfg = config(port=50051)
    assert isinstance(FakeWorld(cfg, SCENARIO), WorldAdapter)


def test_carla_adapter_satisfies_the_adapter_protocol_without_importing_carla():
    """The real adapter must stay structurally compatible even in a CARLA-free env."""
    from carla_driver_interface.runtime.carla_world import CarlaWorldAdapter

    for method in ("setup", "tick", "apply_control", "environment", "events", "close"):
        assert callable(getattr(CarlaWorldAdapter, method)), method


def test_rollouts_are_reproducible_for_a_fixed_seed():
    first, world_a = run(RouteFollowerPolicy(), max_steps=60, seed=11)
    second, world_b = run(RouteFollowerPolicy(), max_steps=60, seed=11)

    path_a = np.stack([s.pose_local_to_rig.position for s in world_a.history])
    path_b = np.stack([s.pose_local_to_rig.position for s in world_b.history])
    assert np.allclose(path_a, path_b)
    assert first.metrics["distance_travelled_m"] == pytest.approx(
        second.metrics["distance_travelled_m"]
    )


def test_a_curving_route_is_actually_curved():
    """Guard the fixture itself: a straight 'curve' would make the tests vacuous."""
    route = straight_then_curve_route()
    headings = np.arctan2(np.diff(route[:, 1]), np.diff(route[:, 0]))
    assert math.degrees(float(headings[-1] - headings[0])) == pytest.approx(90.0, abs=5.0)


def test_an_unsupported_image_format_fails_at_config_construction():
    """Not in a sensor callback thread 200 ticks later, where it is swallowed.

    Left to the encoder, a bad format produced a rollout that ran to completion,
    submitted zero images, and reported success.
    """
    with pytest.raises(ValueError, match="PNG and JPEG only"):
        RuntimeConfig(image_format=ImageFormat.JPEG2000)


def test_ground_truth_and_route_carry_the_same_waypoints():
    """One computation per step, so the driver cannot get two disagreeing routes.

    They used to be computed separately, and only the route got the estimated
    frame correction -- so with egomotion noise on, the two disagreed.
    """

    class _RouteSpy(BaseDriver):
        name = "route_spy"

        def __init__(self) -> None:
            self.pairs: list[tuple[np.ndarray, np.ndarray]] = []

        def drive(self, ctx: DriveContext) -> DriveResult:
            session = ctx.session
            if session.ground_truth_in_rig is not None:
                self.pairs.append(
                    (
                        np.asarray(session.route_waypoints_in_rig),
                        session.ground_truth_in_rig.positions,
                    )
                )
            return DriveResult(trajectory_in_rig=Trajectory.empty())

    spy = _RouteSpy()
    run(spy, max_steps=8, send_ground_truth=True, egomotion_position_noise_m=0.3)

    assert spy.pairs
    for route, ground_truth in spy.pairs:
        assert np.allclose(route, ground_truth)
