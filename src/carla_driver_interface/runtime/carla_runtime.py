# SPDX-License-Identifier: Apache-2.0
"""``CarlaRuntime``: the orchestrator, in alpasim's sense of the word.

alpasim's DESIGN.md puts the Runtime at the centre -- it holds the world state
and drives every other service as a gRPC client.  This class occupies the same
position, but where alpasim fans out to sensorsim, controller, physics and
traffic services, CARLA supplies all four.  The one RPC boundary that survives
is the one that matters: ``egodriver.EgodriverService``.

The per-step order deliberately mirrors alpasim's ``PolicyEvent``
(``runtime/alpasim_runtime/events/policy.py``):

1. submit every observation (images, egomotion, route, ground truth),
2. barrier -- all observations must land before the decision,
3. ``drive(time_now_us, time_query_us)``,
4. map the plan out of the estimated frame back into the true one,
5. actuate, step physics, commit.

Keeping the order identical is what lets a driver written for alpasim behave
the same here.
"""

from __future__ import annotations

import contextlib
import logging
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

import grpc
import numpy as np

from carla_driver_interface.geometry import Pose, Trajectory, dynamic_state_proto
from carla_driver_interface.grpc_api import (
    DriveRequest,
    DriveSessionCloseRequest,
    DriveSessionRequest,
    EgodriverServiceStub,
    Empty,
    GroundTruth,
    GroundTruthRequest,
    RolloutCameraImage,
    RolloutEgoTrajectory,
    RolloutErrorCode,
    Route,
    RouteRequest,
    SimulationReturn,
    Vec3,
    channel_options,
    describe_api_mismatch,
)
from carla_driver_interface.grpc_api.extension import pack_renderer_data, unpack_debug_info
from carla_driver_interface.runtime.config import RuntimeConfig, ScenarioSpec
from carla_driver_interface.runtime.control import TrajectoryFollower, VehicleCommand
from carla_driver_interface.runtime.metrics import MetricsCollector
from carla_driver_interface.runtime.route import RouteProvider
from carla_driver_interface.runtime.world import EgoState, WorldAdapter, WorldSnapshot

logger = logging.getLogger(__name__)

__all__ = ["CarlaRuntime", "RolloutOutcome"]


@dataclass
class RolloutOutcome:
    """What a rollout produced.

    ``rollout_return`` is alpasim's own ``SimulationReturn.RolloutReturn``, so it
    can be handed to alpasim-side tooling unchanged.
    """

    rollout_return: SimulationReturn.RolloutReturn
    steps: int
    terminated_by_driver: bool

    @property
    def success(self) -> bool:
        return self.rollout_return.success

    @property
    def metrics(self) -> dict[str, float]:
        return dict(self.rollout_return.aggregated_metrics)


class CarlaRuntime:
    """Runs one closed-loop rollout against a driver.

    Unlike alpasim's runtime -- an asyncio daemon that load-balances many
    concurrent rollouts across driver replicas -- this is a synchronous,
    single-rollout, in-process class. Concurrency belongs to the caller.
    """

    def __init__(
        self,
        world: WorldAdapter,
        config: RuntimeConfig,
        scenario: ScenarioSpec | None = None,
        driver_channel: grpc.Channel | None = None,
    ) -> None:
        self.world = world
        self.config = config
        self.scenario = scenario or ScenarioSpec()
        self._external_channel = driver_channel

        self._follower = TrajectoryFollower(config.control)
        self._metrics = MetricsCollector()
        self._rng = np.random.default_rng(config.seed)

        #: Newest estimated ("noised") ego pose -- the frame the driver reasons
        #: in. The true pose lives on ``_latest.ego``; keeping a second copy
        #: here is how the two silently drift apart.
        self._last_estimate: Pose | None = None
        self._pending_egomotion: list[tuple[EgoState, Pose]] = []
        self._route: RouteProvider | None = None
        self._latest: WorldSnapshot | None = None

    # -- entry point -------------------------------------------------------

    def run_rollout(self, session_uuid: str | None = None) -> RolloutOutcome:
        """Set up the world, drive it against the driver, and tear it down."""
        session_uuid = session_uuid or str(uuid.uuid4())
        setup = self.world.setup()
        self._route = RouteProvider(
            setup.route_in_local,
            horizon_m=self.config.route_horizon_m,
            resolution_m=self.config.route_resolution_m,
        )
        logger.info(
            "rollout %s on %s: %d camera(s), route %.1f m, rig offset %.3f m",
            session_uuid,
            setup.map_name,
            len(setup.cameras),
            self._route.total_length_m,
            setup.rear_axle_offset_m,
        )

        steps = 0
        terminated_by_driver = False
        error = ""
        error_code = RolloutErrorCode.ROLLOUT_ERROR_CODE_NONE

        try:
            with self._driver_stub() as stub:
                self._start_session(stub, session_uuid, setup)
                try:
                    steps, terminated_by_driver = self._drive_loop(stub, session_uuid)
                finally:
                    with contextlib.suppress(grpc.RpcError):
                        stub.close_session(
                            DriveSessionCloseRequest(session_uuid=session_uuid),
                            timeout=self.config.driver_timeout_s,
                        )
        except grpc.RpcError as exc:
            error = f"driver RPC failed: {exc.code().name}: {exc.details()}"
            error_code = RolloutErrorCode.ROLLOUT_ERROR_CODE_UNSPECIFIED
            logger.error("%s", error)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            error = f"{type(exc).__name__}: {exc}"
            error_code = RolloutErrorCode.ROLLOUT_ERROR_CODE_UNSPECIFIED
            logger.exception("rollout %s failed", session_uuid)
        finally:
            self.world.close()

        events = self.world.events()
        completion = self._route.completion if self._route is not None else 0.0
        success = not error and events.collisions == 0

        return RolloutOutcome(
            rollout_return=self._metrics.build_return(
                rollout_uuid=session_uuid,
                scenario_id=self.scenario.scene_id,
                success=success,
                error=error,
                collisions=events.collisions,
                lane_invasions=events.lane_invasions,
                route_completion=completion,
                error_code=error_code,
            ),
            steps=steps,
            terminated_by_driver=terminated_by_driver,
        )

    # -- session -----------------------------------------------------------

    @contextlib.contextmanager
    def _driver_stub(self) -> Iterator[EgodriverServiceStub]:
        if self._external_channel is not None:
            yield EgodriverServiceStub(self._external_channel)
            return
        with grpc.insecure_channel(
            self.config.driver_address, options=channel_options()
        ) as channel:
            yield EgodriverServiceStub(channel)

    def _start_session(self, stub: EgodriverServiceStub, session_uuid: str, setup) -> None:
        self._check_driver_version(stub)

        request = DriveSessionRequest(
            session_uuid=session_uuid,
            random_seed=self.config.seed,
            debug_info=DriveSessionRequest.DebugInfo(scene_id=self.scenario.scene_id),
            rollout_spec=DriveSessionRequest.RolloutSpec(
                vehicle=DriveSessionRequest.RolloutSpec.VehicleDefinition(
                    available_cameras=setup.cameras,
                ),
            ),
        )
        stub.start_session(request, timeout=self.config.driver_timeout_s)

    def _check_driver_version(self, stub: EgodriverServiceStub) -> None:
        """Log the driver's identity and warn if its API version differs.

        Never fatal: alpasim's API version tracks the ``alpasim_grpc`` release,
        and a mismatch only matters if the messages actually changed.
        """
        try:
            version = stub.get_version(Empty(), timeout=self.config.driver_timeout_s)
        except grpc.RpcError as exc:
            logger.warning("driver get_version failed (%s); continuing", exc.code().name)
            return

        logger.info("driver version_id=%r", version.version_id)
        mismatch = describe_api_mismatch(version.grpc_api_version)
        if mismatch:
            logger.warning("%s", mismatch)

    # -- main loop ---------------------------------------------------------

    def _drive_loop(self, stub: EgodriverServiceStub, session_uuid: str) -> tuple[int, bool]:
        # Priming: let physics settle and the first frames arrive before the
        # driver sees anything, the way alpasim's replay phase does.
        for _ in range(max(0, self.config.warmup_ticks)):
            self._record_tick(self.world.tick())
        if self._latest is None:
            self._record_tick(self.world.tick())
        assert self._latest is not None  # a tick always produces a snapshot

        if self.config.cameras and not self._latest.captures:
            logger.warning(
                "no camera frame arrived during %d warmup ticks despite %d configured "
                "camera(s); the driver will be asked to plan without images",
                self.config.warmup_ticks,
                len(self.config.cameras),
            )

        steps = 0
        for _ in range(self.config.max_steps):
            snapshot = self._latest_snapshot()
            step_start_us = snapshot.timestamp_us
            target_time_us = step_start_us + self.config.policy_timestep_us

            self._submit_observations(stub, session_uuid, snapshot)

            started = time.perf_counter()
            response = stub.drive(
                DriveRequest(
                    session_uuid=session_uuid,
                    time_now_us=step_start_us,
                    time_query_us=target_time_us,
                    renderer_data=pack_renderer_data(self.world.environment(snapshot)),
                ),
                timeout=self.config.driver_timeout_s,
            )
            latency = time.perf_counter() - started
            steps += 1

            plan_in_local = self._plan_to_true_local(
                Trajectory.from_proto(response.trajectory), snapshot
            )
            self._metrics.record_plan(plan_in_local)
            self._log_driver_debug(response)

            command = self._follower.step(
                plan_in_local,
                snapshot.ego.pose_local_to_rig,
                snapshot.ego.speed_mps,
                self.config.policy_timestep_s,
            )
            self._record_metrics(snapshot, command, latency)

            if response.terminate_session:
                logger.info("driver requested termination at t=%d us", step_start_us)
                return steps, True

            self._advance(command)

            if self._route is not None and self._route.completion >= 0.999:
                logger.info("route completed after %d steps", steps)
                return steps, False

        return steps, False

    def _advance(self, command: VehicleCommand) -> None:
        """Apply actuation and run the simulator up to the next policy step."""
        self.world.apply_control(command)
        for _ in range(self.config.ticks_per_policy_step):
            self._record_tick(self.world.tick())

    def _record_tick(self, snapshot: WorldSnapshot) -> None:
        """Fold one simulator tick into the runtime's view of the world."""
        ego = snapshot.ego
        if self._latest is not None and ego.timestamp_us <= self._latest.ego.timestamp_us:
            # A tick that did not advance the clock carries no new information.
            return

        self._last_estimate = self._apply_egomotion_noise(ego.pose_local_to_rig)
        self._pending_egomotion.append((ego, self._last_estimate))
        self._latest = snapshot

    def _latest_snapshot(self) -> WorldSnapshot:
        if self._latest is None:
            raise RuntimeError("no snapshot yet: the world produced no advancing tick")
        return self._latest

    # -- observations ------------------------------------------------------

    def _submit_observations(
        self, stub: EgodriverServiceStub, session_uuid: str, snapshot: WorldSnapshot
    ) -> None:
        """Send everything the driver needs, then return -- the caller's next
        action is ``drive``, which is the barrier alpasim enforces explicitly."""
        for capture in snapshot.captures:
            stub.submit_image_observation(
                RolloutCameraImage(
                    session_uuid=session_uuid,
                    camera_image=RolloutCameraImage.CameraImage(
                        frame_start_us=capture.frame_start_us,
                        frame_end_us=capture.frame_end_us,
                        image_bytes=capture.image_bytes,
                        logical_id=capture.logical_id,
                    ),
                ),
                timeout=self.config.driver_timeout_s,
            )

        self._submit_egomotion(stub, session_uuid)

        waypoints_in_rig = self._route_waypoints_in_rig(snapshot)
        if waypoints_in_rig is not None:
            self._submit_route(stub, session_uuid, snapshot, waypoints_in_rig)
            if self.config.send_ground_truth:
                self._submit_ground_truth(stub, session_uuid, snapshot, waypoints_in_rig)

    def _submit_egomotion(self, stub: EgodriverServiceStub, session_uuid: str) -> None:
        """Send every pose since the last step, as alpasim does.

        Sending the intermediate ticks (not just the newest pose) matters for
        drivers that build an ego-history feature from what they receive.
        """
        if not self._pending_egomotion:
            return

        trajectory = Trajectory(
            [ego.timestamp_us for ego, _ in self._pending_egomotion],
            [estimate for _, estimate in self._pending_egomotion],
        )
        dynamic_states = [
            dynamic_state_proto(
                linear_velocity=ego.linear_velocity_in_rig,
                angular_velocity=ego.angular_velocity_in_rig,
                linear_acceleration=ego.linear_acceleration_in_rig,
            )
            for ego, _ in self._pending_egomotion
        ]
        self._pending_egomotion.clear()

        stub.submit_egomotion_observation(
            RolloutEgoTrajectory(
                session_uuid=session_uuid,
                trajectory=trajectory.to_proto(),
                dynamic_states=dynamic_states,
            ),
            timeout=self.config.driver_timeout_s,
        )

    def _route_waypoints_in_rig(self, snapshot: WorldSnapshot) -> np.ndarray | None:
        """The route window for this step, in the frame the driver reasons in.

        Called once per step and shared by every consumer: ``waypoints_in_rig``
        advances the route's progress marker, and its forward search is bounded,
        so a second call on the same pose can move the marker again.

        Projecting onto the map needs the *true* pose, but the waypoints must
        reach the driver in the frame it reconstructs from the egomotion it was
        sent -- the estimated one. Otherwise the egomotion error shows up as a
        route offset. This is exactly what alpasim's PolicyEvent does.
        """
        if self._route is None:
            return None
        waypoints = self._route.waypoints_in_rig(snapshot.ego.pose_local_to_rig)
        correction = self._estimate_correction(snapshot)
        return waypoints if correction is None else correction.transform_points(waypoints)

    def _submit_route(
        self,
        stub: EgodriverServiceStub,
        session_uuid: str,
        snapshot: WorldSnapshot,
        waypoints_in_rig: np.ndarray,
    ) -> None:
        stub.submit_route(
            RouteRequest(
                session_uuid=session_uuid,
                route=Route(
                    timestamp_us=snapshot.timestamp_us,
                    waypoints=[
                        Vec3(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in waypoints_in_rig
                    ],
                ),
            ),
            timeout=self.config.driver_timeout_s,
        )

    def _submit_ground_truth(
        self,
        stub: EgodriverServiceStub,
        session_uuid: str,
        snapshot: WorldSnapshot,
        waypoints_in_rig: np.ndarray,
    ) -> None:
        """Send the route reference in place of a recorded trajectory.

        There is no recording under CARLA, so this is *not* what alpasim sends.
        It is off by default; see docs/COMPATIBILITY.md before switching it on.
        The waypoints are the same ones ``submit_route`` sent, so the driver
        cannot receive two disagreeing versions of the same polyline.
        """
        step_us = self.config.policy_timestep_us
        reference = Trajectory(
            [snapshot.timestamp_us + i * step_us for i in range(len(waypoints_in_rig))],
            [Pose(point, np.array([0.0, 0.0, 0.0, 1.0])) for point in waypoints_in_rig],
        )
        stub.submit_recording_ground_truth(
            GroundTruthRequest(
                session_uuid=session_uuid,
                ground_truth=GroundTruth(
                    timestamp_us=snapshot.timestamp_us,
                    trajectory=reference.to_proto(),
                ),
            ),
            timeout=self.config.driver_timeout_s,
        )

    # -- frames and noise --------------------------------------------------

    def _apply_egomotion_noise(self, pose_local_to_rig: Pose) -> Pose:
        """Perturb the pose the driver is told about (alpasim's ``rig_est``).

        With both standard deviations at 0 this is the identity, which is the
        default -- CARLA has no proprioceptive error model of its own.
        """
        position_sigma = self.config.egomotion_position_noise_m
        yaw_sigma = self.config.egomotion_yaw_noise_rad
        if position_sigma <= 0.0 and yaw_sigma <= 0.0:
            return pose_local_to_rig

        offset = (
            self._rng.normal(0.0, position_sigma, size=2) if position_sigma > 0.0 else np.zeros(2)
        )
        yaw_offset = float(self._rng.normal(0.0, yaw_sigma)) if yaw_sigma > 0.0 else 0.0
        position = pose_local_to_rig.position + np.array([offset[0], offset[1], 0.0])
        return Pose.from_xyz_yaw(
            position[0], position[1], position[2], pose_local_to_rig.yaw + yaw_offset
        )

    def _estimate_correction(self, snapshot: WorldSnapshot) -> Pose | None:
        """``estimated -> true`` rig correction, or ``None`` when it is identity.

        With no noise model ``_apply_egomotion_noise`` hands back the very pose
        it was given, so an identity check is exact rather than a tolerance --
        and it skips a transform over every waypoint and plan pose on the
        default configuration.
        """
        estimate = self._last_estimate
        true_pose = snapshot.ego.pose_local_to_rig
        if estimate is None or estimate is true_pose:
            return None
        return estimate.inverse() @ true_pose

    def _plan_to_true_local(
        self, plan_in_estimated_local: Trajectory, snapshot: WorldSnapshot
    ) -> Trajectory:
        """Map the driver's plan out of the estimated frame into the true one.

        Identity when no egomotion noise is configured. The formulation matches
        alpasim's ``transform_trajectory_from_noisy_to_true_local_frame``.
        """
        correction = self._estimate_correction(snapshot)
        if correction is None or not plan_in_estimated_local:
            return plan_in_estimated_local
        estimate = self._last_estimate
        assert estimate is not None  # implied by correction being non-None
        return plan_in_estimated_local.transform(
            snapshot.ego.pose_local_to_rig @ estimate.inverse()
        )

    # -- reporting ---------------------------------------------------------

    def _log_driver_debug(self, response) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        debug = unpack_debug_info(response.debug_info.unstructured_debug_info)
        if debug is None:
            return
        logger.debug(
            "driver %s: %.1f ms, %s",
            debug.policy_name,
            debug.inference_seconds * 1e3,
            dict(debug.scalars),
        )

    def _record_metrics(
        self, snapshot: WorldSnapshot, command: VehicleCommand, latency_s: float
    ) -> None:
        self._metrics.record_step(
            timestamp_us=snapshot.timestamp_us,
            pose_local_to_rig=snapshot.ego.pose_local_to_rig,
            speed_mps=snapshot.ego.speed_mps,
            target_speed_mps=command.target_speed_mps,
            route_lateral_error_m=(
                self._route.lateral_error_m(snapshot.ego.pose_local_to_rig) if self._route else 0.0
            ),
            lookahead_lateral_offset_m=command.lookahead_lateral_offset_m,
            route_completion=self._route.completion if self._route else 0.0,
            drive_latency_s=latency_s,
        )
