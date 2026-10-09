"""The ``egodriver.EgodriverService`` servicer that adapts a :class:`BaseDriver`.

:class:`EgodriverServicer` subclasses the servicer generated from the vendored
alpasim proto rather than redeclaring the service, so RPC names, message types and
wire format cannot drift from upstream. It owns the plumbing -- session lifecycle,
frame retention, ego history, rig/local conversion, the CARLA payloads -- and the
policy sees only :mod:`carla_driver_interface.driver`.

It serves in one of two modes (:data:`ServiceMode`), which differ only in the two
opaque ``bytes`` fields the contract leaves to each runtime:

* ``"carla"`` (the default): ``DriveRequest.renderer_data`` carries
  ``CarlaRendererData`` and the response's ``unstructured_debug_info`` a
  ``CarlaDriveDebugInfo``, as autoware_carla_scenario reads and writes them;
* ``"alpasim"``: under an upstream alpasim runtime, ``renderer_data`` is the
  renderer's own payload, so it is never parsed as CARLA ground truth, and
  ``unstructured_debug_info`` is the pickled ``dict`` alpasim's evaluation unpickles
  (``policy_name``, ``inference_seconds``, ``scalars``), as alpasim's own driver
  answers.
"""

from __future__ import annotations

import logging
import pickle
import threading
import time
from typing import Dict, Literal, Optional, get_args

import grpc
import numpy as np

from . import __version__
from .driver import (
    BaseDriver,
    CameraFrame,
    DriveContext,
    LidarFrame,
    SensorFrame,
    SessionState,
)
from .geometry import Pose, Trajectory
from .hdmap import MapFiles
from .protocol import (
    ALPASIM_API_VERSION,
    ALPASIM_REV,
    CarlaDriveDebugInfo,
    CarlaRendererData,
    DriveRequest,
    DriveResponse,
    DriveSessionCloseRequest,
    DriveSessionRequest,
    DynamicState,
    EgodriverServiceServicer,
    Empty,
    GroundTruthRequest,
    RolloutCameraImage,
    RolloutEgoTrajectory,
    RouteRequest,
    SessionRequestStatus,
    VersionId,
    pack_debug_info,
    unpack_renderer_data,
)

logger = logging.getLogger(__name__)

__all__ = ["EgodriverServicer", "SERVICE_MODES", "ServiceMode"]

#: What the runtime on the other end puts in the contract's opaque fields.
ServiceMode = Literal["carla", "alpasim"]
SERVICE_MODES: tuple = get_args(ServiceMode)


class EgodriverServicer(EgodriverServiceServicer):
    """Serves one :class:`BaseDriver` over the alpasim egodriver contract."""

    def __init__(self, driver: BaseDriver, mode: ServiceMode = "carla") -> None:
        if mode not in SERVICE_MODES:
            raise ValueError(f"unknown mode {mode!r}: expected one of {SERVICE_MODES}")
        self._driver = driver
        self._mode = mode
        self._sessions: Dict[str, SessionState] = {}
        self._sessions_lock = threading.Lock()
        #: Opened map sets by ``map_id``; a map does not change under its id.
        self._maps: Dict[str, MapFiles] = {}

    # -- session lifecycle -----------------------------------------------------

    def start_session(
        self, request: DriveSessionRequest, context: grpc.ServicerContext
    ) -> SessionRequestStatus:
        with self._sessions_lock:
            if request.session_uuid in self._sessions:
                context.abort(
                    grpc.StatusCode.ALREADY_EXISTS,
                    f"Session {request.session_uuid} already exists.",
                )
            cameras = {
                cam.logical_id: cam
                for cam in request.rollout_spec.vehicle.available_cameras
            }
            session = SessionState(
                uuid=request.session_uuid,
                seed=int(request.random_seed),
                scene_id=request.debug_info.scene_id,
                cameras=cameras,
                frame_history={logical_id: [] for logical_id in cameras},
            )
            self._sessions[request.session_uuid] = session

        logger.info(
            "start_session uuid=%s scene=%r cameras=%s",
            session.uuid,
            session.scene_id,
            sorted(session.cameras),
        )
        try:
            self._driver.on_session_start(session)
        except Exception as error:
            # A policy that cannot drive this session (wrong cameras, say) must
            # say so to the runtime rather than fail later, mid-rollout.
            with self._sessions_lock:
                self._sessions.pop(session.uuid, None)
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION, f"{self._driver.name}: {error}"
            )
        return SessionRequestStatus()

    def close_session(
        self, request: DriveSessionCloseRequest, context: grpc.ServicerContext
    ) -> Empty:
        with self._sessions_lock:
            session = self._sessions.pop(request.session_uuid, None)
        if session is None:
            logger.warning("close_session for unknown session %s", request.session_uuid)
            return Empty()
        logger.info(
            "close_session uuid=%s after %d drives", session.uuid, session.drive_count
        )
        self._driver.on_session_close(session)
        return Empty()

    def get_version(self, request: Empty, context: grpc.ServicerContext) -> VersionId:
        version = VersionId(
            version_id=f"{self._driver.name}-carla-driver-interface-{__version__}",
            git_hash=ALPASIM_REV,
        )
        version.grpc_api_version.CopyFrom(ALPASIM_API_VERSION)
        return version

    # -- observations ----------------------------------------------------------

    def submit_image_observation(
        self, request: RolloutCameraImage, context: grpc.ServicerContext
    ) -> Empty:
        session = self._require_session(request.session_uuid, context)
        image = request.camera_image
        if image.logical_id not in session.cameras:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Camera {image.logical_id!r} was not declared in start_session; "
                f"known cameras: {sorted(session.cameras)}",
            )
        self._record_frame(
            session,
            CameraFrame(
                logical_id=image.logical_id,
                frame_start_us=int(image.frame_start_us),
                frame_end_us=int(image.frame_end_us),
                image_bytes=image.image_bytes,
            ),
        )
        return Empty()

    def submit_egomotion_observation(
        self, request: RolloutEgoTrajectory, context: grpc.ServicerContext
    ) -> Empty:
        session = self._require_session(request.session_uuid, context)
        new_poses = Trajectory.from_proto(request.trajectory)

        # Upstream sends dynamic states 1:1 with poses but does not require them;
        # pad rather than reject when they are absent.
        states = list(request.dynamic_states)
        if states and len(states) != len(new_poses):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"dynamic_states ({len(states)}) must match trajectory poses "
                f"({len(new_poses)}) when provided",
            )
        with session.lock:
            for i, (timestamp_us, pose) in enumerate(
                zip(new_poses.timestamps_us, new_poses.poses, strict=True)
            ):
                # The runtime may resend the pose at the previous step boundary;
                # skipping it keeps timestamps strictly increasing.
                seen = session.ego_trajectory.timestamps_us
                if seen and timestamp_us <= seen[-1]:
                    continue
                session.ego_trajectory.append(timestamp_us, pose)
                session.dynamic_states.append(states[i] if states else DynamicState())
        self._driver.on_egomotion(session, new_poses)
        return Empty()

    def submit_route(
        self, request: RouteRequest, context: grpc.ServicerContext
    ) -> Empty:
        session = self._require_session(request.session_uuid, context)
        waypoints = request.route.waypoints
        with session.lock:
            session.route_waypoints_in_rig = (
                np.array([[w.x, w.y, w.z] for w in waypoints], dtype=np.float64)
                if waypoints
                else np.zeros((0, 3), dtype=np.float64)
            )
            session.route_timestamp_us = int(request.route.timestamp_us)
        self._driver.on_route(session)
        return Empty()

    def submit_recording_ground_truth(
        self, request: GroundTruthRequest, context: grpc.ServicerContext
    ) -> Empty:
        session = self._require_session(request.session_uuid, context)
        with session.lock:
            session.ground_truth_in_rig = Trajectory.from_proto(
                request.ground_truth.trajectory
            )
        self._driver.on_ground_truth(session)
        return Empty()

    # -- decision --------------------------------------------------------------

    def drive(
        self, request: DriveRequest, context: grpc.ServicerContext
    ) -> DriveResponse:
        session = self._require_session(request.session_uuid, context)
        time_now_us = int(request.time_now_us)
        # alpasim's renderer forwards a payload of its own here; parsing it as
        # CARLA ground truth could "succeed" on foreign bytes.
        renderer_data = (
            None
            if self._mode == "alpasim"
            else unpack_renderer_data(request.renderer_data)
        )
        # The contract has no LiDAR RPC, so sweeps ride in renderer_data; recorded
        # here, before drive, they reach the policy exactly as camera frames do.
        if renderer_data is not None:
            for sweep in renderer_data.lidar:
                self._record_frame(session, LidarFrame.from_proto(sweep))
        ctx = DriveContext(
            session=session,
            time_now_us=time_now_us,
            time_query_us=int(request.time_query_us),
            renderer_data=renderer_data,
            map=self._map(renderer_data),
        )

        started = time.perf_counter()
        result = self._driver.drive(ctx)
        elapsed = time.perf_counter() - started

        with session.lock:
            session.drive_count += 1
            anchor = session.latest_pose()

        # Before the first egomotion observation there is no frame to express a
        # plan in; upstream answers an empty trajectory, which the runtime treats
        # as "hold".
        if anchor is None:
            logger.debug("drive before the first egomotion observation; empty plan")
            trajectory = Trajectory.empty().to_proto()
            sampled = []
        else:
            trajectory = _rig_plan_to_local(
                result.trajectory_in_rig, anchor, time_now_us
            ).to_proto()
            sampled = [
                _rig_plan_to_local(plan, anchor, time_now_us).to_proto()
                for plan in result.sampled_trajectories_in_rig
            ]

        return DriveResponse(
            trajectory=trajectory,
            debug_info=DriveResponse.DebugInfo(
                unstructured_debug_info=self._debug_payload(
                    elapsed, result.debug_scalars
                ),
                sampled_trajectories=sampled,
            ),
            terminate_session=result.terminate_session,
        )

    # -- helpers ---------------------------------------------------------------

    def _debug_payload(self, elapsed: float, scalars: Dict[str, float]) -> bytes:
        """``unstructured_debug_info`` in the form this mode's runtime reads."""
        if self._mode == "alpasim":
            # Builtins only, so unpickling it needs nothing from this package.
            return pickle.dumps(
                {
                    "policy_name": self._driver.name,
                    "inference_seconds": float(elapsed),
                    "scalars": {str(k): float(v) for k, v in scalars.items()},
                }
            )
        return pack_debug_info(
            CarlaDriveDebugInfo(
                policy_name=self._driver.name,
                inference_seconds=elapsed,
                scalars=scalars,
            )
        )

    def _record_frame(self, session: SessionState, frame: SensorFrame) -> None:
        """Record one sensor frame in ``session`` and announce it; every sensor's path."""
        with session.lock:
            history = session.frame_history.setdefault(frame.logical_id, [])
            history.append(frame)
            del history[: -max(1, self._driver.frame_history_length)]
        self._driver.on_frame(session, frame)

    def _map(self, renderer_data: Optional[CarlaRendererData]) -> Optional[MapFiles]:
        """The map set *renderer_data* names, opened once per id."""
        map_dir = self._driver.map_dir
        if map_dir is None or renderer_data is None or not renderer_data.map_id:
            return None
        map_id = renderer_data.map_id
        with self._sessions_lock:
            if map_id not in self._maps:
                self._maps[map_id] = MapFiles.open(
                    map_dir, map_id, self._driver.map_format
                )
                logger.info("opened map %s from %s", map_id, map_dir)
            return self._maps[map_id]

    def _require_session(
        self, uuid: str, context: grpc.ServicerContext
    ) -> SessionState:
        with self._sessions_lock:
            session = self._sessions.get(uuid)
        if session is None:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown session {uuid}")
            raise AssertionError(
                "unreachable: context.abort raises"
            )  # pragma: no cover
        return session


def _rig_plan_to_local(
    plan_in_rig: Trajectory, anchor: Pose, time_now_us: int
) -> Trajectory:
    """Express a rig-frame plan in the ``local`` frame, alpasim style.

    The response is led by the ego pose at ``time_now_us`` and followed by the
    planned waypoints, all as ``local -> rig_est`` transforms -- what alpasim's own
    driver produces and its runtime expects.
    """
    local = Trajectory([time_now_us], [anchor])
    for timestamp_us, pose in zip(
        plan_in_rig.timestamps_us, plan_in_rig.poses, strict=True
    ):
        if timestamp_us <= time_now_us:
            # The anchor already covers t0; duplicate timestamps are not allowed.
            continue
        local.append(timestamp_us, anchor @ pose)
    return local
