# SPDX-License-Identifier: Apache-2.0
"""``egodriver.EgodriverService`` implementation.

:class:`CarlaEgodriverServicer` inherits the *upstream generated*
``EgodriverServiceServicer`` from ``alpasim_grpc`` rather than redeclaring the
service, so the RPC names, request/response types and wire format cannot drift
from alpasim.  ``tests/test_proto_compat.py`` asserts the inheritance and the
service full name.

The servicer owns all the plumbing: session lifecycle, frame retention, ego
history, rig<->local conversion and the CARLA extension payloads.  Policies see
only :mod:`carla_driver_interface.driver.base`.
"""

from __future__ import annotations

import logging
import threading
import time

import grpc
import numpy as np

from carla_driver_interface import ALPASIM_GRPC_REV, __version__
from carla_driver_interface.driver.base import (
    BaseDriver,
    CameraFrame,
    DriveContext,
    SessionState,
)
from carla_driver_interface.geometry import Pose, Trajectory
from carla_driver_interface.grpc_api import (
    API_VERSION_MESSAGE,
    CarlaDriveDebugInfo,
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
)
from carla_driver_interface.grpc_api.extension import pack_debug_info, unpack_renderer_data

logger = logging.getLogger(__name__)

__all__ = ["CarlaEgodriverServicer"]


class CarlaEgodriverServicer(EgodriverServiceServicer):
    """Adapts a :class:`BaseDriver` to the alpasim egodriver contract."""

    def __init__(self, driver: BaseDriver) -> None:
        self._driver = driver
        self._sessions: dict[str, SessionState] = {}
        self._sessions_lock = threading.Lock()

    # -- session lifecycle -------------------------------------------------

    def start_session(
        self, request: DriveSessionRequest, context: grpc.ServicerContext
    ) -> SessionRequestStatus:
        with self._sessions_lock:
            if request.session_uuid in self._sessions:
                context.abort(
                    grpc.StatusCode.ALREADY_EXISTS,
                    f"Session {request.session_uuid} already exists.",
                )
                return SessionRequestStatus()

            cameras = {
                cam.logical_id: cam for cam in request.rollout_spec.vehicle.available_cameras
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
        self._driver.on_session_start(session)
        return SessionRequestStatus()

    def close_session(
        self, request: DriveSessionCloseRequest, context: grpc.ServicerContext
    ) -> Empty:
        with self._sessions_lock:
            session = self._sessions.pop(request.session_uuid, None)
        if session is None:
            logger.warning("close_session for unknown session %s", request.session_uuid)
            return Empty()
        logger.info("close_session uuid=%s after %d drives", session.uuid, session.drive_count)
        self._driver.on_session_close(session)
        return Empty()

    def get_version(self, request: Empty, context: grpc.ServicerContext) -> VersionId:
        return VersionId(
            version_id=f"{self._driver.name}-carla-driver-interface-{__version__}",
            git_hash=ALPASIM_GRPC_REV,
            # Forwarded verbatim so an alpasim runtime sees the API version of
            # the alpasim_grpc build we compiled against.
            grpc_api_version=API_VERSION_MESSAGE,
        )

    # -- observations ------------------------------------------------------

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
            return Empty()

        frame = CameraFrame(
            logical_id=image.logical_id,
            frame_start_us=int(image.frame_start_us),
            frame_end_us=int(image.frame_end_us),
            image_bytes=image.image_bytes,
        )
        with session.lock:
            history = session.frame_history.setdefault(frame.logical_id, [])
            history.append(frame)
            keep = max(1, self._driver.frame_history_length)
            del history[:-keep]

        self._driver.on_image(session, frame)
        return Empty()

    def submit_egomotion_observation(
        self, request: RolloutEgoTrajectory, context: grpc.ServicerContext
    ) -> Empty:
        session = self._require_session(request.session_uuid, context)
        new_poses = Trajectory.from_proto(request.trajectory)

        # Upstream sends dynamic states 1:1 with poses, but tolerates them being
        # absent (the field is not required); pad rather than reject.
        states = list(request.dynamic_states)
        if states and len(states) != len(new_poses):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"dynamic_states ({len(states)}) must match trajectory poses "
                f"({len(new_poses)}) when provided",
            )
            return Empty()

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
                # Keep dynamic_states aligned 1:1 with the trajectory even when
                # the sender omitted them, so index-based access stays valid.
                session.dynamic_states.append(states[i] if states else DynamicState())

        self._driver.on_egomotion(session, new_poses)
        return Empty()

    def submit_route(self, request: RouteRequest, context: grpc.ServicerContext) -> Empty:
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
            session.ground_truth_in_rig = Trajectory.from_proto(request.ground_truth.trajectory)
        self._driver.on_ground_truth(session)
        return Empty()

    # -- decision ----------------------------------------------------------

    def drive(self, request: DriveRequest, context: grpc.ServicerContext) -> DriveResponse:
        session = self._require_session(request.session_uuid, context)
        time_now_us = int(request.time_now_us)

        ctx = DriveContext(
            session=session,
            time_now_us=time_now_us,
            time_query_us=int(request.time_query_us),
            renderer_data=unpack_renderer_data(request.renderer_data),
        )

        started = time.perf_counter()
        result = self._driver.drive(ctx)
        elapsed = time.perf_counter() - started

        with session.lock:
            session.drive_count += 1
            anchor = session.latest_pose()

        # Before the first egomotion observation there is no anchor, so there is
        # no frame to express a plan in. Upstream's driver answers an empty
        # trajectory in the equivalent "not ready" case, and the runtime treats
        # that as "hold".
        if anchor is None:
            logger.debug("drive before first egomotion observation; returning empty plan")
            trajectory_proto = Trajectory.empty().to_proto()
            sampled = []
        else:
            trajectory_proto = _rig_plan_to_local(
                result.trajectory_in_rig, anchor, time_now_us
            ).to_proto()
            sampled = [
                _rig_plan_to_local(traj, anchor, time_now_us).to_proto()
                for traj in result.sampled_trajectories_in_rig
            ]

        debug = CarlaDriveDebugInfo(
            policy_name=self._driver.name,
            inference_seconds=elapsed,
            scalars=result.debug_scalars,
        )
        return DriveResponse(
            trajectory=trajectory_proto,
            debug_info=DriveResponse.DebugInfo(
                unstructured_debug_info=pack_debug_info(debug),
                sampled_trajectories=sampled,
            ),
            terminate_session=result.terminate_session,
        )

    # -- helpers -----------------------------------------------------------

    def _require_session(self, uuid: str, context: grpc.ServicerContext) -> SessionState:
        with self._sessions_lock:
            session = self._sessions.get(uuid)
        if session is None:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown session {uuid}")
            raise AssertionError("unreachable: context.abort raises")  # pragma: no cover
        return session


def _rig_plan_to_local(plan_in_rig: Trajectory, anchor: Pose, time_now_us: int) -> Trajectory:
    """Express a rig-frame plan in the ``local`` frame, alpasim style.

    The response trajectory is led by the ego pose at ``time_now_us`` and
    followed by the planned waypoints, all as ``local -> rig_est`` active
    transforms -- matching what ``alpasim_driver`` produces and what the alpasim
    runtime's ``transform_trajectory_from_noisy_to_true_local_frame`` expects.
    """
    local = Trajectory([time_now_us], [anchor])
    for timestamp_us, pose in zip(plan_in_rig.timestamps_us, plan_in_rig.poses, strict=True):
        if timestamp_us <= time_now_us:
            # The policy may include its own t0 waypoint; the anchor already
            # covers it and duplicate timestamps are not allowed.
            continue
        local.append(timestamp_us, anchor @ pose)
    return local
