"""A closed loop without a simulator, for testing policies.

:class:`FakeLoop` plays the runtime role of the egodriver contract over real gRPC --
``start_session``, egomotion, route and camera frames, ``drive`` -- against any
policy server, on a straight two-lane road:

* the ego follows the returned plan exactly (its pose at the next step is the plan's
  pose at that time), so what is tested is the policy and the wire, not a
  controller;
* camera frames are rendered through each camera's own pinhole model: grey road,
  white lane markings, sky above the horizon. Crude, but a vision policy sees lanes
  where geometry says they are.

It is what CI runs a policy against; anything about real vehicle dynamics or real
images belongs in a scenario against CARLA, e.g. with autoware_carla_scenario
(https://github.com/autowarefoundation/autoware_carla_scenario).
"""

from __future__ import annotations

import io
import math
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import grpc
import numpy as np
from numpy.typing import NDArray

from .geometry import Pose, Trajectory, waypoints_to_proto
from .protocol import (
    AvailableCamera,
    CameraSpec,
    CarlaDriveDebugInfo,
    DriveRequest,
    DriveSessionCloseRequest,
    DriveSessionRequest,
    DynamicState,
    EgodriverServiceStub,
    OpenCVPinholeCameraParam,
    RolloutCameraImage,
    RolloutEgoTrajectory,
    Route,
    RouteRequest,
    ShutterType,
    Vec3,
    channel_options,
    unpack_debug_info,
)

__all__ = ["FakeCamera", "FakeLoop", "LoopResult"]

#: Lane markings of the fake road, as lateral offsets in the local frame [m].
_LANE_MARKINGS_Y = (-5.25, -1.75, 1.75, 5.25)
_ROAD_HALF_WIDTH_M = 7.0
_SKY = (110, 160, 220)
_ROAD = (75, 75, 75)
_VERGE = (70, 110, 60)
_MARKING = (235, 235, 235)


@dataclass(frozen=True)
class FakeCamera:
    """An ideal pinhole camera on the ego, in the rig frame.

    The rig frame is alpasim's: x forward, y left, z up, origin on the ground below
    the rear axle. ``pitch_down_deg`` > 0 tilts the camera towards the road.
    """

    logical_id: str = "camera_front"
    width: int = 640
    height: int = 360
    fov_deg: float = 52.0
    x: float = 2.5
    y: float = 0.0
    z: float = 1.5
    pitch_down_deg: float = 0.0
    yaw_left_deg: float = 0.0

    @property
    def focal_px(self) -> float:
        return self.width / (2.0 * math.tan(math.radians(self.fov_deg) / 2.0))

    def pose_in_rig(self) -> Pose:
        """Camera body pose in the rig (x along the optical axis, y left, z up)."""
        rotation = Pose.from_axis_angle(
            2, math.radians(self.yaw_left_deg)
        ) @ Pose.from_axis_angle(1, math.radians(self.pitch_down_deg))
        return Pose(np.array([self.x, self.y, self.z]), rotation.quat_xyzw)

    def available_camera(self) -> AvailableCamera:
        spec = CameraSpec(
            opencv_pinhole_param=OpenCVPinholeCameraParam(
                principal_point_x=self.width / 2.0,
                principal_point_y=self.height / 2.0,
                focal_length_x=self.focal_px,
                focal_length_y=self.focal_px,
            ),
            logical_id=self.logical_id,
            resolution_w=self.width,
            resolution_h=self.height,
            shutter_type=ShutterType.GLOBAL,
        )
        return AvailableCamera(
            intrinsics=spec,
            rig_to_camera=self.pose_in_rig().to_proto(),
            logical_id=self.logical_id,
        )


@dataclass
class LoopResult:
    """What happened in one :meth:`FakeLoop.run`."""

    #: Ego poses in the local frame, one per step plus the start.
    ego: Trajectory = field(default_factory=Trajectory.empty)
    #: Speed at each recorded pose [m/s].
    speeds_mps: List[float] = field(default_factory=list)
    #: The policy's debug payload per drive call (``None`` if it sent none).
    debug: List[Optional[CarlaDriveDebugInfo]] = field(default_factory=list)
    #: Number of drive calls made.
    steps: int = 0
    terminated_by_policy: bool = False

    @property
    def lateral_offsets_m(self) -> NDArray[np.float64]:
        """The ego's lateral position on the road (0 = centre of the start lane)."""
        return self.ego.positions[:, 1]


class FakeLoop:
    """Drive a policy server through a straight-road closed loop.

    Args:
        target: ``host:port`` of the policy server, or an open gRPC channel.
        cameras: Cameras to declare and render.
        step_s: Simulated time between drive calls.
        initial_speed_mps: Ego speed at the start.
        initial_lateral_offset_m: Start this far left of the lane centre.
        initial_heading_rad: Start at this heading relative to the road.
        image_format: ``"jpeg"`` or ``"png"``.
    """

    def __init__(
        self,
        target: "str | grpc.Channel",
        cameras: Sequence[FakeCamera] = (FakeCamera(),),
        *,
        step_s: float = 0.1,
        initial_speed_mps: float = 0.0,
        initial_lateral_offset_m: float = 0.0,
        initial_heading_rad: float = 0.0,
        image_format: str = "jpeg",
        timeout_s: float = 60.0,
    ) -> None:
        if not cameras:
            raise ValueError("at least one camera is needed")
        self._channel = (
            grpc.insecure_channel(target, options=channel_options())
            if isinstance(target, str)
            else target
        )
        self._owns_channel = isinstance(target, str)
        self._stub = EgodriverServiceStub(self._channel)
        self.cameras = tuple(cameras)
        self.step_s = step_s
        self.initial_speed_mps = initial_speed_mps
        self.initial_lateral_offset_m = initial_lateral_offset_m
        self.initial_heading_rad = initial_heading_rad
        self.image_format = image_format.upper()
        if self.image_format not in ("JPEG", "PNG"):
            raise ValueError("image_format must be 'jpeg' or 'png'")
        self.timeout_s = timeout_s

    def close(self) -> None:
        if self._owns_channel:
            self._channel.close()

    def __enter__(self) -> "FakeLoop":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- the loop --------------------------------------------------------------

    def run(self, steps: int, *, seed: int = 0) -> LoopResult:
        """Run one session of ``steps`` drive calls and close it."""
        session = str(uuid.uuid4())
        request = DriveSessionRequest(session_uuid=session, random_seed=seed)
        request.debug_info.scene_id = "fake:straight_road"
        request.rollout_spec.vehicle.available_cameras.extend(
            camera.available_camera() for camera in self.cameras
        )
        self._stub.start_session(request, timeout=self.timeout_s)

        result = LoopResult()
        pose = Pose.from_xyz_yaw(
            0.0, self.initial_lateral_offset_m, 0.0, self.initial_heading_rad
        )
        speed = self.initial_speed_mps
        step_us = int(round(self.step_s * 1e6))
        try:
            for k in range(steps):
                now_us = k * step_us
                result.ego.append(now_us, pose)
                result.speeds_mps.append(speed)
                self._observe(session, now_us, pose, speed)

                response = self._stub.drive(
                    DriveRequest(
                        session_uuid=session,
                        time_now_us=now_us,
                        time_query_us=now_us + step_us,
                    ),
                    timeout=self.timeout_s,
                )
                result.steps += 1
                result.debug.append(
                    unpack_debug_info(response.debug_info.unstructured_debug_info)
                )
                if response.terminate_session:
                    result.terminated_by_policy = True
                    break

                plan = Trajectory.from_proto(response.trajectory)
                following = (
                    plan.interpolate(now_us + step_us) if len(plan) >= 2 else None
                )
                if following is None:
                    # An empty plan means "hold": stay put.
                    speed = 0.0
                    continue
                speed = (
                    float(np.linalg.norm(following.position[:2] - pose.position[:2]))
                    / self.step_s
                )
                pose = following
        finally:
            self._stub.close_session(
                DriveSessionCloseRequest(session_uuid=session), timeout=self.timeout_s
            )
        return result

    def _observe(self, session: str, now_us: int, pose: Pose, speed: float) -> None:
        state = DynamicState(linear_velocity=Vec3(x=speed, y=0.0, z=0.0))
        egomotion = RolloutEgoTrajectory(session_uuid=session)
        egomotion.trajectory.CopyFrom(Trajectory([now_us], [pose]).to_proto())
        egomotion.dynamic_states.append(state)
        self._stub.submit_egomotion_observation(egomotion, timeout=self.timeout_s)

        # The route is the lane centre ahead, in the rig frame.
        ahead = np.stack(
            [np.arange(0.0, 100.0, 2.0) + pose.position[0], np.zeros(50), np.zeros(50)],
            axis=1,
        )
        in_rig = pose.inverse().transform_points(ahead)
        self._stub.submit_route(
            RouteRequest(
                session_uuid=session,
                route=Route(timestamp_us=now_us, waypoints=waypoints_to_proto(in_rig)),
            ),
            timeout=self.timeout_s,
        )

        for camera in self.cameras:
            image = RolloutCameraImage(session_uuid=session)
            image.camera_image.logical_id = camera.logical_id
            image.camera_image.frame_start_us = now_us
            image.camera_image.frame_end_us = now_us
            image.camera_image.image_bytes = self._encode(render_road(camera, pose))
            self._stub.submit_image_observation(image, timeout=self.timeout_s)

    def _encode(self, rgb: NDArray[np.uint8]) -> bytes:
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(rgb).save(buffer, format=self.image_format, quality=90)
        return buffer.getvalue()


def render_road(camera: FakeCamera, ego_pose: Pose) -> NDArray[np.uint8]:
    """Render the fake road as ``camera`` on an ego at ``ego_pose`` sees it (RGB)."""
    h, w = camera.height, camera.width
    f = camera.focal_px
    camera_in_local = ego_pose @ camera.pose_in_rig()
    rotation = camera_in_local.rotation_matrix  # local <- camera body
    origin = camera_in_local.position

    # Ray of every pixel in the local frame: optical (right, down, forward) -> body.
    u = (np.arange(w) + 0.5 - w / 2.0) / f
    v = (np.arange(h) + 0.5 - h / 2.0) / f
    uu, vv = np.meshgrid(u, v)
    rays_body = np.stack([np.ones_like(uu), -uu, -vv], axis=-1)
    rays = rays_body @ rotation.T

    image = np.empty((h, w, 3), dtype=np.uint8)
    image[:] = _SKY
    down = rays[..., 2] < -1e-6
    t = np.where(down, -origin[2] / np.where(down, rays[..., 2], -1.0), np.inf)
    visible = down & (t < 400.0)
    # Rays that never reach the road get a finite placeholder, not inf * 0.
    t = np.where(visible, t, 0.0)
    ground_x = origin[0] + t * rays[..., 0]
    ground_y = origin[1] + t * rays[..., 1]

    image[visible & (np.abs(ground_y) <= _ROAD_HALF_WIDTH_M)] = _ROAD
    image[visible & (np.abs(ground_y) > _ROAD_HALF_WIDTH_M)] = _VERGE
    # Markings 15 cm wide; dashed in the middle, solid at the edges.
    for y in _LANE_MARKINGS_Y:
        on_line = visible & (np.abs(ground_y - y) < 0.075)
        if abs(y) < 2.0:
            on_line &= np.mod(ground_x, 12.0) < 6.0
        image[on_line] = _MARKING
    return image
