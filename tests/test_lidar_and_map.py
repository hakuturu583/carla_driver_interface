"""What the runtime adds to the alpasim contract: LiDAR frames and map-file stop lines.

The map set here is written by hand -- a manifest, roadgen's IR and two traces --
so the reader is pinned without roadgen; autoware_carla_scenario's tests write
real sets with roadgen and resolve lights against them end to end.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from carla_driver_interface import protocol
from carla_driver_interface.driver import (
    BaseDriver,
    DriveContext,
    DriveResult,
    LidarFrame,
    SensorFrame,
    SessionState,
)
from carla_driver_interface.geometry import Pose, Trajectory
from carla_driver_interface.hdmap import MapFiles
from carla_driver_interface.protocol import (
    AvailableCamera,
    CarlaRendererData,
    DriveRequest,
    DriveSessionRequest,
    LidarSweep,
    RolloutCameraImage,
    StopPoint,
    TrafficLight,
    TrafficLightState,
    Vec3,
    pack_lidar_sweep,
    unpack_lidar_points,
)
from carla_driver_interface.service import EgodriverServicer

# ---------------------------------------------------------------------------
# LiDAR
# ---------------------------------------------------------------------------


def _context(
    data: CarlaRendererData, map_files: Optional[MapFiles] = None
) -> DriveContext:
    session = SessionState(uuid="s", seed=0, scene_id="Town", cameras={})
    return DriveContext(session, 0, 100_000, data, map=map_files)


def test_a_sweep_round_trips_through_the_renderer_payload() -> None:
    points = np.random.default_rng(0).normal(size=(1000, 4)).astype(np.float32)
    sweep = pack_lidar_sweep("lidar_top", 123, points, Pose.identity().to_proto())
    payload = protocol.pack_renderer_data(CarlaRendererData(lidar=[sweep]))

    restored = protocol.unpack_renderer_data(payload)

    assert restored is not None
    out = unpack_lidar_points(restored.lidar[0])
    np.testing.assert_array_equal(out, points)
    assert out.flags.writeable
    frame = LidarFrame.from_proto(restored.lidar[0])
    assert (frame.logical_id, frame.timestamp_us, frame.num_points) == (
        "lidar_top",
        123,
        1000,
    )
    np.testing.assert_array_equal(frame.as_array(), points)


def test_a_truncated_sweep_is_refused_rather_than_read_short() -> None:
    sweep = pack_lidar_sweep(
        "lidar_top", 0, np.zeros((10, 4)), Pose.identity().to_proto()
    )
    broken = LidarSweep()
    broken.CopyFrom(sweep)
    broken.points_xyzi = sweep.points_xyzi[:-4]
    with pytest.raises(ValueError, match="declares 10 points"):
        unpack_lidar_points(broken)
    with pytest.raises(ValueError, match="declares 10 points"):
        LidarFrame.from_proto(broken).as_array()


def test_packing_refuses_the_wrong_number_of_columns() -> None:
    with pytest.raises(ValueError, match=r"\[N, 4\]"):
        pack_lidar_sweep("lidar_top", 0, np.zeros((3, 5)), Pose.identity().to_proto())


# ---------------------------------------------------------------------------
# Map files
# ---------------------------------------------------------------------------


def _write_set(directory: Path) -> Path:
    """A one-light map set: OpenDRIVE signal 5 on lane 3/0/-1, as Lanelet2."""
    directory.mkdir(parents=True)
    files = {
        "manifest.json": {
            "map_id": "Town-000000000000",
            "map_name": "Town",
            "source": "map.xodr",
            "ir": "map.ir.json",
            "read_trace": "map.roadgen.xodr.read.trace.json",
            "formats": {
                "lanelet2": {
                    "path": "lanelet2_map.osm",
                    "trace": "lanelet2_map.osm.trace.json",
                }
            },
        },
        "map.ir.json": {
            "rules": [{"id": "rule/light/5", "objects": ["object/light/5"]}],
        },
        "map.roadgen.xodr.read.trace.json": {
            "links": [
                {"ref": "signal:5", "ir": "object/light/5"},
                {"ref": "lane:3/0/-1", "ir": "lane/3/0/-1"},
            ]
        },
        "lanelet2_map.osm.trace.json": {
            "links": [
                {"ir": "lane/3/0/-1", "ref": "relation:100", "role": "lanelet"},
                {"ir": "lane/3/0/-1", "ref": "way:101", "role": "left_bound"},
                {"ir": "rule/light/5", "ref": "relation:200"},
                {"ir": "object/light/5", "ref": "way:300"},
            ]
        },
    }
    for name, content in files.items():
        (directory / name).write_text(json.dumps(content))
    (directory / "map.xodr").write_text("<OpenDRIVE/>")
    (directory / "lanelet2_map.osm").write_text("<osm/>")
    return directory


def _light(signal: str = "5", lane_id: int = -1) -> TrafficLight:
    return TrafficLight(
        opendrive_id=signal,
        state=TrafficLightState.TRAFFIC_LIGHT_STATE_RED,
        stop_points=[
            StopPoint(
                road_id=3,
                section_id=0,
                lane_id=lane_id,
                position_local=Vec3(x=9.0, y=-1.5),
            )
        ],
    )


def test_a_light_resolves_to_the_formats_own_elements(tmp_path: Path) -> None:
    map_files = MapFiles.open(tmp_path, _write_set(tmp_path / "Town-000000000000").name)
    [stop] = map_files.stop_lines([_light()])

    assert stop.traffic_light_id == "5"
    assert stop.state == TrafficLightState.TRAFFIC_LIGHT_STATE_RED
    assert stop.position_local.tolist() == [9.0, -1.5, 0.0]
    # The lanelet only, not its boundaries; the kind prefix stripped.
    assert stop.lane_ids == ("100",)
    assert stop.rule_ids == ("200",)
    assert stop.light_ids == ("300",)


def test_a_light_the_map_lacks_still_reports_where_to_stop(tmp_path: Path) -> None:
    map_files = MapFiles(_write_set(tmp_path / "set"))
    [stop] = map_files.stop_lines([_light(signal="99", lane_id=-7)])
    assert stop.lane_ids == () and stop.rule_ids == () and stop.light_ids == ()
    assert stop.position_local.tolist() == [9.0, -1.5, 0.0]


def test_a_missing_set_or_format_says_what_to_fix(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="map_dir"):
        MapFiles.open(tmp_path, "Town-000000000000")
    directory = _write_set(tmp_path / "set")
    with pytest.raises(ValueError, match="map_formats"):
        MapFiles(directory, format="clipgt")


def test_stop_lines_need_a_map_and_lights(tmp_path: Path) -> None:
    map_files = MapFiles(_write_set(tmp_path / "set"))
    assert _context(CarlaRendererData(traffic_lights=[_light()])).stop_lines() == []
    assert _context(CarlaRendererData(), map_files).stop_lines() == []
    [stop] = _context(
        CarlaRendererData(traffic_lights=[_light()]), map_files
    ).stop_lines()
    assert stop.lane_ids == ("100",)


# ---------------------------------------------------------------------------
# One path for every sensor
# ---------------------------------------------------------------------------


class _Frames(BaseDriver):
    """Keeps two frames per sensor and notes every frame it is told about."""

    name = "frames"
    frame_history_length = 2

    def __init__(self) -> None:
        self.announced: List[str] = []
        self.seen: List[Dict[str, List[int]]] = []

    def on_frame(self, session: SessionState, frame: SensorFrame) -> None:
        self.announced.append(f"{type(frame).__name__}:{frame.logical_id}")

    def drive(self, ctx: DriveContext) -> DriveResult:
        self.seen.append(
            {
                logical_id: [frame.timestamp_us for frame in frames]
                for logical_id, frames in ctx.session.frame_history.items()
            }
        )
        return DriveResult(trajectory_in_rig=Trajectory.empty())


def test_camera_frames_and_lidar_sweeps_reach_the_policy_the_same_way() -> None:
    policy = _Frames()
    servicer = EgodriverServicer(policy)
    context = MagicMock()
    camera = AvailableCamera(logical_id="camera_front")
    start = DriveSessionRequest(session_uuid="s")
    start.rollout_spec.vehicle.available_cameras.append(camera)
    servicer.start_session(start, context)

    for step in range(3):
        now = step * 100_000
        image = RolloutCameraImage(session_uuid="s")
        image.camera_image.logical_id = "camera_front"
        image.camera_image.frame_end_us = now
        image.camera_image.image_bytes = b"jpeg"
        servicer.submit_image_observation(image, context)
        sweep = pack_lidar_sweep(
            "lidar_top", now, np.ones((4, 4)), Pose.identity().to_proto()
        )
        servicer.drive(
            DriveRequest(
                session_uuid="s",
                time_now_us=now,
                renderer_data=protocol.pack_renderer_data(
                    CarlaRendererData(lidar=[sweep])
                ),
            ),
            context,
        )

    context.abort.assert_not_called()
    # Both kinds are announced, each before the drive that reads it...
    assert policy.announced == ["CameraFrame:camera_front", "LidarFrame:lidar_top"] * 3
    # ...and kept to the same history length, oldest first.
    assert policy.seen[-1] == {
        "camera_front": [100_000, 200_000],
        "lidar_top": [100_000, 200_000],
    }
    lidar = servicer._sessions["s"].latest_frame("lidar_top")  # noqa: SLF001
    assert isinstance(lidar, LidarFrame) and lidar.as_array().shape == (4, 4)
