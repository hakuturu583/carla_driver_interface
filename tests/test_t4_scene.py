"""T4SceneDriver: a devkit T4 driver behind the egodriver contract.

Needs tier4/e2e-devkit, which this package does not depend on; skipped without it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pytest

pytest.importorskip("t4_e2e_devkit.serialization.protobuf")

from t4_e2e_devkit.common.ego import EgoSequence  # noqa: E402
from t4_e2e_devkit.common.scene import SceneMetadata, T4Scene  # noqa: E402
from t4_e2e_devkit.common.trajectory import Trajectory as T4Trajectory  # noqa: E402
from t4_e2e_devkit.driver import DriverOutput  # noqa: E402
from t4_e2e_devkit.serialization.protobuf import to_proto  # noqa: E402
from t4_e2e_devkit.vendor.nuplan.trajectory.trajectory_sampling import (  # noqa: E402
    TrajectorySampling,
)

from carla_driver_interface.driver import (  # noqa: E402
    BaseDriver,
    DriveContext,
    SessionState,
    TurnIndicator,
)
from carla_driver_interface.protocol import CarlaRendererData  # noqa: E402
from carla_driver_interface.t4_scene import T4_MAP_FORMAT, T4SceneDriver  # noqa: E402


class _Model:
    """What T4SceneDriver asks of a devkit T4 driver, recorded."""

    def __init__(self, extras: Optional[dict] = None) -> None:
        self.calls: list = []
        self.extras = extras or {}

    def name(self) -> str:
        return "model"

    def start_session(self, session_id: str) -> None:
        self.calls.append(("start", session_id))

    def close_session(self, session_id: str) -> None:
        self.calls.append(("close", session_id))

    def drive(self, scene: T4Scene, session_id: Optional[str] = None) -> DriverOutput:
        self.calls.append(("drive", session_id, scene))
        poses = np.array([[1.0, 0.0, 0.0], [2.0, 0.5, 0.25]], dtype=np.float32)
        sampling = TrajectorySampling(num_poses=2, interval_length=0.5)
        return DriverOutput(
            trajectory=T4Trajectory(poses=poses, trajectory_sampling=sampling),
            extras=self.extras,
        )


def _scene_bytes(rows: int = 3) -> bytes:
    poses = np.zeros((rows, 3))
    poses[:, 0] = np.arange(rows)
    scene = T4Scene(
        scene_metadata=SceneMetadata(scene_id="rollout"),
        timestamps_ns=np.arange(rows, dtype=np.int64) * 100_000_000,
        current_index=rows - 1,
        ego=EgoSequence(poses=poses),
    )
    return to_proto(scene).SerializeToString()


def _ctx(payload: bytes, *, session: str = "s1", map_files: Any = None) -> DriveContext:
    return DriveContext(
        session=SessionState(uuid=session, seed=0, scene_id="scene", cameras={}),
        time_now_us=1_000_000,
        time_query_us=1_100_000,
        renderer_data=CarlaRendererData(t4_scene=payload),
        map=map_files,
    )


def test_it_is_a_base_driver_named_by_its_model() -> None:
    driver = T4SceneDriver(_Model())
    assert isinstance(driver, BaseDriver)
    assert driver.name == "model" and driver.map_format == T4_MAP_FORMAT
    assert T4SceneDriver(_Model(), name="other").name == "other"


def test_the_scene_reaches_the_model_under_the_session() -> None:
    model = _Model()
    driver = T4SceneDriver(model)
    session = SessionState(uuid="s1", seed=0, scene_id="scene", cameras={})
    driver.on_session_start(session)
    driver.drive(_ctx(_scene_bytes()))
    driver.on_session_close(session)

    assert [call[:2] for call in model.calls] == [
        ("start", "s1"),
        ("drive", "s1"),
        ("close", "s1"),
    ]
    scene = model.calls[1][2]
    assert scene.current_index == 2
    np.testing.assert_array_equal(scene.ego.poses[:, 0], [0.0, 1.0, 2.0])


def test_the_plan_is_timed_from_now_in_the_rig_frame() -> None:
    result = T4SceneDriver(_Model()).drive(_ctx(_scene_bytes()))
    plan = result.trajectory_in_rig

    assert list(plan.timestamps_us) == [1_500_000, 2_000_000]
    np.testing.assert_allclose(plan.positions[:, :2], [[1.0, 0.0], [2.0, 0.5]])
    np.testing.assert_allclose(plan.poses[1].yaw, 0.25, atol=1e-6)
    assert result.turn_indicator == TurnIndicator.NO_COMMAND
    assert result.debug_scalars == {}


def test_the_models_extras_carry_the_indicator_and_scalars() -> None:
    model = _Model({"turn_indicator": 2, "debug_scalars": {"lanes": 3}})
    result = T4SceneDriver(model).drive(_ctx(_scene_bytes()))

    assert result.turn_indicator == TurnIndicator.ENABLE_LEFT
    assert result.debug_scalars == {"lanes": 3.0}


@pytest.mark.parametrize("renderer_data", [None, CarlaRendererData()])
def test_a_drive_without_a_scene_is_refused(renderer_data: Any) -> None:
    ctx = _ctx(b"")
    ctx = DriveContext(
        session=ctx.session,
        time_now_us=ctx.time_now_us,
        time_query_us=ctx.time_query_us,
        renderer_data=renderer_data,
    )
    with pytest.raises(RuntimeError, match="driver.t4_scene"):
        T4SceneDriver(_Model()).drive(ctx)


def test_the_map_resource_is_read_once_per_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from t4_e2e_devkit.serialization import protobuf
    from t4_e2e_devkit.serialization.protobuf import MapResourceStore, t4_scene_pb2

    read: list = []

    class _Store(MapResourceStore):
        def register(self, resource: Any) -> str:
            read.append(resource.sha256)
            return resource.sha256

    monkeypatch.setattr(protobuf, "MapResourceStore", _Store)
    path = tmp_path / "t4_map.pb"
    path.write_bytes(
        t4_scene_pb2.MapResource(sha256="abc", data=b"<osm/>").SerializeToString()
    )
    map_files = SimpleNamespace(map_id="town-1", path=path)

    driver = T4SceneDriver(_Model())
    first = driver._map_resources(map_files)  # noqa: SLF001
    again = driver._map_resources(map_files)  # noqa: SLF001

    assert read == ["abc"] and first is again and isinstance(first, _Store)
    assert driver._map_resources(None) is None  # noqa: SLF001
