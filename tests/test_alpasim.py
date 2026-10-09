"""alpasim mode, and packaging a policy as an alpasim driver."""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml

from carla_driver_interface import cli, protocol
from carla_driver_interface.alpasim import (
    POLICY_ARGS_ENV,
    POLICY_ENV,
    CameraOverride,
    ImageSpec,
    PinholeCamera,
    WizardDriverConfig,
    parse_camera,
    parse_pinhole,
    parse_policy_args,
    write_build_context,
    write_wizard_config,
)
from carla_driver_interface.driver import BaseDriver, DriveContext, DriveResult
from carla_driver_interface.geometry import Pose, Trajectory
from carla_driver_interface.policies import ConstantSpeedPolicy
from carla_driver_interface.protocol import (
    CarlaRendererData,
    DriveRequest,
    DriveSessionRequest,
    RolloutEgoTrajectory,
)
from carla_driver_interface.service import EgodriverServicer


class _Records(BaseDriver):
    name = "records"

    def __init__(self) -> None:
        self.renderer_data: List[Optional[CarlaRendererData]] = []

    def drive(self, ctx: DriveContext) -> DriveResult:
        self.renderer_data.append(ctx.renderer_data)
        plan = Trajectory.empty()
        plan.append(ctx.time_now_us + 100_000, Pose.from_xyz_yaw(1.0, 0.0, 0.0, 0.0))
        return DriveResult(trajectory_in_rig=plan, debug_scalars={"speed": 3.0})


def _drive_once(mode: str, renderer_data: bytes):
    policy = _Records()
    servicer = EgodriverServicer(policy, mode)  # type: ignore[arg-type]
    context = MagicMock()
    servicer.start_session(DriveSessionRequest(session_uuid="s"), context)
    ego = RolloutEgoTrajectory(session_uuid="s")
    ego.trajectory.CopyFrom(Trajectory([0], [Pose.identity()]).to_proto())
    servicer.submit_egomotion_observation(ego, context)
    response = servicer.drive(
        DriveRequest(session_uuid="s", time_now_us=0, renderer_data=renderer_data),
        context,
    )
    context.abort.assert_not_called()
    return policy, response


_CARLA_PAYLOAD = protocol.pack_renderer_data(CarlaRendererData(map_id="Town10HD_Opt"))


def test_carla_mode_reads_ground_truth_and_answers_a_carla_debug_message() -> None:
    policy, response = _drive_once("carla", _CARLA_PAYLOAD)
    assert policy.renderer_data[0] is not None
    assert policy.renderer_data[0].map_id == "Town10HD_Opt"
    debug = protocol.unpack_debug_info(response.debug_info.unstructured_debug_info)
    assert debug is not None and debug.policy_name == "records"


def test_alpasim_mode_never_reads_the_renderer_payload_as_carla_ground_truth() -> None:
    # Even bytes that do parse as CarlaRendererData are alpasim's renderer's own.
    policy, _ = _drive_once("alpasim", _CARLA_PAYLOAD)
    assert policy.renderer_data == [None]


def test_alpasim_mode_answers_the_pickled_dict_alpasim_evaluation_reads() -> None:
    _, response = _drive_once("alpasim", b"")
    extra = pickle.loads(response.debug_info.unstructured_debug_info)
    assert extra["policy_name"] == "records"
    assert extra["scalars"] == {"speed": 3.0}
    assert isinstance(extra["inference_seconds"], float)
    # The plan itself is the same in either mode.
    assert len(response.trajectory.poses) == 2


def test_an_unknown_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        EgodriverServicer(_Records(), "nurec")  # type: ignore[arg-type]


# -- arguments -------------------------------------------------------------------


def test_policy_arguments_are_json_when_they_parse() -> None:
    assert parse_policy_args(["speed=8.5", "name=fast", "flag=true", "s='8'"]) == {
        "speed": 8.5,
        "name": "fast",
        "flag": True,
        "s": "'8'",
    }
    with pytest.raises(ValueError):
        parse_policy_args(["no-equals"])


def test_cameras_parse_with_their_defaults() -> None:
    assert parse_camera("cam") == CameraOverride("cam")
    camera = parse_camera("camera_front_wide_120fov:1920x1080@20")
    assert (camera.width, camera.height, camera.frame_interval_us) == (
        1920,
        1080,
        50_000,
    )
    with pytest.raises(ValueError):
        parse_camera("cam:1920")


# -- the image -------------------------------------------------------------------


def test_the_build_context_copies_local_packages_and_names_the_rest(
    tmp_path: Path,
) -> None:
    project = tmp_path / "my_policy"
    (project / "src").mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname = 'my-policy'\n")
    (project / ".venv").mkdir()
    (project / ".venv" / "huge").write_text("x")
    wheel = tmp_path / "dep-1.0-py3-none-any.whl"
    wheel.write_bytes(b"PK")
    spec = ImageSpec(
        policy="my_policy.driver:MyPolicy",
        policy_args={"speed": 5},
        install=[str(project), str(wheel), "numpy<3"],
        apt_packages=["libgl1"],
    )

    dockerfile = write_build_context(spec, tmp_path / "ctx").read_text()

    packages = tmp_path / "ctx" / "packages"
    assert (packages / "01-my_policy" / "pyproject.toml").exists()
    assert not (packages / "01-my_policy" / ".venv").exists()
    assert (packages / "02-dep-1.0-py3-none-any.whl" / wheel.name).exists()
    assert "RUN python -m pip install carla-driver-interface==" in dockerfile.replace(
        "'", ""
    )
    assert f"./01-my_policy ./02-dep-1.0-py3-none-any.whl/{wheel.name}" in dockerfile
    assert "'numpy<3'" in dockerfile
    assert "apt-get install -y --no-install-recommends libgl1" in dockerfile
    assert f'ENV {POLICY_ENV}="my_policy.driver:MyPolicy"' in dockerfile
    # The policy's module is imported at build time.
    assert 'import_module("my_policy.driver")' in dockerfile
    cmd = json.loads(dockerfile.split("\nCMD ", 1)[1].splitlines()[0])
    assert cmd[:4] == ["carla-driver-interface", "serve", "--mode", "alpasim"]


def test_the_baked_policy_arguments_survive_docker_env_quoting(tmp_path: Path) -> None:
    spec = ImageSpec(policy_args={"name": 'a "b"', "speed": 5})
    dockerfile = write_build_context(spec, tmp_path).read_text()
    line = next(
        x for x in dockerfile.splitlines() if x.startswith(f"ENV {POLICY_ARGS_ENV}=")
    )
    # Docker reads a double-quoted ENV value with JSON's escapes.
    value = json.loads(line.split("=", 1)[1])
    assert json.loads(value) == spec.policy_args


# -- the wizard config -----------------------------------------------------------


def test_the_wizard_config_runs_the_image_as_alpasims_driver(tmp_path: Path) -> None:
    config = WizardDriverConfig(
        name="my_driver",
        image="my-policy:0.1",
        policy="constant_speed",
        policy_args={"cruise_speed_mps": 6.0},
        volumes=["/weights:/mnt/weights"],
        environment={"CUDA_VISIBLE_DEVICES": "0"},
        cameras=[CameraOverride("camera_front_wide_120fov")],
    )
    write_wizard_config(config, tmp_path)

    driver = yaml.safe_load((tmp_path / "driver" / "my_driver.yaml").read_text())
    assert driver["defaults"] == ["my_driver_service", "_self_"]
    assert driver["policy"] == "constant_speed"

    text = (tmp_path / "driver" / "my_driver_service.yaml").read_text()
    # Hydra only reads the header on the first line.
    assert text.startswith("# @package _global_\n")
    service = yaml.safe_load(text)
    driver_service = service["services"]["driver"]
    assert driver_service["image"] == "my-policy:0.1"
    assert driver_service["external_image"] is True
    assert driver_service["volumes"] == [
        "${wizard.log_dir}:/mnt/output",
        "/weights:/mnt/weights",
    ]
    assert driver_service["environments"] == ["CUDA_VISIBLE_DEVICES=0"]
    command = " ".join(driver_service["command"])
    assert "serve --mode alpasim" in command and "--port {port}" in command
    assert "--policy constant_speed" in command
    assert "--policy-arg cruise_speed_mps=6.0" in command
    assert service["runtime"]["simulation_config"]["cameras"] == [
        vars(CameraOverride("camera_front_wide_120fov"))
    ]


def test_the_wizard_config_keeps_alpasims_cameras_unless_told(tmp_path: Path) -> None:
    write_wizard_config(WizardDriverConfig(name="d", image="i"), tmp_path)
    service = yaml.safe_load((tmp_path / "driver" / "d_service.yaml").read_text())
    assert "runtime" not in service
    assert "--policy " not in " ".join(service["services"]["driver"]["command"])


def test_a_driver_name_must_be_a_file_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_wizard_config(WizardDriverConfig(name="../x", image="i"), tmp_path)


# -- serve -----------------------------------------------------------------------


def test_serve_takes_the_images_policy_and_its_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    served: Dict[str, Any] = {}

    def fake_run_server(policy: BaseDriver, **kwargs: object) -> None:
        served.update(policy=policy, **kwargs)

    monkeypatch.setattr(cli, "run_server", fake_run_server)
    monkeypatch.setenv(POLICY_ENV, "constant_speed")
    monkeypatch.setenv(POLICY_ARGS_ENV, json.dumps({"cruise_speed_mps": 3.0}))

    assert cli.main(["serve", "--mode", "alpasim", "--port", "1234"]) == 0
    assert isinstance(served["policy"], ConstantSpeedPolicy)
    assert served["policy"].cruise_speed_mps == 3.0
    assert served["mode"] == "alpasim" and served["port"] == 1234

    # A command-line argument overrides the baked one...
    cli.main(["serve", "--policy-arg", "cruise_speed_mps=4"])
    assert served["policy"].cruise_speed_mps == 4.0
    # ...and another policy does not inherit them.
    cli.main(["serve", "--policy", "route_follower"])
    assert served["policy"].name == "route_follower"


# -- pinhole cameras -------------------------------------------------------------


def test_a_forward_pinhole_looks_down_the_rigs_x_axis() -> None:
    from scipy.spatial.transform import Rotation

    camera = PinholeCamera("cam", 50.0)
    optical = Rotation.from_quat(camera.rotation_xyzw()).as_matrix()
    # Optical z (forward) is the rig's x, optical x (right) its -y, y (down) its -z.
    np.testing.assert_allclose(optical[:, 2], [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(optical[:, 0], [0, -1, 0], atol=1e-12)
    np.testing.assert_allclose(optical[:, 1], [0, 0, -1], atol=1e-12)
    # Positive pitch looks down, positive yaw left.
    down = Rotation.from_quat(
        PinholeCamera("c", 50, rpy_deg=(0, 10, 0)).rotation_xyzw()
    )
    assert down.as_matrix()[2, 2] < 0
    left = Rotation.from_quat(
        PinholeCamera("c", 50, rpy_deg=(0, 0, 10)).rotation_xyzw()
    )
    assert left.as_matrix()[1, 2] > 0


def test_a_pinhole_is_rendered_at_its_cameras_resolution(tmp_path: Path) -> None:
    pinhole = parse_pinhole("camera_front_wide_120fov:50@2.97,-0.02,2.12,0,0.11,0.23")
    assert pinhole.position_m == (2.97, -0.02, 2.12)
    assert pinhole.rpy_deg == (0.0, 0.11, 0.23)
    config = WizardDriverConfig(
        name="d",
        image="i",
        cameras=[parse_camera("camera_front_wide_120fov:1920x1280@10")],
        pinholes=[pinhole],
    )
    write_wizard_config(config, tmp_path)
    runtime = yaml.safe_load((tmp_path / "driver" / "d_service.yaml").read_text())[
        "runtime"
    ]
    (extra,) = runtime["extra_cameras"]
    assert extra["resolution_hw"] == [1280, 1920]
    assert extra["intrinsics"]["model"] == "opencv_pinhole"
    fx, fy = extra["intrinsics"]["opencv_pinhole"]["focal_length"]
    assert fx == fy == pytest.approx(960 / math.tan(math.radians(25)))
    assert extra["intrinsics"]["opencv_pinhole"]["principal_point"] == [960, 640]
    assert extra["rig_to_camera"]["translation_m"] == [2.97, -0.02, 2.12]
    assert extra["shutter_type"] == "GLOBAL"
    # The recorded vehicle's hood does not belong in a redefined camera.
    assert runtime["simulation_config"]["ego_mask_rig_config_id"] is None


def test_a_pinhole_needs_a_resolution_and_a_sane_spec(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="resolution"):
        WizardDriverConfig(
            name="d", image="i", pinholes=[PinholeCamera("c", 50)]
        ).files()
    for bad in ("c", "c:0", "c:200", "c:50@1,2", "c:50@1,2,3,4"):
        with pytest.raises(ValueError):
            parse_pinhole(bad)
