# SPDX-License-Identifier: Apache-2.0
"""Vehicle rigs: TOML files, built-in and plugin names, and the CLI."""

from __future__ import annotations

import argparse
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

from carla_driver_interface import cli
from carla_driver_interface.runtime import rig as rig_module
from carla_driver_interface.runtime.config import CameraConfig, default_camera_rig
from carla_driver_interface.runtime.rig import VehicleRig, available_rigs, load_rig, rig_from_toml

NARROW = """
description = "narrow front camera"
ego_blueprint = "vehicle.lincoln.mkz_2020"

[[cameras]]
logical_id = "camera_front"
width = 1920
height = 1280
fov_deg = 50.0
x = 1.544
y = 0.0243
z = 2.116
pitch_deg = -0.11
yaw_deg = -0.23
roll_deg = -0.10
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "rig.toml"
    path.write_text(text)
    return path


def test_toml_rig(tmp_path: Path) -> None:
    rig = rig_from_toml(write(tmp_path, NARROW))
    assert rig.ego_blueprint == "vehicle.lincoln.mkz_2020"
    assert rig.description == "narrow front camera"
    assert rig.cameras == (
        CameraConfig(
            logical_id="camera_front",
            width=1920,
            height=1280,
            fov_deg=50.0,
            x=1.544,
            y=0.0243,
            z=2.116,
            pitch_deg=-0.11,
            yaw_deg=-0.23,
            roll_deg=-0.10,
        ),
    )
    # load_rig takes the path too, as a string.
    assert load_rig(str(tmp_path / "rig.toml")) == rig


@pytest.mark.parametrize(
    "text, match",
    [
        ('[[cameras]]\nlogical_id = "a"\nfov = 50\n', "unknown keys"),  # misspelt fov_deg
        ('vehicle = "x"\n[[cameras]]\nlogical_id = "a"\n', "unknown keys"),
        ("[[cameras]]\nwidth = 10\n", "logical_id"),
        ('description = "no cameras"\n', "at least one camera"),
        ('[[cameras]]\nlogical_id = "a"\n[[cameras]]\nlogical_id = "a"\n', "duplicate"),
    ],
)
def test_toml_mistakes_are_errors(tmp_path: Path, text: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        rig_from_toml(write(tmp_path, text))


def test_default_rig_is_the_runtime_default() -> None:
    rig = load_rig("default")
    assert list(rig.cameras) == default_camera_rig()
    assert rig.ego_blueprint is None


def test_plugin_rigs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Packages register rigs (or factories) under the entry point group."""
    plugin = VehicleRig(cameras=(CameraConfig(logical_id="cam"),), ego_blueprint="vehicle.x")
    eps = [
        EntryPoint("plugin_value", "tests.fake:value", rig_module.ENTRY_POINT_GROUP),
        EntryPoint("plugin_factory", "tests.fake:factory", rig_module.ENTRY_POINT_GROUP),
        EntryPoint("not_a_rig", "tests.fake:junk", rig_module.ENTRY_POINT_GROUP),
    ]
    loaded = {"plugin_value": plugin, "plugin_factory": lambda: plugin, "not_a_rig": 42}

    def fake_entry_points(*, group: str, name: str | None = None) -> list[EntryPoint]:
        assert group == rig_module.ENTRY_POINT_GROUP
        return [ep for ep in eps if name is None or ep.name == name]

    monkeypatch.setattr(rig_module, "entry_points", fake_entry_points)
    monkeypatch.setattr(EntryPoint, "load", lambda self: loaded[self.name])

    assert available_rigs() == ["default", "not_a_rig", "plugin_factory", "plugin_value"]
    assert load_rig("plugin_value") is plugin
    assert load_rig("plugin_factory") is plugin
    with pytest.raises(TypeError, match="not a VehicleRig"):
        load_rig("not_a_rig")
    with pytest.raises(ValueError, match="unknown rig 'nope'"):
        load_rig("nope")


def _args(**overrides: object) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    cli._add_run_parser(subparsers)
    args = parser.parse_args(["run"])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_cli_rig_sets_cameras_and_vehicle(tmp_path: Path) -> None:
    config = cli._base_config(_args(rig=str(write(tmp_path, NARROW))))
    assert [c.logical_id for c in config.cameras] == ["camera_front"]
    assert config.cameras[0].fov_deg == 50.0
    assert config.ego_blueprint == "vehicle.lincoln.mkz_2020"


def test_cli_default_rig_is_unchanged() -> None:
    config = cli._base_config(_args())
    assert config.cameras == default_camera_rig()
    assert config.ego_blueprint == "vehicle.tesla.model3"
