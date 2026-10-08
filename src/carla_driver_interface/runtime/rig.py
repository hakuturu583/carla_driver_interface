# SPDX-License-Identifier: Apache-2.0
"""Vehicle rigs: which ego vehicle, and which cameras where.

In alpasim the runtime owns the sensor rig; ``EgodriverService`` has no way
for a driver to ask for one. So the rig is runtime configuration, but a policy
usually comes with the rig it was built for -- a monocular ADAS stack expects
its narrow front camera, not a 120 degree wide one. A :class:`VehicleRig`
captures that, and :func:`load_rig` resolves one from

* a TOML file (see :func:`rig_from_toml` for the format),
* a built-in name (``"default"``: alpasim's single wide front camera), or
* a name another package registers under the ``carla_driver_interface.rigs``
  entry point group, so a policy can ship its rig without this project knowing
  about it::

      [project.entry-points."carla_driver_interface.rigs"]
      my_policy = "my_policy.rig:carla_rig"   # a VehicleRig, or a callable returning one
"""

from __future__ import annotations

import tomllib
from collections.abc import Callable
from dataclasses import dataclass, fields
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from carla_driver_interface.runtime.config import CameraConfig, default_camera_rig

__all__ = ["ENTRY_POINT_GROUP", "VehicleRig", "available_rigs", "load_rig", "rig_from_toml"]

#: Entry point group other packages register named rigs under.
ENTRY_POINT_GROUP = "carla_driver_interface.rigs"


@dataclass(frozen=True)
class VehicleRig:
    """An ego vehicle and the cameras mounted on it.

    Camera mounts are relative to the vehicle actor, so they only mean what
    they say on the vehicle they were measured on: hence ``ego_blueprint``
    travels with them. ``None`` leaves the runtime's choice alone.
    """

    cameras: tuple[CameraConfig, ...]
    ego_blueprint: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.cameras:
            raise ValueError("a rig needs at least one camera")
        ids = [camera.logical_id for camera in self.cameras]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate camera logical ids: {ids}")


_CAMERA_FIELDS = {f.name for f in fields(CameraConfig)}
_RIG_KEYS = {"cameras", "ego_blueprint", "description"}


def rig_from_toml(path: str | Path) -> VehicleRig:
    """Read a rig from a TOML file.

    Camera entries take :class:`CameraConfig`'s fields, i.e. CARLA's own
    left-handed mount transform relative to the vehicle actor (metres and
    degrees, exactly what ``carla.Transform`` takes)::

        description = "Narrow front camera"
        ego_blueprint = "vehicle.lincoln.mkz_2020"   # optional

        [[cameras]]
        logical_id = "camera_front"
        width = 1920
        height = 1280
        fov_deg = 50.0
        x = 1.544
        z = 2.116
        pitch_deg = -0.11

    Unknown keys are errors: a misspelt ``fov`` must not silently fall back to
    the default lens.
    """
    path = Path(path)
    with path.open("rb") as f:
        data = tomllib.load(f)

    unknown = set(data) - _RIG_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}; expected {sorted(_RIG_KEYS)}")
    cameras = []
    for i, entry in enumerate(data.get("cameras", [])):
        unknown = set(entry) - _CAMERA_FIELDS
        if unknown:
            raise ValueError(f"{path}: cameras[{i}] has unknown keys {sorted(unknown)}")
        if "logical_id" not in entry:
            raise ValueError(f"{path}: cameras[{i}] needs a logical_id")
        cameras.append(CameraConfig(**entry))
    return VehicleRig(
        cameras=tuple(cameras),
        ego_blueprint=data.get("ego_blueprint"),
        description=data.get("description", ""),
    )


def _builtin_rigs() -> dict[str, Callable[[], VehicleRig]]:
    return {
        "default": lambda: VehicleRig(
            cameras=tuple(default_camera_rig()),
            description="alpasim's single wide (120 degree) front camera",
        ),
    }


def available_rigs() -> list[str]:
    """Names :func:`load_rig` resolves: built-ins and installed entry points."""
    names = set(_builtin_rigs())
    names.update(ep.name for ep in entry_points(group=ENTRY_POINT_GROUP))
    return sorted(names)


def load_rig(spec: str | Path) -> VehicleRig:
    """Resolve ``spec``: a ``.toml`` path, a built-in name, or a registered name."""
    path = Path(spec)
    if path.suffix == ".toml" or path.is_file():
        return rig_from_toml(path)

    name = str(spec)
    builtin = _builtin_rigs().get(name)
    if builtin is not None:
        return builtin()

    matches = entry_points(group=ENTRY_POINT_GROUP, name=name)
    if not matches:
        raise ValueError(f"unknown rig {name!r}; available: {', '.join(available_rigs())}")
    obj: Any = next(iter(matches)).load()
    rig = obj() if callable(obj) else obj
    if not isinstance(rig, VehicleRig):
        raise TypeError(f"rig {name!r} resolved to {type(rig).__name__}, not a VehicleRig")
    return rig
