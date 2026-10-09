"""Read a map set written by a runtime's map export.

The CARLA runtime in autoware_carla_scenario
(https://github.com/autowarefoundation/autoware_carla_scenario) writes these sets with
``autoware_carla_scenario.driver.hdmap_export`` into its ``driver.map_dir``.

The driver half: no roadgen, no CARLA. A :class:`MapFiles` opens one set in
the format a driver reads and turns the traffic lights each step carries --
named by OpenDRIVE -- into :class:`StopLine` records named by that format's
own elements, so a driver never has to know OpenDRIVE was involved.

The translation runs through two roadgen traces the export writes beside the
map: the read trace, from each OpenDRIVE lane (``lane:<road>/<section>/<lane>``)
and signal (``signal:<id>``) to the IR element it became, and the format's
trace, from each IR element to the elements of the format it became. The IR
itself says which rules name a light.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .protocol import TrafficLight, TrafficLightState

__all__ = ["MapFiles", "StopLine"]

_MANIFEST_FILE = "manifest.json"

#: The trace role of the element a format makes of an IR lane, where the
#: format gives a lane several (Lanelet2 also traces its boundaries and centre
#: line to it). A format not listed keeps every element traced to the lane.
_LANE_ROLES = {"lanelet2": "lanelet"}


@dataclass(frozen=True)
class StopLine:
    """Where to stop for one traffic light on one lane, and the light's state.

    The ``*_ids`` name elements of the driver's map format, as that format
    numbers them -- for Lanelet2, lanelet, regulatory element and traffic-light
    line string ids.
    """

    #: The light's OpenDRIVE signal id.
    traffic_light_id: str
    state: TrafficLightState
    #: The stop point on the lane's centre line, ``(3,)``, ``local`` frame.
    position_local: np.ndarray
    #: The lane the point lies on, as OpenDRIVE names it.
    road_id: int
    section_id: int
    lane_id: int
    #: The lane the stop point lies on -- the approach to the line -- in the
    #: map format; empty when the format has no such lane.
    lane_ids: tuple[str, ...]
    #: The format's rule elements for this light (Lanelet2: regulatory
    #: elements). They hang off the lanes the light governs, which for a
    #: CARLA town are the junction lanes past the line, not ``lane_ids``.
    rule_ids: tuple[str, ...]
    #: The format's elements for the light itself (Lanelet2: its line strings).
    light_ids: tuple[str, ...]


class MapFiles:
    """One map set, opened in one format.

    Args:
        directory: The set's directory, ``<map_dir>/<map_id>``.
        format: A format the set was written in, e.g. ``"lanelet2"``.
    """

    def __init__(self, directory: str | Path, format: str = "lanelet2") -> None:
        self.directory = Path(directory)
        manifest_path = self.directory / _MANIFEST_FILE
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(
                f"no map set at {self.directory} (no {_MANIFEST_FILE}); the runtime writes "
                "one per map under driver.map_dir -- is the driver's map_dir a copy "
                "of the runtime's?"
            ) from None
        written = manifest.get("formats", {})
        if format not in written:
            raise ValueError(
                f"map {manifest.get('map_id')!r} was not written as {format!r}, only as "
                f"{sorted(written)}; add it to driver.map_formats"
            )
        self.map_id: str = manifest["map_id"]
        self.map_name: str = manifest.get("map_name", "")
        self.format = format
        #: The map file (or, for ClipGT, directory) in ``format``.
        self.path = self.directory / written[format]["path"]
        #: The OpenDRIVE the set was converted from.
        self.opendrive_path = self.directory / manifest["source"]

        ir = json.loads((self.directory / manifest["ir"]).read_text(encoding="utf-8"))
        trace = json.loads(
            (self.directory / written[format]["trace"]).read_text(encoding="utf-8")
        )
        read = json.loads(
            (self.directory / manifest["read_trace"]).read_text(encoding="utf-8")
        )
        self._from_opendrive = _read_trace(read)
        self._rules_by_object = _ir_rules_by_object(ir)
        self._refs = _trace_refs(trace)
        self._lane_role = _LANE_ROLES.get(format)

    @classmethod
    def open(
        cls, map_dir: str | Path, map_id: str, format: str = "lanelet2"
    ) -> MapFiles:
        """The set ``map_id`` under ``map_dir``."""
        return cls(Path(map_dir) / map_id, format)

    def stop_lines(self, lights: Iterable[TrafficLight]) -> list[StopLine]:
        """Each light's stop points as stop lines in this format, in arrival order."""
        stop_lines = []
        for light in lights:
            objects = self._from_opendrive.get(f"signal:{light.opendrive_id}", ())
            rules = dict.fromkeys(
                r for obj in objects for r in self._rules_by_object.get(obj, ())
            )
            rule_ids = tuple(ref for rule in rules for ref in self._ids(rule))
            light_ids = tuple(ref for obj in objects for ref in self._ids(obj))
            for point in light.stop_points:
                lanes = self._from_opendrive.get(
                    f"lane:{point.road_id}/{point.section_id}/{point.lane_id}", ()
                )
                p = point.position_local
                stop_lines.append(
                    StopLine(
                        traffic_light_id=light.opendrive_id,
                        state=light.state,
                        position_local=np.array([p.x, p.y, p.z], dtype=np.float64),
                        road_id=point.road_id,
                        section_id=point.section_id,
                        lane_id=point.lane_id,
                        lane_ids=tuple(
                            ref
                            for lane in lanes
                            for ref in self._ids(lane, self._lane_role)
                        ),
                        rule_ids=rule_ids,
                        light_ids=light_ids,
                    )
                )
        return stop_lines

    def _ids(self, ir_id: str, role: str | None = None) -> tuple[str, ...]:
        """The format's ids for one IR element, optionally of one role only."""
        return tuple(
            ref
            for ref, ref_role in self._refs.get(ir_id, ())
            if role is None or ref_role == role
        )


def _read_trace(trace: dict) -> dict[str, list[str]]:
    """OpenDRIVE element (``signal:943``, ``lane:21/0/-1``) -> the IR elements it became."""
    became: dict[str, list[str]] = defaultdict(list)
    for link in trace.get("links", ()):
        if link["ir"] not in became[link["ref"]]:
            became[link["ref"]].append(link["ir"])
    return became


def _ir_rules_by_object(ir: dict) -> dict[str, list[str]]:
    """IR object id -> the IR rules that name it, each once."""
    rules: dict[str, list[str]] = defaultdict(list)
    for rule in ir.get("rules", ()):
        for obj in dict.fromkeys(rule.get("objects", ())):
            rules[obj].append(rule["id"])
    return rules


def _trace_refs(trace: dict) -> dict[str, list[tuple[str, str | None]]]:
    """IR id -> ``(format id, role)`` for each element it became, ``kind:`` stripped."""
    refs: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
    for link in trace.get("links", ()):
        ref = str(link["ref"]).split(":", 1)[-1]
        refs[link["ir"]].append((ref, link.get("role")))
    return refs
