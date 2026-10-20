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

A Lanelet2 map given to the runtime rather than converted
(``driver.lanelet2_path``) comes with an empty trace: nothing relates it to the
OpenDRIVE. Its lights are found on the map instead (:class:`_GivenLanelet2`):
each stop point lies on a lanelet, and the light is the traffic-light
regulatory element of the first lanelet on or ahead of it that carries one.
Its ``local_x`` / ``local_y`` must be the ``local`` frame.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
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

#: How far ahead of a stop point a given Lanelet2 map is searched for its light.
#: CARLA puts a light's stop points at its trigger volume, a few metres short of
#: the painted line; the light's element may hang off the junction lanes past it.
_LIGHT_SEARCH_M = 50.0
#: Lanelets join where one's end and the next's start are this close (m).
_JOIN_M = 0.3


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
        #: A Lanelet2 map given as it is: its lights are found on the map.
        self._given = (
            _GivenLanelet2(self.path)
            if format == "lanelet2" and written[format].get("provided")
            else None
        )

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
            if self._given is not None:
                stop_lines += self._given.stop_lines(light)
                continue
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


class _GivenLanelet2:
    """A Lanelet2 map's lanelets and traffic lights, read for resolving lights.

    Only what that takes: each lanelet's outline, which lanelets follow it and
    its traffic-light regulatory elements (with the line strings they refer to).
    """

    def __init__(self, path: Path) -> None:
        root = ET.parse(path).getroot()
        xy: dict[str, tuple[float, float]] = {}
        for node in root.iter("node"):
            tags = _tags(node)
            if "local_x" in tags and "local_y" in tags:
                xy[node.get("id", "")] = (
                    float(tags["local_x"]),
                    float(tags["local_y"]),
                )
        ways = {
            way.get("id", ""): [nd.get("ref", "") for nd in way.iter("nd")]
            for way in root.iter("way")
        }
        relations = list(root.iter("relation"))
        #: Traffic-light element id -> the line strings it refers to (the lights).
        self.lights: dict[str, tuple[str, ...]] = {
            rel.get("id", ""): tuple(
                m.get("ref", "")
                for m in rel.iter("member")
                if m.get("role") == "refers"
            )
            for rel in relations
            if _tags(rel).get("subtype") == "traffic_light"
        }
        #: Lanelet id -> (outline ``(N, 2)``, centre-line length, its light elements).
        self.lanelets: dict[str, tuple[np.ndarray, float, tuple[str, ...]]] = {}
        starts: dict[tuple[int, int], list[str]] = defaultdict(list)
        ends: dict[str, np.ndarray] = {}
        for rel in relations:
            if _tags(rel).get("type") != "lanelet":
                continue
            members = {m.get("role"): m.get("ref", "") for m in rel.iter("member")}
            try:
                left = np.array([xy[n] for n in ways[members["left"]]])
                right = np.array([xy[n] for n in ways[members["right"]]])
            except KeyError:
                continue  # a lanelet without both bounds, or off the local frame
            # The left bound gives the direction; the right runs with it.
            if np.linalg.norm(left[0] - right[-1]) + np.linalg.norm(
                left[-1] - right[0]
            ) < np.linalg.norm(left[0] - right[0]) + np.linalg.norm(
                left[-1] - right[-1]
            ):
                right = right[::-1]
            lanelet = rel.get("id", "")
            rules = tuple(
                m.get("ref", "")
                for m in rel.iter("member")
                if m.get("role") == "regulatory_element" and m.get("ref") in self.lights
            )
            centre = (left[[0, -1]] + right[[0, -1]]) / 2
            length = 0.5 * (_length(left) + _length(right))
            self.lanelets[lanelet] = (np.vstack([left, right[::-1]]), length, rules)
            starts[_cell(centre[0])].append(lanelet)
            ends[lanelet] = centre[1]
        #: Each lanelet's outline's bounding box, ``(N, 4)``: min x, min y, max x, max y.
        self._ids = list(self.lanelets)
        self._boxes = np.array(
            [
                [*outline.min(axis=0), *outline.max(axis=0)]
                for outline, _, _ in self.lanelets.values()
            ]
        ).reshape(-1, 4)
        #: A light's stop point -> its lanelets and light elements; the points do
        #: not move, and every step asks again.
        self._resolved: dict[
            tuple[float, float], tuple[tuple[str, ...], tuple[str, ...]]
        ] = {}
        #: Lanelet id -> the lanelets that start where it ends.
        self.successors: dict[str, list[str]] = {}
        for lanelet, end in ends.items():
            cx, cy = _cell(end)
            near = [
                other
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                for other in starts.get((cx + dx, cy + dy), ())
            ]
            self.successors[lanelet] = [
                other
                for other in near
                if other != lanelet
                and np.linalg.norm(self._start(other) - end) < _JOIN_M
            ]

    def _start(self, lanelet: str) -> np.ndarray:
        outline = self.lanelets[lanelet][0]
        return (outline[0] + outline[-1]) / 2

    def lanelets_at(self, point: tuple[float, float]) -> list[str]:
        """The lanelets whose outline holds ``point``."""
        x, y = point
        b = self._boxes
        near = np.flatnonzero(
            (b[:, 0] <= x) & (x <= b[:, 2]) & (b[:, 1] <= y) & (y <= b[:, 3])
        )
        return [
            self._ids[i] for i in near if _inside(point, self.lanelets[self._ids[i]][0])
        ]

    def rules_ahead(self, lanelets: Iterable[str]) -> list[str]:
        """The light elements of the first lanelets on or ahead that carry one.

        Each branch stops at its first lanelet with a light, or once
        ``_LIGHT_SEARCH_M`` of lanelets past ``lanelets`` lie behind it: a stop
        point sits near the end of its own lanelet, however long that is.
        """
        found: dict[str, None] = {}
        start = set(lanelets)
        # (lanelet, metres of lanelets between the start's end and it)
        queue = [(lanelet, 0.0) for lanelet in start]
        seen = set()
        while queue:
            lanelet, travelled = queue.pop(0)
            if lanelet in seen:
                continue
            seen.add(lanelet)
            _, length, rules = self.lanelets[lanelet]
            if rules:
                found.update(dict.fromkeys(rules))
                continue
            ahead = travelled if lanelet in start else travelled + length
            if ahead <= _LIGHT_SEARCH_M:
                queue += [(nxt, ahead) for nxt in self.successors[lanelet]]
        return list(found)

    def stop_lines(self, light: TrafficLight) -> list[StopLine]:
        """``light``'s stop points, named by this map's lanelets and elements."""
        points = []
        rules: dict[str, None] = {}
        for point in light.stop_points:
            p = (point.position_local.x, point.position_local.y)
            if p not in self._resolved:
                lanes = tuple(self.lanelets_at(p))
                self._resolved[p] = (lanes, tuple(self.rules_ahead(lanes)))
            lanes, ahead = self._resolved[p]
            rules.update(dict.fromkeys(ahead))
            points.append((point, lanes))
        rule_ids = tuple(rules)
        light_ids = tuple(
            dict.fromkeys(ref for rule in rule_ids for ref in self.lights[rule])
        )
        return [
            StopLine(
                traffic_light_id=light.opendrive_id,
                state=light.state,
                position_local=np.array(
                    [
                        point.position_local.x,
                        point.position_local.y,
                        point.position_local.z,
                    ],
                    dtype=np.float64,
                ),
                road_id=point.road_id,
                section_id=point.section_id,
                lane_id=point.lane_id,
                lane_ids=lanes,
                rule_ids=rule_ids,
                light_ids=light_ids,
            )
            for point, lanes in points
        ]


def _tags(element: ET.Element) -> dict[str, str]:
    return {t.get("k", ""): t.get("v", "") for t in element.iter("tag")}


def _length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _cell(point: np.ndarray) -> tuple[int, int]:
    return (math.floor(point[0]), math.floor(point[1]))


def _inside(point: tuple[float, float], outline: np.ndarray) -> bool:
    """Even-odd test of ``point`` against the closed polygon ``outline``."""
    x, y = point
    xs, ys = outline[:, 0], outline[:, 1]
    xn, yn = np.roll(xs, -1), np.roll(ys, -1)
    crosses = (ys > y) != (yn > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        at = xs + (y - ys) * (xn - xs) / (yn - ys)
    return bool(np.count_nonzero(crosses & (x < at)) % 2)
