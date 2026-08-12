# SPDX-License-Identifier: Apache-2.0
"""Where this project agrees with alpasim, and where it does not.

The point of this module is that "how compatible is it, exactly?" has a
*runnable* answer rather than only a prose one.  :data:`COMPAT_ENTRIES` is the
machine-readable twin of ``docs/COMPATIBILITY.md``; ``carla-driver-interface
compat-report`` prints it, and ``tests/test_proto_compat.py`` checks that the
claims marked :data:`CompatLevel.COMPAT_LEVEL_EXACT` really are exact.
"""

from __future__ import annotations

from carla_driver_interface import ALPASIM_GRPC_REV, __version__
from carla_driver_interface.grpc_api import (
    API_VERSION_MESSAGE,
    CarlaCompatReport,
    CompatEntry,
    CompatLevel,
    describe_api_mismatch,
)

__all__ = [
    "COMPAT_ENTRIES",
    "build_report",
    "describe_api_mismatch",
    "format_report",
]


def _entry(area: str, alpasim: str, carla: str, level: CompatLevel, *symbols: str) -> CompatEntry:
    return CompatEntry(
        area=area,
        alpasim_behaviour=alpasim,
        carla_behaviour=carla,
        level=level,
        symbols=list(symbols),
    )


#: The full difference list. Keep in sync with docs/COMPATIBILITY.md.
COMPAT_ENTRIES: tuple[CompatEntry, ...] = (
    # -- exact ----------------------------------------------------------
    _entry(
        "Egodriver service",
        "8 RPCs on egodriver.EgodriverService",
        "All 8 implemented by inheriting the upstream generated servicer; "
        "same service path and message types",
        CompatLevel.COMPAT_LEVEL_EXACT,
        "egodriver.EgodriverService",
    ),
    _entry(
        "Message types",
        "Defined in alpasim_grpc",
        "The same package is a pinned git dependency; no message is redefined",
        CompatLevel.COMPAT_LEVEL_EXACT,
        "common",
        "egodriver",
        "nre.grpc.protos.sensorsim",
    ),
    _entry(
        "Observation ordering",
        "Images, egomotion, route and ground truth all land before drive(), "
        "enforced by an explicit barrier",
        "Same order, same barrier -- the runtime sends every observation before issuing drive()",
        CompatLevel.COMPAT_LEVEL_EXACT,
        "egodriver.DriveRequest",
    ),
    _entry(
        "Response frame",
        "DriveResponse.trajectory holds local->rig_est, led by the pose at time_now_us",
        "Identical: the servicer anchors the plan on the ego pose at time_now_us",
        CompatLevel.COMPAT_LEVEL_EXACT,
        "egodriver.DriveResponse",
    ),
    _entry(
        "Early termination",
        "DriveResponse.terminate_session ends the rollout immediately",
        "Honoured: the loop returns without stepping further",
        CompatLevel.COMPAT_LEVEL_EXACT,
        "egodriver.DriveResponse.terminate_session",
    ),
    _entry(
        "API version",
        "get_version reports alpasim_grpc.API_VERSION_MESSAGE",
        "Forwarded verbatim; the runtime compares it and warns on mismatch",
        CompatLevel.COMPAT_LEVEL_EXACT,
        "common.VersionId",
    ),
    _entry(
        "Rollout results",
        "SimulationReturn.RolloutReturn with timestep and aggregated metrics",
        "The same message is produced, so alpasim-side tooling can read it",
        CompatLevel.COMPAT_LEVEL_EXACT,
        "SimulationReturn.RolloutReturn",
    ),
    # -- partial --------------------------------------------------------
    _entry(
        "Camera model",
        "ftheta, OpenCV fisheye and OpenCV pinhole",
        "Pinhole only -- CARLA renders an ideal pinhole, so ftheta parameters "
        "cannot be filled honestly",
        CompatLevel.COMPAT_LEVEL_PARTIAL,
        "nre.grpc.protos.sensorsim.CameraSpec",
    ),
    _entry(
        "Shutter",
        "Rolling shutter; frame_start_us and frame_end_us bracket the sweep",
        "Global shutter; frame_start_us == frame_end_us",
        CompatLevel.COMPAT_LEVEL_PARTIAL,
        "nre.grpc.protos.sensorsim.ShutterType",
    ),
    _entry(
        "Recording ground truth",
        "submit_recording_ground_truth carries the real car's recorded path",
        "No recording exists. Off by default; when enabled it sends the route "
        "reference instead, which is a different quantity",
        CompatLevel.COMPAT_LEVEL_PARTIAL,
        "egodriver.GroundTruthRequest",
    ),
    _entry(
        "Egomotion error model",
        "rig_est diverges from rig via a proprioceptive noise model",
        "Identity by default; optional Gaussian position/yaw noise reproduces "
        "the divergence and the runtime-side correction",
        CompatLevel.COMPAT_LEVEL_PARTIAL,
        "egodriver.RolloutEgoTrajectory",
    ),
    _entry(
        "scene_id",
        "UUID of a recorded clip",
        "'<map>:<scenario>' -- there are no recordings to identify",
        CompatLevel.COMPAT_LEVEL_PARTIAL,
        "egodriver.DriveSessionRequest.DebugInfo",
    ),
    # -- structural -----------------------------------------------------
    _entry(
        "Renderer",
        "SensorsimService: NRE neural reconstruction over gRPC",
        "CARLA's rasterizer, in process. The sensorsim RPCs are never called; "
        "only its message types are reused to describe cameras",
        CompatLevel.COMPAT_LEVEL_STRUCTURAL,
        "nre.grpc.protos.sensorsim.SensorsimService",
    ),
    _entry(
        "Controller / vehicle model",
        "VDCService over gRPC turns the plan into motion",
        "In-process pure-pursuit plus speed PID, with CARLA's own vehicle dynamics behind it",
        CompatLevel.COMPAT_LEVEL_STRUCTURAL,
        "controller.VDCService",
    ),
    _entry(
        "Physics",
        "PhysicsService performs ground-intersection correction",
        "CARLA's physics engine keeps the vehicle on the ground; no RPC needed",
        CompatLevel.COMPAT_LEVEL_STRUCTURAL,
        "physics.PhysicsService",
    ),
    _entry(
        "Traffic",
        "TrafficService simulates other agents",
        "CARLA TrafficManager",
        CompatLevel.COMPAT_LEVEL_STRUCTURAL,
        "traffic.TrafficService",
    ),
    _entry(
        "Coordinate frames",
        "Right-handed ENU local frame; rig at the rear axle centre on the ground",
        "CARLA is left-handed with the actor origin at the vehicle centre; the "
        "conversion layer mirrors y and shifts to the rear axle",
        CompatLevel.COMPAT_LEVEL_STRUCTURAL,
        "common.Pose",
    ),
    _entry(
        "Orchestration",
        "Asyncio RuntimeService daemon, many concurrent rollouts, load balanced "
        "across driver replicas",
        "Synchronous in-process class running one rollout; concurrency is the caller's problem",
        CompatLevel.COMPAT_LEVEL_STRUCTURAL,
        "RuntimeService",
    ),
    # -- extension ------------------------------------------------------
    _entry(
        "Renderer payload",
        "DriveRequest.renderer_data is free-form and NRE-specific",
        "Carries a serialized carla_driver.v0.CarlaRendererData (map, weather, "
        "traffic light, speed limit, actors). Drivers that ignore it are unaffected",
        CompatLevel.COMPAT_LEVEL_EXTENSION,
        "egodriver.DriveRequest.renderer_data",
    ),
    _entry(
        "Driver debug payload",
        "DebugInfo.unstructured_debug_info is free-form",
        "Carries a serialized carla_driver.v0.CarlaDriveDebugInfo",
        CompatLevel.COMPAT_LEVEL_EXTENSION,
        "egodriver.DriveResponse.DebugInfo",
    ),
    # -- unimplemented --------------------------------------------------
    _entry(
        "Structured logging",
        "logging.proto records every request/response into an ASL log",
        "Not implemented; the runtime logs through the standard logging module",
        CompatLevel.COMPAT_LEVEL_UNIMPLEMENTED,
        "logging.LogEntry",
    ),
    _entry(
        "Video model",
        "video_model.proto drives a generative video model",
        "Not implemented",
        CompatLevel.COMPAT_LEVEL_UNIMPLEMENTED,
        "video_model",
    ),
    _entry(
        "LiDAR",
        "SensorsimService.render_lidar produces point clouds",
        "Not implemented -- the egodriver contract has no LiDAR submission RPC, "
        "so there is nowhere to deliver it",
        CompatLevel.COMPAT_LEVEL_UNIMPLEMENTED,
        "nre.grpc.protos.sensorsim.LidarRenderRequest",
    ),
    _entry(
        "Runtime gRPC surface",
        "RuntimeService.simulate / prefetch_scene / get_runtime_info / shut_down",
        "Not served; CarlaRuntime is used as a Python class",
        CompatLevel.COMPAT_LEVEL_UNIMPLEMENTED,
        "RuntimeService",
    ),
)


def build_report() -> CarlaCompatReport:
    """The difference list as a protobuf message."""
    return CarlaCompatReport(
        carla_driver_interface_version=__version__,
        alpasim_grpc_rev=ALPASIM_GRPC_REV,
        alpasim_grpc_api_version=API_VERSION_MESSAGE,
        entries=list(COMPAT_ENTRIES),
    )


_LEVEL_LABELS = {
    CompatLevel.COMPAT_LEVEL_EXACT: "exact",
    CompatLevel.COMPAT_LEVEL_PARTIAL: "partial",
    CompatLevel.COMPAT_LEVEL_STRUCTURAL: "structural",
    CompatLevel.COMPAT_LEVEL_EXTENSION: "extension",
    CompatLevel.COMPAT_LEVEL_UNIMPLEMENTED: "unimplemented",
    CompatLevel.COMPAT_LEVEL_UNSPECIFIED: "unspecified",
}


def format_report(report: CarlaCompatReport | None = None) -> str:
    """Render the report for a terminal, grouped by compatibility level."""
    report = report or build_report()
    api = report.alpasim_grpc_api_version
    lines = [
        f"carla_driver_interface {report.carla_driver_interface_version}",
        f"alpasim rev           {report.alpasim_grpc_rev}",
        f"alpasim_grpc API      {api.major}.{api.minor}.{api.patch}",
        "",
    ]
    order = (
        CompatLevel.COMPAT_LEVEL_EXACT,
        CompatLevel.COMPAT_LEVEL_PARTIAL,
        CompatLevel.COMPAT_LEVEL_STRUCTURAL,
        CompatLevel.COMPAT_LEVEL_EXTENSION,
        CompatLevel.COMPAT_LEVEL_UNIMPLEMENTED,
    )
    for level in order:
        entries = [e for e in report.entries if e.level == level]
        if not entries:
            continue
        lines.append(f"== {_LEVEL_LABELS[level].upper()} ({len(entries)}) ==")
        for entry in entries:
            lines.append(f"  {entry.area}")
            lines.append(f"    alpasim: {entry.alpasim_behaviour}")
            lines.append(f"    carla  : {entry.carla_behaviour}")
            if entry.symbols:
                lines.append(f"    protos : {', '.join(entry.symbols)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
