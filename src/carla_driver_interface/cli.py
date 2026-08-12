# SPDX-License-Identifier: Apache-2.0
"""Command line entry points: ``serve``, ``run``, ``demo`` and ``compat-report``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace

from carla_driver_interface import __version__
from carla_driver_interface.driver.policies import POLICY_REGISTRY
from carla_driver_interface.driver.server import run_server
from carla_driver_interface.runtime.config import (
    CameraConfig,
    RuntimeConfig,
    ScenarioSpec,
    default_camera_rig,
)
from carla_driver_interface.runtime.images import parse_image_format

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    return args.handler(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="carla-driver-interface",
        description=("An alpasim-compatible egodriver server and a CARLA closed-loop runtime."),
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="logging verbosity (default: info)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_serve_parser(subparsers)
    _add_run_parser(subparsers)
    _add_demo_parser(subparsers)
    _add_compat_parser(subparsers)
    return parser


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


def _add_serve_parser(subparsers) -> None:
    serve = subparsers.add_parser(
        "serve",
        help="run an egodriver gRPC server (compatible with an alpasim runtime)",
    )
    serve.add_argument(
        "--policy",
        default="route_follower",
        choices=sorted(POLICY_REGISTRY),
        help="reference policy to serve (default: route_follower)",
    )
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=50051)
    serve.add_argument("--max-workers", type=int, default=8)
    serve.add_argument(
        "--cruise-speed",
        type=float,
        default=None,
        help="override the policy's cruise speed, in m/s",
    )
    serve.set_defaults(handler=_cmd_serve)


def _cmd_serve(args: argparse.Namespace) -> int:
    policy_cls = POLICY_REGISTRY[args.policy]
    kwargs = {}
    if args.cruise_speed is not None:
        # Both reference policies name their speed knob differently.
        kwargs = (
            {"cruise_speed_mps": args.cruise_speed}
            if args.policy == "route_follower"
            else {"target_speed_mps": args.cruise_speed}
        )
    run_server(policy_cls(**kwargs), port=args.port, host=args.host, max_workers=args.max_workers)
    return 0


# ---------------------------------------------------------------------------
# run / demo
# ---------------------------------------------------------------------------


def _add_common_rollout_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--driver", default="localhost:50051", help="driver host:port")
    parser.add_argument("--steps", type=int, default=400, help="max policy steps")
    parser.add_argument("--policy-timestep", type=float, default=0.1, help="seconds")
    parser.add_argument("--fixed-delta", type=float, default=0.05, help="seconds per tick")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-format", default="jpeg", help="png or jpeg (default: jpeg)")
    parser.add_argument(
        "--metrics-json",
        default=None,
        help="write the rollout's aggregated metrics to this path",
    )


def _add_run_parser(subparsers) -> None:
    run = subparsers.add_parser("run", help="run a closed-loop rollout against a CARLA server")
    run.add_argument("--carla-host", default="localhost")
    run.add_argument("--carla-port", type=int, default=2000)
    run.add_argument(
        "--carla-python-path",
        default=None,
        help="directory holding a CARLA PythonAPI (needed for 0.10.x, which is not on PyPI)",
    )
    run.add_argument("--map", dest="map_name", default="Town10HD_Opt")
    run.add_argument("--scenario", default="default")
    run.add_argument("--spawn-point", type=int, default=None)
    run.add_argument("--traffic", type=int, default=30, help="background vehicles")
    run.add_argument("--weather", default=None, help="CARLA weather preset, e.g. ClearNoon")
    run.add_argument("--ego-blueprint", default="vehicle.tesla.model3")
    run.add_argument(
        "--rear-axle-offset",
        type=float,
        default=None,
        help=(
            "signed metres from the actor origin to the rear axle (negative for a car). "
            "Overrides the wheel-physics derivation, which is unreliable on CARLA 0.9.x."
        ),
    )
    run.add_argument("--camera-width", type=int, default=960)
    run.add_argument("--camera-height", type=int, default=604)
    run.add_argument("--camera-fov", type=float, default=120.0)
    _add_common_rollout_args(run)
    run.set_defaults(handler=_cmd_run)


def _cmd_run(args: argparse.Namespace) -> int:
    from carla_driver_interface.runtime.carla_runtime import CarlaRuntime
    from carla_driver_interface.runtime.world import CarlaWorldAdapter

    scenario = ScenarioSpec(
        map_name=args.map_name,
        name=args.scenario,
        spawn_point_index=args.spawn_point,
        num_background_vehicles=args.traffic,
        weather_preset=args.weather,
    )
    config = _base_config(args)
    config = replace(
        config,
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        ego_blueprint=args.ego_blueprint,
        rear_axle_offset_m=args.rear_axle_offset,
        cameras=[
            CameraConfig(
                logical_id=default_camera_rig()[0].logical_id,
                width=args.camera_width,
                height=args.camera_height,
                fov_deg=args.camera_fov,
            )
        ],
    )

    world = CarlaWorldAdapter(config, scenario, carla_python_path=args.carla_python_path)
    outcome = CarlaRuntime(world, config, scenario).run_rollout()
    return _report(outcome, args.metrics_json)


def _add_demo_parser(subparsers) -> None:
    demo = subparsers.add_parser(
        "demo",
        help="run a closed-loop rollout against the built-in fake world (no CARLA needed)",
    )
    _add_common_rollout_args(demo)
    demo.set_defaults(handler=_cmd_demo, steps=120)


def _cmd_demo(args: argparse.Namespace) -> int:
    from carla_driver_interface.fakes import FakeWorld
    from carla_driver_interface.runtime.carla_runtime import CarlaRuntime

    scenario = ScenarioSpec(map_name="FakeTown", name="straight_then_curve")
    config = _base_config(args)
    world = FakeWorld(config, scenario)
    outcome = CarlaRuntime(world, config, scenario).run_rollout()
    return _report(outcome, args.metrics_json)


def _base_config(args: argparse.Namespace) -> RuntimeConfig:
    return RuntimeConfig(
        driver_address=args.driver,
        max_steps=args.steps,
        policy_timestep_s=args.policy_timestep,
        fixed_delta_s=args.fixed_delta,
        seed=args.seed,
        image_format=parse_image_format(args.image_format),
    )


def _report(outcome, metrics_path: str | None) -> int:
    metrics = outcome.metrics
    print(f"success            : {outcome.success}")
    print(f"steps              : {outcome.steps}")
    print(f"terminated by driver: {outcome.terminated_by_driver}")
    if outcome.rollout_return.error:
        print(f"error              : {outcome.rollout_return.error}", file=sys.stderr)
    for name in sorted(metrics):
        print(f"  {name:<26} {metrics[name]:.4f}")

    if metrics_path:
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True)
        print(f"metrics written to {metrics_path}")

    return 0 if outcome.success else 1


# ---------------------------------------------------------------------------
# compat-report
# ---------------------------------------------------------------------------


def _add_compat_parser(subparsers) -> None:
    compat = subparsers.add_parser(
        "compat-report",
        help="print exactly where this project matches alpasim and where it differs",
    )
    compat.add_argument(
        "--json", action="store_true", help="emit the report as JSON instead of text"
    )
    compat.set_defaults(handler=_cmd_compat)


def _cmd_compat(args: argparse.Namespace) -> int:
    from google.protobuf.json_format import MessageToJson

    from carla_driver_interface.compat import build_report, format_report

    if args.json:
        print(MessageToJson(build_report(), preserving_proto_field_name=True))
    else:
        print(format_report(), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
