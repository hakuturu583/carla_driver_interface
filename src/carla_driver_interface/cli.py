"""``carla-driver-interface``: serve a policy, drive one without CARLA, or package one
as an alpasim driver."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .alpasim import (
    DEFAULT_BASE_IMAGE,
    POLICY_ARGS_ENV,
    POLICY_ENV,
    ImageSpec,
    WizardDriverConfig,
    build_image,
    default_context_dir,
    parse_camera,
    parse_pinhole,
    parse_policy_args,
    write_build_context,
    write_wizard_config,
)
from .policies import POLICY_REGISTRY, load_policy
from .server import run_server
from .service import SERVICE_MODES
from .testing import FakeCamera, FakeLoop

__all__ = ["main"]

_POLICY_HELP = f"one of {sorted(POLICY_REGISTRY)}, or package.module:Class"


def _add_policy_arguments(
    parser: argparse.ArgumentParser, default: Optional[str]
) -> None:
    parser.add_argument("--policy", default=default, help=_POLICY_HELP)
    parser.add_argument(
        "--policy-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="keyword argument for the policy (repeatable); VALUE is JSON when it parses",
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="carla-driver-interface", description=__doc__)
    parser.add_argument("--log-level", default="INFO")
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="serve a policy over gRPC")
    _add_policy_arguments(serve, os.environ.get(POLICY_ENV, "route_follower"))
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=50051)
    serve.add_argument("--cruise-speed", type=float, default=None, help="m/s")
    serve.add_argument(
        "--mode",
        choices=SERVICE_MODES,
        default="carla",
        help="the runtime on the other end: carla (autoware_carla_scenario; the "
        "default) or alpasim (an upstream alpasim runtime)",
    )
    serve.add_argument(
        "--max-workers", type=int, default=8, help="gRPC handler threads"
    )
    serve.add_argument(
        "--log-file", type=Path, default=None, help="also log to this file"
    )

    demo = commands.add_parser(
        "demo", help="drive a policy server through the CARLA-free fake loop"
    )
    demo.add_argument("--driver", default="localhost:50051", help="policy host:port")
    demo.add_argument("--steps", type=int, default=100)
    demo.add_argument("--step", type=float, default=0.1, help="seconds per drive call")
    demo.add_argument("--camera-width", type=int, default=FakeCamera.width)
    demo.add_argument("--camera-height", type=int, default=FakeCamera.height)
    demo.add_argument("--camera-fov", type=float, default=FakeCamera.fov_deg)
    demo.add_argument("--camera-id", default=FakeCamera.logical_id)

    alpasim = commands.add_parser(
        "alpasim", help="package a policy as an alpasim driver service"
    )
    alpasim_commands = alpasim.add_subparsers(dest="alpasim_command", required=True)

    build = alpasim_commands.add_parser(
        "build-image",
        help="build a Docker image serving a policy in alpasim mode",
        description="Build a Docker image that serves a policy in alpasim mode: the "
        "policy and carla-driver-interface installed with pip on a Python base image.",
    )
    build.add_argument("--tag", required=True, help="image tag, e.g. my-policy:0.1")
    _add_policy_arguments(build, "route_follower")
    build.add_argument(
        "--install",
        action="append",
        default=[],
        metavar="REQUIREMENT",
        help="pip requirement for the policy (repeatable): a name, a URL, or a local "
        "project directory or wheel, which is copied into the build context",
    )
    build.add_argument(
        "--carla-driver-interface",
        dest="carla_driver_interface",
        default=None,
        metavar="REQUIREMENT",
        help="how to install this package (default: this version from PyPI); a local "
        "checkout works too",
    )
    build.add_argument("--base-image", default=DEFAULT_BASE_IMAGE)
    build.add_argument(
        "--apt", action="append", default=[], metavar="PACKAGE", help="Debian package"
    )
    build.add_argument(
        "--no-check-policy",
        dest="check_policy",
        action="store_false",
        help="do not import the policy's module at build time",
    )
    build.add_argument(
        "--context-dir",
        type=Path,
        default=None,
        help="where to write the build context (default: under ~/.cache)",
    )
    build.add_argument(
        "--no-build",
        dest="build",
        action="store_false",
        help="write the build context and its Dockerfile, but do not build",
    )
    build.add_argument(
        "--docker-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="extra argument for `docker build` (repeatable), "
        "e.g. --docker-arg=--no-cache",
    )

    wizard = alpasim_commands.add_parser(
        "wizard-config",
        help="write a driver config for alpasim's wizard that runs an image",
        description="Write the config group `driver=NAME` for alpasim's wizard, which "
        "runs IMAGE as alpasim's driver service. Use it with "
        "`alpasim_wizard driver=NAME hydra.searchpath=[file://OUT] ...`.",
    )
    wizard.add_argument("--image", required=True, help="image from `build-image`")
    wizard.add_argument("--out", type=Path, required=True, help="search-path directory")
    wizard.add_argument("--name", default="carla_driver_interface", help="driver=NAME")
    _add_policy_arguments(wizard, None)
    wizard.add_argument(
        "--volume", action="append", default=[], metavar="HOST:CONTAINER"
    )
    wizard.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    wizard.add_argument(
        "--camera",
        action="append",
        default=[],
        metavar="ID[:WxH][@HZ]",
        help="replace alpasim's cameras with these (repeatable)",
    )
    wizard.add_argument(
        "--pinhole",
        action="append",
        default=[],
        metavar="ID:HFOV[@X,Y,Z[,ROLL,PITCH,YAW]]",
        help="render camera ID through an undistorted pinhole model of HFOV degrees, "
        "mounted at X,Y,Z m in the rig frame, rotated ROLL,PITCH,YAW degrees "
        "(repeatable; give its resolution with --camera)",
    )
    wizard.add_argument("--max-workers", type=int, default=8)

    args = parser.parse_args(argv)
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if args.command == "serve" and args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )

    if args.command == "serve":
        return _serve(parser, args)
    if args.command == "alpasim":
        return _alpasim(parser, args)
    return _demo(args)


def _policy_kwargs(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> Dict[str, Any]:
    try:
        return parse_policy_args(args.policy_arg)
    except ValueError as exc:
        parser.error(str(exc))


def _serve(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    kwargs: Dict[str, Any] = {}
    # An image's baked-in arguments belong to its baked-in policy only.
    if args.policy == os.environ.get(POLICY_ENV):
        kwargs.update(json.loads(os.environ.get(POLICY_ARGS_ENV) or "{}"))
    kwargs.update(_policy_kwargs(parser, args))
    if args.cruise_speed is not None:
        kwargs["cruise_speed_mps"] = args.cruise_speed
    try:
        policy = load_policy(args.policy, **kwargs)
    except ValueError as exc:
        parser.error(str(exc))
    run_server(
        policy,
        port=args.port,
        host=args.host,
        max_workers=args.max_workers,
        mode=args.mode,
    )
    return 0


def _alpasim(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.alpasim_command == "wizard-config":
        return _wizard_config(parser, args)

    spec = ImageSpec(
        policy=args.policy,
        policy_args=_policy_kwargs(parser, args),
        install=args.install,
        carla_driver_interface=args.carla_driver_interface,
        base_image=args.base_image,
        apt_packages=args.apt,
        check_policy=args.check_policy,
    )
    context_dir = args.context_dir or default_context_dir(args.tag)
    if not args.build:
        print(write_build_context(spec, context_dir))
        return 0
    try:
        build_image(spec, args.tag, context_dir, extra_args=args.docker_arg)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    except subprocess.CalledProcessError as exc:
        print(
            f"docker build failed ({exc.returncode}); context: {context_dir}",
            file=sys.stderr,
        )
        return exc.returncode
    print(f"built {args.tag} (context: {context_dir})")
    return 0


def _wizard_config(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    environment = {}
    for item in args.env:
        key, sep, value = item.partition("=")
        if not sep:
            parser.error(f"--env {item!r} is not KEY=VALUE")
        environment[key] = value
    try:
        cameras = [parse_camera(c) for c in args.camera]
        pinholes = [parse_pinhole(p) for p in args.pinhole]
    except ValueError as exc:
        parser.error(str(exc))
    config = WizardDriverConfig(
        name=args.name,
        image=args.image,
        policy=args.policy,
        policy_args=_policy_kwargs(parser, args),
        volumes=args.volume,
        environment=environment,
        cameras=cameras,
        pinholes=pinholes,
        max_workers=args.max_workers,
    )
    try:
        written = write_wizard_config(config, args.out)
    except ValueError as exc:
        parser.error(str(exc))
    for path in written:
        print(path)
    print(
        f"\nrun: alpasim_wizard driver={args.name} "
        f"'hydra.searchpath=[file://{args.out.resolve()}]' ..."
    )
    return 0


def _demo(args: argparse.Namespace) -> int:
    camera = FakeCamera(
        logical_id=args.camera_id,
        width=args.camera_width,
        height=args.camera_height,
        fov_deg=args.camera_fov,
    )
    with FakeLoop(args.driver, [camera], step_s=args.step) as loop:
        result = loop.run(args.steps)
    final = result.ego.poses[-1].position
    print(f"steps              : {result.steps}")
    print(f"terminated by policy: {result.terminated_by_policy}")
    print(f"distance travelled : {final[0]:.2f} m")
    print(f"final lateral offset: {final[1]:+.2f} m")
    print(f"max lateral offset : {max(abs(y) for y in result.lateral_offsets_m):.2f} m")
    print(f"final speed        : {result.speeds_mps[-1]:.2f} m/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
