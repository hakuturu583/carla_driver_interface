# carla_driver_interface

A Python package with two halves: a gRPC driver compatible with the
[NVlabs/alpasim](https://github.com/NVlabs/alpasim) driver module
(`egodriver.EgodriverService`), and a CARLA Runtime that closes the loop around
it.

- alpasim's protos are **imported as a pinned git dependency** and the generated
  servicer is **subclassed**. Nothing is redefined, so wire compatibility —
  messages and service path alike — holds by construction rather than by
  vigilance.
- Where an exact match is impossible, the difference is **stated in
  [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md) and printed by
  `carla-driver-interface compat-report`**, and a test fails if the two drift
  apart.
- Four dependencies: `alpasim-grpc`, `grpcio`, `numpy`, `pillow`. `carla` is an
  optional extra.

```
CarlaRuntime ──── egodriver.EgodriverService (same as upstream) ────► Driver
  │                                                                     │
  ├ carla.Sensor(rgb) ......... replaces alpasim SensorsimService       ├ ConstantSpeedPolicy
  ├ pure pursuit + PID ........ replaces alpasim VDCService             └ RouteFollowerPolicy
  ├ carla physics engine ...... replaces alpasim PhysicsService
  └ carla.TrafficManager ...... replaces alpasim TrafficService
```

### Which half you need

The two halves are independent, and most uses need only one.

| | Driver half | Runtime half |
| --- | --- | --- |
| What it is | `BaseDriver`, the servicer, `run_server` | `CarlaRuntime`, `CarlaWorldAdapter` |
| Use it to | write a policy | close the loop around one |
| Imports `carla` | **no, never** | yes |
| Owns | nothing | the client, map, ego, clock and background traffic |

A policy built on the driver half is a gRPC server and nothing else. It never
touches CARLA, and the `carla` extra is not needed to write, test or ship one.

The runtime half is *a* way to close the loop, not the only one. Anything that
speaks `egodriver.EgodriverService` can drive the same policy unmodified — a
scenario runner that already owns a CARLA world plays the Runtime role itself
and never constructs `CarlaWorldAdapter`. Everything in the left-hand box above
then goes unused, and the replacement brings its own.

`carla.TrafficManager` is worth naming there, because it is easy to assume it
has something to do with the ego. It does not. It drives the background vehicles
`--traffic` spawns; the ego is never registered with it, and follows the
policy's plan through `TrajectoryFollower`.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full picture.

## Setup

Python 3.11 or 3.12 (the `alpasim-grpc` constraint).

```console
$ uv sync --extra dev
```

With CARLA (0.9.x is on PyPI):

```console
$ uv sync --extra dev --extra carla
```

CARLA 0.10.x is not on PyPI; point `--carla-python-path` at its bundled
PythonAPI instead.

## Running it

### Closing the loop without CARLA

The built-in `FakeWorld` (a kinematic bicycle model) runs the entire loop with
no CARLA server.

```console
$ uv run carla-driver-interface serve --policy route_follower --port 50051 &
$ uv run carla-driver-interface demo --driver localhost:50051 --steps 200
success            : True
steps              : 132
terminated by driver: False
  collisions                 0.0000
  distance_travelled_m       86.6028
  lane_invasions             0.0000
  route_completion           1.0000
  route_lateral_error_m      0.1687
  speed_mps                  6.5671
```

### Against a real CARLA server

With the server already running:

```console
$ uv run carla-driver-interface serve --policy route_follower --port 50051 &
$ uv run carla-driver-interface run \
      --carla-host localhost --carla-port 2000 \
      --map Town10HD_Opt --traffic 30 \
      --driver localhost:50051 --steps 400 \
      --metrics-json /tmp/rollout.json
```

On CARLA 0.10.x, add `--carla-python-path /path/to/CARLA/PythonAPI/carla/dist`.

> **About the vehicle geometry.** alpasim puts the rig origin at the rear axle
> centre projected onto the ground; CARLA's actor origin is the vehicle centre.
> By default the difference is derived from the wheel physics and logged. In
> CARLA 0.9.x `WheelPhysicsControl.position` is reported in world coordinates
> and centimetres, so if the logged value looks wrong, set it explicitly:
> `--rear-axle-offset -1.4`.

### Compatibility report

```console
$ uv run carla-driver-interface compat-report
$ uv run carla-driver-interface compat-report --json
```

## Writing your own policy

Subclass `BaseDriver` and implement `drive()`. gRPC, session management, frame
retention and the rig/local conversion all belong to the servicer.

```python
from carla_driver_interface.driver import BaseDriver, DriveContext, DriveResult, run_server
from carla_driver_interface.geometry import Pose, Trajectory


class MyPolicy(BaseDriver):
    name = "my_policy"
    frame_history_length = 4  # frames retained per camera

    def drive(self, ctx: DriveContext) -> DriveResult:
        frame = ctx.session.latest_frame("camera_front_wide_120fov")
        image = frame.as_array()  # (H, W, 3) uint8 RGB
        route = ctx.session.route_waypoints_in_rig  # (N, 3), rig frame
        speed = ctx.session.speed_mps()

        # CARLA-specific ground truth. None when driven by an alpasim runtime.
        if ctx.renderer_data is not None:
            speed_limit = ctx.renderer_data.speed_limit_mps

        # Return the plan in the rig frame; the servicer converts it to local.
        plan = Trajectory.empty()
        for i in range(1, 41):
            dt = i * 0.1
            plan.append(ctx.time_now_us + int(dt * 1e6), Pose.from_xyz_yaw(8.0 * dt, 0.0, 0.0, 0.0))
        return DriveResult(trajectory_in_rig=plan)


run_server(MyPolicy(), port=50051)
```

This driver can be **called by an alpasim runtime as-is**, and conversely
`CarlaRuntime` can drive alpasim's own `alpasim_driver` (just point `--driver`
at it). For the caveats, see
[docs/COMPATIBILITY.md](docs/COMPATIBILITY.md#interoperability-in-practice).

## Using it from Python

```python
from carla_driver_interface.runtime import (
    CarlaRuntime,
    CarlaWorldAdapter,
    RuntimeConfig,
    ScenarioSpec,
)

config = RuntimeConfig(driver_address="localhost:50051", max_steps=400)
scenario = ScenarioSpec(map_name="Town10HD_Opt", num_background_vehicles=30)

outcome = CarlaRuntime(CarlaWorldAdapter(config, scenario), config, scenario).run_rollout()
print(outcome.success, outcome.metrics)
# outcome.rollout_return is alpasim's SimulationReturn.RolloutReturn itself
```

## Development

```console
$ uv run pytest                                   # no CARLA needed
$ uv run ruff check . && uv run ruff format --check .
$ uv run mypy src
$ uv run python scripts/compile_protos.py         # after editing a proto
```

The generated protobuf modules are committed, so installing the package needs
no `grpcio-tools`. `tests/test_proto_compat.py` regenerates them and fails if
what is committed has gone stale.

## Licence

Apache-2.0, as is alpasim.
