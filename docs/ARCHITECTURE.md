# Architecture

[alpasim's DESIGN.md](https://github.com/NVlabs/alpasim/blob/main/docs/DESIGN.md)
describes a distributed system of microservices with the Runtime at the centre:
it holds the world state and drives every other service as a gRPC client.

**This repository keeps the Driver from that diagram exactly as it is, and
replaces everything else with CARLA.**

## How alpasim is arranged

```
                       ┌───────────────────┐
                       │      Runtime      │  holds the world state
                       │   (gRPC client)   │  advances the rollout
                       └─────────┬─────────┘
          ┌────────────┬─────────┼─────────┬────────────┐
          ▼            ▼         ▼         ▼            ▼
    ┌───────────┐ ┌─────────┐ ┌─────┐ ┌─────────┐ ┌───────────┐
    │ Sensorsim │ │ Driver  │ │ VDC │ │ Physics │ │ Trafficsim│
    │   (NRE)   │ │(policy) │ │ctrl │ │ ground  │ │           │
    └───────────┘ └─────────┘ └─────┘ └─────────┘ └───────────┘
      All gRPC. Protocol definitions in alpasim_grpc/v0/*.proto
```

## How this repository is arranged

```
┌──────────────── CarlaRuntime process ────────────────┐   ┌─── Driver process ────┐
│                                                      │   │                       │
│  CarlaRuntime                                        │   │ CarlaEgodriverServicer│
│   ├ RouteProvider ......... alpasim RouteGenerator   │   │  (inherits upstream's │
│   ├ TrajectoryFollower .... alpasim VDCService       │   │   EgodriverService-   │
│   │    pure pursuit + speed PID                      │   │   Servicer)           │
│   ├ MetricsCollector ...... alpasim eval             │   │          │            │
│   └ WorldAdapter                                     │   │          ▼            │
│        ├ CarlaWorldAdapter (carla python API)        │   │      BaseDriver       │
│        │    ├ carla.Sensor(rgb) .... Sensorsim       │   │       ├ ConstantSpeed │
│        │    ├ carla physics engine . PhysicsService  │   │       └ RouteFollower │
│        │    └ carla.TrafficManager . Trafficsim      │   │                       │
│        └ FakeWorld (for CI / no CARLA needed)        │   │                       │
│                                                      │   │                       │
└────────────────────────┬─────────────────────────────┘   └───────────▲───────────┘
                         │                                             │
                         └──── egodriver.EgodriverService (gRPC) ──────┘
                              byte-for-byte the upstream wire contract
```

**The one RPC boundary that remains is the one worth keeping.** Sensorsim,
controller, physics and traffic are all roles CARLA already fills, so there is
nothing to gain from putting a process boundary in front of them.

### The left-hand box is replaceable

`WorldAdapter` is the seam that makes it so. `CarlaWorldAdapter` and `FakeWorld`
are two implementations of it, and a third can come from outside this
repository.

A runtime that already owns a CARLA world — a scenario runner, say, with its own
tick loop, its own NPCs and its own pass/fail conditions — has no use for
`CarlaRuntime`, which would insist on connecting a second client and loading the
map over again. It plays the Runtime role itself, speaking the same contract to
the same unmodified policy. The driver process on the right does not know the
difference, and cannot: that is what byte-for-byte wire compatibility buys.

Everything inside the left box then goes unused, `carla.TrafficManager`
included. That costs less than it sounds: the traffic manager drives the
background vehicles, never the ego, so a runtime bringing its own traffic gives
up nothing by not using this one. The ego is driven by the policy's plan in
either arrangement.

## One policy step

The order follows alpasim's `PolicyEvent`
(`runtime/alpasim_runtime/events/policy.py`). Matching it is *why* a driver
written for alpasim behaves the same here.

```
 CarlaRuntime                                        Driver
      │
      │  1. one image per camera         submit_image_observation
      ├───────────────────────────────────────────────────►
      │  2. every pose since last step   submit_egomotion_observation
      │     plus velocities
      ├───────────────────────────────────────────────────►
      │  3. route waypoints, rig frame   submit_route
      ├───────────────────────────────────────────────────►
      │  4. (optional) reference path    submit_recording_ground_truth
      ├───────────────────────────────────────────────────►
      │
      │  ===== barrier: every observation lands before the decision =====
      │
      │  5. drive(time_now_us, time_query_us, renderer_data)
      ├───────────────────────────────────────────────────►
      │◄─────────────────────────────────────────────────── DriveResponse.trajectory
      │                                                     (local -> rig_est)
      │  6. map rig_est back into the true local frame
      │  7. TrajectoryFollower -> carla.VehicleControl
      │  8. world.tick() x ticks_per_policy_step
      │  9. update metrics
      ▼
```

Step 6 is the subtle one. The driver is only ever told an *estimated* ego pose,
so the trajectory it returns is expressed in that estimated frame. Mapping it
back to the true `local` frame is the runtime's job, using the same formulation
as alpasim's `transform_trajectory_from_noisy_to_true_local_frame`. With
egomotion noise at zero the correction is the identity, and the runtime skips it
outright.

## Coordinate frames

| frame | definition |
|---|---|
| `local` | Inertial, right-handed, ENU-like. Obtained from the CARLA world by mirroring y |
| `rig` | Body-fixed. x forward, y left, z up. Origin at the **rear axle centre projected onto the ground** |
| CARLA world | Left-handed. x forward, **y right**, z up. Actor origin at the vehicle centre |

For the conversion itself, and for the CARLA 0.9.x wheel-position units trap,
see [COMPATIBILITY.md](COMPATIBILITY.md#the-coordinate-conversion-in-detail).

## Module map

| file | role | alpasim counterpart |
|---|---|---|
| `grpc_api/__init__.py` | Single import window for the wire contract | `alpasim_grpc` |
| `grpc_api/extension.py` | Pack/unpack for the extension payloads | (none) |
| `geometry.py` | Pose / Trajectory and proto conversion, numpy only | `alpasim_utils.geometry` (Rust extension) |
| `polyline.py` | Arc length, resampling, curvature; shared by driver and runtime | `utils_rs.Polyline` |
| `driver/service.py` | `EgodriverService` implementation | `alpasim_driver.main.EgoDriverService` |
| `driver/base.py` | Policy-facing API | `alpasim_driver.models.base` |
| `runtime/carla_runtime.py` | The orchestrator | `alpasim_runtime.worker.runtime` + `events/` |
| `runtime/world.py` | Simulator abstraction: protocol and dataclasses only | `alpasim_runtime.services.*` |
| `runtime/carla_world.py` | The CARLA implementation of it | `alpasim_runtime.services.*` |
| `runtime/control.py` | Trajectory tracking | `controller.VDCService` |
| `runtime/route.py` | Route generation and slicing | `alpasim_runtime.route_generator` |
| `runtime/metrics.py` | Scoring | `alpasim_eval` |
| `compat.py` | Machine-readable definition of the differences | (none) |

## Testing strategy

`carla` is an optional extra and is not installed in CI. A `WorldAdapter`
protocol sits in the way so `FakeWorld` (a kinematic bicycle model) can be
substituted, which means **the whole closed loop runs over real gRPC in CI**:
session setup, image submission, egomotion, route, `drive`, control application
and metrics. The contract (`runtime/world.py`) and the CARLA implementation
(`runtime/carla_world.py`) are separate files, so a CARLA-free process never
imports the latter.

`FakeWorld` is not a physics model. Anything that depends on real vehicle
dynamics has to be checked against a real CARLA server — see the manual steps in
the README.

But **the fake is not allowed to simply not implement something**. Camera mount
poses are the cautionary case: both adapters go through
`conversions.camera_pose_in_rig`, and `tests/test_conversions.py` asserts they
agree. When a fake carries its own hand-rolled approximation instead, what CI
believes it is protecting stops matching what runs on real hardware.
