# Compatibility with alpasim

Tracked upstream: [NVlabs/alpasim](https://github.com/NVlabs/alpasim)
`68709245a5dc0f2eda4f8cb2c3aa8cbdfa913043` (`alpasim_grpc` 0.55.0).

This document is prose, but **the same content exists in machine-readable form
in [`src/carla_driver_interface/compat.py`](../src/carla_driver_interface/compat.py)**.
If the two disagree,
`tests/test_proto_compat.py::test_compat_entries_are_mirrored_in_the_docs`
fails. To read it yourself:

```console
$ uv run carla-driver-interface compat-report
$ uv run carla-driver-interface compat-report --json
```

The vocabulary:

| level | meaning |
|---|---|
| **exact** | Identical to upstream, and tested |
| **partial** | Same messages, narrower behaviour or different meaning |
| **structural** | Same contract as seen by the driver; different implementation behind it |
| **extension** | Extra information carried in an extension point upstream already provides. Ignoring it is safe |
| **unimplemented** | Present upstream, absent here |

---

## exact — identical to upstream

| Area | alpasim | carla_driver_interface |
|---|---|---|
| **Egodriver service** | 8 RPCs on `egodriver.EgodriverService` | All 8 implemented by inheriting the upstream generated servicer. Service path `/egodriver.EgodriverService/*` and message types are the same |
| **Message types** | Defined in `alpasim_grpc` | The same package, as a rev-pinned git dependency. Not one message is redefined |
| **Observation ordering** | Images, egomotion, route and ground truth all land before `drive()`, enforced by an explicit barrier | Same order, same barrier |
| **Response frame** | `DriveResponse.trajectory` holds `local -> rig_est`, led by the ego pose at `time_now_us` | Identical: the servicer anchors the plan on the ego pose at `time_now_us` |
| **Early termination** | `DriveResponse.terminate_session` ends the rollout immediately | Honoured: the loop returns without stepping further |
| **API version** | `get_version` reports `alpasim_grpc.API_VERSION_MESSAGE` | Forwarded verbatim; the runtime compares it at startup and warns on a mismatch |
| **Rollout results** | `SimulationReturn.RolloutReturn` | The same message is produced, so alpasim-side tooling can read a CARLA rollout unchanged |

**Why this cannot quietly break.** The servicer inherits
`alpasim_grpc.v0.egodriver_pb2_grpc.EgodriverServiceServicer` by Python
inheritance, not by duck typing. If upstream adds an RPC,
`test_all_upstream_rpcs_exist_and_are_implemented` fails rather than the new
method silently returning `UNIMPLEMENTED`.

---

## partial — same messages, different behaviour

| Area | alpasim | carla_driver_interface |
|---|---|---|
| **Camera model** | ftheta, OpenCV fisheye and OpenCV pinhole | **Pinhole only** (`OpenCVPinholeCameraParam`). CARLA renders an ideal pinhole, so ftheta coefficients cannot be filled in honestly. Distortion coefficients are left empty rather than faked |
| **Shutter** | Rolling shutter; `frame_start_us` and `frame_end_us` bracket the sweep | **Global shutter**; `frame_start_us == frame_end_us`, `ShutterType.GLOBAL` |
| **Recording ground truth** | `submit_recording_ground_truth` carries the real car's recorded path | No recording exists. **Off by default.** With `send_ground_truth=True` it sends the route reference instead, which is a different quantity |
| **Egomotion error model** | `rig_est` diverges from `rig` via a proprioceptive noise model | Identity by default. `egomotion_position_noise_m` / `egomotion_yaw_noise_rad` reproduce both the divergence and the runtime-side correction, through the same code path upstream uses |
| **scene_id** | UUID of a recorded clip | `"<map>:<scenario>"` — there are no recordings to identify |

A driver that requires ftheta (alpasim's own `alpasim_driver` uses it for
rectification) will receive CARLA's pinhole. For most implementations pinhole is
the rectification *target*, so the practical impact is small — but the
ftheta-specific code path is not exercised.

---

## structural — same contract, different implementation

| Area | alpasim | carla_driver_interface |
|---|---|---|
| **Renderer** | `SensorsimService`: NRE neural reconstruction over gRPC | CARLA's rasterizer, in process. The sensorsim RPCs are never called; only its message types are reused, to describe cameras |
| **Controller / vehicle model** | `VDCService` over gRPC turns the plan into motion | In-process pure pursuit plus a speed PID, with CARLA's own vehicle dynamics behind it |
| **Physics** | `PhysicsService` performs ground-intersection correction | CARLA's physics engine keeps the vehicle on the ground; no RPC needed |
| **Traffic** | `TrafficService` simulates other agents | CARLA TrafficManager |
| **Coordinate frames** | Right-handed ENU `local` frame; rig origin at the rear axle centre projected onto the ground | CARLA is left-handed with the actor origin at the vehicle centre. The conversion layer mirrors y and shifts to the rear axle |
| **Orchestration** | Asyncio `RuntimeService` daemon, many concurrent rollouts, load balanced across driver replicas | A synchronous in-process class running one rollout. Concurrency is the caller's problem |

### The coordinate conversion in detail

CARLA: x forward, **y right**, z up; rotations as degrees `(pitch, yaw, roll)`.
alpasim: right-handed, rig is x forward, **y left**, z up.

```
position:  (x, y, z)          -> (x, -y, z)
rotation:  (pitch, yaw, roll) -> (-pitch, -yaw, roll)   [degrees -> radians]
```

This is the mirror `M = diag(1, -1, 1)` about the y axis, i.e.
`R_alpasim = M R_carla M`.
`tests/test_conversions.py::test_rotation_matches_the_mirrored_carla_matrix`
checks it against a reproduction of CARLA's own `Transform.get_matrix()` rather
than against our reasoning about it.

**Rig origin.** Set it explicitly with `RuntimeConfig.rear_axle_offset_m`. Left
unset, it is derived from `get_physics_control().wheels` — but in CARLA 0.9.x
`WheelPhysicsControl.position` is reported in **world coordinates,
centimetres**, so the derived value is easy to get wrong. It is always logged at
startup; if it looks implausible, override it.

Every adapter builds camera mounts through `conversions.camera_pose_in_rig`.
Doing the mirror by hand is how a mount rotation gets dropped — a bare `-y` on
the position looks right for a forward camera and silently turns a side camera
into a forward one.

---

## extension — additions carried in upstream's extension points

| Area | alpasim | carla_driver_interface |
|---|---|---|
| **Renderer payload** | `DriveRequest.renderer_data` is free-form and NRE-specific | Carries a serialized `carla_driver.v0.CarlaRendererData`: map, weather, traffic light, speed limit, other actors. Drivers that ignore it are unaffected |
| **Driver debug payload** | `DebugInfo.unstructured_debug_info` is free-form | Carries a serialized `carla_driver.v0.CarlaDriveDebugInfo` |

`proto/carla_driver/v0/carla_driver.proto` **declares no service at all**.
Declaring one would create a second, incompatible way to talk to a driver;
`test_our_proto_declares_no_service` prevents it. The extension messages
`import` upstream types and compose them rather than copying, so
`CarlaDriveSessionInfo.base` *is* upstream's `egodriver.DriveSessionRequest`.

Unpacking is deliberately tolerant. An upstream alpasim runtime may put
something else entirely in `renderer_data`; the driver treats an unparseable
payload as "no CARLA data" and keeps driving
(`test_foreign_renderer_data_is_ignored_not_fatal`). Both directions share one
codec, `grpc_api/extension.py`, so the two ends cannot drift apart on that
policy.

---

## unimplemented — present upstream, absent here

| Area | alpasim | carla_driver_interface |
|---|---|---|
| **Structured logging** | `logging.proto` records every request and response into an ASL log | Not implemented; the runtime logs through the standard `logging` module |
| **Video model** | `video_model.proto` drives a generative video model | Not implemented |
| **LiDAR** | `SensorsimService.render_lidar` produces point clouds | Not implemented — **the egodriver contract has no LiDAR submission RPC**, so there would be nowhere to deliver it |
| **Runtime gRPC surface** | `RuntimeService.simulate` / `prefetch_scene` / `get_runtime_info` / `shut_down` | Not served; `CarlaRuntime` is used as a Python class |

---

## Interoperability in practice

**Driving this driver from an alpasim runtime.** It should work as-is: the
service name, the messages and the observation ordering are the same, and
`get_version` reports upstream's API version. The CARLA extension simply will
not arrive, so `RouteFollowerPolicy` drives without seeing traffic lights or
speed limits.

**Driving an alpasim driver from `CarlaRuntime`.** Run `alpasim_driver` in its
own process and point `--driver <host>:<port>` at it. The *partial* rows above
apply — pinhole only, global shutter, no recorded ground truth — so check any
model that assumes ftheta before relying on the result.

**Both packages in one process.** Fine. Upstream `alpasim_grpc` is used as a
dependency and its descriptors are never duplicated, so nothing collides in the
descriptor pool.
