# Vendored protobuf definitions

The `.proto` files here define the wire contract `carla-driver-interface` speaks: NVlabs/alpasim's
`egodriver.EgodriverService` gRPC contract, copied **verbatim**, and the CARLA ground-truth
extension a policy reads alongside it. Vendoring them means neither this package nor
the runtimes built on it (such as
[autoware_carla_scenario](https://github.com/autowarefoundation/autoware_carla_scenario))
has to depend on alpasim's distribution.

| Directory | Upstream | Pinned revision | Upstream path |
| --- | --- | --- | --- |
| `alpasim_grpc/` | [NVlabs/alpasim](https://github.com/NVlabs/alpasim) | `68709245a5dc0f2eda4f8cb2c3aa8cbdfa913043` | `src/grpc/alpasim_grpc/v0/` |
| `carla_driver/` | this repository, as of carla-driver-interface 0.1.0 | `af1dcd3d3ddae7739f811e414a642d72d0440386` | `proto/carla_driver/v0/` |

Both are Apache-2.0; see `LICENSE.alpasim`.

`carla_driver/` is not vendored but maintained here: it is the extension this repository
has always defined, carried over from 0.1.0 and extended in place. `CarlaRendererData`
fields 9–12 (`lidar`, `map_id`, `traffic_lights`, field 9 reserved) and the `LidarSweep`,
`TrafficLight` and `StopPoint` messages carry over [#12](https://github.com/hakuturu583/carla_driver_interface/pull/12)'s
`driver_extension.proto` with the same field numbers.

## Why vendor instead of depend

`alpasim-grpc` declares `requires-python = ">=3.11,<3.13"`, while this package supports
`>=3.10,<3.15` — Autoware's own environment is 3.10, which `alpasim-grpc` excludes.
Depending on it would drop 3.10 (and 3.13+) support, so the wire contract is vendored and
compiled locally instead. (carla-driver-interface 0.1.0 did depend on `alpasim-grpc`, and
was limited to Python 3.11–3.12 for that reason.)

Only the transitive closure of the two entry points is vendored:

```
egodriver.proto              carla_driver.proto
├── common.proto             ├── common.proto
└── sensorsim.proto          ├── egodriver.proto
    └── common.proto         └── sensorsim.proto
```

`carla_driver.proto` defines the CARLA-specific messages that ride inside the two `bytes`
extension points alpasim already provides — `DriveRequest.renderer_data` and
`DriveResponse.DebugInfo.unstructured_debug_info` — so nothing about the alpasim wire
format changes.

The `alpasim_grpc/` files are **never edited**. Field numbers, package names, and the
`alpasim_grpc/v0/...` import paths are preserved exactly, which is what keeps the generated
messages wire-compatible with an upstream alpasim runtime. The `carla_driver/` extension
only ever grows: existing field numbers keep their meaning, and retired ones are
`reserved`.

## Regenerating the Python stubs

Generated modules are committed to `src/carla_driver_interface/_proto/` so that the
runtime never needs `grpcio-tools`. Regenerate them with `scripts/compile_protos.py`, from
the repository root, on Python 3.10–3.12 (where the dev group installs `grpcio-tools`;
it stays on the protobuf 4.25 line the runtime floor loads):

```bash
uv sync --group dev --python 3.12
uv run python scripts/compile_protos.py
```

`--check` verifies the committed output is up to date without writing anything; the same
check runs in `tests/test_proto_generated.py` (skipped where `grpcio-tools` is not
installed).

## Updating to a newer upstream revision

1. Copy alpasim's `.proto` files from the new revision into `alpasim_grpc/v0/`.
2. Update the pinned revision in this file, in `scripts/compile_protos.py`
   (`ALPASIM_REV`) and in `carla_driver_interface/protocol.py` (`ALPASIM_REV`,
   `ALPASIM_API_VERSION`).
3. Re-run the generator and the test suite; check the runtimes that pin this package
   (autoware_carla_scenario) against the new revision too.

The `carla_driver` pin matters for behaviour, not just for messages: `af1dcd3` is the
revision that made the runtime report a traffic light *before* the ego reaches its stop
line, and autoware_carla_scenario's `driver/renderer.py` reproduces that logic. A policy pinned to the same
revision — as [stl_driver](https://github.com/hakuturu583/stl_driver) is — expects it.
