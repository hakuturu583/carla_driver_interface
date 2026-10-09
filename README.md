# carla-driver-interface

Both ends of [NVlabs/alpasim](https://github.com/NVlabs/alpasim)'s
`egodriver.EgodriverService` gRPC contract, light enough for a driving policy to depend on:

- **policy side**: implement `BaseDriver.drive()` and serve it with `run_server()` /
  `serving()` (or `carla-driver-interface serve --policy route_follower`, or any
  `package.module:Class` through `load_policy()`); session bookkeeping, frame retention,
  ego history and the rig/local conversion are handled for you;
- **a CARLA-free runtime** for tests: `testing.FakeLoop` drives any policy server over
  real gRPC on a straight road, rendering each declared pinhole camera
  (`carla-driver-interface demo --driver localhost:50051`).

The CARLA runtime lives in
[autoware_carla_scenario](https://github.com/autowarefoundation/autoware_carla_scenario):
with `ego.entity=carla_driver` it drives the same policy server inside a scenario. An
upstream alpasim runtime drives it too — nothing about the alpasim wire format is changed.

## Install

```console
$ pip install carla-driver-interface
```

Python 3.10–3.14. Dependencies: grpcio, protobuf (4.25+), numpy, scipy, Pillow. No CARLA,
no alpasim distribution: the wire contract is vendored ([`proto/README.md`](https://github.com/hakuturu583/carla_driver_interface/blob/main/proto/README.md))
with field numbers and service names unchanged.

## Writing a policy

```python
from carla_driver_interface.driver import BaseDriver, DriveContext, DriveResult
from carla_driver_interface.geometry import Pose, Trajectory
from carla_driver_interface.server import run_server


class MyPolicy(BaseDriver):
    name = "my_policy"

    def drive(self, ctx: DriveContext) -> DriveResult:
        frame = ctx.session.latest_frame("camera_front")  # .as_array() -> RGB
        plan = Trajectory.empty()
        for i in range(1, 41):  # 4 s ahead, in the rig frame (x forward, y left)
            plan.append(
                ctx.time_now_us + i * 100_000, Pose.from_xyz_yaw(0.5 * i, 0.0, 0.0, 0.0)
            )
        return DriveResult(trajectory_in_rig=plan)


run_server(MyPolicy(), port=50051)
```

Every sensor reaches a policy the same way: as a frame in `session.frame_history`
(`CameraFrame`, `LidarFrame`), announced through `on_frame`, each with `as_array()`.
Beyond alpasim's cameras, a policy run by autoware_carla_scenario gets LiDAR sweeps (rig
frame) that way and, with `map_dir` set to its copy of the runtime's `driver.map_dir`, the
world's map as files (`ctx.map`) and every traffic light resolved into that map's own
elements (`ctx.stop_lines()`, `carla_driver_interface.hdmap`). CARLA ground truth arrives
in `ctx.renderer_data` (`None` when an alpasim runtime drives the policy).

## Testing a policy without CARLA

```console
$ carla-driver-interface serve --policy route_follower --port 50051 &
$ carla-driver-interface demo --driver localhost:50051 --steps 200
```

or, in a test, `FakeLoop("localhost:50051", [FakeCamera()])` from
`carla_driver_interface.testing` (a context manager; `.run(steps)` returns a
`LoopResult`). The ego follows the returned plan exactly, so what is tested is the policy
and the wire, not a controller.

## Successor to carla-driver-interface 0.1.0

This package replaces the previous `carla-driver-interface` 0.1.0 of this repository,
which had two halves: a driver built on the `alpasim-grpc` distribution (Python 3.11–3.12
only), and a CARLA runtime (`CarlaRuntime`, `CarlaWorldAdapter`, `FakeWorld`,
`carla-driver-interface run`). From 1.0.0:

- the **driver half** is this package. It was developed as `autoware-carla-egodriver`, a
  workspace member of autoware_carla_scenario, and moves here under the old name so
  policies and scenario packages version independently;
- the **runtime half** is autoware_carla_scenario's `ego.entity=carla_driver`, which owns
  the CARLA world, the ego and the traffic, and drives the policy over the same contract.
  `testing.FakeLoop` replaces `FakeWorld` for CARLA-free tests.

### Migrating from 0.1.0

Policies port by changing imports; `BaseDriver`, `DriveContext`, `DriveResult`,
`SessionState`, `CameraFrame`, `Pose` and `Trajectory` keep their names and fields.

| 0.1.0 | 1.0.0 |
| --- | --- |
| `carla_driver_interface.driver.base` (`BaseDriver`, `DriveContext`, ...) | `carla_driver_interface.driver` |
| `carla_driver_interface.driver.server` (`run_server`, `serving`, `build_server`) | `carla_driver_interface.server` |
| `carla_driver_interface.driver.service.CarlaEgodriverServicer` | `carla_driver_interface.service.EgodriverServicer` |
| `carla_driver_interface.driver.policies` | `carla_driver_interface.policies` (plus `load_policy`) |
| `carla_driver_interface.grpc_api` (messages, stubs) | `carla_driver_interface.protocol` |
| `carla_driver_interface.grpc_api.extension` (`pack_renderer_data`, `unpack_debug_info`, ...) | `carla_driver_interface.protocol` |
| `carla_driver_interface.grpc_api.API_VERSION_MESSAGE` | `carla_driver_interface.protocol.ALPASIM_API_VERSION` |
| `alpasim_grpc.v0.{common,egodriver,sensorsim}_pb2` | `carla_driver_interface.protocol.{common,egodriver,sensorsim,carla_driver}_pb2` |
| `carla_driver_interface.geometry` (`Pose`, `Trajectory`), `.polyline` | unchanged |
| `geometry.quat_mul`, `yaw_to_quat_xyzw`, `quat_to_yaw`, ... | `Pose` methods, or `scipy.spatial.transform.Rotation` |
| `BaseDriver.on_image(session, frame)` | `BaseDriver.on_frame(session, frame)` (cameras and LiDAR) |
| `carla_driver_interface.fakes.FakeWorld` | `carla_driver_interface.testing.FakeLoop` |
| `carla_driver_interface.runtime` (`CarlaRuntime`, ...), `carla-driver-interface run` | autoware_carla_scenario, `ego.entity=carla_driver` |
| `carla_driver_interface.compat`, `carla-driver-interface compat-report` | removed |

`alpasim_grpc.v0.runtime_pb2` (`RolloutSpec`, `SimulationReturn`, ...) is not vendored: it
is the runtime's own contract, not the egodriver one. Messages not re-exported by name in
`protocol` (`CarlaDriveSessionInfo`, `CarlaCompatReport`, ...) remain available from
`protocol.carla_driver_pb2`.

## Versioning

carla-driver-interface follows [semantic versioning](https://semver.org) on its own,
independently of autoware_carla_scenario or any policy. The compatibility unit is the gRPC
wire contract (`proto/`): any 1.x policy server and any runtime built against 1.x
interoperate, so depend on it with a range (`carla-driver-interface>=1.0,<2`), not an
exact pin.

- **patch**: fixes that change neither the wire nor the Python API;
- **minor**: additions — new fields or messages in the `carla_driver` extension (old peers
  ignore them), new Python API;
- **major**: anything that breaks a peer or a policy — a changed or removed field number,
  a renamed service, a removed Python API.

The vendored alpasim contract is never edited; following a newer alpasim revision is a
minor release when it is wire-compatible and a major one when it is not.

## Development

```console
$ uv sync --group dev
$ uv run ruff check . && uv run ruff format --check .
$ uv run mypy src tests scripts
$ uv run pytest -q
```

The generated protobuf modules in `src/carla_driver_interface/_proto` are committed;
regenerate them with `uv run python scripts/compile_protos.py` (on Python 3.10–3.12, see
[`proto/README.md`](https://github.com/hakuturu583/carla_driver_interface/blob/main/proto/README.md)). `--check`, also run by the test suite, verifies
they are current.

## Releasing

Releases are cut by merging, and go to
[PyPI](https://pypi.org/project/carla-driver-interface/) from
[`.github/workflows/release.yml`](https://github.com/hakuturu583/carla_driver_interface/blob/main/.github/workflows/release.yml)
through Trusted Publishing — no API token is stored anywhere.

**Every PR carries exactly one of the labels `bump patch`, `bump minor` or `bump major`**
(the *Check Version Bump Label* check fails otherwise), chosen by the
[versioning](#versioning) rules above. Do not edit `version` in `pyproject.toml` by hand.
When the PR is merged to `main`, the release workflow:

1. bumps `version` in `pyproject.toml` by the label, re-locks (`uv lock`) and pushes the
   commit `chore: bump version to X.Y.Z` to `main`;
2. tags it `vX.Y.Z` and creates the GitHub Release;
3. builds the sdist and wheel (`uv build`), checks them with `twine check --strict`,
   installs the wheel into a clean virtual environment on every supported Python
   (3.10–3.14) and runs `carla-driver-interface --help`;
4. publishes to PyPI.

A merge without a bump label releases nothing.

**Re-running a failed release.** If a step after the bump failed (the tag, the Release,
PyPI), do not re-run the push run — it would bump again. Run the workflow by hand instead
(Actions → Release → Run workflow, on `main`) with the `version` it bumped to, e.g.
`1.2.3`. It builds from that version's tag, or else from its
`chore: bump version to 1.2.3` commit, never from a later `main`, and reuses an existing
tag or Release; files already on PyPI are skipped.

The successor keeps the predecessor's version line: it starts from 0.1.0, and the PR
that introduces it is merged with `bump major`, which releases 1.0.0 like any other
release.

**Pushing to `main`.** The bump commit and the tag are pushed with `GITHUB_TOKEN`, which
works while `main` has no branch protection rules. If `main` gets a ruleset whose bypass
list does not include `github-actions[bot]`, add a `GH_PAT` repository secret (a token
that may push to `main` and create tags); the workflow uses it whenever it exists.

**One-time PyPI setup.** PyPI needs a (pending) trusted publisher for the project
`carla-driver-interface` (the distribution name in `pyproject.toml`) with owner
`hakuturu583`, repository `carla_driver_interface`, workflow `release.yml` and environment
`pypi` (PyPI → Your account → Publishing), and this repository needs a `pypi` environment
(Settings → Environments). Until then the publish job fails and the built distributions
remain available as the run's `pypi-dist` artifact.

## License

Apache-2.0. The vendored alpasim protos are Apache-2.0 as well
([`proto/LICENSE.alpasim`](https://github.com/hakuturu583/carla_driver_interface/blob/main/proto/LICENSE.alpasim)).
