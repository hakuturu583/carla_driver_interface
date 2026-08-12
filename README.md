# carla_driver_interface

[NVlabs/alpasim](https://github.com/NVlabs/alpasim) の **driver module (`egodriver.EgodriverService`)**
と互換な gRPC driver と、それを CARLA で closed loop に回す **CARLA Runtime** の Python パッケージ。

- alpasim の proto を **git 依存としてそのまま import** し、生成 servicer を **Python 継承**して実装する。
  メッセージも service パスも upstream と同一なので、ワイヤ互換は構造的に保証される。
- 完全一致できない部分は推測に頼らず、**[docs/COMPATIBILITY.md](docs/COMPATIBILITY.md) と
  `carla-driver-interface compat-report` の両方で明示**する。両者のずれはテストで検出される。
- 依存は 4 つだけ (`alpasim-grpc`, `grpcio`, `numpy`, `pillow`)。`carla` は optional extra。

```
CarlaRuntime ──── egodriver.EgodriverService (upstream と同一) ────► Driver
  │                                                                    │
  ├ carla.Sensor(rgb) ......... alpasim SensorsimService の代替        ├ ConstantSpeedPolicy
  ├ pure pursuit + PID ........ alpasim VDCService の代替              └ RouteFollowerPolicy
  ├ carla 物理エンジン ......... alpasim PhysicsService の代替
  └ carla.TrafficManager ...... alpasim TrafficService の代替
```

詳細は [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## セットアップ

Python 3.11 / 3.12（`alpasim-grpc` の制約）。

```console
$ uv sync --extra dev
```

CARLA を使う場合（0.9.x は PyPI にある）:

```console
$ uv sync --extra dev --extra carla
```

CARLA 0.10.x は PyPI に無いので、同梱の PythonAPI を `--carla-python-path` で指す。

## 動かす

### CARLA 抜きで closed loop を確認する

内蔵の `FakeWorld`（自転車モデル）で、CARLA サーバ無しに loop 全体が回る。

```console
$ uv run carla-driver-interface serve --policy route_follower --port 50051 &
$ uv run carla-driver-interface demo --driver localhost:50051 --steps 200
success            : True
steps              : 132
terminated by driver: False
  collisions                   0.0000
  distance_travelled_m         86.6028
  lane_invasions               0.0000
  route_completion             1.0000
  route_lateral_error_m        0.1687
  speed_mps                    6.5671
```

### 実 CARLA に対して

CARLA サーバを起動しておいてから:

```console
$ uv run carla-driver-interface serve --policy route_follower --port 50051 &
$ uv run carla-driver-interface run \
      --carla-host localhost --carla-port 2000 \
      --map Town10HD_Opt --traffic 30 \
      --driver localhost:50051 --steps 400 \
      --metrics-json /tmp/rollout.json
```

CARLA 0.10.x の場合は `--carla-python-path /path/to/CARLA/PythonAPI/carla/dist` を足す。

> **車両ジオメトリについて**: alpasim の rig 原点は後輪軸中心の接地投影だが、CARLA の actor
> 原点は車両中心。既定では wheel physics から差分を導出してログに出す。CARLA 0.9.x の
> `WheelPhysicsControl.position` はワールド座標かつ cm 単位という癖があるので、
> ログの値が怪しければ `--rear-axle-offset -1.4` のように明示すること。

### 互換性レポート

```console
$ uv run carla-driver-interface compat-report
$ uv run carla-driver-interface compat-report --json
```

## 自分の policy を書く

`BaseDriver` を継承して `drive()` だけ実装すればよい。gRPC・セッション管理・フレーム保持・
rig↔local 変換は servicer 側が持つ。

```python
from carla_driver_interface.driver import BaseDriver, DriveContext, DriveResult, run_server
from carla_driver_interface.geometry import Pose, Trajectory


class MyPolicy(BaseDriver):
    name = "my_policy"
    frame_history_length = 4  # カメラごとに保持するフレーム数

    def drive(self, ctx: DriveContext) -> DriveResult:
        frame = ctx.session.latest_frame("camera_front_wide_120fov")
        image = frame.as_array()  # (H, W, 3) uint8 RGB
        route = ctx.session.route_waypoints_in_rig  # (N, 3), rig 系
        speed = ctx.session.speed_mps()

        # CARLA 固有の ground truth。alpasim runtime 相手なら None になる。
        if ctx.renderer_data is not None:
            speed_limit = ctx.renderer_data.speed_limit_mps

        plan = Trajectory.empty()  # rig 系で返す。local 変換は servicer が行う
        for i in range(1, 41):
            dt = i * 0.1
            plan.append(ctx.time_now_us + int(dt * 1e6), Pose.from_xyz_yaw(8.0 * dt, 0.0, 0.0, 0.0))
        return DriveResult(trajectory_in_rig=plan)


run_server(MyPolicy(), port=50051)
```

この driver は **alpasim の runtime からもそのまま呼べる**。逆に alpasim の `alpasim_driver` を
`CarlaRuntime` から呼ぶこともできる（`--driver` で指すだけ）。制約は
[docs/COMPATIBILITY.md](docs/COMPATIBILITY.md#相互運用の実際) を参照。

## Python API から使う

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
# outcome.rollout_return は alpasim の SimulationReturn.RolloutReturn そのもの
```

## 開発

```console
$ uv run pytest                                   # CARLA 不要
$ uv run ruff check . && uv run ruff format --check .
$ uv run mypy src
$ uv run python scripts/compile_protos.py         # proto 変更時
```

生成された protobuf モジュールはリポジトリにコミットしてある（インストール時に
`grpcio-tools` を要求しないため）。`tests/test_proto_compat.py` が再生成結果と
突き合わせて、古くなっていれば落ちる。

## ライセンス

Apache-2.0。alpasim も Apache-2.0。
