# アーキテクチャ

[alpasim の DESIGN.md](https://github.com/NVlabs/alpasim/blob/main/docs/DESIGN.md) は、
Runtime を中心に据えた分散マイクロサービス構成を示している。Runtime が world state を持ち、
gRPC クライアントとして他のサービスをすべて駆動する。

このリポジトリは**その図のうち Driver をそのまま残し、残りを CARLA 一つで置き換える**。

## alpasim の構成

```
                     ┌──────────────────┐
                     │     Runtime      │  world state を保持
                     │  (gRPC client)   │  ロールアウトを進行させる
                     └────────┬─────────┘
          ┌───────────┬───────┼────────┬────────────┐
          ▼           ▼       ▼        ▼            ▼
    ┌──────────┐ ┌────────┐ ┌────┐ ┌────────┐ ┌──────────┐
    │ Sensorsim│ │ Driver │ │VDC │ │Physics │ │ Trafficsim│
    │  (NRE)   │ │ (policy)│ │ctrl│ │ ground │ │           │
    └──────────┘ └────────┘ └────┘ └────────┘ └──────────┘
      すべて gRPC。プロトコル定義は alpasim_grpc/v0/*.proto
```

## このリポジトリの構成

```
┌─────────────── CarlaRuntime プロセス ───────────────┐      ┌──── Driver プロセス ────┐
│                                                     │      │                         │
│  CarlaRuntime                                       │      │  CarlaEgodriverServicer │
│   ├ RouteProvider ......... alpasim RouteGenerator  │      │   (upstream の          │
│   ├ TrajectoryFollower .... alpasim VDCService      │      │    EgodriverService-    │
│   │    pure pursuit + 速度 PID                      │      │    Servicer を継承)     │
│   ├ MetricsCollector ...... alpasim eval            │      │          │              │
│   └ WorldAdapter                                    │      │          ▼              │
│        ├ CarlaWorldAdapter (carla python API)       │      │      BaseDriver         │
│        │    ├ carla.Sensor(rgb) ... Sensorsim       │      │       ├ ConstantSpeed   │
│        │    ├ carla 物理エンジン ... PhysicsService  │      │       └ RouteFollower   │
│        │    └ carla.TrafficManager  Trafficsim      │      │                         │
│        └ FakeWorld (CI 用 / CARLA 不要)             │      │                         │
│                                                     │      │                         │
└───────────────────────┬─────────────────────────────┘      └────────────▲────────────┘
                        │                                                 │
                        └────── egodriver.EgodriverService (gRPC) ────────┘
                                 upstream と完全に同一のワイヤ契約
```

**残る唯一の RPC 境界が、残すべき唯一の境界**である。sensorsim / controller / physics /
traffic はいずれも CARLA が同じ役目を果たすので、プロセス間通信にする理由が無い。

## 1 policy step の流れ

alpasim の `runtime/alpasim_runtime/events/policy.py` (`PolicyEvent`) と同じ順序を踏む。
順序を揃えていることが、alpasim 向けに書かれた driver が同じ挙動になる根拠になっている。

```
 CarlaRuntime                                        Driver
      │
      │  ① 画像を全カメラ分                submit_image_observation
      ├───────────────────────────────────────────────────►
      │  ② 前ステップ以降の全 pose + 速度   submit_egomotion_observation
      ├───────────────────────────────────────────────────►
      │  ③ rig 系の route waypoint         submit_route
      ├───────────────────────────────────────────────────►
      │  ④ (任意) 参照軌跡                 submit_recording_ground_truth
      ├───────────────────────────────────────────────────►
      │
      │  ===== バリア: 観測が全部届いてから決定 =====
      │
      │  ⑤ drive(time_now_us, time_query_us, renderer_data)
      ├───────────────────────────────────────────────────►
      │◄─────────────────────────────────────────────────── DriveResponse.trajectory
      │                                                     (local -> rig_est)
      │  ⑥ rig_est → 真の local へ写像
      │  ⑦ TrajectoryFollower → carla.VehicleControl
      │  ⑧ world.tick() × ticks_per_policy_step
      │  ⑨ メトリクス更新
      ▼
```

⑥ が要点。driver には「推定された」自車位置しか渡していないので、返ってきた軌道は推定フレーム
に乗っている。真の local へ戻すのは runtime の責務であり、alpasim の
`transform_trajectory_from_noisy_to_true_local_frame` と同じ式を使っている。
egomotion ノイズが 0 なら恒等変換になる。

## 座標系

| フレーム | 定義 |
|---|---|
| `local` | 慣性系。右手系 ENU 相当。CARLA world から y 反転で得る |
| `rig` | 車体固定。x 前 / y 左 / z 上。原点は**後輪軸中心を地面に投影した点** |
| CARLA world | 左手系。x 前 / **y 右** / z 上。actor 原点は車両中心 |

変換の詳細と、CARLA 0.9.x の wheel position 単位の罠については
[COMPATIBILITY.md](COMPATIBILITY.md#座標変換の詳細) を参照。

## モジュール対応表

| ファイル | 役割 | alpasim の対応物 |
|---|---|---|
| `grpc_api/__init__.py` | 契約の単一 import 窓口 | `alpasim_grpc` |
| `geometry.py` | Pose / Trajectory と proto 変換 (numpy のみ) | `alpasim_utils.geometry` (Rust 拡張) |
| `driver/service.py` | `EgodriverService` 実装 | `alpasim_driver.main.EgoDriverService` |
| `driver/base.py` | ポリシー向け API | `alpasim_driver.models.base` |
| `runtime/carla_runtime.py` | オーケストレータ | `alpasim_runtime.worker.runtime` + `events/` |
| `runtime/world.py` | シミュレータ抽象 | `alpasim_runtime.services.*` |
| `runtime/control.py` | 軌道追従 | `controller.VDCService` |
| `runtime/route.py` | route 生成・切り出し | `alpasim_runtime.route_generator` |
| `runtime/metrics.py` | スコアリング | `alpasim_eval` |
| `compat.py` | 差分の機械可読な定義 | (対応物なし) |

## テスト戦略

`carla` は optional extra で CI には入らない。そのため `WorldAdapter` プロトコルを一枚挟み、
`FakeWorld`（自転車モデル）で差し替える。これにより **closed loop 全体が実 gRPC 越しに
CI で回る**: セッション確立、画像送信、egomotion、route、`drive`、制御適用、メトリクス。

`FakeWorld` は物理モデルではない。実際の車両ダイナミクスに依存する検証は、
実 CARLA サーバに対して行うこと（README の手動確認手順を参照）。
