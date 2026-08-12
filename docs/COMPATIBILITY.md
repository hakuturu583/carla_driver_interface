# alpasim との互換性 / Compatibility with alpasim

対象 upstream: [NVlabs/alpasim](https://github.com/NVlabs/alpasim) `68709245a5dc0f2eda4f8cb2c3aa8cbdfa913043`
(`alpasim_grpc` 0.55.0)

このドキュメントは散文だが、**同じ内容が
[`src/carla_driver_interface/compat.py`](../src/carla_driver_interface/compat.py) に機械可読な形で
入っている**。両者がずれると `tests/test_proto_compat.py::test_compat_entries_are_mirrored_in_the_docs`
が落ちる。手元で確認するには:

```console
$ uv run carla-driver-interface compat-report
$ uv run carla-driver-interface compat-report --json
```

互換度の語彙:

| レベル | 意味 |
|---|---|
| **exact** | upstream と同一。テストで検証済み |
| **partial** | 同じメッセージだが挙動が狭い / 意味が違う |
| **structural** | driver から見た契約は同じ。裏の実装が別物 |
| **extension** | upstream が用意した拡張点に載せた追加情報。無視しても壊れない |
| **unimplemented** | upstream にあってこちらに無い |

---

## exact — 完全互換

| Area | alpasim | carla_driver_interface |
|---|---|---|
| **Egodriver service** | `egodriver.EgodriverService` の 8 RPC | upstream 生成 servicer を継承して 8 本すべて実装。サービスパス `/egodriver.EgodriverService/*` もメッセージ型も同一 |
| **Message types** | `alpasim_grpc` で定義 | 同パッケージを rev 固定の git 依存として使用。メッセージを一つも再定義しない |
| **Observation ordering** | 画像 → egomotion → route → GT を送り切ってバリア、その後 `drive()` | 同じ順序・同じバリア |
| **Response frame** | `DriveResponse.trajectory` は `local -> rig_est`、先頭が `time_now_us` の自車 pose | 同一。servicer が `time_now_us` の自車 pose を anchor にして rig→local 変換する |
| **Early termination** | `DriveResponse.terminate_session` で即終了 | 対応。残りのステップを実行せずに抜ける |
| **API version** | `get_version` が `alpasim_grpc.API_VERSION_MESSAGE` を返す | そのまま転送。runtime 側は起動時に照合し、不一致なら警告 |
| **Rollout results** | `SimulationReturn.RolloutReturn` | 同じメッセージを生成するので alpasim 側ツールでそのまま読める |

**なぜ壊れないか**: servicer は `alpasim_grpc.v0.egodriver_pb2_grpc.EgodriverServiceServicer` を
Python 継承している。ダックタイピングではないので、upstream が RPC を増やせば
`test_all_upstream_rpcs_exist_and_are_implemented` が落ちて気付ける。

---

## partial — メッセージは同じ、挙動が違う

| Area | alpasim | carla_driver_interface |
|---|---|---|
| **Camera model** | ftheta / OpenCV fisheye / OpenCV pinhole | **pinhole のみ** (`OpenCVPinholeCameraParam`)。CARLA は理想 pinhole を描画するので ftheta 係数を正直に埋められない。歪み係数は空のまま置く |
| **Shutter** | ローリングシャッター (`frame_start_us` != `frame_end_us`) | **グローバルシャッター** (`frame_start_us` == `frame_end_us`)。`ShutterType.GLOBAL` |
| **Recording ground truth** | `submit_recording_ground_truth` は実車の走行記録 | 記録が存在しない。**既定で送らない**。`send_ground_truth=True` にすると route 参照軌跡を送るが、これは別物 |
| **Egomotion error model** | 自己位置推定誤差モデルで `rig_est` が `rig` からずれる | 既定は恒等。`egomotion_position_noise_m` / `egomotion_yaw_noise_rad` でガウスノイズを入れると、ずれと runtime 側の補正の両方が upstream と同じ経路で再現される |
| **scene_id** | 録画クリップの UUID | `"<map>:<scenario>"`。同定すべき録画が無い |

ftheta を要求する driver（alpasim の `alpasim_driver` は rectification に使う）は、
CARLA 側の pinhole を受け取ることになる。多くの実装では pinhole が rectification のターゲット
形式なので実害は小さいが、ftheta 前提のコードパスは通らない。

---

## structural — 契約は同じ、実装が別物

| Area | alpasim | carla_driver_interface |
|---|---|---|
| **Renderer** | `SensorsimService`: NRE ニューラル再構成を gRPC 越しに | CARLA のラスタライザを in-process で。sensorsim の RPC は一度も呼ばない。カメラ記述のためにメッセージ型だけ再利用する |
| **Controller / vehicle model** | `VDCService` を gRPC 越しに呼んで軌道を運動に変換 | in-process の pure pursuit + 速度 PID。その先は CARLA 自身の車両ダイナミクス |
| **Physics** | `PhysicsService` が地面交差補正 | CARLA の物理エンジンが接地を保つので RPC 不要 |
| **Traffic** | `TrafficService` が他エージェントを模擬 | CARLA TrafficManager |
| **Coordinate frames** | 右手系 ENU の `local`、rig 原点は後輪軸中心の接地投影 | CARLA は左手系で actor 原点が車両中心。変換層が y を反転し、rig を後輪軸へずらす |
| **Orchestration** | asyncio の `RuntimeService` デーモン。多数 rollout を driver レプリカに負荷分散 | 同期の in-process クラスで 1 rollout。並列化は呼び出し側の責任 |

### 座標変換の詳細

CARLA: x 前 / **y 右** / z 上、回転は度の `(pitch, yaw, roll)`。
alpasim: 右手系、rig は x 前 / **y 左** / z 上。

```
位置: (x, y, z)          -> (x, -y, z)
回転: (pitch, yaw, roll) -> (-pitch, -yaw, roll)   [度 -> ラジアン]
```

これは y 軸に関する鏡映 `M = diag(1,-1,1)` で `R_alpasim = M R_carla M` と書ける。
`tests/test_conversions.py::test_rotation_matches_the_mirrored_carla_matrix` が、
CARLA 本家の `Transform.get_matrix()` を再現した行列と突き合わせて検証している。

**rig 原点**: `RuntimeConfig.rear_axle_offset_m` で明示指定できる。未指定なら
`get_physics_control().wheels` から導出するが、**CARLA 0.9.x では
`WheelPhysicsControl.position` はワールド座標かつセンチメートル**という癖があるため、
導出値は必ず起動時にログへ出す。値が怪しければ config で上書きすること。

---

## extension — upstream の拡張点に載せた追加情報

| Area | alpasim | carla_driver_interface |
|---|---|---|
| **Renderer payload** | `DriveRequest.renderer_data` は自由形式（NRE 固有） | `carla_driver.v0.CarlaRendererData` をシリアライズして格納（map / weather / 信号 / 制限速度 / 他 actor）。無視する driver には影響なし |
| **Driver debug payload** | `DebugInfo.unstructured_debug_info` は自由形式 | `carla_driver.v0.CarlaDriveDebugInfo` を格納 |

`proto/carla_driver/v0/carla_driver.proto` は **service を一つも定義しない**。
定義すれば driver と話す二つ目の非互換な経路ができてしまう
(`test_our_proto_declares_no_service` が防いでいる)。
拡張メッセージは upstream 型を `import` して合成する（コピーしない）ので、
`CarlaDriveSessionInfo.base` の型は upstream の `egodriver.DriveSessionRequest` そのもの。

パースは寛容にしてある: upstream の runtime が `renderer_data` に別のものを入れてきても、
driver 側は `None` として扱って走り続ける (`test_foreign_renderer_data_is_ignored_not_fatal`)。

---

## unimplemented — upstream にあってこちらに無い

| Area | alpasim | carla_driver_interface |
|---|---|---|
| **Structured logging** | `logging.proto` が全 request/response を ASL ログへ記録 | 未実装。標準 `logging` モジュールのみ |
| **Video model** | `video_model.proto` で生成動画モデルを駆動 | 未実装 |
| **LiDAR** | `SensorsimService.render_lidar` | 未実装。**egodriver の契約に LiDAR 送信 RPC が無い**ので、生成しても届け先が無い |
| **Runtime gRPC surface** | `RuntimeService.simulate` / `prefetch_scene` / `get_runtime_info` / `shut_down` | gRPC サーバとしては提供しない。`CarlaRuntime` は Python クラスとして使う |

---

## 相互運用の実際

**この driver を alpasim runtime から使う**: そのまま動くはず。サービス名・メッセージ・
observation の順序が同一で、`get_version` も upstream の API バージョンを返す。
ただし CARLA 拡張は届かないので、`RouteFollowerPolicy` は信号や制限速度を見ずに走る。

**alpasim の driver を `CarlaRuntime` から使う**: `alpasim_driver` を別プロセスで立て、
`--driver <host>:<port>` を指すだけ。上の partial 項目（pinhole のみ・グローバルシャッター・
GT 無し）が効いてくるので、ftheta 前提のモデルは事前に確認すること。

**同一プロセスで本家 `alpasim_grpc` と共存**: 問題ない。本家をそのまま依存として使っており、
descriptor を複製していないため。
