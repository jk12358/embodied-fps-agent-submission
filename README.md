# Embodied FPS Agent

MuJoCo のロボットアームの先端に取り付けたマウスを机の上で動かし、FPS の射撃場（[ValTrainer](https://github.com/idoraz1/ValTrainer) 互換の仕様で作った一人称 3D 射撃場）のボットを撃つシステムです。

AI に渡すのは一人称の画面画像だけです。ボットの 3D 座標や正解の角度は、AI にも制御にも渡しません（命中判定と集計にだけ使います）。

## 仕組み

1. **画面を見る**：一人称の画面画像（960×540 の PNG）を Claude API（`claude-haiku-5-5`）に渡し、ボットの頭の位置を画像座標で答えさせます。
2. **視点を回す量を求める**：水平視野角 103° のカメラモデルで、画像座標を視点の回転角（Yaw / Pitch）に直します。800 DPI・感度 0.35（マウスを 46.65 cm 動かすと 1 回転）の条件で、必要なマウスの移動量を計算します。
3. **マウスを動かす**：2 自由度のアームが逆運動学と PD 制御でマウスを机の上で動かします。マウスが机に接している間（または光学センサの浮き上がり許容 1.0 mm 以内）だけ、移動量を視点の回転に変えます。
4. **クリックする**：指先の 1 自由度の部品が左ボタンを押し込みます。2.0 mm 以上押し込んで離すと 1 発の射撃になります。

左右に動くボットには、色マスクで頭を追う学習なしの高速制御（`ShootingGSM`）を使います。アームの移動とクリックには時間がかかるため（平均 0.47 秒）、そのままでは撃つときにボットがずれた位置にいます。そこで Round 1 の計測値（動作時間・角度誤差・追従誤差）と画面画像を Claude Haiku に渡し、予測時間（`lead_seconds`）を含む 3 つの数値を 1 回だけ返させて Round 2 で使います。予測時間を動作時間に合わせるという方針はプロンプトで与えています。これは [GameSense（Lu et al., ACL 2025）](https://aclanthology.org/2025.acl-long.643/) の「遅い VLM が速い制御を調整する」という考え方を簡略化したもので、本家のように学習済みモデルを使ったり、VLM が制御コードを書き直したりはしません。

## 結果（固定シード `seed = 7`）

| 課題 | 条件 | 命中 | 頭部命中 | 平均角度誤差 | API 費用 |
| --- | --- | ---: | ---: | ---: | ---: |
| 止まっているボット 20 体 | Claude Haiku が直接指定 | 20 / 20 | 20 / 20 | 0.183° | US$0.003915 |
| 左右に動くボット 8 体 | デジタル入力（遅れなし） | 8 / 8 | 8 / 8 | 0.112° | — |
| 左右に動くボット 8 体 | 物理マウス、補正なし（Round 1） | 0 / 8 | 0 / 8 | 5.023° | — |
| 左右に動くボット 8 体 | 物理マウス、Claude Haiku が予測時間を 0.4703 秒に設定（Round 2） | 8 / 8 | 7 / 8 | 0.328° | US$0.000299 |

物理クリックはすべての試行で成功しました。今回確認したのは、射撃場で 1 体ずつ撃つ課題（止まっているボットと左右に動くボット）です。連射の反動、複数のボットの同時出現、実際のゲームクライアントや実戦のマップでの評価はしていません。

## 実行方法

Python 3.12 と [uv](https://docs.astral.sh/uv/) を使います。

```bash
uv sync --locked --all-groups
uv run pytest -q
```

API を使わない確認（射撃場のプレビュー画像と、高速制御だけでの射撃）：

```bash
uv run python scripts/simulation/preview_range.py
uv run python scripts/evaluation/run_range.py --mode gamesense_gsm --drill head_flicks --targets 10 --render
uv run python scripts/evaluation/run_range.py --mode gamesense_gsm --drill strafe_bots --targets 10 --lead-seconds 0.47 --render
```

Claude API を使う評価では、`.env.example` を参考に `.env` を作り、`ANTHROPIC_API_KEY` を設定します。実行前に入力トークンを数えて料金を見積もり、予算（`--budget-usd`）を超える場合は呼び出しません。使ったトークンと料金は出力ディレクトリの `api-costs.json` に記録します。

```bash
# 見積もりだけ（生成しない）
uv run python scripts/evaluation/run_range.py --mode vlm_direct --targets 1 --estimate-only
# 止まっているボットを Claude Haiku が直接狙う
uv run python scripts/evaluation/run_range.py --mode vlm_direct --drill head_flicks --targets 20 --max-calls 20 --budget-usd 0.03
# 動くボット：デジタル / 物理マウス補正なし / Claude Haiku が設定した予測時間 の比較
uv run python scripts/evaluation/run_range.py --mode gamesense_tune --drill strafe_bots --targets 8 --max-calls 2 --budget-usd 0.02 --render
```

結果は `outputs/evaluation/` に保存されます（`--render` を付けると、アームと一人称画面を並べた録画 `session.mp4` も保存します）。

## 構成

- `assets/mujoco/mounted_mouse.xml`：アーム・机・マウス・クリック用の指の MuJoCo シーン
- `configs/range.toml`：画面、マウス感度、制御、射撃場、API 予算の設定
- `src/embodied_fps_agent/`
  - `robot.py`：アームの逆運動学、机との接触、ボタンの状態
  - `fps_bridge.py`：感度の換算（cm/360°）と、机の上の移動量から視点の回転角への変換
  - `range_env.py`：一人称 3D 射撃場の描画と命中判定（頭は半径 0.14 m の球、胴は円柱）
  - `episode.py`：物理マウスでのフリックとクリック、比較用のデジタル入力
  - `policy.py`：Claude Haiku による照準と数値の設定、色マスクで頭を追う高速制御
  - `costs.py`：トークンの見積もり、予算の検査、料金の記録
  - `recording.py` / `runner.py`：録画と評価のループ
- `scripts/`：実行の入口（`simulation/preview_range.py`、`evaluation/run_range.py`）
- `tests/`：単体テスト
