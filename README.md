# 2644.T

2644.T の翌朝判断の予測システムとして始めたプロジェクト。2026-10-01 に予測モデルの開発を終え、運用は予測を使わない売買ルールの「朝の合図」にした。

## 現状（2026-10-01）

- 予測モデル: 開発終了。フェーズ4の判定で不合格（DSR 0.008、PBO 0.69、毎朝買うだけのやり方にも、買う確率を合わせたランダム売買にも負けた）
- 運用: 売買ルールの朝の合図（`ops/`）。毎朝買う・前日終値が平均取得価格の4%下なら買い増し・平均取得価格×1.02 の全口売り指値、上限300口。使い方は `ops/README.md`、値は `semi2644/config/config.yaml` の `ops` 節と `v2` 節の `position_*`
- 一時は一定額保有（84万円を買ったまま持つ）に決めたが、同じ日に売買ルールの合図に戻した（記録は残してある）
- 経緯と数字: `research/trial_log/`（`2026-10-01_phase4.md`・`2026-10-01_rule_variants.md`・`2026-10-01_fixed_amount.md`・`2026-10-01_operation_rule.md` など）

## 構成

| 場所 | 中身 |
|---|---|
| `ops/` | 朝の合図（`morning_signal.py`・`run_morning.bat`）、約定記録の書式（`fills.example.csv`）、使い方（`README.md`） |
| `common_utils/morning_signal.py` | 約定記録から保有状態を作り、今日の注文を出す（シミュレータと同じ売買になることをテスト済み） |
| `data/adjust.py` | 価格の補正（分割）。補正はここ1か所だけ |
| `data/cache/` | Yahoo・FRED の取得キャッシュ |
| `common_utils/data_fetcher.py` | データ取得（キーは環境変数か `semi2644/config/secrets.env`） |
| `generate/features_v2.py` | 判断時刻（D の 08:50 JST）基準のデータセット（特徴量・価格・ラベル） |
| `common_utils/target_utils.py` | v2 のラベル |
| `common_utils/position_simulator.py`・`rule_risk.py`・`fixed_amount.py` | 運用ルールと一定額保有の時価評価・リスク測定 |
| `common_utils/selection_bias.py`・`cpcv.py` | DSR・PBO・CPCV（過学習の判定） |
| `research/*.py` | 検証スクリプト（フェーズ2〜4、ルールのリスク、一定額保有の保有額） |
| `research/trial_log/` | 試行の記録と結果 |
| `research/v2_model/` | v2 モデル（不合格）。フェーズ3の結果を再現できるように動く状態で残す |
| `research/legacy/` | 旧予測パイプライン。保守しないし、そのままでは動かない |
| `audit/` | 監査報告（2026-09-30）と再現スクリプト。再現スクリプトは旧パイプラインに依存する |
| `semi2644/` | 設定（`config/config.yaml`・`config/corporate_actions.yaml`）と東証カレンダー |
| `logs/` | 旧パイプラインの学習結果（監査報告が参照するので残す） |
| `tests/` | テスト |

## 使い方

テスト:

```bash
c:/2644.T/.venv/Scripts/python.exe -m pytest -q tests
cd semi2644 && c:/2644.T/.venv/Scripts/python.exe -m pytest -q
```

朝の合図（PC で平日 8:30。手順は `ops/README.md`）:

```bash
c:/2644.T/ops/run_morning.bat
```

検証の再実行（データは `data/cache/` を使う。結果は `research/trial_log/` に書く）:

```bash
c:/2644.T/.venv/Scripts/python.exe research/phase4_eval.py
c:/2644.T/.venv/Scripts/python.exe research/rule_variants_report.py
c:/2644.T/.venv/Scripts/python.exe research/fixed_amount_sizing.py
```

## 免責事項

本プロジェクトは情報提供・研究目的であり、投資助言ではありません。
