# 2644.T

2644.T の翌朝判断の予測システムとして始めたプロジェクト。2026-10-01 に予測モデルの開発を終え、運用は一定額保有に決めた。

## 現状（2026-10-01）

- 予測モデル: 開発終了。フェーズ4の判定で不合格（DSR 0.008、PBO 0.69、毎朝買うだけのやり方にも、買う確率を合わせたランダム売買にも負けた）
- 運用: 84万円分を一度買い、買い足さず売らずに持ち続ける。1年たって買値を下回っていても売らない。値は `semi2644/config/config.yaml` の v2 節（`fixed_*`）
- 経緯と数字: `research/trial_log/`（`2026-10-01_phase4.md`・`2026-10-01_rule_variants.md`・`2026-10-01_fixed_amount.md` など）

## 構成

| 場所 | 中身 |
|---|---|
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

検証の再実行（データは `data/cache/` を使う。結果は `research/trial_log/` に書く）:

```bash
c:/2644.T/.venv/Scripts/python.exe research/phase4_eval.py
c:/2644.T/.venv/Scripts/python.exe research/rule_variants_report.py
c:/2644.T/.venv/Scripts/python.exe research/fixed_amount_sizing.py
```

## 免責事項

本プロジェクトは情報提供・研究目的であり、投資助言ではありません。
