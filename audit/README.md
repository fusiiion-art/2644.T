# audit

2026-09-30 の監査（A〜F）。報告は `AUDIT_REPORT.md`、数値の出どころは `results/`。

## 再現スクリプトの状態（2026-10-01 の片付け以後）

旧予測パイプラインを `research/legacy/` へ移したので、旧パイプラインを import するスクリプトは今のままでは動かない。
報告と `results/` の数値は監査時点のもので、変わらない。

| スクリプト | 状態 |
|---|---|
| `optuna_trials.py` | 動く（`logs/optuna_db/` を一時ディレクトリにコピーして読むだけ） |
| `summarize_wf.py` | 動く（`results/` の集計だけ） |
| `check_data.py` | 動かない（`timescale_modules.daily_2644t.config_daily_2644t` を import する） |
| `wf_ridge_eval.py` | 動かない（`timescale_modules` と `models.linear_model` を import する） |

動かない2本を再現したい場合は、監査時点のコミット（main の 8384ee4）を checkout して実行する。
