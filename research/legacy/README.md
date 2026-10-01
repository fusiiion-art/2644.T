# research/legacy

本流から外した旧予測パイプライン。保守しない。2回に分けて移した。

- ファイルは元の場所と同じ相対パスに置いてある（例: `common_utils/trainer_model.py` → `research/legacy/common_utils/trainer_model.py`）。
  ただし本流の import パスから外したので、そのままでは動かない（相互の import も元のパスのまま）。
- `audit/` の再現スクリプト（`check_data.py`・`wf_ridge_eval.py`）はここにあるモジュールを元のパスで import するので、今のままでは動かない。
- 旧パイプラインの学習結果（`logs/`）は、監査報告が参照するので元の場所に残した。

## 1回目（2026-09-30、設計書 v2 フェーズ3「旧モデル群を research/ に移す」、ユーザー了承）

本番に戻す条件は「Huber＋浅い LightGBM の2本の平均を、コスト込み DSR で上回る」こととしていたが、
その v2 モデル自体がフェーズ4で不合格になったので、この条件は意味を失った。

| 移したもの | 元の場所 |
|---|---|
| DLinear / NLinear | `models/dlinear_model.py` |
| TCN + Attention | `models/tcn_attention_model.py` |
| 深層モデル（iTransformer・PatchTST・S5・TimeMixer・TimesNet） | `models/archive_heavy/` |
| dynamic_ensemble（AttentionGate） | `common_utils/dynamic_ensemble.py` |
| supervisor（MetaLearner・SupervisorEnsemble） | `common_utils/supervisor_model.py` |
| meta_model（スタブ） | `trading_pipeline/meta_model.py` |
| 日次 predicter（supervisor 統合・可視化） | `common_utils/predicter_model.py` の DailyPredicter 以下、`timescale_modules/daily_2644t/predicter_daily_2644t.py` |
| supervisor 学習・指値コントローラ | `timescale_modules/daily_2644t/supervisor_daily_2644t.py` |
| 学習済みアセットの読み込み | `common_utils/asset_loader.py` |
| SHAP ツール（supervisor・深層モデル用） | `analysis/shap_analyzer.py`、`analysis/feature_suggester.py` |
| テスト | `tests/test_dynamic_ensemble.py` |

## 2回目（2026-10-01、予測モデルの開発終了後の片付け、ユーザー依頼）

フェーズ4でモデルが不合格になり、運用は一定額保有に決まったので、1回目で本流に残した旧パイプラインもすべて移した。

| 移したもの | 元の場所 |
|---|---|
| CLI | `main.py` |
| 日次モジュール（設定・学習） | `timescale_modules/`（`config_daily_2644t.py`・`trainer_daily_2644t.py`） |
| 売買パイプライン | `trading_pipeline/` |
| モデル（ridge / elasticnet / lightgbm） | `models/` |
| 評価・分析（監査 G-9 の再学習つきウォークフォワードを含む） | `analysis/` |
| 学習・推論・特徴量の共通部品 | `common_utils/` の `base_model`・`clustered_importance`・`config_model`・`cusum_filter`・`drift_detector`・`feature_base`・`feature_registry`・`limit_engine`・`metrics`・`purged_cv`・`regime_detector`・`risk_manager`・`trainer_model`・`triple_barrier` |
| ExpertPredicter | `common_utils/predicter_model.py` → `common_utils/predicter_model_expert.py`（1回目に移した同名ファイルと区別するため改名） |
| 旧ターゲット生成（create_targets など） | `common_utils/target_utils.py` の v2 以外の部分（v2 のラベルは本流に残した） |
| 旧特徴量生成 | `generate/` の `alternative_data`・`check_data`・`feature_engine_core`・`features_2644t`・`features_daily_features` |
| 旧特徴量ファイル | `data/features_daily_2644t.parquet` |
| テスト（14ファイル） | `tests/` の `test_clustered_importance`・`test_cusum_filter`・`test_feature_base`・`test_limit_engine`・`test_main_cli`・`test_multi_target_pipeline`・`test_prediction_output`・`test_prediction_smoke`・`test_purged_cv`・`test_rfe_deprecation`・`test_trainer_integration`・`test_trainer_model`・`test_triple_barrier`・`test_walk_forward_oof` |

v2 モデル（`trainer_v2`・`signal_rule`）は `research/v2_model/` に移した。こちらはフェーズ3の結果を再現できるように動く状態で残し、テスト（`tests/test_cpcv_v2.py`）も本流で回している。
