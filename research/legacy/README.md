# research/legacy

設計書 v2 フェーズ3「旧モデル群を research/ に移す」で本流から外したコード。保守しない（2026-09-30、ユーザー了承）。

- 本流の import パスから外したので、ここにあるファイルはそのままでは動かない（相互の import も元のパスのまま）。
- 本番に戻す条件（設計書 v2 モデル構成）: 「Huber＋浅い LightGBM の2本の平均を、コスト込み DSR で上回る」ことを示せたときだけ。
- 旧パイプラインで本流に残したもの: `models/linear_model.py`・`models/lightgbm_model.py`（ridge / elasticnet / lightgbm）、
  `common_utils/trainer_model.py`、`common_utils/predicter_model.py`（ExpertPredicter のみ）、
  `analysis/walk_forward_validator.py`（監査 G-9 の再学習つき版）、`audit/` の再現スクリプト。

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
