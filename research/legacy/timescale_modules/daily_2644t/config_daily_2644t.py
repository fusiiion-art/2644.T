from typing import Dict, List
from common_utils.config_model import (
    ALL_YF_SYMBOLS,           # 旧名: COMMON_YF_SYMBOLS_MODEL
    ALL_FRED_SYMBOLS,         # 旧名: COMMON_FRED_SYMBOLS_MODEL
    COMMON_COLUMN_MAP,
    COMMON_FEATURE_PARAMS,
    COMMON_BLACKLIST_FEATURES,
    COMMON_TRAINING_PARAMS,
    COMMON_REGIME_EXPERTS,
    COMMON_BEST_PARAMS,
    get_merged_symbols,
    get_column_map
)
# registry_2644t.py was removed, features are defined locally below
# 後方互換性エイリアス
COMMON_YF_SYMBOLS_MODEL = ALL_YF_SYMBOLS
COMMON_FRED_SYMBOLS_MODEL = ALL_FRED_SYMBOLS

# === 1. 基本設定 ===
MODULE_NAME = "daily_2644t"
DATA_START_DATE = COMMON_TRAINING_PARAMS["start_date"]
OUTPUT_FILENAME = "features_daily_2644t.parquet"
DATA_INTERVAL = COMMON_TRAINING_PARAMS["data_interval"]

# キャッシュファイル名
YF_CACHE_FILENAME = "yf_data_daily_2644t.parquet"
FRED_CACHE_FILENAME = "fred_data_daily_2644t.parquet"

# === 2. 予測ターゲット定義 ===
# 4対象の予測を想定: T+1日始値変化, 高値, 安値, 当日終値
MAIN_ASSET = "2644t_adj_close"       # 特徴量計算用（2644.T）
OPEN_COL = "2644t_open"              # T+1日始値変化の元データ
HIGH_COL = "2644t_high"             # 高値予測用
LOW_COL = "2644t_low"               # 安値予測用
CLOSE_COL = "2644t_adj_close"       # 終値予測用
TARGET_TYPE = "multi_target"        # 'multi_target' / 'open_to_open' / 'high_low' / 'close_to_close'
TARGET_COLUMN = f"{MAIN_ASSET}_return"  # 特徴量としてのリターン（入力用）
PREDICTION_HORIZON = 1              # T日から T+1日への変化を予測
TARGET_NAV_COLUMN = MAIN_ASSET

# ★新規: タスクタイプ（回帰 or 二値分類）
TASK_TYPE = "regression"  # 'regression' or 'classification'
# ★損失関数: LightGBM では MSE / Huber が実務的
LOSS_TYPE = "mse"

# === 3. データソース定義 ===
YF_SYMBOLS_MODEL = COMMON_YF_SYMBOLS_MODEL.copy()
FRED_SYMBOLS_MODEL = COMMON_FRED_SYMBOLS_MODEL.copy()

TIINGO_SYMBOLS: Dict[str, str] = {}
STOOQ_SYMBOLS = {}

# マージ
YF_SYMBOLS, FRED_SYMBOLS = get_merged_symbols(YF_SYMBOLS_MODEL, FRED_SYMBOLS_MODEL)

# === 4. カラム名マッピング ===
ALL_SYMBOLS = YF_SYMBOLS.copy()
ALL_SYMBOLS.update(FRED_SYMBOLS)
COLUMN_MAP = get_column_map(ALL_SYMBOLS)

# === 5. 特徴量エンジニアリング設定 ===
FEATURE_PARAMS = COMMON_FEATURE_PARAMS.copy()
FEATURE_PARAMS.update({
    "lag_periods": [1, 2],
    "ma_window": 25,
    "atr_window": 14,
})

# === 6. 特徴量グループ定義 ===
FEATURE_SETS_DEFINITIONS = {
    "TARGET_LAGS": [r"^(?:semi|2644t)_.*_return_lag\d+"],
    "SOX_FEATURES": [
        r"^sox_.*_return",
        r"^SOX_Overnight_Gap",
        r"^SOX_Intraday_Force",
    ],
    "NVDA_FEATURES": [
        r"^nvda_.*_return",
        r"^NVDA_Impact_Factor",
    ],
    "TSM_FEATURES": [
        r"^tsm_.*_return",
    ],
    "VOLATILITY": [
        r"^vix_.*",
        r"^vxn_.*",
        r"^Tech_Risk_Premium",
        r"realized_vol_.*",
    ],
    "TECHNICAL_INDICATORS": [
        r"^RSI_14$",
        r".*_macd.*",
        r".*_bb_.*",
    ],
    "INTERMARKET": [
        r".*_corr_.*d$",
        r".*_ret_spread$",
        r"^Sector_Relative_Strength",
    ],
    "REPORT_ALPHA_SIGNALS": [
        r"^SOX_Overnight_Gap",         
        r"^SOX_Intraday_Force",        
        r"^Tech_Risk_Premium",         
        r"^Sector_Relative_Strength",  
        r"^NVDA_Impact_Factor",        
        r"^JP_Gap_Fill_Force",         
    ],
    "CALENDAR": [
        r"^day_of_week$",
        r"^month$",
        r"^is_holiday_jp$"
    ]
}

FEATURE_SETS = list(FEATURE_SETS_DEFINITIONS.keys())
BLACKLIST_FEATURES = COMMON_BLACKLIST_FEATURES.copy()

# ★スクリーニング結果から50%以下（ノイズ）の特徴量を除外 (82件)
SCREENING_BLACKLIST = [
    "semi_bb_upper", "semi_bb_lower", "main_ret_skew_30d", "sox_vs_vxn_corr_20d",
    "nasdaq_100_adj_close_return_lag1", "mu_adj_close_return", "ema_return_20_lag1",
    "SOX_Overnight_Gap", "vt_adj_close_return_lag1", "semi_vs_advantest_corr_20d",
    "usd_jpy_adj_close_return_lag2", "nikkei_225_adj_close_return_lag2",
    "eur_usd_adj_close_return_lag1", "semi_vs_disco_corr_20d", "vt_adj_close_return_lag2",
    "disco_adj_close_return_lag2", "semi_bb_mid", "semi_adj_close_return_lag2",
    "semi_vs_usd_jpy_corr_60d", "semi_atr", "rolling_mean_ret_20", "realized_vol_20",
    "gold_adj_close_return_lag1", "tsm_adj_close_return", "semi_vs_nikkei_225_corr_20d",
    "sox_adj_close_return_lag2", "gold_adj_close_return", "semi_vs_nvda_corr_20d",
    "disco_adj_close_return_lag1", "semi_vs_sox_corr_20d", "us_10y_yield_adj_close_return_lag1",
    "credit_spread_diff", "semi_vs_nvda_corr_60d", "nvda_adj_close_return_lag2",
    "semi_vs_nikkei_225_corr_60d", "semi_vs_tokyo_electron_corr_20d",
    "us_10y_yield_adj_close_return_lag2", "main_ret_skew_60d", "semi_adj_close_return_lag1",
    "tsm_adj_close_return_lag2", "semi_vs_tokyo_electron_corr_60d", "sox_adj_close_return_lag1",
    "usd_jpy_adj_close_return_lag1", "vxn_adj_close_return_lag2", "ema_return_5_lag2",
    "nasdaq_100_adj_close_return_lag2", "semi_vs_sox_corr_60d", "us_10y_yield_adj_close_return",
    "main_ret_kurt_60d", "semi_macd_signal", "semi_bb_bw", "semi_vs_mu_corr_20d",
    "semi_macd", "advantest_adj_close_return_lag2", "mu_adj_close_return_lag2",
    "sox_vs_vxn_corr_60d", "eur_jpy_adj_close_return_lag2", "realized_vol_10",
    "ema_return_20_lag2", "semi_vs_tsm_corr_60d", "vxn_adj_close_return_lag1",
    "ema_return_5_lag1", "semi_vs_usd_jpy_corr_20d", "eur_usd_adj_close_return_lag2",
    "main_ret_kurt_30d", "semi_adx", "gold_adj_close_return_lag2", "ema_return_10_lag1",
    "eur_jpy_adj_close_return_lag1", "us_yield_spread", "ema_return_10_lag2",
    "nikkei_225_adj_close_return_lag1", "semi_vs_advantest_corr_60d",
    "advantest_adj_close_return_lag1", "semi_vs_tsm_corr_20d", "eur_usd_adj_close_return",
    "realized_vol_60", "semi_vs_disco_corr_60d", "expected_inflation_diff",
    "month", "JP_Gap_Fill_Force", "day_of_week",
]
# BLACKLIST_FEATURES.extend(SCREENING_BLACKLIST)  # ★一時解除: 全特徴量を試す

# ★新規: 必ず選択する特徴量（RFE結果に関わらず強制追加）
# 相関分析で高い予測力を持つと判明した特徴量
# === CFI (Clustered Feature Importance) Settings ===
CFI_LINKAGE_METHOD = 'ward'
CFI_MAX_CLUSTERS = 20
CFI_IMPORTANCE_THRESHOLD = 0.0
CFI_CV_PURGE_GAP = PREDICTION_HORIZON + 1
CFI_CV_EMBARGO_GAP = 5

# === 7. モデル学習 & ハイパーパラメータ設定 ===
REGIME_VIX_COLUMN = COMMON_TRAINING_PARAMS["regime_vix_col"]
REGIME_VIX_THRESHOLD = COMMON_TRAINING_PARAMS["regime_vix_threshold"]
REGIMES = COMMON_TRAINING_PARAMS["regimes"]

# ★上書き: daily_2644tは小データ(~1000行)向けなので軽量モデルを使用
# trainer_daily_2644t.pyのMODEL_MAPと一致させる
REGIME_EXPERTS = {
    "risk_on": ["ridge", "elasticnet", "lightgbm"],
    "risk_off": ["ridge", "elasticnet", "lightgbm"],
}
EXPERTS_CONFIG: Dict[str, Dict] = REGIME_EXPERTS

SEQUENCE_LENGTH = 32 # Changed from 20 to 32 for 16-bit mode stability
SEQUENCE_LENGTH_S5 = COMMON_TRAINING_PARAMS["sequence_length_s5"]
N_TRIALS = COMMON_TRAINING_PARAMS["n_trials"]
MAX_EPOCHS = COMMON_TRAINING_PARAMS["max_epochs"]
BATCH_SIZE = COMMON_TRAINING_PARAMS["batch_size"]
# LOSS_TYPE は42行目で "directional" に設定済み（重複定義削除）
PRECISION = COMMON_TRAINING_PARAMS["precision"]
GRADIENT_CLIP_VAL = COMMON_TRAINING_PARAMS["gradient_clip_val"]
ACCUMULATE_GRAD_BATCHES = COMMON_TRAINING_PARAMS["accumulate_grad_batches"]

ALL_EXPERTS_FLAT = list(set(expert for experts in REGIME_EXPERTS.values() for expert in experts))
EXPERTS_CONFIG: List[Dict[str, str]] = [{"name": expert, "log_dir_name": expert} for expert in ALL_EXPERTS_FLAT]

STACKER_INPUT_COLUMNS: List[str] = [
    "ridge", "elasticnet", "lightgbm",
    "meta_experts_std_5d",
    "meta_ridge_abs_error_5d",
    "meta_elasticnet_abs_error_5d",
    "meta_lightgbm_abs_error_5d",
]

# === 9. 最強パラメータ (初期値) ===
BEST_PARAMS = {
    "risk_on": COMMON_BEST_PARAMS.copy(),
    "risk_off": COMMON_BEST_PARAMS.copy() 
}
