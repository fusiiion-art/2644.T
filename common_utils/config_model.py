from typing import Dict, List, Any

# ============================================================
# 共通データソース定義（一元管理）
# ============================================================
# ★全シンボルをここで一元管理。各モジュールはここから参照する。
# ★重複を防ぐため、他ファイルでのシンボル定義は禁止。

# === Yahoo Finance シンボル（全て）===
ALL_YF_SYMBOLS: Dict[str, str] = {
    # --- ターゲット ---
    "semi": "2644.T",           # 日本半導体ETF
    
    # --- 戦略の中核：先行指標 ---
    "nikkei_225": "^N225",      # 日経平均
    "nasdaq_100": "^NDX",       # 米国NASDAQ (半導体メイン)
    "sox": "^SOX",              # 米国半導体指数
    "nvda": "NVDA",             # NVIDIA (半導体の王様)
    
    # --- 為替 ---
    "usd_jpy": "JPY=X",         # ドル円
    "eur_jpy": "EURJPY=X",      # ユーロ円 (輸出企業用)
    "eur_usd": "EURUSD=X",      # ユーロドル
    
    # --- 日本半導体構成銘柄 (ETF 2644の主要銘柄) ---
    "tokyo_electron": "8035.T", # 東京エレクトロン
    "advantest": "6857.T",      # アドバンテスト
    "disco": "6146.T",          # ディスコ
    
    # --- サプライチェーン・メモリ市況 ---
    "tsm": "TSM",               # TSMC ADR (製造装置需要の先行指標)
    "mu": "MU",                 # Micron (メモリサイクル)
    
    # --- センチメント・ボラティリティ ---
    "vxn": "^VXN",              # 米国テック恐怖指数 (VIXより半導体向け)
    # "vix": "^VIX",            # 削除: VXNと重複
    
    # --- 債券・コモディティ ---
    "us_10y_yield": "^TNX",     # 米10年債利回り
    "gold": "GC=F",             # 金
    
    # --- その他 ---
    "vt": "VT",                 # 全世界株
    # "spy": "SPY",             # 削除: NASDAQ100と重複
    # "sp500": "^GSPC",         # 削除: NASDAQ100と重複
    # "crude_oil": "CL=F",      # 削除: 半導体への影響が薄い
}

# === FRED (Federal Reserve Economic Data) シンボル ===
ALL_FRED_SYMBOLS: Dict[str, str] = {
    # --- 金利 ---
    "us_10y": "DGS10",          # 米10年国債利回り
    "us_2y": "DGS2",            # 米2年国債利回り
    "effr": "FEDFUNDS",         # FF金利
    "jp_10y_yield": "IRLTLT01JPM156N",  # 日本10年債利回り
    
    # --- インフレ期待 ---
    "us_10y_breakeven": "T10YIE",      # 10年ブレークイーブン・インフレ率
    "expected_inflation": "T5YIE",      # 5年期待インフレ率
    
    # --- 信用・流動性 ---
    "credit_spread": "BAMLH0A0HYM2",   # ハイイールド債スプレッド
    "market_liquidity": "TEDRATE",      # TED Spread
    
    # --- 為替 ---
    "dollar_index": "DTWEXBGS",         # ドルインデックス
}

# === カラム名マッピング（シンボル→カラム名）===
COMMON_COLUMN_MAP: Dict[str, str] = {
    ticker.lower(): internal_name 
    for internal_name, ticker in {**ALL_YF_SYMBOLS, **ALL_FRED_SYMBOLS}.items()
}
COMMON_COLUMN_MAP.update({
    'adj close': 'adj_close',
    'open': 'open',
    'high': 'high',
    'low': 'low',
    'close': 'close',
    'volume': 'volume'
})

# ============================================================
# 後方互換性のためのエイリアス（新規コードでは ALL_* を使用推奨）
# ============================================================
COMMON_YF_SYMBOLS_MODEL = ALL_YF_SYMBOLS
COMMON_FRED_SYMBOLS_MODEL = ALL_FRED_SYMBOLS

# === 共通特徴量パラメータ ===
COMMON_FEATURE_PARAMS: Dict[str, Any] = {
    "REQUIRED_COLUMNS": {
        "semi_adj_close",
        "nikkei_225_adj_close", 
        "nasdaq_100_adj_close",  # sp500から変更
        "sox_adj_close", 
        "nvda_adj_close",
        "usd_jpy_adj_close",   
        "vxn_close",  # vixからvxnに変更
        # 日本構成銘柄
        "tokyo_electron_adj_close",
        "advantest_adj_close",
        "disco_adj_close",
        # サプライチェーン・メモリ
        "tsm_adj_close",  # TSMC
        "mu_adj_close",   # Micron
    },

    "rsi_window": 14,
    "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
    "bb_window": 20, "bb_std": 2,
    "adx_window": 14,
    "atr_window": 14,
    
    # Intermarket Features
    "intermarket_corr_windows": [20, 60], 
    "intermarket_pairs": [
        ("semi_adj_close", "sox_adj_close", "return_vs_return"),
        ("semi_adj_close", "nvda_adj_close", "return_vs_return"),
        ("semi_adj_close", "nikkei_225_adj_close", "return_vs_return"),
        ("semi_adj_close", "usd_jpy_adj_close", "return_vs_return"),
        ("sox_adj_close", "vxn_close", "return_vs_raw"),  # vix→vxn
        # 日本構成銘柄との相関
        ("semi_adj_close", "tokyo_electron_adj_close", "return_vs_return"),
        ("semi_adj_close", "advantest_adj_close", "return_vs_return"),
        ("semi_adj_close", "disco_adj_close", "return_vs_return"),
        # ★新規: サプライチェーン(先行指標)
        ("semi_adj_close", "tsm_adj_close", "return_vs_return"),  # TSMC
        ("semi_adj_close", "mu_adj_close", "return_vs_return"),   # Micron
    ],
    
    "rolling_ret_windows": [5, 10, 20],
    "ema_spans": [5, 10, 20],
    "vol_windows": [10, 20, 60],
    "statistical_windows": [30, 60],
    
    "trend_window": 30, 
    "regime_window": 90,
    "norm_window": 252, 
    "REGIME_VIX_COLUMN": "vxn_close",  # vix→vxn (テック恐怖指数)
    "REGIME_MA_WINDOW": 200,
}

# === 共通除外特徴量 ===
# 重複削除: sp500関連、vix関連、crude_oil関連を完全除外
COMMON_BLACKLIST_FEATURES: List[str] = [
    # 削除したシンボルの派生特徴量
    "sp500_adj_close_return",
    "sp500_adj_close",
    "vix_adj_close_return",
    "vix_adj_close",
    "vix_close",
    "spy_adj_close_return",
    "crude_oil_adj_close_return",
    "crude_oil_adj_close",
    # 既存の除外（生リターンは除外、相関は使用）
    "nasdaq_100_adj_close_return",
    "sox_adj_close_return", 
    "nvda_adj_close_return",
    "vt_adj_close_return",
    "vxn_adj_close_return",
    "gold_adj_close_return",
    "us_10y_yield_adj_close_return",
    "eur_usd_adj_close_return", 
    "usd_jpy_adj_close_return",
    "eur_jpy_adj_close_return",
    # 日本構成銘柄のリターン（相関のみ使用）
    "tokyo_electron_adj_close_return",
    "advantest_adj_close_return",
    "disco_adj_close_return",
    # サプライチェーン（相関のみ使用）
    "tsm_adj_close_return",
    "mu_adj_close_return",
]

# === 共通学習パラメータ ===
COMMON_TRAINING_PARAMS = {
    "start_date": "2021-09-28",
    "data_interval": "1d",
    "regime_vix_col": "vxn_close",  # vix→vxnに変更
    "regime_vix_threshold": 22.0,  # ★変更: 20→22（データバランス改善）
    "regimes": ["risk_on", "risk_off"],
    "sequence_length": 32,
    "sequence_length_s5": 120,
    "n_trials": 2000,  # ★変更: 500→1000（より広範な探索）
    "max_epochs": 100,
    "batch_size": 64,
    "precision": 32,               # 精度: 16 or 32
    "gradient_clip_val": 1.0,      # 勾配クリッピング閾値
    "accumulate_grad_batches": 4,  # 勾配蓄積バッチ数（本番学習用）
}

# === 共通REGIME_EXPERTS (TCN/BDH -> TimesNet/iTransformer/TimeMixer) ===
COMMON_REGIME_EXPERTS = {
    "risk_on": ["timesnet", "itransformer", "timemixer", "s5", "patchtst"],
    "risk_off": ["timesnet", "itransformer", "timemixer", "s5", "patchtst"], 
}

# === 共通Hyperparameters (BEST_PARAMS デフォルト値) ===
# 各モデルの推奨初期パラメータ
COMMON_BEST_PARAMS = {
    "timesnet": {
        "d_model": 32, "d_ff": 64, "top_k": 3, "num_kernels": 6, "n_layers": 2, 
        "lr": 0.001, "dropout": 0.1, "max_epochs": 50
    },
    "itransformer": {
        "d_model": 128, "n_heads": 4, "d_ff": 256, "n_layers": 2, 
        "lr": 0.001, "dropout": 0.1, "max_epochs": 50
    },
    "timemixer": {
        "d_model": 32, "down_sampling_window": 2, "down_sampling_layers": 2,
        "lr": 0.001, "dropout": 0.1, "max_epochs": 50
    },
    "s5": {
        "d_model": 64, "n_layers": 4, "ssm_size": 32, "dropout": 0.2, 
        "lr": 0.001, "max_epochs": 60, "initialization": "hippo"
    },
    "patchtst": {
        "d_model": 64, "n_layers": 3, "n_heads": 4, "d_ff": 128,
        "lr": 0.001, "dropout": 0.2, "max_epochs": 50
    }
}

def get_merged_symbols(model_yf_symbols: Dict, model_fred_symbols: Dict):
    """モジュール固有のシンボルと共通シンボルをマージする（後方互換性用）"""
    # ALL_*には既に全シンボルが含まれているため、model_*は空でも動作する
    yf_symbols = {**ALL_YF_SYMBOLS, **model_yf_symbols}
    fred_symbols = {**ALL_FRED_SYMBOLS, **model_fred_symbols}
    return yf_symbols, fred_symbols

def get_column_map(all_symbols: Dict):
    col_map = COMMON_COLUMN_MAP.copy()
    model_col_map = {
        ticker.lower(): internal_name 
        for internal_name, ticker in all_symbols.items()
        if ticker.lower() not in col_map 
    }
    col_map.update(model_col_map)
    generic_keywords = ['adj close', 'open', 'high', 'low', 'close', 'volume']
    return {k: v for k, v in col_map.items() if k not in generic_keywords}
