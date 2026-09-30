"""
共通ターゲット生成関数
- create_targets: Close→Closeリターン
- create_open_targets: Open-to-Open/Close
- create_binary_targets: 二値分類用

注: 現在の主要方針は T+1 を対象とするため、5日先用の実装は削除済み。
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Optional
from common_utils.cusum_filter import symmetric_cusum_filter
from common_utils.triple_barrier import TripleBarrierLabeler


def create_targets(df: pd.DataFrame, horizons: List[int], target_col: str) -> pd.DataFrame:
    """
    従来のClose→Closeリターンターゲット生成（後方互換性用）
    
    Args:
        df: データフレーム
        horizons: 予測ホライズンのリスト（例: [1, 5]）
        target_col: ターゲット列名（リターン列）
    
    Returns:
        target_1 等の列が追加されたデータフレーム
    """
    if not target_col or target_col not in df.columns:
        logging.error(f"Target source column '{target_col}' not found. Cannot create targets.")
        return df

    for h in horizons:
        df[f'target_{h}'] = df[target_col].rolling(window=h, min_periods=1).sum().shift(-h)
    
    logging.info(f"Successfully created target columns for horizons: {horizons}")
    return df


def create_open_targets(
    df: pd.DataFrame,
    horizons: List[int],
    open_col: str,
    target_type: str = 'open_to_open',
    high_col: Optional[str] = None,
    low_col: Optional[str] = None,
    close_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Open/High/Low/Close ベースのターゲット生成。

    Args:
        df: データフレーム
        horizons: 予測ホライズンのリスト（例: [1, 5]）
        open_col: 始値列名（例: 'semi_open'）
        target_type:
            'open_to_open' - T日始値 → T+H日始値
            'open_to_close' - T日始値 → T日終値
            'high_low' - T日高値/安値をターゲットとする（現在は高値/安値の変化率を返す）
            'close_to_close' - T日終値 → T+H日終値
    Returns:
        target_1 等の列が追加されたデータフレーム
    """
    if open_col not in df.columns:
        logging.error(f"Open column '{open_col}' not found. Cannot create open-based targets.")
        return df

    epsilon = 1e-9
    if close_col is None:
        close_col = open_col.replace('_open', '_close').replace('_open', '_adj_close')
        if close_col not in df.columns:
            close_col = open_col.replace('_open', '_adj_close')
        if close_col not in df.columns:
            logging.warning(f"Close column for '{open_col}' not found. Using open-to-open only.")
            close_col = None

    if high_col is None:
        high_col = open_col.replace('_open', '_high')
    if low_col is None:
        low_col = open_col.replace('_open', '_low')

    for h in horizons:
        if target_type == 'open_to_open':
            future_open = df[open_col].shift(-h)
            current_open = df[open_col]
            df[f'target_{h}'] = np.log((future_open + epsilon) / (current_open + epsilon))
            logging.info(f"Created target_{h}: {open_col} -> {open_col} (T+{h}日) [open_to_open]")

        elif target_type == 'open_to_close' and close_col:
            if h == 1:
                df[f'target_{h}'] = np.log((df[close_col] + epsilon) / (df[open_col] + epsilon))
            else:
                future_close = df[close_col].shift(-(h-1))
                df[f'target_{h}'] = np.log((future_close + epsilon) / (df[open_col] + epsilon))
            logging.info(f"Created target_{h}: {open_col} -> {close_col} [open_to_close]")

        elif target_type == 'close_to_close' and close_col:
            future_close = df[close_col].shift(-h)
            current_close = df[close_col]
            df[f'target_{h}'] = np.log((future_close + epsilon) / (current_close + epsilon))
            logging.info(f"Created target_{h}: {close_col} -> {close_col} (T+{h}日) [close_to_close]")

        elif target_type == 'high_low' and high_col in df.columns and low_col in df.columns:
            current_high = df[high_col]
            current_low = df[low_col]
            future_high = df[high_col].shift(-h)
            future_low = df[low_col].shift(-h)
            # 高値・安値の方向性を捉えるため、未来の高値/安値比率を単一ターゲットにまとめる
            df[f'target_{h}'] = np.log((future_high + epsilon) / (current_high + epsilon)) - np.log((future_low + epsilon) / (current_low + epsilon))
            logging.info(f"Created target_{h}: {high_col}/{low_col} spread (T+{h}日) [high_low]")

        else:
            logging.error(f"Invalid target_type: {target_type}")

    return df


def create_multi_targets(
    df: pd.DataFrame,
    horizons: List[int],
    open_col: str,
    high_col: Optional[str] = None,
    low_col: Optional[str] = None,
    close_col: Optional[str] = None,
) -> pd.DataFrame:
    """Create multiple target columns for training/prediction pipeline.

    The current setup produces one primary target `target_{h}` plus aliases for the
    specific sub-targets used by the daily 2644t pipeline.
    """
    out = df.copy()
    if not horizons:
        return out

    # primary target is the open-to-open return by default
    out = create_open_targets(out, horizons, open_col, target_type='open_to_open', high_col=high_col, low_col=low_col, close_col=close_col)

    for h in horizons:
        base_col = f'target_{h}'
        if base_col not in out.columns:
            continue

        # Alias for the specific sub-targets so training code can select a target by name.
        if open_col in out.columns:
            out[f'{base_col}_open_to_open'] = out[base_col]
        if high_col and low_col and high_col in out.columns and low_col in out.columns:
            out[f'{base_col}_high_low'] = out[base_col]
        if close_col and close_col in out.columns:
            out[f'{base_col}_close_to_close'] = out[base_col]

    target_cols = [f'target_{h}' for h in horizons]
    out = out.dropna(subset=target_cols).reset_index(drop=True)
    return out


def create_binary_targets(df: pd.DataFrame, horizons: List[int], source_target_col: str) -> pd.DataFrame:
    """
    二値分類用のターゲット生成
    
    Args:
        df: データフレーム
        horizons: 予測ホライズンのリスト
        source_target_col: 元のリターン列名（例: 'target_1'）
    
    Returns:
        target_binary_1 等の列が追加されたデータフレーム
        1 = リターン > 0 (上昇), 0 = リターン <= 0 (下落)
    """
    for h in horizons:
        source_col = f'target_{h}'
        if source_col not in df.columns:
            logging.warning(f"Source target column '{source_col}' not found. Skipping binary target creation.")
            continue
        
        # 二値化: リターン > 0 なら 1, そうでなければ 0
        df[f'target_binary_{h}'] = (df[source_col] > 0).astype(float)
        
        # 統計情報をログ出力
        pos_ratio = df[f'target_binary_{h}'].mean() * 100
        logging.info(f"Created target_binary_{h}: {pos_ratio:.1f}% positive (rising)")
    
    return df


def create_triple_barrier_targets(
    df: pd.DataFrame,
    price_col: str,
    return_col: str,
    cusum_threshold_multiplier: float = 1.0,
    upper_multiplier: float = 1.5,
    lower_multiplier: float = 1.5,
    max_holding_period: int = 5,
    vol_window: int = 20
) -> pd.DataFrame:
    """
    CUSUMフィルタ + トリプルバリア法でイベント駆動ラベルを生成。
    既存のtarget列と共存可能（列名: target_tb_label, target_tb_return など）
    """
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not in dataframe")

    prices = df[price_col].copy()
    returns = df[return_col].fillna(0).values if return_col in df.columns else prices.pct_change().fillna(0).values

    # use dynamic CUSUM thresholding based on realized vol
    event_idx = symmetric_cusum_filter(np.asarray(returns), threshold=-1.0, vol_window=vol_window, multiplier=cusum_threshold_multiplier)

    # map integer positions to timestamps
    event_dates = prices.index[event_idx[event_idx < len(prices)]] if len(event_idx) > 0 else []

    labeler = TripleBarrierLabeler(upper_multiplier=upper_multiplier,
                                   lower_multiplier=lower_multiplier,
                                   max_holding_period=max_holding_period,
                                   vol_window=vol_window)

    tb_df = labeler.label(prices, pd.DatetimeIndex(event_dates))

    # merge results into original df
    df = df.copy()
    df['target_tb_label'] = 0
    df['target_tb_return'] = 0.0
    df['target_tb_holding'] = 0
    for idx, row in tb_df.iterrows():
        if idx in df.index:
            df.at[idx, 'target_tb_label'] = int(row['label'])
            df.at[idx, 'target_tb_return'] = float(row['return'])
            df.at[idx, 'target_tb_holding'] = int(row['holding_period'])

    return df

def infer_open_column(df: pd.DataFrame, open_col: Optional[str] = None) -> Optional[str]:
    if open_col and open_col in df.columns:
        return open_col

    candidates = [c for c in df.columns if c.endswith('_open') and c.startswith(('semi', '2644t'))]
    if len(candidates) == 1:
        logging.info(f"Inferred open column '{candidates[0]}' from data.")
        return candidates[0]
    if len(candidates) > 1:
        logging.warning(f"Multiple open columns found: {candidates}. Defaulting to the first one.")
        return candidates[0]

    if open_col:
        logging.error(f"Configured OPEN_COL '{open_col}' not found in dataframe.")
        return None

    logging.error("Unable to infer open column from dataframe. Please configure OPEN_COL.")
    return None


def v2_forward_returns(prices: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    """v2 のラベル（設計書 v2 ラベリング設計・監査 G-1/B）。行 D（判断日 = 設計書の t+1）に直接置く。

    prices は東京営業日（XTKS）で連続した index を持ち、分割調整済みの adj_open / adj_close を含むこと。
    欠損営業日は NaN の行として残す（shift が営業日単位で正しく数えられるように）。

    fwd_oc_{h}: ln(C[D+h-1] / O[D])  D の寄付で買い、h 営業日目の大引けで評価
    fwd_oo_{h}: ln(O[D+h]   / O[D])  D の寄付で買い、h 営業日後の寄付で評価
    gap       : ln(O[D] / C[D-1])    前営業日の大引け→D の寄付（判断時刻には未知。分析専用で特徴量にしない）
    """
    o = prices["adj_open"].astype(float)
    c = prices["adj_close"].astype(float)
    out = pd.DataFrame(index=prices.index)
    for h in horizons:
        out[f"fwd_oc_{h}"] = np.log(c.shift(-(h - 1)) / o)
        out[f"fwd_oo_{h}"] = np.log(o.shift(-h) / o)
    out["gap"] = np.log(o / c.shift(1))
    return out.astype("float64")
