"""
価格補正の一元化モジュール（規約3）

すべての価格補正（株式分割・配当落ち）はこのファイルのみで行う。
- 指標計算には調整済み価格 (adj_*) を使用
- 発注価格には生値 (raw) を使用

corporate_actions.yaml からアクションを読み込み、
yfinance の adj_close が不整合な期間を検出・修正する。
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# semi2644 パッケージ側のデフォルト corporate_actions.yaml
_DEFAULT_CA_PATH = Path(__file__).resolve().parents[1] / "semi2644" / "config" / "corporate_actions.yaml"


def load_corporate_actions(path: Optional[Path] = None) -> list[dict]:
    """corporate_actions.yaml を読み込む。

    Returns:
        コーポレートアクションのリスト（各要素は dict）
    """
    ca_path = path or _DEFAULT_CA_PATH
    if not ca_path.exists():
        logger.warning(f"Corporate actions file not found: {ca_path}")
        return []
    with open(ca_path, "r", encoding="utf-8") as f:
        actions = yaml.safe_load(f)
    if not isinstance(actions, list):
        logger.error(f"Invalid corporate actions format in {ca_path}")
        return []
    logger.info(f"Loaded {len(actions)} corporate actions from {ca_path.name}")
    return actions


def get_splits(actions: list[dict]) -> list[dict]:
    """分割アクションのみ抽出（日付昇順）。"""
    splits = [a for a in actions if a.get("type") == "split"]
    return sorted(splits, key=lambda a: a.get("date", ""))


def get_dividends(actions: list[dict]) -> list[dict]:
    """配当アクションのみ抽出（record 日昇順）。"""
    divs = [a for a in actions if a.get("type") == "dividend"]
    return sorted(divs, key=lambda a: a.get("record", ""))


def adjust_for_splits(
    df: pd.DataFrame,
    price_cols: list[str],
    volume_cols: list[str] | None = None,
    actions: list[dict] | None = None,
    ca_path: Path | None = None,
) -> pd.DataFrame:
    """分割を反映して価格・出来高を遡及調整する。

    yfinance の adj_close は分割を反映しているが、
    raw OHLC は反映していない場合がある。
    この関数は corporate_actions.yaml の情報を使い、
    指定された価格列を分割調整する。

    Args:
        df: 日付列 'Date' を含む DataFrame
        price_cols: 調整対象の価格列名リスト
        volume_cols: 出来高列名リスト（逆方向に調整）
        actions: 事前に読み込んだアクションリスト（省略時はファイルから読み込み）
        ca_path: corporate_actions.yaml のパス

    Returns:
        分割調整済み DataFrame（コピー）
    """
    if actions is None:
        actions = load_corporate_actions(ca_path)
    splits = get_splits(actions)
    if not splits:
        return df.copy()

    out = df.copy()
    if "Date" not in out.columns:
        logger.error("DataFrame must have a 'Date' column for split adjustment.")
        return out

    out["Date"] = pd.to_datetime(out["Date"])

    for split in splits:
        split_date = pd.Timestamp(split["date"])
        ratio = float(split["ratio"])
        if ratio <= 0 or ratio == 1.0:
            continue

        mask_before = out["Date"] < split_date
        for col in price_cols:
            if col in out.columns:
                out.loc[mask_before, col] = out.loc[mask_before, col] / ratio
        if volume_cols:
            for col in volume_cols:
                if col in out.columns:
                    out.loc[mask_before, col] = out.loc[mask_before, col] * ratio

        logger.info(
            f"Applied split adjustment: date={split_date.date()}, "
            f"ratio={ratio}, cols={price_cols}"
        )

    return out


def validate_adj_consistency(
    df: pd.DataFrame,
    raw_close_col: str,
    adj_close_col: str,
    tolerance: float = 0.02,
) -> pd.DataFrame:
    """生終値と調整済み終値の整合性をチェックする。

    分割後のデータでは raw_close ≈ adj_close であるべき。
    大きな乖離がある行を警告として報告する。

    Returns:
        乖離が tolerance を超える行のみを含む DataFrame
    """
    if raw_close_col not in df.columns or adj_close_col not in df.columns:
        logger.warning(
            f"Cannot validate: {raw_close_col} or {adj_close_col} not found"
        )
        return pd.DataFrame()

    ratio = (df[adj_close_col] / df[raw_close_col].replace(0, np.nan)).dropna()
    # 最新行の ratio を基準に乖離を計算
    if ratio.empty:
        return pd.DataFrame()

    latest_ratio = ratio.iloc[-1]
    deviation = (ratio / latest_ratio - 1.0).abs()
    bad = df.loc[deviation[deviation > tolerance].index].copy()
    if not bad.empty:
        logger.warning(
            f"Found {len(bad)} rows where adj/raw ratio deviates by >{tolerance*100:.0f}%"
        )
    return bad
