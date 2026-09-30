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

# semi2644 パッケージ側のデフォルト corporate_actions.yaml / config.yaml
_DEFAULT_CA_PATH = Path(__file__).resolve().parents[1] / "semi2644" / "config" / "corporate_actions.yaml"
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "semi2644" / "config" / "config.yaml"


def load_price_config(path: Optional[Path] = None) -> dict:
    """semi2644/config/config.yaml を読む（数値パラメータの一元管理先）。"""
    with open(path or _DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def raw_ohlc_from_cache(cache: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """yfinance キャッシュ（'Date' 列＋'<ticker>_open' 形式の列）から1銘柄の生 OHLCV を取り出す。

    値の無い日（その取引所の休場日や欠損日）は落とす。index は tz-naive の正規化日付。
    """
    fields = ["open", "high", "low", "close", "volume"]
    present = [f for f in fields if f"{ticker}_{f}" in cache.columns]
    df = cache[[f"{ticker}_{f}" for f in present]].copy()
    df.columns = present
    df.index = pd.DatetimeIndex(pd.to_datetime(cache["Date"])).tz_localize(None).normalize()
    df.index.name = "Date"
    return df.dropna(subset=["close"]).astype(float).sort_index()


def split_adjust_ohlc(
    raw: pd.DataFrame,
    actions: list[dict] | None = None,
    jump_bounds: list[float] | None = None,
    ca_path: Path | None = None,
    config_path: Path | None = None,
) -> pd.DataFrame:
    """生 OHLC(V) を分割調整する（監査 B）。価格補正はこの関数に一元化する。

    - 分割日（権利落ち日）より前の価格を ratio で割り、出来高は掛ける。
    - 取得元が既に遡及調整済みなら二重に割らない（分割日前後の終値比が 1 と 1/ratio のどちらに近いかで判定）。
    - 調整後の日次終値比が jump_bounds を外れたら ValueError（説明できない段差）。
    - 配当は調整しない。corporate_actions.yaml の分配金に未転記分（TODO）があるため。

    Returns:
        adj_open / adj_high / adj_low / adj_close（分割調整済み。指標・ラベル用）、
        adj_volume（volume があれば）、raw_open / raw_close（生値。発注価格の計算専用）
    """
    if not isinstance(raw.index, pd.DatetimeIndex) or raw.index.tz is not None:
        raise ValueError("index は tz-naive の DatetimeIndex であること")
    if actions is None:
        actions = load_corporate_actions(ca_path)
    if jump_bounds is None:
        jump_bounds = load_price_config(config_path)["data"]["jump_bounds"]
    lo, hi = float(jump_bounds[0]), float(jump_bounds[1])

    df = raw.sort_index()
    factor = pd.Series(1.0, index=df.index)
    for split in get_splits(actions):
        split_date = pd.Timestamp(split["date"])
        ratio = float(split["ratio"])
        before = df.index[df.index < split_date]
        after = df.index[df.index >= split_date]
        if ratio <= 0 or ratio == 1.0 or len(before) == 0 or len(after) == 0:
            continue
        jump = df.at[after[0], "close"] / df.at[before[-1], "close"]
        if abs(np.log(jump * ratio)) >= abs(np.log(jump)):
            logger.info(f"Split {split_date.date()} already reflected in source (jump={jump:.3f}); skipped.")
            continue
        factor[df.index < split_date] /= ratio
        logger.info(f"Applied split adjustment: date={split_date.date()}, ratio={ratio}")

    out = pd.DataFrame(index=df.index)
    for col in ["open", "high", "low", "close"]:
        out[f"adj_{col}"] = df[col] * factor
    if "volume" in df.columns:
        out["adj_volume"] = df["volume"] / factor
    out["raw_open"] = df["open"]
    out["raw_close"] = df["close"]

    daily = (out["adj_close"] / out["adj_close"].shift(1)).dropna()
    bad = daily[(daily < lo) | (daily > hi)]
    if not bad.empty:
        raise ValueError(f"調整後も説明できない段差があります（jump_bounds={jump_bounds}）: "
                         f"{[(str(d.date()), round(float(v), 3)) for d, v in bad.items()]}")
    return out
