"""毎朝の合図（平日 8:30 ごろに実行。寄付の前に注文を出すため）。

    python ops/morning_signal.py                 # 今日（日本時間）の合図。2644.T の日足を取り直してから計算
    python ops/morning_signal.py --date 2026-10-02
    python ops/morning_signal.py --no-fetch      # 取り直さず、キャッシュのまま計算

- ルールと注文の出し方は common_utils/morning_signal.py（バックテストのシミュレータと同じ売買になることをテスト済み）
- 設定は semi2644/config/config.yaml の ops 節と v2 節の position_*・fixed_loss_limit_jpy
- 約定したら ops/fills.csv に1行ずつ記入する（書式は ops/fills.example.csv）
- 合図は ops/signals/<日付>.md と ops/signals/latest.md に保存し、画面にも出す
本ツールは情報提供であり投資助言ではない。
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_utils.morning_signal import SignalRules, build_signal, format_report, load_fills  # noqa: E402
from data.adjust import get_splits, load_corporate_actions, load_price_config, raw_ohlc_from_cache, split_adjust_ohlc  # noqa: E402


def xtks_sessions(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    import exchange_calendars as xcals
    s = xcals.get_calendar("XTKS").sessions_in_range(start, end)
    return pd.DatetimeIndex(s).tz_localize(None).normalize()


def main(argv=None) -> str:
    ap = argparse.ArgumentParser(description="2644.T 朝の合図")
    ap.add_argument("--date", help="判断日 YYYY-MM-DD（既定は日本時間の今日）")
    ap.add_argument("--no-fetch", action="store_true", help="データを取り直さない")
    ap.add_argument("--fills", help="約定記録の CSV（既定は config の ops.fills_path）")
    ap.add_argument("--cache", help="価格キャッシュのファイル名（data/cache 内。既定は config の ops.cache_filename）")
    args = ap.parse_args(argv)

    cfg = load_price_config()
    ops = cfg["ops"]
    if ops["mode"] != "rule":
        raise SystemExit(f"ops.mode が rule ではない: {ops['mode']}")
    day = pd.Timestamp(args.date) if args.date else pd.Timestamp(datetime.now(ZoneInfo("Asia/Tokyo")).date())
    cache_name = args.cache or ops["cache_filename"]
    if not args.no_fetch:
        from common_utils.data_fetcher import get_yfinance_data
        get_yfinance_data({cfg["symbol"]: cfg["symbol"]}, cfg["data"]["start"], "1d", cache_name)
    cache_file = ROOT / "data" / "cache" / cache_name
    if not cache_file.exists():
        raise SystemExit(f"価格キャッシュが無い: {cache_file}（--no-fetch を外して取得する）")
    prices = split_adjust_ohlc(raw_ohlc_from_cache(pd.read_parquet(cache_file), cfg["symbol"]))

    fills_path = Path(args.fills) if args.fills else ROOT / ops["fills_path"]
    fills = (load_fills(fills_path) if fills_path.exists()
             else pd.DataFrame({"date": pd.to_datetime([]), "side": [], "units": [], "price": []}))
    splits = get_splits(load_corporate_actions())
    sessions = xtks_sessions(pd.Timestamp(cfg["data"]["start"]), day + pd.Timedelta(days=10))

    rules = SignalRules.from_config()
    text = format_report(build_signal(day, prices, fills, rules, sessions, splits), rules)
    out_dir = ROOT / ops["signals_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in (f"{day.strftime('%Y-%m-%d')}.md", "latest.md"):
        (out_dir / name).write_text(text, encoding="utf-8")
    print(text)
    return text


if __name__ == "__main__":
    main()
