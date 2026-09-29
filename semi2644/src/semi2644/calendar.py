"""XTKS営業日と、国内ETFの権利付最終日/権利落ち日の計算。"""
from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd


@lru_cache(maxsize=1)
def xtks():
    return xcals.get_calendar("XTKS")


def exdiv_dates(record: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """決算日から (権利付最終日, 権利落ち日) を返す。T+2決済前提。

    決算日が休日なら直前営業日を起点にし、その2営業日前が権利付最終日。
    """
    cal = xtks()
    settle = cal.date_to_session(pd.Timestamp(record), direction="previous")
    last_cum = cal.session_offset(settle, -2)
    return last_cum, cal.next_session(last_cum)
