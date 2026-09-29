import pandas as pd

from semi2644.calendar import exdiv_dates, xtks


def T(s):
    return pd.Timestamp(s)


def test_exdiv_weekday_record():
    assert exdiv_dates("2024-10-24") == (T("2024-10-22"), T("2024-10-23"))


def test_exdiv_saturday_record():
    assert exdiv_dates("2026-10-24") == (T("2026-10-21"), T("2026-10-22"))


def test_exdiv_dates_are_sessions():
    cal = xtks()
    for rec in ["2022-04-24", "2023-04-24", "2025-10-24", "2026-04-24"]:
        for d in exdiv_dates(rec):
            assert cal.is_session(d)
