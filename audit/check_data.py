"""監査 A-方法2 / B / 付随データチェック（読み取り専用）。

- A-方法2: 特徴量ファイルの行 t の米国系列が、米国日付 t と t-1(直前の米国営業日) のどちらと一致するかを直近100行で照合
- B     : 2024-10-09 の 1:2 分割前後の段差（生値・adj_close・ターゲット・ラグ特徴量）
- 付随  : FRED 系列の頻度、先頭行の bfill、東京休場日の行、定数列、Sector_Relative_Strength_5d の中身
結果は audit/results/data_checks.json に保存。
"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]

import exchange_calendars as xc
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

yf = pd.read_parquet(ROOT / "data/cache/yf_data_daily_2644t.parquet").set_index("Date")
fred = pd.read_parquet(ROOT / "data/cache/fred_data_daily_2644t.parquet").set_index("Date")
feat = pd.read_parquet(ROOT / "data/features_daily_2644t.parquet").set_index("Date")
out = {}


def prev_session(s, dates):
    """dates の各日付について、それより厳密に前の最後の値（=直前の米国営業日）。"""
    s = s.dropna()
    pos = s.index.searchsorted(dates, side="left") - 1
    return pd.Series([s.iloc[p] if p >= 0 else np.nan for p in pos], index=dates)


# ---------------- A-方法2 ----------------
last = feat.index[-100:]
a = {"period": [str(last[0].date()), str(last[-1].date())], "n_rows": len(last), "returns": {}, "levels": {}, "normalized": {}}
for tk, col in [("^SOX", "sox_adj_close_return"), ("NVDA", "nvda_adj_close_return"),
                ("TSM", "tsm_adj_close_return"), ("^VXN", "vxn_adj_close_return")]:
    px = yf[f"{tk}_adj_close"].dropna()
    r = np.log(px / px.shift(1))
    f = feat.loc[last, col]
    same = r.reindex(last)                   # 米国日付 t
    prev = prev_session(r, last)             # 米国日付 t-1（直前の米国営業日）
    m_same, m_prev = np.isclose(f, same, atol=1e-9), np.isclose(f, prev, atol=1e-9)
    ok = same.notna()
    a["returns"][col] = {
        "match_US_t": int(m_same.sum()), "match_US_t-1": int(m_prev.sum()),
        "pearson_US_t": float(np.corrcoef(f[ok], same[ok])[0, 1]),
        "pearson_US_t-1": float(np.corrcoef(f, prev)[0, 1]),
        "non_matching_t-1": {str(d.date()): {"feature": float(f[d]), "US_t-1": float(prev[d])} for d in last[~m_prev]},
    }
for tk, col in [("^VXN", "vxn_close"), ("^VXN", "vxn_high"), ("^VXN", "vxn_low")]:
    src = yf[f"{tk}_{col.split('_')[1]}"]
    f = feat.loc[last, col]
    a["levels"][col] = {"match_US_t": int(np.isclose(f, src.reindex(last)).sum()),
                        "match_US_t-1": int(np.isclose(f, prev_session(src, last)).sum())}
so, sc = yf["^SOX_open"].dropna(), yf["^SOX_close"].dropna()
for name, raw in [("SOX_Intraday_Force", sc / so - 1), ("SOX_Overnight_Gap", so / sc.shift(1) - 1)]:
    f = feat.loc[last, name]
    same, prev = raw.reindex(last), prev_session(raw, last)
    ok = same.notna()
    a["normalized"][name] = {"spearman_US_t": float(spearmanr(f[ok], same[ok]).statistic),
                             "spearman_US_t-1": float(spearmanr(f, prev).statistic)}
a["vix_columns_in_yf_cache"] = [c for c in yf.columns if "VIX" in c.upper()]
a["fred_columns_in_feature_file"] = [c for c in feat.columns if c in fred.columns or c.startswith(tuple(fred.columns))]
out["A_method2"] = a

# ---------------- B ----------------
w = yf.loc["2024-10-04":"2024-10-15", ["2644.T_open", "2644.T_close", "2644.T_adj_close", "2644.T_volume"]]
b = {"window": {str(d.date()): {k: (None if pd.isna(v) else float(v)) for k, v in row.items()} for d, row in w.iterrows()}}
b["close_ratio_1009_over_1008"] = float(yf.at[pd.Timestamp("2024-10-09"), "2644.T_close"] / yf.at[pd.Timestamp("2024-10-08"), "2644.T_close"])
b["adjclose_ratio_1009_over_1008"] = float(yf.at[pd.Timestamp("2024-10-09"), "2644.T_adj_close"] / yf.at[pd.Timestamp("2024-10-08"), "2644.T_adj_close"])
b["open_ratio_1009_over_1008"] = float(yf.at[pd.Timestamp("2024-10-09"), "2644.T_open"] / yf.at[pd.Timestamp("2024-10-08"), "2644.T_open"])
r = np.log(yf["2644.T_adj_close"].dropna()).diff().abs().sort_values(ascending=False)
b["largest_abs_logret_adj_close"] = {str(d.date()): float(v) for d, v in r.head(3).items()}
ratio = (yf["2644.T_adj_close"] / yf["2644.T_close"]).dropna()
b["adj_over_raw_ratio"] = {"before_split_mean": float(ratio[:"2024-10-08"].mean()), "after_split_mean": float(ratio["2024-10-09":].mean())}
b["feature_file"] = {
    "target_1@2024-10-08": float(feat.at[pd.Timestamp("2024-10-08"), "target_1"]),
    "log(open1009/open1008)": float(np.log(yf.at[pd.Timestamp("2024-10-09"), "2644.T_open"] / yf.at[pd.Timestamp("2024-10-08"), "2644.T_open"])),
    "semi_adj_close_return_lag1@2024-10-10": float(feat.at[pd.Timestamp("2024-10-10"), "semi_adj_close_return_lag1"]),
    "semi_adj_close_return_lag2@2024-10-11": float(feat.at[pd.Timestamp("2024-10-11"), "semi_adj_close_return_lag2"]),
    "target_1_std_all": float(feat.target_1.std()),
    "target_1_std_excl_1008": float(feat.target_1.drop(pd.Timestamp("2024-10-08")).std()),
    "target_1_abs_top5": {str(d.date()): float(v) for d, v in feat.target_1.abs().sort_values(ascending=False).head(5).items()},
}
b["target_is_open_to_open"] = bool(np.isclose(b["feature_file"]["target_1@2024-10-08"], b["feature_file"]["log(open1009/open1008)"], atol=1e-6))
out["B_split"] = b

# ---------------- 付随チェック ----------------
x = {}
x["fred_frequency"] = {c: {"first": str(s.index.min().date()), "last": str(s.index.max().date()), "n": int(len(s)),
                           "only_day1": bool((s.index.day == 1).all())} for c in fred.columns for s in [fred[c].dropna()]}
x["feature_file_first_date"] = str(feat.index.min().date())
x["feature_file_rows"] = int(len(feat))
x["head_rows_identical_bfill_evidence"] = feat.iloc[:4][["semi_adj_close_return_lag1", "semi_adj_close_return_lag2",
                                                         "sox_adj_close_return_lag2", "Sector_Relative_Strength_5d"]].round(6).astype(float).to_dict(orient="list")
cal = xc.get_calendar("XTKS")
sess = set(cal.sessions_in_range(feat.index.min(), feat.index.max()))
x["non_tokyo_session_rows_in_feature_file"] = int((~feat.index.isin(sess)).sum())
x["examples_non_session_rows"] = [str(d.date()) for d in feat.index[~feat.index.isin(sess)][:8]]
x["target_exact_zero_rows"] = int((feat.target_1 == 0).sum())
x["constant_columns"] = [c for c in feat.columns if feat[c].std() == 0]
n = yf["^N225_adj_close"].dropna()
n5 = np.log(n / n.shift(1)).reindex(pd.bdate_range(n.index.min(), n.index.max())).fillna(0).rolling(5).sum()
f = feat["Sector_Relative_Strength_5d"]
x["Sector_Relative_Strength_5d_spearman_vs_minus_nikkei5d"] = float(spearmanr(f.iloc[10:], -n5.reindex(f.index).iloc[10:], nan_policy="omit").statistic)
out["misc"] = x

# ---------------- A-方法1: どの生列が shift(1) されるか（コード写経） ----------------
import re
sys.path.insert(0, str(ROOT))
import timescale_modules.daily_2644t.config_daily_2644t as cfg   # 純 Python（外部 I/O なし）


def apply_column_mapping(cols, column_map):  # common_utils/data_fetcher.py:297-320
    rename = {}
    sorted_map = sorted(column_map.items(), key=lambda it: len(it[0]), reverse=True)
    for oc in cols:
        if oc == "Date":
            continue
        low = str(oc).lower()
        low = re.sub(r"adjclose$", "_adj_close", low)
        low = re.sub(r"(?<!_)(open|high|low|close|volume)$", r"_\1", low)
        for kw, new in sorted_map:
            if kw.lower() in low:
                if any(s in new for s in ["_nav", "_open", "_high", "_low", "_close"]):
                    rename[oc] = new
                else:
                    rename[oc] = f"{new.lower()}{low.replace(kw.lower(), '').replace(' ', '_')}"
                break
    return [rename.get(c, c) for c in cols]


LAG_TARGETS = ['sp500', 'nasdaq', 'sox', 'nvda', 'vix', 'vxn', 'us_10y', 'us_2y',   # feature_base.py:103-107
               'vt', 'crude_oil', 'gold', 'eur_usd', 'dollar_index', 'apple', 'google',
               'aapl', 'msft', 'tsm', 'avgo', 'asml', 'amd', 'txn', 'qcom', 'intc', 'mu', 'amat']
raw_cols = [c for c in list(yf.columns) + list(fred.columns) if c != "Date"]
mapped = apply_column_mapping(raw_cols, cfg.COLUMN_MAP)
shifted = sorted({re.sub(r"_(adj_close|close|open|high|low|volume)$", "", c) for c in mapped
                  if any(t in c for t in LAG_TARGETS) and "target" not in c})
not_shifted = sorted({re.sub(r"_(adj_close|close|open|high|low|volume)$", "", c) for c in mapped
                      if not any(t in c for t in LAG_TARGETS)})
out["A_method1_shift_map"] = {"shifted_by_1_weekday": shifted, "NOT_shifted": not_shifted,
                              "matched_lag_substring": {c: [t for t in LAG_TARGETS if t in c] for c in shifted}}

# ---------------- 学習ラベルの整合（休場日を含む平日グリッドの影響） ----------------
# 学習ラベル: 窓の最終行 k -> target[k+1] (trainer_model.py:223-225)。正しくは open(次営業日)->open(次々営業日)。
o = yf["2644.T_open"].dropna()
o_adj = o.copy()
o_adj[o_adj.index < pd.Timestamp("2024-10-09")] /= 2.0
sessions = o.index
spos = {d: i for i, d in enumerate(sessions)}
lab = feat["target_1"].shift(-1)
rows = []
for d in feat.index:
    if d not in spos or spos[d] + 2 >= len(sessions):
        continue
    i = spos[d]
    if cal.next_session(d) != sessions[i + 1] or cal.next_session(sessions[i + 1]) != sessions[i + 2]:
        continue
    true_next = np.log(o_adj.iloc[i + 2] / o_adj.iloc[i + 1])      # 正しい区間
    same_day_start = np.log(o_adj.iloc[i + 1] / o_adj.iloc[i])      # 当日始値から始まる区間（予測時点で一部既知）
    rows.append((d, lab[d], true_next, same_day_start))
la = pd.DataFrame(rows, columns=["Date", "label", "true", "starts_at_open_t"]).dropna()
ok = np.isclose(la.label, la.true, atol=1e-6)
la_bad = la[~ok]
out["label_alignment"] = {
    "n_rows_checked": int(len(la)), "n_label_equals_true_next_session": int(ok.sum()),
    "n_mismatch": int((~ok).sum()),
    "n_mismatch_label_is_zero": int((la_bad.label == 0).sum()),
    "n_mismatch_label_starts_at_open_t": int(np.isclose(la_bad.label, la_bad.starts_at_open_t, atol=1e-6).sum()),
    "n_mismatch_split_row": int((la_bad.Date == pd.Timestamp("2024-10-07")).sum()),
    "examples": la_bad.head(6).assign(Date=lambda x: x.Date.astype(str)).round(6).to_dict(orient="records"),
}

od = Path(__file__).resolve().parent / "results"
od.mkdir(exist_ok=True)
(od / "data_checks.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
