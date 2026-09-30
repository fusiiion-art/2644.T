"""監査 F: Optuna 総試行数（DSR 用）。

logs/optuna_db/*.db を一時ディレクトリへコピーしてから読み取り専用で開く（元ファイルは変更しない）。
結果は audit/results/optuna_trials.json。
"""
import json
import shutil
import sqlite3
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
out = {}
with tempfile.TemporaryDirectory() as td:
    for src in sorted((ROOT / "logs" / "optuna_db").glob("*.db")):
        dst = Path(td) / src.name
        shutil.copy2(src, dst)
        con = sqlite3.connect(f"file:{dst}?mode=ro", uri=True)
        q = lambda sql, *a: con.execute(sql, a).fetchall()
        best = q("""select t.number, v.value from trials t join trial_values v on t.trial_id=v.trial_id
                    where t.state='COMPLETE' and v.value_type='FINITE' order by v.value asc limit 1""")
        best_params = {}
        if best:
            tid = q("select trial_id from trials where number=?", best[0][0])[0][0]
            best_params = dict(q("select param_name, param_value from trial_params where trial_id=?", tid))
        out[src.name] = {
            "direction": q("select direction from study_directions")[0][0],
            "total_trials": q("select count(*) from trials")[0][0],
            "by_state": dict(q("select state, count(*) from trials group by state")),
            "by_value_type": dict(q("select value_type, count(*) from trial_values group by value_type")),
            "finite_complete": q("""select count(*) from trials t join trial_values v on t.trial_id=v.trial_id
                                    where t.state='COMPLETE' and v.value_type='FINITE'""")[0][0],
            "pruned": q("select count(*) from trials where state='PRUNED'")[0][0],
            "best_finite_trial": {"number": best[0][0], "value_mse": best[0][1], "params": best_params} if best else None,
            "datetime_range": q("select min(datetime_start), max(datetime_complete) from trials")[0],
        }
        con.close()
# CSV ログ（all_trials_summary.csv）の行数も記録
csv_rows = {}
for p in sorted((ROOT / "logs").glob("*/version_*/all_trials_summary.csv")):
    csv_rows[str(p.relative_to(ROOT))] = sum(1 for _ in open(p)) - 1
out["all_trials_summary_csv_rows"] = csv_rows
out["grand_total_trials_in_db"] = sum(v["total_trials"] for k, v in out.items() if k.endswith(".db"))
(Path(__file__).resolve().parent / "results" / "optuna_trials.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(json.dumps(out, ensure_ascii=False, indent=2))
