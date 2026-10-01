# ops: 朝の合図

使い方（準備・毎朝の流れ・約定記録の書き方・困ったとき）は、トップの [README.md](../README.md) の「使い方ガイド」にまとめてある。

| ファイル | 中身 |
|---|---|
| `morning_signal.py` | 2644.T の日足を取り直し、約定記録から今日の注文を作って `signals/` に保存・表示する |
| `run_morning.bat` | タスクスケジューラから呼ぶ起動ファイル（合図をメモ帳で開く） |
| `fills.example.csv` | 約定記録の書式。`fills.csv` にコピーして使う |
| `fills.csv`・`signals/` | 自分の約定記録と毎朝の合図（git に入れない） |

ルールと注文の計算は `common_utils/morning_signal.py`、テストは `tests/test_morning_signal.py`。本ツールは情報提供であり投資助言ではない。
