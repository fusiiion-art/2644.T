# 2644.T 日足予測 監査レポート

- 監査日: 2026-09-30　対象: ブランチ `ccr-3920bedc-2piypf`（HEAD `2615715`）
- 対象データ: `data/features_daily_2644t.parquet`（783行, 2023-07-04〜2026-07-02）、`data/cache/yf_data_daily_2644t.parquet`（〜2026-07-03）、`data/cache/fred_data_daily_2644t.parquet`、`logs/` 配下の学習済み Ridge と Optuna DB
- 既存コードは無変更。新規ファイルは `audit/` 以下のみ（スクリプト4本＋`results/`）。Optuna DB は一時コピーを読み取り専用で開き、実行前後で md5 一致を確認。
- 実行環境の制約: GPU なし（学習は `accelerator="cpu"` に置換）。Yahoo Finance は本環境のネットワークポリシーで遮断（HTTP 403）されたため、**キャッシュ外の外部データとの突き合わせは未実施**。
- 本レポートは情報提供であり投資助言ではない。

## 0. 判定一覧

| 項目 | 判定 | 要点（根拠は各節） |
|---|---|---|
| A-1 SOX/NVDA/TSM/VXN の日付結合 | **PASS** | 行 t には**米国 t-1**（直前の米国営業日）の値。直近100行で t-1 と 96/100 完全一致、t とは 0/100 |
| A-2 VIX | **PASS（該当なし）** | ^VIX は取得対象から削除済み。キャッシュ・特徴量に VIX 列なし |
| A-3 FRED | **FAIL（潜在）** | 9系列中5系列が shift なし＝米国日付 t、月次2系列は期初日付で ffill（最大1〜2か月の先読み）。**現行特徴量ファイルには FRED 列ゼロ**のため現行モデルへの影響はなし |
| A-方法3 | PASS を支持 | 米国系列を追加で1日遅らせても指標は悪化しない（IC・シャープの差はシード間ばらつきより小さい） |
| B-1 分割前後の段差 | **FAIL** | `adj_close` も生値も 2024-10-09 に ×0.509。`data/adjust.py` はどこからも呼ばれていない |
| B-2 ターゲット/約定価格の生値・調整値 | **FAIL** | ターゲット＝分割未調整の生始値（2024-10-08 の target_1 = −0.681）、特徴量＝配当のみ調整の adj_close。発注価格の経路は列不在で機能していない |
| C 15:00 / 15:30 のハードコード | **PASS** | 実行コードには無し。文書2件とTODOコメント1件のみ（下に列挙） |
| D-1 feature_screener | **PASS** | 全期間で1回実行された痕跡（`SCREENING_BLACKLIST` 82件）はあるが、適用行がコメントアウトで CV に未使用 |
| D-2 clustered_importance | **PASS** | `USE_CFI` 未定義のため実行されない（有効化すると先読みになる潜在問題あり） |
| D-3 SHAP | **PASS** | 出力は `reports/` へのファイル書き出しのみ。学習・CV から読まれない |
| E regime_detector | **PASS** | 平滑化確率でもフィルタ確率でもない。行 t の `vxn_close`（＝米国 t-1 の VXN 終値）> 22 の決定的な閾値判定で、未来を見ない |
| F 現行モデルの実力（WF OOS） | **FAIL** | 492日の OOS で的中率 48.2%（10シード中央値）＜「常に上昇」53.1%、IC −0.015（p=0.68）、年率シャープ 0.35 ＜ バイ&ホールド 0.90。10シードすべてで両基準に負け |

A–F の範囲外で見つけた先読み・バグは「7. 追加所見」に記載（休場日を含む平日グリッドによる**ラベルの先読み**など、FAIL 相当が複数ある）。

---

## 1. A. 米国系列の日付結合リーク

### 方法1: 結合コード

| 段階 | 場所 | 内容 |
|---|---|---|
| 取得 | `common_utils/data_fetcher.py:133,152` | 全シンボルを1回の `yf.download` で取得し、`Date` を tz-naive 化。各系列の `Date` はその取引所の**現地の取引日** |
| 結合キー | `generate/features_daily_features.py:155-158` | `pd.merge(..., on='Date', how='outer')`（暦日の一致で結合。時差補正なし） |
| 再索引 | `generate/features_daily_features.py:166-167,172` | `pd.bdate_range`（月〜金の**平日**。東京営業日ではない）に reindex → `ffill()` |
| shift | `common_utils/feature_base.py:68,101-116` | `run_all` の最初に `_apply_timezone_lag`。列名に `lag_targets`（103-107行）の**部分文字列**を含む列を `shift(1)`（=1平日遅らせる） |
| 派生特徴量 | `generate/features_2644t.py:35,41` | `SOX_Overnight_Gap` / `SOX_Intraday_Force` は shift 後の `sox_open`/`sox_close` から計算 → 米国 t-1 |

コードを写経して判定を機械的に再現した結果（`audit/results/data_checks.json` → `A_method1_shift_map`）:

- **shift(1) される（行 t = 米国 t-1）**: `sox, nvda, tsm, vxn, nasdaq_100, mu, vt, gold, eur_usd, us_10y_yield`（yfinance）、`us_10y, us_2y, us_10y_breakeven, dollar_index`（FRED）
- **shift されない（行 t = 日付 t の値）**: FRED の `effr, jp_10y_yield, expected_inflation, credit_spread, market_liquidity`、為替 `usd_jpy, eur_jpy`（と東京系列 `semi, nikkei_225, tokyo_electron, advantest, disco`）
- VIX: `common_utils/config_model.py:36` で `"vix": "^VIX"` がコメントアウト。キャッシュに VIX 列なし。`Tech_Risk_Premium`（`features_2644t.py:47`）は `vix_close` が無いので生成されない。
- FRED の追加問題: `effr`(FEDFUNDS) と `jp_10y_yield`(IRLTLT01JPM156N) は**月次で毎月1日付**（`data_checks.json` → `misc.fred_frequency.only_day1 = true`）。月平均値が月初日付で ffill されるので、当月中に当月平均を参照する先読みになる。日次 FRED（DGS10 等）も公表は翌営業日の米国夕方なので、t-1 でも東京の大引け時点では未公表の日がある。
- **ただし現行の特徴量ファイルには FRED 列も為替列も1つも無い**（`FEATURE_SETS_DEFINITIONS` = `config_daily_2644t.py:70-113` の正規表現に一致しないため）。現行モデルが使う米国系は SOX/NVDA/TSM/VXN のみ。

### 方法2: 実データ照合（直近100行: 2026-02-13〜2026-07-02）

`audit/check_data.py` の出力（`results/data_checks.json` → `A_method2`）:

| 特徴量（特徴量ファイルの値） | 米国 t と一致 | 米国 t-1 と一致 | 相関 vs t | 相関 vs t-1 |
|---|---|---|---|---|
| `sox_adj_close_return` | 0/100 | **96/100** | −0.096 | **0.980** |
| `nvda_adj_close_return` | 0/100 | **96/100** | −0.025 | **0.984** |
| `tsm_adj_close_return` | 0/100 | **96/100** | −0.188 | **0.973** |
| `vxn_adj_close_return` | 0/100 | **96/100** | −0.179 | **0.992** |
| `vxn_close`（水準・未正規化） | 0/100 | **100/100** | – | – |
| `SOX_Intraday_Force`（正規化済み, Spearman） | – | – | 0.006 | **1.000** |
| `SOX_Overnight_Gap`（正規化済み, Spearman） | – | – | −0.158 | **0.953** |

t-1 と不一致の4行（2026-02-17, 04-06, 05-26, 06-22）はいずれも米国祝日の翌東京営業日で、特徴量は 0.0（平日グリッド上で祝日の価格を ffill した結果のリターン0）。先読みではない。`SOX_Overnight_Gap` の 0.953 も同じ4行の差によるもので、「平日グリッド→ffill→shift(1)」で再構成した値とは直近100行で Spearman 1.000。

### 方法3: 米国系特徴量を追加で1日遅らせて再評価

`sox_ / nvda_ / tsm_ / vxn_ / SOX_ / NVDA_` で始まる全列（レジーム判定用 `vxn_close` を含む）をさらに1行遅らせ、F と同じウォークフォワードで比較（10シード）。

| 指標（10シード中央値 [最小, 最大]） | 現行（asis） | 米国系を追加で1日遅延（uslag1） |
|---|---|---|
| 方向的中率 | 48.2% [46.7%, 51.3%] | 48.0% [45.8%, 52.9%] |
| IC (Spearman) | −0.015 [−0.075, +0.018] | −0.013 [−0.099, +0.036] |
| 年率シャープ 往復0% | +0.35 [−0.28, +0.60] | +0.28 [−0.57, +0.79] |
| 年率シャープ 往復0.10% | +0.16 [−0.46, +0.46] | +0.11 [−0.73, +0.64] |
| 取引回数（往復） | 111 [103, 123] | 104 [99, 112] |

差はどれもシード間のばらつき（最小〜最大の幅）よりずっと小さい。同日リークがあれば「現行だけ成績が良く、遅らせると大きく落ちる」はずだが、そうなっていない。ただし現行モデル自体に予測力が無いため、この検定の検出力は低い（方法1・2の直接照合が主な根拠）。全指標は F 節の uslag1 表を参照。

**判定: A-1 PASS。** 現行モデルが使う米国系列は米国 t-1 で、東京 t の大引け（15:00/15:30）より前に確定している値のみ。追加の遅延で成績が落ちないことも、同日リークが無いことと矛盾しない。
**A-3 FAIL（潜在）。** FRED の shift 漏れ・月次系列の期初日付・公表ラグ無視。現在は特徴量に入っていないが、`FEATURE_SETS` に追加した瞬間に先読みになる。

---

## 2. B. 2024-10-09 の 1:2 分割

### B-1 調整済み系列の段差

`data/cache/yf_data_daily_2644t.parquet`（`results/data_checks.json` → `B_split`）:

| 日付 | Open | Close | Adj Close |
|---|---|---|---|
| 2024-10-08 | 3745 | 3725 | 3648.22 |
| 2024-10-09 | 1896 | 1896 | 1856.92 |

- 終値比 0.5090、**adj_close 比も 0.5090**。adj_close の最大の日次対数リターンは 2024-10-09 の −0.675（2番目は 2024-08-05 の 0.177）。
- adj/raw 比は分割前平均 0.974、分割後 0.992 → adj_close は**配当しか調整されておらず、分割は未調整**。
- `data/adjust.py:56-113` の `adjust_for_splits` は定義のみで、リポジトリ内に呼び出し元がない（grep で0件）。`semi2644/config/corporate_actions.yaml:4` の分割情報はどこにも使われていない。
- 原因の推定: `common_utils/data_fetcher.py:88-94` の差分キャッシュは「最終キャッシュ日＋1日」以降しか取り直さないため、Yahoo 側が過去分を遡って分割調整しても分割前の行は更新されない（Yahoo に到達できないため外部照合は未実施。**推定**）。

### B-2 ターゲット・約定価格の生値/調整値

- ターゲット: `target_utils.py:136` → `create_open_targets(..., 'open_to_open')`、`target_utils.py:84-86` で `log(open[t+1]/open[t])`。`OPEN_COL="2644t_open"`（`config_daily_2644t.py:32`）は実在しないので `infer_open_column`（`target_utils.py:231-241`）が `semi_open` を採用 → **分割未調整・配当未調整の生始値**。
- 実データ: 特徴量ファイルの `target_1@2024-10-08 = −0.6807` = `log(1896/3745)`（一致を確認）。target_1 の標準偏差はこの1行を含むと 0.0370、除くと 0.0279。
- 特徴量側: `semi_adj_close_return_lag1@2024-10-10 = −0.6753`、`lag2@2024-10-11 = −0.6753`（配当調整済み・分割未調整の adj_close 由来）。
- つまり**ターゲット＝生始値、特徴量＝adj_close（配当のみ調整）で基準が混在し、両方とも分割は未調整**。
- 約定（発注）価格: `timescale_modules/daily_2644t/supervisor_daily_2644t.py:59-66` は `open` / `ATR_14` / `BB_lower_2` / `SOX_return_prev_night` 列を参照するが、どれも特徴量ファイルに存在しないため常に `None` を返す。`predicter_daily_2644t.py:69-73` の `latest_price` も存在しない列（`Close` / `close`）を見るので 0.0。**発注価格を生値で出す経路は現状機能していない。**

**判定: B-1 FAIL、B-2 FAIL。**（分割は 2644 の実データ期間の中にあり、現行 risk_off モデルの学習データ（2024-10-08 は risk_off）に −0.68 の外れ値ラベルとして入っている。）

---

## 3. C. 大引け時刻（2024-11-05 に 15:00→15:30）

grep 対象: `*.py, *.yaml, *.yml, *.md, *.toml, *.json, *.cfg, *.ini, *.txt`。パターン: `15:00|15:30|14:50|15:25|1500|1530|hour=15|time(15|T15:|close_time|market_close|session_close|大引け|引け|closing` と、`datetime.time|.hour|minute|Asia/Tokyo|America/New_York|tz_convert|between_time|at_time`。

| 場所 | 内容 | 種別 |
|---|---|---|
| `CLAUDE.md:3` | 「t行にはtの大引け(15:30)時点で既知の情報のみ」 | 規約文書 |
| `semi2644/CLAUDE.md:3` | 同上 | 規約文書 |
| `trading_pipeline/filters.py:17` | `# TODO: 時間帯フィルタの実装 (例: 9:00-9:15, 11:25-12:35, 14:50-15:00 を除外)` | コメント（本体は `pass`）。**2024-11-05以降の大引け 15:30 と合っていない** |

- 実行コードで大引け時刻を持つ箇所は**無い**（日足のみで時刻を扱わない）。`semi2644/src/semi2644/calendar.py` も営業日判定のみ。
- 関連リスク（UNKNOWN）: `data_fetcher.py:88-94,103` は UTC の当日までを取得し、差分キャッシュは取り直さない。東京の場中（15:30 より前）に実行すると 2644.T・^N225 等の当日**途中足**が保存され、以後更新されない。外部データに到達できず、キャッシュに途中足が混入しているかは検証できなかった。

**判定: PASS**（ハードコード箇所は上表の3件のみで、いずれも実行されない）。

---

## 4. D. 特徴量選択の先読み

| ツール | 全期間で1回実行か | 結果を CV/学習で使うか | 根拠 |
|---|---|---|---|
| `analysis/feature_screener.py` | **はい**（`:50` で全行を読み、`:93` で特徴量ごとに前70%/後30%分割） | **いいえ** | 結果（`reports/.../feature_screening_report.csv`）を読むコードは無い。その結果とみられる `SCREENING_BLACKLIST` 82件（`config_daily_2644t.py:118-147`）の適用行 `:148` はコメントアウト。実際に risk_off の `features.json` には `semi_adj_close_return_lag1`・`sox_adj_close_return_lag1` 等のブラックリスト掲載列が入っている |
| `common_utils/clustered_importance.py` | 実行されない | いいえ | `USE_CFI` は config に存在せず（grep でテストのみ）、`trainer_model.py:121,186,310` の `getattr(config,'USE_CFI',False)` が False |
| SHAP（`analysis/shap_analyzer.py`, `analysis/feature_suggester.py`） | suggester は全期間のレジーム行で計算（`:162-199`） | いいえ | 出力は `reports/.../importance_full_list_*.json`（`:250`）への書き出しのみで、読み手なし。shap_analyzer は `REGIME_SUPERVISOR_CONFIG` 未定義（`:42`）で終了 |

- `SCREENING_BLACKLIST` と実際に適用される `BLACKLIST_FEATURES`（`config_model.py` の共通リスト）の重なりは5件（`eur_usd/gold/mu/tsm/us_10y_yield_adj_close_return`）だけ。共通リストは「外生系列の生リターンは全部除外」という一律ルールで15系列すべての生リターンを並べたもの（`SCREENING_BLACKLIST` に載っていない `sox_/nvda_adj_close_return` 等も含む）なので、重なりはこのルールで説明できる。ただし共通リストがどうやって作られたかの記録は無い。
- 現行 risk_off モデルの `features.json` には `SCREENING_BLACKLIST` 掲載の11列（`SOX_Overnight_Gap, day_of_week, month, semi_adj_close_return_lag1/2, sox_adj_close_return_lag1/2` 等）が含まれており、スクリーニング結果が適用されていないことを実物で確認。

補足（判定には含めないが F の解釈に影響）:
- 実際に効いている特徴量選択は「候補をアルファベット順に並べた先頭 N 個」（`trainer_model.py:139-143, 328`）で、**N（`n_features_to_select`）と正則化などを Optuna が全期間の 5-fold Purged CV で選んでいる**（`trainer_model.py:300, 332-337`）。F のウォークフォワードはこの固定パラメータを使うので、**F の OOS 成績は楽観側に偏っている**（それでも優位性は出ていない）。
- 潜在問題: CFI を有効にすると、重要度は `df_regime` の先頭80%（`trainer_model.py:305-307`）で計算され、同じ 5-fold CV の検証 fold 1〜4 を含む → 先読みになる。
- `feature_screener.py:82` は目的変数に `cfg.TARGET_COLUMN`（同日の入力リターン）を使っており、将来リターンを評価していない（バグ。しかも現在の特徴量ファイルにはこの列が無く `:52` で KeyError になる）。

**判定: D-1 / D-2 / D-3 すべて PASS**（どれも「全期間で1回実行し、その結果を CV で使う」状態にはない）。

---

## 5. E. regime_detector

- `common_utils/regime_detector.py:13-32` `detect_simple_vix`: `np.where(df[vix_column] > threshold, 'risk_off', 'risk_on')`（`:30`）。行ごとの**決定的な閾値判定**で、確率も平滑化も無い。学習時の呼び出しは `trainer_model.py:890-893`（列 `vxn_close`、閾値 22.0）。
- `detect_complex`（`:35-`）は rolling / ewm / rolling(252).quantile のみで因果的。呼び出し元なし。
- HMM・マルコフ切替・平滑化確率の実装はリポジトリに無い（`hmm|markov|smoothed_marginal|filtered_marginal|viterbi|forward_backward` で grep 0件）。
- 使う値 `vxn_close` の行 t は米国 t-1 の VXN 終値（A 方法2で 100/100 一致）なので、東京 t の大引け時点で既知。
- 付記（バグ・運用）: `predicter_model.py:120-121` は列名に `vix` を含む列を探すが、実際は `vxn_close` なので見つからず `latest_vix=15.0` → **DailyPredicter は常に risk_on**。閾値 22 は「データバランス改善」で 20 から変更（`config_model.py:180`）されたが `research/trial_log` に記録が無い。

**判定: PASS**（平滑化確率でもフィルタ確率でもなく、その時点の値だけを使う閾値判定）。

---

## 6. F. 現行モデルの実力（ウォークフォワード OOS のみ）

### 評価の設計（`audit/wf_ridge_eval.py`）

- **既存の学習済みモデルは OOS 評価に使えない**: 保存済み scaler の `n_samples_seen_` は risk_on 506 + risk_off 277 = 783 で、特徴量ファイルの全行。既存の `analysis/walk_forward_validator.py` もこのモデルで後半50%を予測するだけの in-sample 評価（G-9）。そのため**拡大窓で再学習するウォークフォワードを新たに実施した**。
- データ: 現行モデルの学習データそのもの（`data/features_daily_2644t.parquet`, 783行）。
- 初期学習窓 252行（2023-07-04〜2024-06-19）、以後 21行ごとに拡大窓で再学習（26回）。再学習時点 j では行 0..j-1 のラベル（open(j) までで確定）だけを使う。
- 学習手順は `trainer_model.py:549-628`（`_train_full_period`）を写経: レジーム別、定数列除外→候補をアルファベット順に並べた先頭 N 個、StandardScaler、リポジトリの `RidgeLightning` を 100 epoch（AdamW + OneCycleLR、accumulate 4、grad clip 1.0、MSE）。ハイパーパラメータは現行 ckpt と同じ値で固定（risk_on = Optuna #2303: N=4, dropout 0.146, lr 0.00509, wd 0.000733／risk_off = #2000: N=20, dropout 0.370, lr 0.000798, wd 3.29）。GPU 指定のみ CPU に置換。
- 予測: 行 k の `vxn_close` でレジームを選び、本番の predicter と同じく直近32行（連続行）を入力。
- 実現リターン: 生データの 2644.T 始値（2024-10-09 より前は ÷2）で、**東京営業日の open(翌営業日)→open(翌々営業日)**（規約2「約定は t+1 の始値」）。これはパイプラインの学習ラベル `target[k+1]` と 492日中437日で完全一致（残り55日は休場日前後と分割日＝G-1 の不具合）。
- OOS: 2024-06-20〜2026-07-01 の東京営業日 492日（休場日36行、データ欠損日 2025-10-24 をまたぐ2行、最終行を除外）。予測ありは478日（最初の2回の再学習では risk_off の行が29行しかなく（32行必要）risk_off モデルを作れないため、2024-07-25〜08-14 の14日はノーポジション扱い。この14日のバイ&ホールド損益は複利で +0.75% で、評価への影響は小さい）。
- 売買ルール: 予測 > 0 でロング、それ以外はノーポジション（`predicter_model.py:210, 379-380` と同じロング/フラット）。コストは建て・手仕舞いのそれぞれに往復コストの半分。年率シャープ = 日次平均 / 日次標準偏差 × √252（ノーポジション日を含む）。取引回数 = 建て回数（＝往復数）。
- 学習がシード依存（初期値・ドロップアウト・シャッフル）なので 10 シードで実行し、中央値 [最小, 最大] と、10シードの予測平均で1本にした結果を示す。

### 結果（現行: asis）

| 指標 | 10シード中央値 [最小, 最大] | シード平均予測 |
|---|---|---|
| 方向的中率 | 48.2% [46.7%, 51.3%] | 46.9% |
| 「常に上昇」の的中率 | 53.1% | 53.1% |
| IC (Spearman) | −0.015 [−0.075, +0.018] | −0.038 |
| IC の p 値 | 0.68 [0.10, 0.85] | 0.41 |
| 年率シャープ 往復0% | +0.35 [−0.28, +0.60] | +0.36 |
| 年率シャープ 往復0.05% | +0.25 [−0.37, +0.53] | +0.28 |
| 年率シャープ 往復0.10% | +0.16 [−0.46, +0.46] | +0.20 |
| 取引回数（往復） | 111 [103, 123] | 108 |
| 保有日数（/492） | 256 [214, 279] | 260 |
| 参考: バイ&ホールドのシャープ | +0.90 | +0.90 |

- 的中率は **10シードすべてで「常に上昇」を下回る**。IC が正のシードは 3/10（最大 +0.018, p=0.69）。p が最も小さいシード（p=0.10）は IC が**負**（−0.075）。
- シャープは **10シードすべてでバイ&ホールド未満**。保有比率が同じ（52%）ランダムなロング/ノーポジのシャープ分布（5,000回）は中央値 0.65・5〜95% が −0.17〜1.50 で、10シードすべてがこの中央値以下（パーセンタイル 3〜46%）。プラスのシャープは上昇相場に約半分の日数乗っていたことで説明でき、予測力の証拠にはならない。
- レジーム別（シード平均予測）: risk_on 262日は的中率 45.4% 対「常に上昇」52.3%、IC −0.030／risk_off 216日は 48.6% 対 54.2%、IC −0.084。
- 取引は2年で約111往復。往復0.10%でシャープは約0.19下がる。
- 学習データ上では保存済み risk_on モデルの予測とラベルの相関が 0.484 あるが（パラメータ128個に対して学習サンプル474個）、OOS では ≈0。過学習の典型的な形。
- **楽観バイアス**: N とハイパーパラメータは全期間の CV で選ばれている（D 補足）ため、この表は実力の上限寄り。

### 米国系を1日遅らせた版（uslag1）

A は PASS なので必須ではないが、方法3で使った版の全指標を示す。

| 指標 | 10シード中央値 [最小, 最大] | シード平均予測 |
|---|---|---|
| 方向的中率 | 48.0% [45.8%, 52.9%] | 48.3% |
| 「常に上昇」の的中率 | 53.1% | 53.1% |
| IC (Spearman) | −0.013 [−0.099, +0.036] | −0.072 |
| IC の p 値 | 0.46 [0.03, 0.95] | 0.12 |
| 年率シャープ 往復0% | +0.28 [−0.57, +0.79] | −0.06 |
| 年率シャープ 往復0.05% | +0.20 [−0.65, +0.71] | −0.13 |
| 年率シャープ 往復0.10% | +0.11 [−0.73, +0.64] | −0.20 |
| 取引回数（往復） | 104 [99, 112] | 106 |
| 保有日数（/492） | 243 [195, 262] | 245 |

（p < 0.05 の2シード（p=0.030, 0.041）はどちらも IC が**負**（−0.100, −0.094）。）

### 参考: 分割ラベルだけ直した版（bfix）

`target_1@2024-10-08` と `semi_adj_close_return_lag1@10-10` / `lag2@10-11` の3セルだけを分割調整値に置き換えたもの（risk_on モデルは影響を受けない）。的中率 48.3% [46.2%, 53.1%]、IC −0.022 [−0.077, +0.022]、シャープ（往復0%）+0.33 [−0.18, +0.68]、（往復0.10%）+0.17 [−0.37, +0.55]、取引 112 [100, 120]。ラベルを1つ直しただけでは結論は変わらない。

### Optuna 試行数（DSR 用に記録）

`audit/optuna_trials.py` → `results/optuna_trials.json`（DB は一時コピーを読み取り専用で開き、元ファイルの md5 は実行前後で同一）

| study | 総試行 | 状態 | 値が有限 | 最良（MSE） |
|---|---|---|---|---|
| `daily_2644t_risk_on_ridge_optimization` | 2,323 | COMPLETE 2,016 / PRUNED 305 / FAIL 1 / RUNNING 1 | 321（完了16＋枝刈り305） | #2303: 0.000631 |
| `daily_2644t_risk_off_ridge_optimization` | 2,001 | COMPLETE 2,001 | 1 | #2000: 0.004138 |
| **合計** | **4,324** | | **322** | |

- 各 study の最初の 2,000 試行は値が `+inf`（CSV 上は `FAILED`）で、何も評価していない。DSR の N は**保守的には 4,324、実際に評価された構成数なら 322** を使う。
- どちらの study も、コードとデータが変わっても同じ DB に試行を足し続けている（`load_if_exists=True`, `trainer_model.py:746-752`）。risk_off の現行モデルは有限値が1件しかない study の「最良」。
- ElasticNet・DLinear など他モデルの study DB はリポジトリに無く、他にどれだけ試したかは **UNKNOWN**。
- DSR 用の他の入力（シード平均予測・往復0%の日次戦略リターン）: T = 492日、歪度 −0.07、尖度（Pearson）9.38。

---

## 7. 追加所見（A–F の範囲外。先読み・バグ・コスト関連）

| # | 判定 | 内容 | 根拠 |
|---|---|---|---|
| G-1 | **FAIL（ラベルの先読み）** | 特徴量・ラベルが**平日グリッド**（東京の休場日51行を含む）上で作られる。学習ラベル（窓の最終行 k → `target[k+1]`）は全期間 728 行中 **81 行で正しい「翌営業日始値→翌々営業日始値」と不一致**：46 行はラベル 0、**34 行は「当日始値→翌営業日始値」で予測時点（t の大引け）に一部既知の区間**、1 行は分割。規約1（index は東京営業日）にも反する | `features_daily_features.py:166-167`、`trainer_model.py:223-225`、`results/data_checks.json` → `label_alignment`, `misc.non_tokyo_session_rows_in_feature_file=51` |
| G-2 | FAIL（軽微） | `run_all` の最後で `bfill()` → 先頭行のウォームアップ NaN が**未来の値**で埋まる（先頭3行の `semi_adj_close_return_lag1/lag2` 等が同値）。後段の `dropna()`（`features_daily_features.py:204`）は効かない | `feature_base.py:90`、`data_checks.json` → `misc.head_rows_identical_bfill_evidence` |
| G-3 | FAIL（軽微） | 正規化（median/IQR, ±10 クリップ）を**全期間の統計**で計算してから保存 | `feature_base.py:313-331` |
| G-4 | バグ（データ損失） | FRED `credit_spread`（BAMLH0A0HYM2）がキャッシュ上 2023-07-04 開始のため、結合後の `dropna()` で **2021-09〜2023-07 の約21か月が全て削除**（FRED は特徴量に使われていないのに） | `features_daily_features.py:176`、`data_checks.json` → `misc.fred_frequency` |
| G-5 | バグ | `MAIN_ASSET="2644t_adj_close"` 等（`config_daily_2644t.py:31-37`）が実列名 `semi_*` と不一致 → 主資産の RSI/MACD/BB/ATR/統計量が生成されない。`Sector_Relative_Strength_5d` は `semi_ret` が 0 系列（`features_2644t.py:51-55`）のため**−日経5日リターンそのもの**（Spearman 1.000）。`NVDA_Impact_Factor` は恒等的に 0 | `data_checks.json` → `misc` |
| G-6 | バグ | レジームで間引いた非連続の行でシーケンスを作るため、各レジームで 32 窓のラベルが翌営業日ではない（最大 51 / 119 行先） | `trainer_model.py:905,907` → `:359,585` |
| G-7 | 潜在バグ | Objective が失敗時に `-np.inf` を返す（`trainer_model.py:294, 491`）が study は minimize（`trainer_model.py:751`）→ そうした試行が best になり得る。現 DB の失敗 2,000 件は `+inf`（INF_POS）なので現モデルへの影響はなし | `results/optuna_trials.json` |
| G-8 | バグ（学習/推論の不一致） | `main.py predict` は保存済み scaler を使わず、直近64行で `StandardScaler` を fit し直す | `main.py:71-72` |
| G-9 | FAIL（評価の先読み） | `analysis/walk_forward_validator.py` は全期間で学習した本番モデルで後半50%を予測（in-sample）し、予測 k を `target[k]` と比較（学習ラベル `target[k+1]` と1日ずれ）。その結果を supervisor が「OOF」として読む | `walk_forward_validator.py:72,81,95`、`supervisor_daily_2644t.py:101-139` |
| G-10 | 規約違反 | `tests/` に先読みテストが無い（規約5）。`research/trial_log/` は README のみで、閾値 20→22 などの変更記録が無い（規約6） | grep 0件、`config_model.py:180,184` |
| G-11 | セキュリティ | FRED API キーをソースに平文で保存し、SSL 検証を全体で無効化 | `data_fetcher.py:27,40` |
| G-12 | 規約違反 | `LimitEngine` の数値パラメータ（0.6 / 1.5 / −0.05 / −0.03 / 0.002 / 0.985）がハードコード（規約4） | `common_utils/limit_engine.py:7-10,28,36` |

---

## 8. FAIL 項目の修正方針（各3行以内）

- **A-3 FRED**: 列名の部分一致（`feature_base.py:103-112`）をやめ、`config` に系列ごとの「市場/公表ラグ」を持たせて as-of 結合する。
  日次 FRED は公表日（ALFRED の `realtime_start`）基準、月次（FEDFUNDS, IRLTLT01JPM156N）は参照月末＋公表日で日付を付け直す。
  「行 t の値は東京 t 15:30 以前に公表済み」を検証する先読みテストを追加してから `FEATURE_SETS` に入れる。
- **B-1 分割段差**: 特徴量生成で `data/adjust.py` の `adjust_for_splits`（`corporate_actions.yaml` 読み込み）を必ず通し、他の場所では補正しない。
  差分キャッシュは `config.yaml` の `overlap_days: 15` 分を毎回取り直し、`validate_adj_consistency` で adj/raw 比の段差を検知したら全期間を再取得する。
  分割日前後の調整済みリターンが `jump_bounds` 内に収まることをテストにする。
- **B-2 生値/調整値の混在**: ターゲット（と評価リターン）は `adjust.py` で作った分割・配当調整済み始値から作り、特徴量と同じ基準にそろえる。
  生の始値は発注専用の別列（例 `raw_open`）として保持し、`DailyDecisionController` の参照列を実在する列名に直す。
- **F 実力**: 現状は「常に上昇」とバイ&ホールドに勝っておらず、運用判断には使えない。B・G-1・G-2 を直した後で、
  Optuna（N 選択を含む）を各学習窓の内側だけで回す入れ子ウォークフォワードで再評価し、試行総数を N とした DSR で判定する。
- **G-1 平日グリッド/ラベル先読み**: index を XTKS 営業日（`semi2644.calendar.xtks`）で作り直し、ラベルは行 t に「翌営業日始値→翌々営業日始値」を直接置く。
  `_create_sequences` の暗黙の +1 ずらしに頼らない。休場日前後でラベルが当日始値から始まらないことをテストする。
- **G-2 bfill**: `feature_base.py:90` の `bfill()` を削除し、ウォームアップで NaN の行は `dropna` する。先頭行に未来値が入らないことをテストする。
- **G-3 全期間正規化**: 保存前の RobustScaler（`feature_base.py:313-331`）をやめ、正規化は学習窓の中で fit する（モデル側の StandardScaler に一本化）。
- **G-9 既存 WF スクリプト**: 各時点までのデータで再学習して翌日を予測する形に作り直し、`actual` を学習ラベルと同じ区間（`target[k+1]`）にそろえる。
  supervisor はその真の OOF 予測だけを使う（`audit/wf_ridge_eval.py` の方式を流用できる）。
- **G-10〜G-12（規約・セキュリティ）**: 上記の先読みテストを `tests/` に追加し、閾値変更などは `research/trial_log` に記録する。
  FRED API キーは `semi2644/config/secrets.env`（`semi2644/.gitignore` で除外済み）へ移してキーを失効・再発行し、`ssl._create_unverified_context` を削除する。`LimitEngine` の数値は `config.yaml` へ移す。

---

## 9. 再現方法

```bash
# 依存（監査用。リポジトリの .venv とは別に用意）
pip install pandas numpy scipy scikit-learn pyarrow pyyaml exchange_calendars torch pytorch_lightning optuna joblib
python audit/check_data.py          # A-方法1/2, B, 追加所見 → audit/results/data_checks.json
python audit/optuna_trials.py       # F: Optuna 試行数 → audit/results/optuna_trials.json
python audit/wf_ridge_eval.py --variant asis   --seeds 0,1,2,3,4,5,6,7,8,9   # F（現行）
python audit/wf_ridge_eval.py --variant uslag1 --seeds 0,1,2,3,4,5,6,7,8,9   # A-方法3（米国系+1日）
python audit/wf_ridge_eval.py --variant bfix   --seeds 0,1,2,3,4,5,6,7,8,9   # 参考（分割ラベル修正）
python audit/summarize_wf.py        # F の表
```

- 各スクリプトは `sys.dont_write_bytecode=True` で、リポジトリに `__pycache__` を作らない。Lightning の出力先は `--root-dir` でリポジトリ外を指定可能。
- `wf_ridge_eval.py` の学習手順が本番と同じであることの検証: 全期間データで再現学習したモデルと保存済み `logs/daily_2644t_risk_on_ridge_final_training/version_3/swa.ckpt` の予測相関は 0.959（seed0）/ 0.976（seed1）。先頭 N 特徴量の再現結果は両レジームとも `features.json` と完全一致。
