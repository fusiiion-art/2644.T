# 2644.T Trading System

このプロジェクトは、AI予測とルールベースの意思決定を組み合わせた簡易トレーディング実験基盤です。

## 役割の簡潔な整理

- common_utils: 共通ユーティリティ、ターゲット生成、ルールエンジン
- generate: 特徴量生成パイプライン
- models: 予測モデル本体
- timescale_modules: T+1戦略モジュール（daily_2644t を主要ターゲット）
- analysis: 評価・分析スクリプト
- tests: 回帰テストと smoke test

## すぐ始める方法

```bash
c:/2644.T/.venv/Scripts/python.exe main.py demo
```

## 主要コマンド

### 1. 特徴量生成

```bash
c:/2644.T/.venv/Scripts/python.exe main.py features --module daily_2644t
```

### 2. 学習

```bash
c:/2644.T/.venv/Scripts/python.exe main.py train --module daily_2644t --models ridge --n-jobs 1
```

### 3. 推論

```bash
c:/2644.T/.venv/Scripts/python.exe main.py predict --module daily_2644t --model ridge --regime risk_on
```

結果をファイルに保存する場合:

```bash
c:/2644.T/.venv/Scripts/python.exe main.py predict --module daily_2644t --model ridge --regime risk_on --output logs/prediction.json
```

### 4. テスト

```bash
c:/2644.T/.venv/Scripts/python.exe -m pytest -q
```

## 主要な実装ポイント

- multi-target 生成: target_1 などのターゲット列を特徴量生成時に作成
- 既存の RFE 経路は非推奨化され、CFI/first-N フォールバックで動作
- 学習済みモデルと特徴量リストは logs 配下に保存される

## 免責事項

本プロジェクトは情報提供・研究目的であり、投資助言ではありません。
