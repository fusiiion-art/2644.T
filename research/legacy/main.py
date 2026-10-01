import argparse
import json
import subprocess
import sys
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description="2644.T trading pipeline entry point")
    parser.add_argument("command", nargs="?", default="demo", choices=["demo", "test", "features", "train", "predict"])
    parser.add_argument("--module", default="daily_2644t")
    parser.add_argument("--models", default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--model", default="ridge")
    parser.add_argument("--regime", default="risk_on")
    parser.add_argument("--output", default=None)
    return parser


def run_features(module_name: str):
    script = Path(__file__).resolve().parent / "generate" / "features_daily_features.py"
    cmd = [sys.executable, str(script), "--modules", module_name]
    return subprocess.run(cmd, check=False)


def run_train(module_name: str, models: str | None, n_jobs: int):
    if module_name != "daily_2644t":
        raise ValueError(f"Unsupported module: {module_name}")
    script = Path(__file__).resolve().parent / "timescale_modules" / "daily_2644t" / "trainer_daily_2644t.py"
    cmd = [sys.executable, str(script)]
    if models:
        cmd.extend(["--models", models])
    cmd.extend(["--n-jobs", str(n_jobs)])
    return subprocess.run(cmd, check=False)


def run_predict(module_name: str, model_name: str, regime: str, output_path: str | None = None):
    if module_name != "daily_2644t":
        raise ValueError(f"Unsupported module: {module_name}")

    project_root = Path(__file__).resolve().parent
    import pandas as pd
    import timescale_modules.daily_2644t.config_daily_2644t as cfg
    from common_utils.predicter_model import ExpertPredicter
    from models.linear_model import RidgeLightning
    from sklearn.preprocessing import StandardScaler

    data_path = project_root / "data" / cfg.OUTPUT_FILENAME
    df = pd.read_parquet(data_path)
    df = df[df["Date"] >= cfg.DATA_START_DATE].copy()
    df = df.tail(max(cfg.SEQUENCE_LENGTH, 64)).copy()

    model_dir = project_root / "logs" / f"daily_2644t_{regime}_{model_name}_final_training"
    version_dirs = sorted([p for p in model_dir.glob("version_*") if p.is_dir()], key=lambda p: int(p.name.split("_")[-1]))
    if not version_dirs:
        raise FileNotFoundError(f"No trained model directory found for {regime}/{model_name}")
    latest_dir = version_dirs[-1]

    features_path = latest_dir / "features.json"
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    feature_cols = json.loads(features_path.read_text())

    ckpt_path = latest_dir / "swa.ckpt"
    if not ckpt_path.exists():
        ckpt_path = latest_dir / "best.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found in {latest_dir}")

    model = RidgeLightning.load_from_checkpoint(ckpt_path, map_location="cpu")
    scaler = StandardScaler()
    scaler.fit(df[feature_cols].fillna(0).to_numpy())

    assets = {"model": model, "scaler": scaler, "features": feature_cols}
    predicter = ExpertPredicter(model_name, assets, cfg)
    mu, sigma = predicter.predict(df)
    result = {"model": model_name, "regime": regime, "mu": mu, "sigma": sigma}
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps({**result, "output": str(output_file)}, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False))
    return 0


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "demo":
        print("2644.T project ready")
    elif args.command == "test":
        import pytest
        raise SystemExit(pytest.main(["-q", "tests" ]))
    elif args.command == "features":
        result = run_features(args.module)
        raise SystemExit(result.returncode)
    elif args.command == "train":
        result = run_train(args.module, args.models, args.n_jobs)
        raise SystemExit(result.returncode)
    elif args.command == "predict":
        result = run_predict(args.module, args.model, args.regime, args.output)
        raise SystemExit(result)


if __name__ == "__main__":
    main()
