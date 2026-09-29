import main


def test_cli_parser_supports_workflow_commands():
    parser = main.build_parser()

    features_args = parser.parse_args(["features", "--module", "daily_2644t"])
    assert features_args.command == "features"
    assert features_args.module == "daily_2644t"

    train_args = parser.parse_args(["train", "--module", "daily_2644t", "--models", "ridge", "--n-jobs", "2"])
    assert train_args.command == "train"
    assert train_args.module == "daily_2644t"
    assert train_args.models == "ridge"
    assert train_args.n_jobs == 2

    predict_args = parser.parse_args(["predict", "--module", "daily_2644t", "--model", "ridge", "--regime", "risk_on"])
    assert predict_args.command == "predict"
    assert predict_args.model == "ridge"
    assert predict_args.regime == "risk_on"
