import json
import main


def test_predict_command_supports_output_path_argument():
    parser = main.build_parser()
    args = parser.parse_args(["predict", "--module", "daily_2644t", "--model", "ridge", "--regime", "risk_on", "--output", "out.json"])
    assert args.output == "out.json"
