import pandas as pd
from types import SimpleNamespace
import pytorch_lightning as pl

from common_utils.trainer_model import ExpertTrainer


class DummyModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()


def test_get_feature_columns_uses_config_feature_sets_when_present():
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "target_1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_b": [0.5, 1.0, 1.5, 2.0, 2.5],
            "regime": [0, 0, 0, 0, 0],
        }
    )
    config = SimpleNamespace(
        FEATURE_SETS=["TEST_SET"],
        FEATURE_SETS_DEFINITIONS={"TEST_SET": [r"^feature_.*"]},
        PREDICTION_HORIZON=1,
        SEQUENCE_LENGTH=4,
        TARGET_COLUMN="target_1",
        FEATURE_PARAMS={},
        USE_CFI=False,
        BLACKLIST_FEATURES=[],
        MAX_EPOCHS=1,
        BATCH_SIZE=2,
        LOSS_TYPE="mse",
        PRECISION=32,
        GRADIENT_CLIP_VAL=0.5,
        ACCUMULATE_GRAD_BATCHES=1,
        REGIMES=[0],
        REGIME_VIX_COLUMN="vix_close",
        REGIME_VIX_THRESHOLD=20.0,
        REGIME_EXPERTS={"risk_on": ["ridge"]},
        N_TRIALS=1,
    )

    trainer = ExpertTrainer(
        model_name="ridge",
        model_class=DummyModel,
        df_regime=df,
        config=config,
        project_root=".",
        run_name="test_run",
        log_base_dir=".",
    )

    assert "feature_a" in trainer.all_feature_candidates
    assert "feature_b" in trainer.all_feature_candidates
