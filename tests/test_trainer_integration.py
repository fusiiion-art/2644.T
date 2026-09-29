import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVR
from common_utils.trainer_model import ExpertTrainer

class SVRModel:
    """Light wrapper to provide a minimal interface compatible with trainer usage."""
    @staticmethod
    def define_param_space(trial):
        return {
            'C': trial.suggest_float('svr_C', 1e-2, 1e2, log=True),
            'epsilon': trial.suggest_float('svr_epsilon', 1e-4, 1e-1, log=True),
            'gamma': trial.suggest_categorical('svr_gamma', ['scale', 'auto'])
        }

    def __init__(self, **params):
        allowed = {'C', 'epsilon', 'kernel', 'degree', 'gamma', 'coef0', 'shrinking', 'tol', 'cache_size', 'max_iter'}
        svr_kwargs = {k: v for k, v in params.items() if k in allowed}
        self.params = params
        self.model = SVR(**svr_kwargs)

    def fit_lgbm(self, X_tr, y_tr, X_val=None, y_val=None):
        self.model.fit(X_tr.reshape(len(X_tr), -1), y_tr.ravel())

    def predict(self, X):
        return self.model.predict(X.reshape(len(X), -1))


class DummyConfig:
    PREDICTION_HORIZON = 1
    FEATURE_SETS = []
    SEQUENCE_LENGTH = 4
    SEQUENCE_LENGTH_S5 = 120
    MAX_EPOCHS = 1
    BATCH_SIZE = 16
    PRECISION = 32
    GRADIENT_CLIP_VAL = 0.5
    ACCUMULATE_GRAD_BATCHES = 1
    N_TRIALS = 1
    OBJECTIVE_METRIC = 'val_loss'
    TASK_TYPE = 'regression'


def test_trainer_objective_runs():
    # synthetic dataset
    n = 200
    rng = np.random.RandomState(0)
    df = pd.DataFrame({f'f{i}': rng.normal(size=n) for i in range(8)})
    # create a target column
    df['target_1'] = 0.5 * df['f0'] + 0.2 * df['f1'] + rng.normal(scale=0.01, size=n)
    # no Date column to avoid LightGBM dtype issues in feature importance

    config = DummyConfig()
    project_root = Path('.')
    run_name = 'test_run'
    log_dir = project_root / 'logs' / 'test'
    log_dir.mkdir(parents=True, exist_ok=True)

    trainer = ExpertTrainer(
        model_name='svr',
        model_class=SVRModel,
        df_regime=df,
        config=config,
        project_root=project_root,
        run_name=run_name,
        log_base_dir=log_dir
    )

    class DummyTrial:
        def __init__(self):
            self.params = {}
            self.number = 0
        def suggest_loguniform(self, name, a, b):
            val = (a + b) / 2.0
            self.params[name] = val
            return val
        def suggest_float(self, name, a, b, log=False):
            val = (a + b) / 2.0
            self.params[name] = val
            return val
        def suggest_categorical(self, name, choices):
            val = choices[0]
            self.params[name] = val
            return val
        def suggest_int(self, name, a, b):
            val = max(a, min(10, b))
            self.params[name] = val
            return val
        def set_user_attr(self, k, v):
            pass
        def report(self, *args, **kwargs): pass
        def should_prune(self): return False

    trial = DummyTrial()
    res = trainer.objective(trial)
    assert np.isfinite(res)
