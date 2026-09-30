import numpy as np
import pandas as pd
from common_utils.dynamic_ensemble import AttentionGateEnsemble
from common_utils.dynamic_ensemble import AttentionGateTrainer


def test_attention_gate_forward_and_train():
    rng = np.random.RandomState(0)
    n = 200
    n_experts = 4
    state_dim = 3

    # synthetic expert preds and state
    expert_preds = rng.normal(size=(n, n_experts)).astype(float)
    state = rng.normal(size=(n, state_dim)).astype(float)
    # synthetic target is weighted sum
    true_w = np.array([0.1, 0.2, 0.3, 0.4])
    targets = (expert_preds * true_w).sum(axis=1) + rng.normal(scale=0.01, size=n)

    model = AttentionGateEnsemble(n_experts=n_experts, state_dim=state_dim, hidden_dim=16)
    trainer = AttentionGateTrainer(model, lr=1e-2, batch_size=32)

    # forward before training
    pred0, w0 = model.__call__(__import__('torch').from_numpy(expert_preds[:5]).float(), __import__('torch').from_numpy(state[:5]).float())
    assert pred0.shape[0] == 5

    # train for a few epochs
    trainer.fit(expert_preds, state, targets, epochs=5)
    pred1, w1 = trainer.predict(expert_preds[:10], state[:10])
    assert pred1.shape[0] == 10
