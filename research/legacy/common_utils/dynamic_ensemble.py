import torch
import torch.nn as nn
import numpy as np
from typing import List, Any


def generate_oos_predictions(models: List[Any], X, y, cv_splitter) -> 'np.ndarray':
    """
    Generate out-of-sample predictions for each model using cv_splitter.
    Returns a 2D numpy array (n_samples, n_models) with NaN for in-sample rows.
    """
    import pandas as pd
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    oos = np.full((len(X_df), len(models)), np.nan, dtype=float)
    for i, model in enumerate(models):
        preds = np.full(len(X_df), np.nan, dtype=float)
        for train_idx, test_idx in cv_splitter.split(X_df, y):
            X_tr = X_df.iloc[train_idx]
            X_te = X_df.iloc[test_idx]
            y_tr = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            try:
                model.fit(X_tr, y_tr)
                p = model.predict(X_te)
                preds[test_idx] = p
            except Exception:
                continue
        oos[:, i] = preds
    return oos


class AttentionGateTrainer:
    def __init__(self, model: nn.Module, lr: float = 1e-3, batch_size: int = 32, device: str = None):
        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def fit(self, expert_preds: np.ndarray, state: np.ndarray, targets: np.ndarray, epochs: int = 10):
        import torch
        X_e = torch.from_numpy(expert_preds).float()
        X_s = torch.from_numpy(state).float()
        y = torch.from_numpy(targets).float().reshape(-1, 1)

        dataset = torch.utils.data.TensorDataset(X_e, X_s, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for _ in range(epochs):
            for xb_e, xb_s, by in loader:
                xb_e = xb_e.to(self.device)
                xb_s = xb_s.to(self.device)
                by = by.to(self.device)
                pred, _ = self.model(xb_e, xb_s)
                loss = self.loss_fn(pred.view(-1,1), by)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

    def predict(self, expert_preds: np.ndarray, state: np.ndarray):
        import torch
        self.model.eval()
        with torch.no_grad():
            xe = torch.from_numpy(expert_preds).float().to(self.device)
            xs = torch.from_numpy(state).float().to(self.device)
            pred, weights = self.model(xe, xs)
            return pred.cpu().numpy(), weights.cpu().numpy()


class AttentionGateEnsemble(nn.Module):
    def __init__(self, n_experts: int, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.n_experts = n_experts
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.attn = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, expert_preds: torch.Tensor, state: torch.Tensor):
        """
        expert_preds: (batch, n_experts)
        state: (batch, state_dim)
        returns: (ensemble_pred, weights)
        """
        # ensure tensors are on the same device as model
        try:
            dev = next(self.parameters()).device
        except StopIteration:
            dev = torch.device('cpu')
        if expert_preds.device != dev:
            expert_preds = expert_preds.to(dev)
        if state.device != dev:
            state = state.to(dev)

        weights = self.attn(state)
        ensemble = (weights * expert_preds).sum(dim=-1)
        return ensemble, weights
