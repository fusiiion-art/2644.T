import numpy as np
import torch

# 評価指標はNumpyとTorchの両方に対応できるように修正

def directional_accuracy(y_pred, y_true):
    """
    予測の方向性（符号）がどれだけ正しかったかを評価する指標。
    y_pred, y_trueはNumpy配列またはTorchテンソルのいずれか。
    """
    if isinstance(y_pred, torch.Tensor):
        pred_signs = torch.sign(y_pred)
        true_signs = torch.sign(y_true)
        correct_directions = (pred_signs == true_signs).float().mean().item()
    else: # Numpy配列を想定
        pred_signs = np.sign(y_pred)
        true_signs = np.sign(y_true)
        correct_directions = np.mean(pred_signs == true_signs)
    return correct_directions
