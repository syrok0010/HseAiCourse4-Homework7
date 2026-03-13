from __future__ import annotations
import numpy as np
import torch
from torch import nn
from sklearn.base import BaseEstimator, RegressorMixin


class BaseNeuralRegressor(BaseEstimator, RegressorMixin):
    """Базовый класс для обертки PyTorch моделей в Scikit-Learn стиль."""

    def __init__(self, model: nn.Module, scaler: object | None = None) -> None:
        self.model = model
        self.scaler = scaler

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Предсказание с обратным лог-преобразованием.

        Аргументы:
            x: матрица признаков для предсказания.

        Вернёт:
            Вектор предсказаний зарплат.
        """
        self.model.eval()
        x_proc = self.scaler.transform(x) if self.scaler else x
        x_tensor = torch.tensor(x_proc, dtype=torch.float32)
        with torch.no_grad():
            y_log = self.model(x_tensor).cpu().numpy()
        return np.expm1(y_log).flatten()
