from __future__ import annotations
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error


class MetricsHelper:
    """Утилита для вычисления метрик качества."""

    def __init__(self):
        pass

    @staticmethod
    def calculate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Расчет R2, MAE и MAPE.

        Аргументы:
            y_true: истинные значения.
            y_pred: предсказанные значения.

        Вернёт:
            Словарь с вычисленными метриками.
        """
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        return {
            "r2_score_test": float(r2),
            "mae_test": float(mae),
            "mape_test": float(
                np.mean(
                    np.abs(
                        (y_true - y_pred) / y_true)) * 100)}
