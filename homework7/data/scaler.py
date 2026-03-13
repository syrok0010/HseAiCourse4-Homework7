from __future__ import annotations
import numpy as np
from sklearn.preprocessing import StandardScaler


class DataScaler:
    """Утилита для нормализации данных."""

    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Обучение и трансформация данных.

        Аргументы:
            x: входные признаки.

        Вернёт:
            Отмасштабированные признаки.
        """
        return self.scaler.fit_transform(x)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Трансформация новых данных.

        Аргументы:
            x: входные признаки.

        Вернёт:
            Отмасштабированные признаки.
        """
        return self.scaler.transform(x)
