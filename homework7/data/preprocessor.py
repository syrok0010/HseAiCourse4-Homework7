from __future__ import annotations
import numpy as np
from sklearn.model_selection import train_test_split


def prepare_data(x_path: str,
                 y_path: str) -> tuple[np.ndarray,
                                       np.ndarray,
                                       np.ndarray,
                                       np.ndarray]:
    """Загрузка и разделение данных на train и test.

    Аргументы:
        x_path: путь к файлу с признаками.
        y_path: путь к файлу с таргетом.

    Вернёт:
        Кортеж (x_train, x_test, y_train, y_test).
    """
    x = np.load(x_path)
    y = np.load(y_path)
    return train_test_split(x, y, test_size=0.2, random_state=42)
