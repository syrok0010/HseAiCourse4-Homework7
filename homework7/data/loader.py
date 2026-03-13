from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def create_dataloader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True
) -> DataLoader:
    """Создает DataLoader из numpy массивов.

    Аргументы:
        x: матрица признаков.
        y: вектор целевой переменной.
        batch_size: размер батча.
        shuffle: перемешивать ли данные.

    Вернёт:
        Объект DataLoader для итерации.
    """
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(np.log1p(y), dtype=torch.float32).reshape(-1, 1)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
