from __future__ import annotations
import torch
import numpy as np
from torch import nn
from homework7.data.loader import create_dataloader


class NeuralTrainer:
    """Класс для обучения PyTorch моделей."""

    def __init__(
            self,
            model: nn.Module,
            epochs: int = 100,
            lr: float = 0.001) -> None:
        """Инициализация тренера.

        Аргументы:
            model: архитектура для обучения.
            epochs: количество эпох.
            lr: скорость обучения.
        """
        self.model = model
        self.epochs = epochs
        self.lr = lr

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Обучение модели на данных.

        Аргументы:
            x: матрица признаков.
            y: вектор таргета.
        """
        loader = create_dataloader(x, y, batch_size=256)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        criterion = nn.MSELoss()
        self.model.train()
        for _ in range(self.epochs):
            epoch_loss = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step(epoch_loss / len(loader))
