from __future__ import annotations
import torch
from torch import nn


class FCNModule(nn.Module):
    """Полносвязная нейронная сеть для задачи регрессии."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [512, 256, 128, 64],
        dropout: float = 0.2
    ) -> None:
        """Инициализация слоев сети.

        Аргументы:
            input_dim: количество входных признаков.
            hidden_dims: список размеров скрытых слоев.
            dropout: вероятность зануления нейрона.
        """
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход.

        Аргументы:
            x: входной тензор признаков.

        Вернёт:
            Предсказание (в логарифмической шкале).
        """
        return self.net(x)
