from __future__ import annotations
from torch import nn, Tensor


class CNNModule(nn.Module):
    """Сверточная нейронная сеть для табличных данных."""

    def __init__(self, input_dim: int, out_channels: int = 32) -> None:
        """Инициализирует архитектуру CNN.

        Аргументы:
            input_dim: количество признаков.
            out_channels: количество каналов в первом слое.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels * 2),
            nn.GELU())
        self.fc = nn.Linear(out_channels * 2 * input_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Прямой проход.

        Аргументы:
            x: входной тензор признаков.

        Вернёт:
            Предсказание (в логарифмической шкале).
        """
        x_cnn = x.unsqueeze(1)
        features = self.conv(x_cnn)
        return self.fc(features.flatten(1))
