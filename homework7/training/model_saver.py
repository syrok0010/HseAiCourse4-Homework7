from __future__ import annotations
from pathlib import Path
import torch


class ModelSaver:
    """Утилита для сохранения весов модели."""

    @staticmethod
    def save(model: torch.nn.Module, path: Path) -> None:
        """Сохранение state_dict в указанный путь.

        Аргументы:
            model: сохраняемая модель.
            path: путь к файлу.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
