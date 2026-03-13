from __future__ import annotations

from torch import nn

from homework7.models.nn.cnn_module import CNNModule
from homework7.models.nn.fcn_module import FCNModule


class ModelRegistry:
    """Реестр доступных архитектур нейросетей."""

    _models = {
        "fcn": FCNModule,
        "cnn": CNNModule
    }

    @classmethod
    def get(cls, name: str) -> type[nn.Module]:
        """Возвращает класс модели по имени."""
        if name not in cls._models:
            raise ValueError(f"Неизвестная модель: {name}")
        return cls._models[name]

    @classmethod
    def list_models(cls) -> list[str]:
        """Возвращает список имен моделей."""
        return list(cls._models.keys())
