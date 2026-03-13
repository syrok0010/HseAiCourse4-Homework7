from __future__ import annotations

import numpy as np
from torch import nn

from homework7.config import ModelParams
from homework7.models.wrappers.base_regressor import BaseNeuralRegressor
from homework7.training.metrics import MetricsHelper
from homework7.training.neural_trainer import NeuralTrainer


class TrainerOrchestrator:
    """Оркестратор процесса обучения одной модели."""

    def __init__(self,
                 model_module: type[nn.Module],
                 config: ModelParams) -> None:
        """Инициализация оркестратора.

        Аргументы:
            model_module: класс нейронной сети.
            config: параметры обучения.
        """
        self.model = model_module(config.input_dim)
        self.config = config
        self.trainer = NeuralTrainer(self.model, config.epochs, config.lr)

    def run(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        scaler: object
    ) -> tuple:
        """Полный цикл для одной модели.

        Аргументы:
            x_train: тренировочные признаки.
            y_train: тренировочный таргет.
            x_test: тестовые признаки.
            y_test: тестовый таргет.
            scaler: объект для масштабирования.

        Вернёт:
            Кортеж (обученная модель, словарь метрик).
        """
        self.trainer.fit(scaler.transform(x_train), y_train)
        wrapper = BaseNeuralRegressor(self.model, scaler)
        y_pred = wrapper.predict(x_test)
        return self.model, MetricsHelper.calculate(y_test, y_pred)
