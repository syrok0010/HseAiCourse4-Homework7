from __future__ import annotations

import mlflow
import mlflow.pytorch
from torch import nn

from homework7.config import EXPERIMENT_NAME, TRACKING_URI


class MLFlowTracker:
    """Класс для логирования экспериментов в MLflow."""

    def __init__(self):
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

    def log_run(
        self,
        run_name: str,
        params: dict,
        metrics: dict,
        model: nn.Module
    ) -> str:
        """Старт рана и логирование всех данных.

        Аргументы:
            run_name: имя запуска.
            params: словарь параметров.
            metrics: словарь метрик.
            model: обученная модель.

        Вернёт:
            Идентификатор run_id.
        """
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.pytorch.log_model(model, artifact_path="model")
            return run.info.run_id
        return None
