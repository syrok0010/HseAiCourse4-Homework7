from __future__ import annotations
from pathlib import Path
from homework7.config import ModelParams, DEFAULT_X_PATH, DEFAULT_Y_PATH
from homework7.data.preprocessor import prepare_data
from homework7.data.scaler import DataScaler
from homework7.models.registry import ModelRegistry
from homework7.training.orchestrator import TrainerOrchestrator
from homework7.tracking.tracker import MLFlowTracker
from homework7.training.model_saver import ModelSaver


def main() -> None:
    """Точка входа: обучение всех нейросетей из реестра."""
    xt, xv, yt, yv = prepare_data(str(DEFAULT_X_PATH), str(DEFAULT_Y_PATH))
    scaler = DataScaler()
    scaler.fit_transform(xt)
    config = ModelParams()
    tracker = MLFlowTracker()
    for m_name in ModelRegistry.list_models():
        orch = TrainerOrchestrator(ModelRegistry.get(m_name), config)
        model, metrics = orch.run(xt, yt, xv, yv, scaler)
        run_name = f"syrov_vadim_{m_name}"
        rid = tracker.log_run(run_name, config.__dict__, metrics, model)
        ModelSaver.save(model, Path(f"resources/{run_name}.pt"))
        print(
            f"Модель {run_name} сохранена с R2: {
                metrics['r2_score_test']:.4f}, ID запуска: {rid}")


if __name__ == "__main__":
    main()
