from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

# Константы путей
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_X_PATH = BASE_DIR.parent / "Homework4" / "x_data.npy"
DEFAULT_Y_PATH = BASE_DIR.parent / "Homework4" / "y_data.npy"

# Параметры MLflow
EXPERIMENT_NAME = "Line Regression HH"
TRACKING_URI = "http://kamnsv.com:55000/"
FCN_MODEL_NAME = "syrov_vadim_fcn"
CNN_MODEL_NAME = "syrov_vadim_cnn"


@dataclass(frozen=True)
class ModelParams:
    """Общие параметры для нейронных сетей."""

    input_dim: int = 52
    epochs: int = 200
    batch_size: int = 256
    lr: float = 0.0005
    random_state: int = 42
