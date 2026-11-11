import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

MODEL_PATH = Path(__file__).resolve().parent / "models" / "churn_model.joblib"


@dataclass
class ChurnModelBundle:
    model: Pipeline
    trained_at: datetime
    metrics: dict[str, float]


def save_churn_model(bundle: ChurnModelBundle, path: Path = MODEL_PATH) -> None:
    """Save pipeline and metadata together, replacing the old file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".joblib", delete=False) as file:
        temporary_path = Path(file.name)
    try:
        joblib.dump(bundle, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_churn_model(path: Path = MODEL_PATH) -> ChurnModelBundle:
    bundle = joblib.load(path)
    if not isinstance(bundle, ChurnModelBundle) or not isinstance(bundle.model, Pipeline):
        raise ValueError("Файл не содержит Pipeline churn с метаданными")
    check_is_fitted(bundle.model)
    if not isinstance(bundle.trained_at, datetime) or bundle.trained_at.tzinfo is None:
        raise ValueError("В файле отсутствует дата обучения с часовым поясом")
    if not isinstance(bundle.metrics, dict) or any(
        not isinstance(bundle.metrics.get(name), (int, float))
        or not 0 <= bundle.metrics[name] <= 1
        for name in ("accuracy", "f1")
    ):
        raise ValueError("В файле отсутствуют корректные метрики accuracy и f1")
    return bundle
