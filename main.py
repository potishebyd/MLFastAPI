import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Query

from dataset import dataset_info, load_dataset
from preprocessing import prepare_data, split_info
from schemas import DatasetRowChurn, FeatureVectorChurn
from training import train_and_evaluate
from model_storage import ChurnModelBundle, load_churn_model, save_churn_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_bundle = None
    app.state.model_load_error = None
    try:
        app.state.model_bundle = load_churn_model()
    except FileNotFoundError:
        pass
    except Exception:
        logging.getLogger(__name__).exception("Не удалось загрузить churn модель")
        app.state.model_load_error = "Не удалось загрузить сохранённую модель; запустите обучение заново"
    yield


app = FastAPI(lifespan=lifespan)
app.state.model_bundle = None
app.state.model_load_error = None


@app.get("/")
def root():
    return {"message": "ml churn service is running"}


@app.post("/predict", response_model=FeatureVectorChurn)
def predict(features: FeatureVectorChurn) -> FeatureVectorChurn:
    return features


def get_dataset() -> list[dict]:
    try:
        return load_dataset()
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Датасет data/churn_dataset.csv не найден") from error
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=500, detail="Не удалось загрузить churn_dataset.csv") from error


@app.get("/dataset/preview", response_model=list[DatasetRowChurn])
def preview_dataset(
    n: int = Query(default=5, ge=1, description="Количество первых строк"),
    rows: list[dict] = Depends(get_dataset),
) -> list[dict]:
    return rows[:n]


@app.get("/dataset/info")
def get_dataset_info(rows: list[dict] = Depends(get_dataset)) -> dict:
    return dataset_info(rows)


@app.get("/dataset/split-info")
def get_split_info(rows: list[dict] = Depends(get_dataset)) -> dict:
    try:
        return split_info(prepare_data(rows))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/model/train")
def train_model(rows: list[dict] = Depends(get_dataset)) -> dict:
    try:
        pipeline, metrics = train_and_evaluate(rows)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    bundle = ChurnModelBundle(pipeline, datetime.now(timezone.utc), metrics)
    try:
        save_churn_model(bundle)
    except Exception as error:
        logging.getLogger(__name__).exception("Не удалось сохранить churn модель")
        raise HTTPException(status_code=500, detail="Не удалось сохранить обученную модель") from error
    app.state.model_bundle = bundle
    app.state.model_load_error = None
    return metrics


@app.get("/model/status")
def model_status() -> dict:
    bundle = app.state.model_bundle
    return {
        "is_trained": bundle is not None,
        "trained_at": bundle.trained_at if bundle else None,
        "metrics": bundle.metrics if bundle else None,
        "load_error": app.state.model_load_error,
    }
