import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Body, Depends, FastAPI, Query

from dataset import dataset_info, load_dataset
from errors import ServiceError, error_docs, register_error_handlers
from preprocessing import feature_schema, prepare_data, split_info
from schemas import DatasetRowChurn, FeatureVectorChurn, PredictionResponseChurn, TrainingConfigChurn
from inference import predict_churn
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


app = FastAPI(lifespan=lifespan, responses={
    422: error_docs("Некорректные данные запроса"),
    500: error_docs("Внутренняя ошибка сервиса"),
})
register_error_handlers(app)
app.state.model_bundle = None
app.state.model_load_error = None


@app.get(
    "/model/schema",
    summary="Схема входных признаков churn",
    description="Доступна без обученной модели. Порядок полей JSON произвольный; сервис упорядочивает признаки сам.",
)
def get_model_schema() -> dict:
    return feature_schema()


@app.get("/")
def root():
    return {"message": "ml churn service is running"}


PREDICT_EXAMPLE = {
    "monthly_fee": 29.9,
    "usage_hours": 45.5,
    "support_requests": 2,
    "account_age_months": 12,
    "failed_payments": 0,
    "region": "europe",
    "device_type": "mobile",
    "payment_method": "card",
    "autopay_enabled": 1,
}


@app.post(
    "/predict",
    response_model=PredictionResponseChurn | list[PredictionResponseChurn],
    summary="Предсказать отток одного или нескольких клиентов",
    description=(
        "Принимает объект клиента или непустой список объектов. Возвращает объект "
        "или список прогнозов в том же порядке. churn: 0 — останется, 1 — уйдёт. "
        "probabilities содержит вероятности обоих классов. Сначала обучите модель "
        "через POST /model/train либо загрузите сохранённую модель при старте. "
        "Числа в примерах ответа иллюстративные."
    ),
    responses={
        200: {"content": {"application/json": {"examples": {
            "single": {"summary": "Один клиент", "value": {
                "churn": 0, "probabilities": {"0": 0.8, "1": 0.2},
            }},
            "batch": {"summary": "Список клиентов", "value": [
                {"churn": 0, "probabilities": {"0": 0.8, "1": 0.2}},
                {"churn": 1, "probabilities": {"0": 0.3, "1": 0.7}},
            ]},
        }}}},
        422: error_docs("Неверные признаки, типы или пустой список", {
            "invalid_features": {"value": {
                "code": "VALIDATION_ERROR", "message": "Некорректные данные запроса",
                "details": [{"location": ["body", "FeatureVectorChurn", "monthly_fee"],
                             "type": "float_parsing", "message": "Input should be a valid number, unable to parse string as a number"}],
            }},
            "empty_list": {"value": {
                "code": "EMPTY_CLIENT_LIST", "message": "Список клиентов не должен быть пустым", "details": None,
            }},
        }),
        503: error_docs("Модель не обучена или не загружена", {
            "model_unavailable": {"value": {
                "code": "MODEL_NOT_READY",
                "message": "Модель не обучена или не загружена. Выполните POST /model/train",
                "details": None,
            }},
        }),
    },
)
def predict(
    features: FeatureVectorChurn | list[FeatureVectorChurn] = Body(
        ...,
        openapi_examples={
            "single": {"summary": "Один клиент", "value": PREDICT_EXAMPLE},
            "batch": {"summary": "Два клиента", "value": [
                PREDICT_EXAMPLE,
                {**PREDICT_EXAMPLE, "usage_hours": 1, "failed_payments": 3, "autopay_enabled": 0},
            ]},
        },
    ),
) -> PredictionResponseChurn | list[PredictionResponseChurn]:
    if isinstance(features, list) and not features:
        raise ServiceError(422, "EMPTY_CLIENT_LIST", "Список клиентов не должен быть пустым")
    bundle = app.state.model_bundle
    if bundle is None:
        raise ServiceError(
            503, "MODEL_NOT_READY",
            "Модель не обучена или не загружена. Выполните POST /model/train",
        )
    clients = features if isinstance(features, list) else [features]
    try:
        predictions = predict_churn(bundle.model, clients)
    except ValueError as error:
        logging.getLogger(__name__).exception("Ошибка предсказания модели")
        raise ServiceError(422, "PREDICTION_ERROR", "Не удалось получить прогноз по переданным признакам") from error
    return predictions if isinstance(features, list) else predictions[0]


def get_dataset() -> list[dict]:
    try:
        return load_dataset()
    except FileNotFoundError as error:
        raise ServiceError(404, "DATASET_NOT_FOUND", "Датасет data/churn_dataset.csv не найден") from error
    except ValueError as error:
        raise ServiceError(422, "INVALID_DATASET", "CSV содержит некорректные данные") from error
    except OSError as error:
        raise ServiceError(500, "DATASET_READ_ERROR", "Не удалось прочитать churn_dataset.csv") from error


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
    if not rows:
        raise ServiceError(422, "EMPTY_DATASET", "Датасет пуст")
    try:
        return split_info(prepare_data(rows))
    except ValueError as error:
        raise ServiceError(422, "DATA_PREPARATION_ERROR", "Не удалось разделить датасет; проверьте данные и распределение классов") from error


@app.post(
    "/model/train",
    summary="Обучить выбранную модель churn",
    description="Без тела запроса используется logreg. Некорректные настройки возвращают 422.",
    responses={
        404: error_docs("Файл датасета не найден"),
        422: error_docs("Некорректные настройки или датасет", {
            "empty_dataset": {"value": {
                "code": "EMPTY_DATASET", "message": "Датасет пуст", "details": None,
            }},
            "invalid_config": {"value": {
                "code": "TRAINING_ERROR",
                "message": "Не удалось обучить модель. Проверьте гиперпараметры и данные",
                "details": None,
            }},
        }),
    },
)
def train_model(
    config: TrainingConfigChurn | None = Body(
        default=None,
        openapi_examples={
            "logreg": {"summary": "Логистическая регрессия", "value": {
                "model_type": "logreg", "hyperparameters": {"C": 0.5, "max_iter": 1000},
            }},
            "random_forest": {"summary": "Случайный лес", "value": {
                "model_type": "random_forest",
                "hyperparameters": {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 2},
            }},
        },
    ),
    rows: list[dict] = Depends(get_dataset),
) -> dict:
    config = config or TrainingConfigChurn()
    if not rows:
        raise ServiceError(422, "EMPTY_DATASET", "Датасет пуст")
    try:
        pipeline, metrics = train_and_evaluate(rows, config)
    except ValueError as error:
        logging.getLogger(__name__).exception("Ошибка обучения модели")
        raise ServiceError(422, "TRAINING_ERROR", "Не удалось обучить модель. Проверьте гиперпараметры и данные") from error
    bundle = ChurnModelBundle(
        model=pipeline,
        trained_at=datetime.now(timezone.utc),
        metrics=metrics,
        model_type=config.model_type,
        hyperparameters=pipeline.named_steps["classifier"].get_params(deep=False),
    )
    try:
        save_churn_model(bundle)
    except Exception as error:
        logging.getLogger(__name__).exception("Не удалось сохранить churn модель")
        raise ServiceError(500, "MODEL_SAVE_ERROR", "Не удалось сохранить обученную модель") from error
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
        "model_type": bundle.model_type if bundle else None,
        "hyperparameters": bundle.hyperparameters if bundle else None,
    }
