from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class ErrorResponse(BaseModel):
    code: str = Field(description="Машинный код ошибки")
    message: str = Field(description="Понятное описание ошибки")
    details: JsonValue = Field(default=None, description="Дополнительные сведения или null")


class FeatureVectorChurn(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int


class DatasetRowChurn(FeatureVectorChurn):
    monthly_fee: float | None
    usage_hours: float | None
    support_requests: int | None
    account_age_months: int | None
    failed_payments: int | None
    region: str | None
    device_type: str | None
    payment_method: str | None
    autopay_enabled: int | None
    churn: int


class PredictionResponseChurn(BaseModel):
    churn: Literal[0, 1] = Field(description="0 — клиент останется, 1 — уйдёт")
    probabilities: dict[Literal["0", "1"], float] = Field(
        description='Вероятности классов: "0" — останется, "1" — уйдёт'
    )


class TrainingConfigChurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_type: Literal["logreg", "random_forest"] = Field(
        default="logreg", description="Тип классификатора"
    )
    hyperparameters: dict[str, JsonValue] = Field(
        default_factory=dict, description="Параметры конструктора выбранного классификатора"
    )
