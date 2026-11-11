from pydantic import BaseModel


class FeatureVectorChurn(BaseModel):
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
