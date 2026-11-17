from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from schemas import FeatureVectorChurn

NUMERIC_FEATURES = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "autopay_enabled",
]
CATEGORICAL_FEATURES = ["region", "device_type", "payment_method"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "churn"


def build_feature_frame(rows: list[dict]) -> pd.DataFrame:
    """Use the same raw feature names, order and conversions for fit and predict."""
    if not rows:
        raise ValueError("Список объектов пуст")
    frame = pd.DataFrame(rows)
    missing = set(FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Отсутствуют признаки: {', '.join(sorted(missing))}")
    X = frame.loc[:, FEATURE_COLUMNS].copy()
    for column in NUMERIC_FEATURES:
        X[column] = pd.to_numeric(X[column], errors="raise")
    for column in CATEGORICAL_FEATURES:
        X[column] = X[column].astype(object).where(X[column].notna(), float("nan"))
    return X


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ], remainder="drop")


def feature_schema() -> dict:
    return {
        "feature_order": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "features": [
            {
                "name": name,
                "type": FeatureVectorChurn.model_fields[name].annotation.__name__,
                "required": FeatureVectorChurn.model_fields[name].is_required(),
                "group": "numeric" if name in NUMERIC_FEATURES else "categorical",
            }
            for name in FEATURE_COLUMNS
        ],
    }


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer


def extract_features(rows: list[dict]) -> tuple[pd.DataFrame, pd.Series]:
    if not rows:
        raise ValueError("Датасет пуст")
    frame = pd.DataFrame(rows)
    X = build_feature_frame(rows)
    y = frame[TARGET]
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("Таргет churn должен содержать только 0 и 1 без пропусков")
    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError("Для разбиения нужны оба класса churn и минимум две строки каждого класса")

    return X, y.astype(int)


def split_dataset(rows: list[dict], test_size: float = 0.2, random_state: int = 42):
    X, y = extract_features(rows)
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def prepare_data(
    rows: list[dict], test_size: float = 0.2, random_state: int = 42
) -> PreparedData:
    """Split first; learn imputation values on train only to avoid data leakage."""
    X_train, X_test, y_train, y_test = split_dataset(rows, test_size, random_state)
    preprocessor = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), NUMERIC_FEATURES),
            ("categorical", SimpleImputer(strategy="constant", fill_value="unknown"), CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    return PreparedData(X_train, X_test, y_train, y_test, preprocessor)


def split_info(data: PreparedData) -> dict:
    def describe(X: pd.DataFrame, y: pd.Series) -> dict:
        counts = {str(label): int((y == label).sum()) for label in (0, 1)}
        return {
            "row_count": len(X),
            "feature_count": X.shape[1],
            "churn_distribution": counts,
            "churn_proportions": {label: count / len(y) for label, count in counts.items()},
        }

    return {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "train": describe(data.X_train, data.y_train),
        "test": describe(data.X_test, data.y_test),
    }
