from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    extract_features,
    split_dataset,
)


def train_churn_model(rows: list[dict]) -> Pipeline:
    """Fit a complete pipeline on the supplied training rows."""
    X, y = extract_features(rows)
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipeline.fit(X, y)
    return pipeline


def train_and_evaluate(rows: list[dict]) -> tuple[Pipeline, dict]:
    X_train, X_test, y_train, y_test = split_dataset(rows)
    train_rows = X_train.assign(**{TARGET: y_train}).to_dict(orient="records")
    pipeline = train_churn_model(train_rows)
    predictions = pipeline.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, pos_label=1, zero_division=0)),
    }
    return pipeline, metrics
