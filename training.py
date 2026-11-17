from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from schemas import TrainingConfigChurn

from preprocessing import (
    TARGET,
    build_preprocessor,
    extract_features,
    split_dataset,
)


def create_classifier(config: TrainingConfigChurn):
    if config.model_type == "logreg":
        classifier_class = LogisticRegression
        defaults = {"max_iter": 1000, "random_state": 42}
    else:
        classifier_class = RandomForestClassifier
        defaults = {"n_estimators": 100, "random_state": 42}
    try:
        return classifier_class(**{**defaults, **config.hyperparameters})
    except TypeError as error:
        raise ValueError(f"Некорректные гиперпараметры {config.model_type}: {error}") from error


def train_churn_model(
    rows: list[dict], config: TrainingConfigChurn | None = None
) -> Pipeline:
    """Fit a complete pipeline on the supplied training rows."""
    X, y = extract_features(rows)
    classifier = create_classifier(config or TrainingConfigChurn())
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier),
    ])
    pipeline.fit(X, y)
    return pipeline


def train_and_evaluate(
    rows: list[dict], config: TrainingConfigChurn | None = None
) -> tuple[Pipeline, dict]:
    X_train, X_test, y_train, y_test = split_dataset(rows)
    train_rows = X_train.assign(**{TARGET: y_train}).to_dict(orient="records")
    pipeline = train_churn_model(train_rows, config)
    predictions = pipeline.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, pos_label=1, zero_division=0)),
    }
    return pipeline, metrics
