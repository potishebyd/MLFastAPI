from sklearn.pipeline import Pipeline

from preprocessing import build_feature_frame
from schemas import FeatureVectorChurn, PredictionResponseChurn


def predict_churn(
    model: Pipeline, clients: list[FeatureVectorChurn]
) -> list[PredictionResponseChurn]:
    X = build_feature_frame([client.model_dump() for client in clients])
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    return [
        PredictionResponseChurn(
            churn=int(prediction),
            probabilities={
                str(int(label)): float(probability)
                for label, probability in zip(model.classes_, class_probabilities)
            },
        )
        for prediction, class_probabilities in zip(predictions, probabilities)
    ]
