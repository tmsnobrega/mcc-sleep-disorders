# src/model.py
import pickle
import pandas as pd

MODEL_PATH = "models/xgboost_model.pkl"
LABELS_PATH = "models/label_mappings.pkl"


def load_model():
    """Load trained model pipeline and label mappings."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(LABELS_PATH, "rb") as f:
        labels = pickle.load(f)

    return model, labels["inv_label_mapping"]


def predict_single(payload: dict, model, inv_label_mapping):
    """Run inference for a single observation."""
    df = pd.DataFrame([payload])

    y_pred_num = model.predict(df)[0]
    prediction = inv_label_mapping[int(y_pred_num)]

    y_proba = model.predict_proba(df)[0]
    probabilities = {
        inv_label_mapping[i]: float(prob)
        for i, prob in enumerate(y_proba)
    }

    return prediction, probabilities
