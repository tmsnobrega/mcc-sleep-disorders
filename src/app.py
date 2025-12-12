# src/app.py
from fastapi import FastAPI
from pydantic import BaseModel

from src.model import load_model, predict_single

app = FastAPI(title="Sleep Disorder Classification API")

model, inv_label_mapping = load_model()


class SleepFeatures(BaseModel):
    gender: str
    age: int
    occupation: str
    sleep_duration: float
    quality_of_sleep: int
    physical_activity_level: int
    stress_level: int
    bmi_category: str
    heart_rate: int
    daily_steps: int
    systolic_blood_pressure: int
    diastolic_blood_pressure: int


@app.post("/predict")
def predict(features: SleepFeatures):
    prediction, probabilities = predict_single(
        features.dict(), model, inv_label_mapping
    )

    return {
        "sleep_disorder_prediction": prediction,
        "class_probabilities": probabilities,
    }


@app.get("/health")
def health():
    return {"status": "ok"}
