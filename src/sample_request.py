# src/sample_request.py

"""
Sample client script to test the FastAPI prediction endpoint.

This script sends a POST request to the locally running API.
It is intended for manual testing and demonstration purposes only.

Usage:
    uvicorn src.app:app --reload
    python src/sample_request.py
"""

import requests

url = "http://localhost:8000/predict"

# Example input sample for prediction (as extracted in the notebook.ipynb)
# Note: The target variable "sleep_disorder" is intentionally left out, as this is what we want to predict.
# For this sample, note that this person does not have a sleep disorder.
sample = {
    # "sleep_disorder": "no_disorder",
    "age": 32,
    "bmi_category": "normal",
    "daily_steps": 5000,
    "diastolic_blood_pressure": 80,
    "gender": "male",
    "heart_rate": 72,
    "occupation": "doctor",
    "physical_activity_level": 30,
    "quality_of_sleep": 6,
    "sleep_duration": 6.0,
    "stress_level": 8,
    "systolic_blood_pressure": 125
}

response = requests.post(url, json=sample)

if response.status_code != 200:
    raise RuntimeError(
        f"Request failed ({response.status_code}): {response.text}"
    )

print("Response:", response.json())

