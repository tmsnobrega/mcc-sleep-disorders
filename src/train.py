# src/train.py
import os
import pickle

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


# Configuration
RANDOM_STATE = 42
DATA_PATH = "data/processed/data_clean.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# Load processed data
df = pd.read_csv(DATA_PATH, index_col=0)

TARGET = "sleep_disorder"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Label mappings (required for XGBoost)
label_mapping = {
    "no_disorder": 0,
    "sleep_apnea": 1,
    "insomnia": 2,
}
inv_label_mapping = {v: k for k, v in label_mapping.items()}
y_num = y.map(label_mapping)


# Train / Validation / Test split (60/20/20)
X_train, X_temp, y_train, y_temp, y_train_num, y_temp_num = train_test_split(
    X,
    y,
    y_num,
    test_size=0.4,
    stratify=y,
    random_state=RANDOM_STATE,
)

X_val, X_test, y_val, y_test, y_val_num, y_test_num = train_test_split(
    X_temp,
    y_temp,
    y_temp_num,
    test_size=0.5,
    stratify=y_temp,
    random_state=RANDOM_STATE,
)


# Preprocessing
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)


# Logistic Regression (Baseline / Tuned equivalent)
logreg = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )),
    ]
)

logreg.fit(X_train, y_train)


# Random Forest (Tuned equivalent)
rf = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            random_state=RANDOM_STATE,
        )),
    ]
)

rf.fit(X_train, y_train)


# XGBoost (Final Model)
xgb = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_estimators=300,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
        )),
    ]
)

xgb.fit(X_train, y_train_num)


# Save models
with open(f"{MODEL_DIR}/logreg_model.pkl", "wb") as f:
    pickle.dump(logreg, f)

with open(f"{MODEL_DIR}/random_forest_model.pkl", "wb") as f:
    pickle.dump(rf, f)

with open(f"{MODEL_DIR}/xgboost_model.pkl", "wb") as f:
    pickle.dump(xgb, f)

# Save label mappings (required for inference)
label_artifacts = {
    "label_mapping": label_mapping,
    "inv_label_mapping": inv_label_mapping,
}

with open(f"{MODEL_DIR}/label_mappings.pkl", "wb") as f:
    pickle.dump(label_artifacts, f)

print("Training complete. Models saved to ./models")
