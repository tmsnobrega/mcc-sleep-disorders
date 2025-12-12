# **💤 Sleep Disorder Classification**

**Health Data • Multi-class Classification • Machine Learning • FastAPI • Docker • Cloud Deployment**

## **Introduction**

Sleep disorders are widespread and often underdiagnosed, yet they have a significant impact on physical health, mental well-being, and overall quality of life. Early identification using routinely collected health and lifestyle data can support preventative care, risk stratification, and more efficient clinical decision-making. From a data perspective, this problem is well-suited for supervised learning: the features are structured, the target is categorical, and model outputs can be directly translated into actionable insights when used responsibly.

## **Project Overview**

This project implements a **multi-class machine learning system** to predict sleep disorders using health, lifestyle, and demographic data. It follows a structured ML workflow: data cleaning, exploratory analysis, feature preparation, model training and comparison, and preparation for deployment via an API.

## **Dataset**

The dataset used in this project is the Sleep Health and Lifestyle Dataset, publicly available on Kaggle:

🔗 https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset

It captures a broad range of attributes related to sleep and daily habits, including:

- Sleep metrics such as duration and quality  
- Lifestyle factors including physical activity, stress levels, and daily steps  
- Health indicators like BMI category, blood pressure, and heart rate  
- Demographic characteristics such as age, gender, and occupation  
- Sleep disorder classification, which serves as the target variable for modeling

### Column Reference

- Person ID – Unique identifier for each individual  
- Gender – Male or Female  
- Age – Age in years  
- Occupation – Profession or job title  
- Sleep Duration – Average hours of sleep per day  
- Quality of Sleep – Self-reported rating from 1 (poor) to 10 (excellent)  
- Physical Activity Level – Minutes of physical activity per day  
- Stress Level – Self-reported rating from 1 (low) to 10 (high)  
- BMI Category – Weight classification based on BMI (e.g., Underweight, Normal, Overweight)
- Blood Pressure – Systolic/diastolic measurement  
- Heart Rate – Resting heart rate in bpm  
- Daily Steps – Average number of steps per day  
- Sleep Disorder – Target variable: None, Insomnia, or Sleep Apnea

### Sleep Disorder Classes

- None – No diagnosed sleep disorder  
- Insomnia – Persistent difficulty falling or staying asleep, resulting in inadequate rest  
- Sleep Apnea – Breathing interruptions during sleep, leading to fragmented sleep and potential health risks  

## **Repository Structure**

```text
mcc_sleep_disorders/
│
├── data/
│   └── processed/
│       └── data_clean.csv                      # Cleaned dataset used for training
│   └── raw/
│       └── sleep_health_and_lifestyle.csv      # Raw dataset
│
├── models/
│   ├── logreg_model.pkl                        # Logistic Regression pipeline
│   ├── random_forest_model.pkl                 # Random Forest pipeline
│   ├── xgboost_model.pkl                       # Final selected model
│   └── label_mappings.pkl                      # Label encoders for inference (XGBoost model)
│
├── src/
│   ├── train.py                                # Reproducible training script
│   ├── model.py                                # Model loading & inference logic
│   ├── app.py                                  # FastAPI application
│   └── sample_request.py                       # Sample client request script
│
├── notebook.ipynb                              # EDA, cleaning, feature analysis
├── pyproject.toml                              # Dependencies (uv)
├── Dockerfile                                  
├── uv.lock
└── README.md
```

## **Workflow**

- Perform EDA and feature engineering in a notebook

- Export a cleaned dataset for reproducible training

- Train and compare multiple classification models

- Select and persist the best-performing model

- Prepare inference logic for API-based predictions

- ⚠️ FUTURE: [Add Docker]

- ⚠️ FUTURE: [Add cloud deployment]


## **Key EDA Insights**

The whole EDA is documented in `notebook.ipynb`. Key analysis areas include:

**1) Target Imbalance**  
- The dataset is imbalanced: *no disorder* (~59%) is the majority class, while *sleep apnea* and *insomnia* each represent ~21%. This motivates the use of macro-averaged evaluation metrics.

**2) Sleep Metrics Differentiate Classes**  
- Insomnia shows the lowest sleep duration and sleep quality.  
- Sleep apnea exhibits moderate but variable sleep metrics.  
- No disorder maintains higher overall sleep quality.

**3) Lifestyle & Stress Patterns Are Predictive**  
- Higher stress and lower physical activity/steps are common in both disorder groups, especially insomnia.  
- No disorder individuals report lower stress and higher activity levels.

**4) Cardiovascular Indicators Distinguish Sleep Apnea**  
- Elevated heart rate and systolic/diastolic blood pressure are strongly associated with sleep apnea.

**5) BMI and Age Show Relevant Associations**  
- Overweight/obese individuals appear more frequently in sleep apnea and insomnia cases.  
- Age contributes meaningfully to class separation.

**6) Correlation & Pairwise Analysis Suggest Nonlinear Class Boundaries**  
- Strong observed correlations: sleep duration ↔ sleep quality, activity ↔ steps, systolic ↔ diastolic.  
- Pairplots show class clustering but with nonlinear separation.

**7) Mutual Information Identifies Top Predictors**  
- Highest MI features include blood pressure, age, sleep duration, daily steps, heart rate, physical activity, and BMI.  
- Gender and most occupation categories contribute minimally.

## **Model Comparison Summary**

The table below compares the **tuned versions** of all three models evaluated in this notebook:  
- **Logistic Regression** (linear baseline)  
- **Random Forest** (tree-based ensemble)  
- **XGBoost** (gradient boosting)

Metrics are based on validation set performance, using macro-averaged scores to ensure equal weight across classes.

### Validation Performance Comparison

| Model               | Accuracy | F1 (Macro) | Recall (Macro) | Precision (Macro) | ROC AUC (Macro OVR) |
|--------------------|----------|------------|------------------|--------------------|-----------------------|
| Logistic Regression | 0.867    | 0.845      | 0.854           | 0.841             | 0.338                 |
| Random Forest       | 0.907    | 0.878      | 0.878           | 0.878             | 0.358                 |
| XGBoost             | **0.907** | **0.878**   | **0.878**        | **0.878**          | **0.899**             |


### Interpretation and Final Model Selection

Across all evaluated models, Random Forest and XGBoost achieve the highest accuracy and macro-averaged scores, clearly outperforming logistic regression.  
However, **XGBoost provides significantly higher ROC AUC**, indicating superior probability calibration and better ability to separate classes in a multiclass setting.

### Final Model Choice: XGBoost

**XGBoost is selected as the final model** because it provides the best overall validation performance, combining high predictive accuracy, balanced class performance, and excellent probability calibration.


## **Deployment**

This section describes how to run the project **locally**, **using Docker**, and **in the cloud**.  
All options start from the same GitHub repository and use the same trained model artifacts.

### Clone the Repository

```
git clone https://github.com/<your-username>/mcc_sleep_disorders.git
cd mcc_sleep_disorders
```

### Run Locally (Without Docker)

This option is useful for development, debugging, or inspection of the API.

#### Prerequisites

- **Python 3.13**  
- **uv** (Python package manager)

#### Sync Dependencies

```
uv sync
```

This installs all dependencies defined in `pyproject.toml` and locked in `uv.lock`.

#### Start the API

```
uv run uvicorn src.app:app --host 0.0.0.0 --port 8000
```

The API will be available at:

```
http://localhost:8000/docs
```

### Run with Docker

**Docker** provides a fully isolated and reproducible runtime environment.

#### Prerequisites

- **Docker** (Docker Desktop or Docker Engine)

#### Build the Image

With Docker running, from the project root:

```
docker build -t sleep-disorder-classifier .
```

#### Run the Container

```
docker run -p 8000:8000 sleep-disorder-classifier
```

The FastAPI service will be accessible at:

```
http://localhost:8000/docs
```

### Cloud Deployment (Render)

This project can be deployed as a containerized web service using **Render**.

#### Push the Repository to GitHub

Ensure your repository contains:

- `Dockerfile`  
- Trained model artifacts in `models/`  
- Application code in `src/`

#### Create a Web Service on Render

**1)** Go to the Render dashboard  
**2)** Click **Add New → Web Service**  
**3)** Select the GitHub repository => mcc-sleep-disorders
**4)** Choose **Docker** as the runtime  

#### Render Configuration

- **Dockerfile Path:** `./Dockerfile`  
- **Port:** `8000`  
- **Start Command:**

```
uv run uvicorn src.app:app --host 0.0.0.0 --port 8000
```

#### Render Will

- Build the Docker image  
- Start the FastAPI application  
- Provide a public service URL 

The deployed service is available at:
👉 [https://mcc-sleep-disorders.onrender.com/docs](https://mcc-sleep-disorders.onrender.com/docs)

### Using the API

POST/predict

- Accepts a JSON payload with health and lifestyle features  
- Returns the predicted sleep disorder class  

#### Example Client Request

A sample request script is provided to test the FastAPI prediction endpoint:

```
src/sample_request.py
```

This script sends a POST request to the locally running API.
It is intended for manual testing and demonstration purposes only.

You can run it with:

```
python src/sample_request.py
```
