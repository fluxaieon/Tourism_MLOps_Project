
# Importing libraries required for model development and experiment tracking.
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


# ------------------------------------------------------------
# 1. Defining paths and reproducible settings
# ------------------------------------------------------------

RANDOM_STATE = 42
CV_FOLDS = 5
TARGET_COLUMN = "ProdTaken"
EXPERIMENT_NAME = "Tourism_Package_Prediction"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"
ARTIFACT_DIR = PROJECT_ROOT / "model_artifacts"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"

TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"
MODEL_PATH = DEPLOYMENT_DIR / "tourism_model.joblib"
SCHEMA_PATH = DEPLOYMENT_DIR / "feature_schema.json"

DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. Loading and validating the prepared datasets
# ------------------------------------------------------------

for required_path in [TRAIN_DATA_PATH, TEST_DATA_PATH]:
    if not required_path.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found at: {required_path}"
        )

train_dataset = pd.read_csv(TRAIN_DATA_PATH)
test_dataset = pd.read_csv(TEST_DATA_PATH)

if TARGET_COLUMN not in train_dataset.columns:
    raise ValueError(f"{TARGET_COLUMN} is missing from the training dataset.")

if TARGET_COLUMN not in test_dataset.columns:
    raise ValueError(f"{TARGET_COLUMN} is missing from the testing dataset.")

print(f"Training dataset shape: {train_dataset.shape}")
print(f"Testing dataset shape: {test_dataset.shape}")


# ------------------------------------------------------------
# 3. Selecting features available before customer contact
# ------------------------------------------------------------

interaction_columns = [
    "TypeofContact",
    "DurationOfPitch",
    "NumberOfFollowups",
    "ProductPitched",
    "PitchSatisfactionScore",
]

pre_contact_features = [
    column
    for column in train_dataset.columns
    if column not in interaction_columns + [TARGET_COLUMN]
]

X_train = train_dataset[pre_contact_features]
y_train = train_dataset[TARGET_COLUMN]

X_test = test_dataset[pre_contact_features]
y_test = test_dataset[TARGET_COLUMN]

if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Training and testing feature schemas do not match.")

print(f"Pre-contact predictors used: {len(pre_contact_features)}")
print(f"Excluded interaction variables: {interaction_columns}")


# ------------------------------------------------------------
# 4. Defining numerical and categorical preprocessing
# ------------------------------------------------------------

categorical_features = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numerical_features = [
    column
    for column in X_train.columns
    if column not in categorical_features
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_features,
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
    ],
    remainder="drop",
)

print(f"Numerical features: {numerical_features}")
print(f"Categorical features: {categorical_features}")


# ------------------------------------------------------------
# 5. Configuring stratified cross-validation
# ------------------------------------------------------------

cross_validation = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# ------------------------------------------------------------
# 6. Establishing a Logistic Regression baseline
# ------------------------------------------------------------

baseline_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

baseline_scores = cross_val_score(
    baseline_pipeline,
    X_train,
    y_train,
    scoring="roc_auc",
    cv=cross_validation,
    n_jobs=1,
)

baseline_cv_roc_auc = float(baseline_scores.mean())

print(
    f"\nBaseline Logistic Regression CV ROC-AUC: "
    f"{baseline_cv_roc_auc:.4f}"
)


# ------------------------------------------------------------
# 7. Defining and tuning the XGBoost model
# ------------------------------------------------------------

negative_records = int((y_train == 0).sum())
positive_records = int((y_train == 1).sum())
class_imbalance_ratio = negative_records / positive_records

xgboost_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=class_imbalance_ratio,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
        ),
    ]
)

hyperparameter_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.05, 0.10],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8],
}

xgboost_search = GridSearchCV(
    estimator=xgboost_pipeline,
    param_grid=hyperparameter_grid,
    scoring="roc_auc",
    cv=cross_validation,
    n_jobs=1,
    refit=True,
    verbose=1,
)

xgboost_search.fit(X_train, y_train)

xgboost_cv_roc_auc = float(xgboost_search.best_score_)

print(f"Best XGBoost CV ROC-AUC: {xgboost_cv_roc_auc:.4f}")
print(f"Best XGBoost parameters: {xgboost_search.best_params_}")


# ------------------------------------------------------------
# 8. Selecting the best model using training-only CV results
# ------------------------------------------------------------

if xgboost_cv_roc_auc >= baseline_cv_roc_auc:
    selected_model_name = "Tuned XGBoost"
    selected_model = xgboost_search.best_estimator_
    selected_cv_roc_auc = xgboost_cv_roc_auc
else:
    selected_model_name = "Logistic Regression"
    selected_model = baseline_pipeline.fit(X_train, y_train)
    selected_cv_roc_auc = baseline_cv_roc_auc

print(f"\nSelected model: {selected_model_name}")
print(f"Selected CV ROC-AUC: {selected_cv_roc_auc:.4f}")


# ------------------------------------------------------------
# 9. Evaluating the selected model on untouched test data
# ------------------------------------------------------------

test_predictions = selected_model.predict(X_test)
test_probabilities = selected_model.predict_proba(X_test)[:, 1]

test_metrics = {
    "accuracy": float(accuracy_score(y_test, test_predictions)),
    "precision": float(
        precision_score(y_test, test_predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, test_predictions, zero_division=0)
    ),
    "f1_score": float(
        f1_score(y_test, test_predictions, zero_division=0)
    ),
    "roc_auc": float(
        roc_auc_score(y_test, test_probabilities)
    ),
}

print("\nTest metrics:")
for metric_name, metric_value in test_metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        test_predictions,
        digits=4,
        zero_division=0,
    )
)


# ------------------------------------------------------------
# 10. Saving evaluation artefacts
# ------------------------------------------------------------

metrics_path = ARTIFACT_DIR / "test_metrics.json"
report_path = ARTIFACT_DIR / "classification_report.json"
confusion_matrix_path = ARTIFACT_DIR / "confusion_matrix.png"

with metrics_path.open("w", encoding="utf-8") as file:
    json.dump(test_metrics, file, indent=4)

classification_results = classification_report(
    y_test,
    test_predictions,
    output_dict=True,
    zero_division=0,
)

with report_path.open("w", encoding="utf-8") as file:
    json.dump(classification_results, file, indent=4)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    test_predictions,
    display_labels=["Not Purchased", "Purchased"],
    cmap="Blues",
)

plt.title(f"Confusion Matrix — {selected_model_name}")
plt.tight_layout()
plt.savefig(confusion_matrix_path, dpi=150)
plt.close()


# ------------------------------------------------------------
# 11. Creating the deployment feature schema
# ------------------------------------------------------------

feature_schema = {
    "target": TARGET_COLUMN,
    "features": pre_contact_features,
    "numerical_features": numerical_features,
    "categorical_features": categorical_features,
    "numerical_defaults": {
        column: float(X_train[column].median())
        for column in numerical_features
    },
    "numerical_minimums": {
        column: float(X_train[column].min())
        for column in numerical_features
    },
    "numerical_maximums": {
        column: float(X_train[column].max())
        for column in numerical_features
    },
    "categorical_values": {
        column: sorted(
            X_train[column].dropna().astype(str).unique().tolist()
        )
        for column in categorical_features
    },
}

with SCHEMA_PATH.open("w", encoding="utf-8") as file:
    json.dump(feature_schema, file, indent=4)


# ------------------------------------------------------------
# 12. Tracking experiments and the selected model with MLflow
# ------------------------------------------------------------

mlflow.set_tracking_uri(MLFLOW_DIR.resolve().as_uri())
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="baseline_logistic_regression"):
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("cv_folds", CV_FOLDS)
    mlflow.log_metric("mean_cv_roc_auc", baseline_cv_roc_auc)

with mlflow.start_run(run_name="tuned_xgboost"):
    mlflow.log_param("model", "XGBClassifier")
    mlflow.log_param("cv_folds", CV_FOLDS)
    mlflow.log_param("scale_pos_weight", class_imbalance_ratio)

    for parameter_name, parameter_value in (
        xgboost_search.best_params_.items()
    ):
        mlflow.log_param(
            parameter_name.replace("model__", ""),
            parameter_value,
        )

    mlflow.log_metric("best_cv_roc_auc", xgboost_cv_roc_auc)

with mlflow.start_run(run_name="selected_deployment_model"):
    mlflow.log_param("selected_model", selected_model_name)
    mlflow.log_param("feature_count", len(pre_contact_features))
    mlflow.log_param(
        "excluded_interaction_features",
        ", ".join(interaction_columns),
    )
    mlflow.log_metric("selected_cv_roc_auc", selected_cv_roc_auc)

    for metric_name, metric_value in test_metrics.items():
        mlflow.log_metric(f"test_{metric_name}", metric_value)

    mlflow.log_artifact(str(metrics_path))
    mlflow.log_artifact(str(report_path))
    mlflow.log_artifact(str(confusion_matrix_path))
    mlflow.log_artifact(str(SCHEMA_PATH))

    model_signature = infer_signature(X_test, test_predictions)

    mlflow.sklearn.log_model(
        sk_model=selected_model,
        name="model",
        signature=model_signature,
        input_example=X_train.head(5),
    )


# ------------------------------------------------------------
# 13. Saving the complete deployment pipeline
# ------------------------------------------------------------

joblib.dump(selected_model, MODEL_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "The trained deployment model was not saved."
    )

print(f"\nDeployment model saved to: {MODEL_PATH}")
print(f"Feature schema saved to: {SCHEMA_PATH}")
print(f"MLflow tracking directory: {MLFLOW_DIR}")
print("Model training and tracking status: PASSED")
