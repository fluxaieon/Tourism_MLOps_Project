
# Importing libraries required for automated pipeline validation.
import json
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "tourism.csv"
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

MODEL_PATH = PROJECT_ROOT / "deployment" / "tourism_model.joblib"
SCHEMA_PATH = PROJECT_ROOT / "deployment" / "feature_schema.json"
APP_PATH = PROJECT_ROOT / "deployment" / "app.py"
DEPLOYMENT_REQUIREMENTS_PATH = (
    PROJECT_ROOT / "deployment" / "requirements.txt"
)
WORKFLOW_PATH = (
    REPOSITORY_ROOT / ".github" / "workflows" / "pipeline.yml"
)

TARGET_COLUMN = "ProdTaken"

EXPECTED_PRE_CONTACT_FEATURES = [
    "Age",
    "CityTier",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
]

EXCLUDED_COLUMNS = {
    "Unnamed: 0",
    "CustomerID",
    "TypeofContact",
    "DurationOfPitch",
    "NumberOfFollowups",
    "ProductPitched",
    "PitchSatisfactionScore",
}


def test_registered_dataset_is_valid():
    """Confirm that the registered raw dataset has the expected target."""
    assert RAW_DATA_PATH.exists()

    raw_dataset = pd.read_csv(RAW_DATA_PATH)

    assert not raw_dataset.empty
    assert TARGET_COLUMN in raw_dataset.columns
    assert set(raw_dataset[TARGET_COLUMN].unique()) == {0, 1}
    assert raw_dataset["CustomerID"].is_unique
    assert raw_dataset.duplicated().sum() == 0


def test_prepared_splits_are_valid():
    """Confirm that the prepared splits preserve records and target balance."""
    assert TRAIN_DATA_PATH.exists()
    assert TEST_DATA_PATH.exists()

    train_dataset = pd.read_csv(TRAIN_DATA_PATH)
    test_dataset = pd.read_csv(TEST_DATA_PATH)
    raw_dataset = pd.read_csv(RAW_DATA_PATH)

    assert len(train_dataset) + len(test_dataset) == len(raw_dataset)
    assert TARGET_COLUMN in train_dataset.columns
    assert TARGET_COLUMN in test_dataset.columns
    assert not train_dataset.isna().any().any()
    assert not test_dataset.isna().any().any()

    for excluded_column in ["Unnamed: 0", "CustomerID"]:
        assert excluded_column not in train_dataset.columns
        assert excluded_column not in test_dataset.columns

    class_proportion_difference = abs(
        train_dataset[TARGET_COLUMN].mean()
        - test_dataset[TARGET_COLUMN].mean()
    )

    assert class_proportion_difference < 0.01


def test_deployment_schema_uses_pre_contact_features():
    """Confirm that the deployment schema excludes interaction leakage."""
    assert SCHEMA_PATH.exists()

    with SCHEMA_PATH.open("r", encoding="utf-8") as file:
        feature_schema = json.load(file)

    assert feature_schema["features"] == EXPECTED_PRE_CONTACT_FEATURES
    assert len(feature_schema["features"]) == 13
    assert not EXCLUDED_COLUMNS.intersection(
        feature_schema["features"]
    )


def test_deployment_model_generates_valid_prediction():
    """Confirm that the saved model accepts schema-compliant input."""
    assert MODEL_PATH.exists()
    assert SCHEMA_PATH.exists()

    deployment_model = joblib.load(MODEL_PATH)

    with SCHEMA_PATH.open("r", encoding="utf-8") as file:
        feature_schema = json.load(file)

    sample_customer = {}

    for feature in feature_schema["features"]:
        if feature in feature_schema["numerical_features"]:
            sample_customer[feature] = (
                feature_schema["numerical_defaults"][feature]
            )
        else:
            sample_customer[feature] = (
                feature_schema["categorical_values"][feature][0]
            )

    sample_data = pd.DataFrame(
        [sample_customer],
        columns=feature_schema["features"],
    )

    prediction = int(deployment_model.predict(sample_data)[0])
    probability = float(
        deployment_model.predict_proba(sample_data)[0, 1]
    )

    assert prediction in {0, 1}
    assert 0 <= probability <= 1


def test_streamlit_application_is_deployment_ready():
    """Confirm that the application and dependencies are valid."""
    assert APP_PATH.exists()
    assert DEPLOYMENT_REQUIREMENTS_PATH.exists()

    app_source = APP_PATH.read_text(encoding="utf-8")

    # Compiling in memory validates syntax without creating pycache files.
    compile(app_source, str(APP_PATH), "exec")

    required_app_content = [
        "st.form(",
        "pd.DataFrame(",
        "model.predict(",
        "model.predict_proba(",
        'feature_schema["features"]',
    ]

    for required_content in required_app_content:
        assert required_content in app_source

    deployment_requirements = (
        DEPLOYMENT_REQUIREMENTS_PATH.read_text(encoding="utf-8")
    )

    assert "streamlit==1.60.0" in deployment_requirements
    assert "scikit-learn==1.6.1" in deployment_requirements
    assert "xgboost==2.1.4" in deployment_requirements


def test_github_actions_workflow_is_complete():
    """Confirm that the workflow contains no unfinished placeholders."""
    assert WORKFLOW_PATH.exists()

    workflow_source = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "<add_code_here>" not in workflow_source
    assert "data-registration:" in workflow_source
    assert "data-preparation:" in workflow_source
    assert "model-training:" in workflow_source
    assert "deployment-validation:" in workflow_source
    assert "python -m pytest tourism_project/tests" in workflow_source
