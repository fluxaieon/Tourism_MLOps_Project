
from pathlib import Path

import pandas as pd


# ------------------------------------------------------------
# 1. Locating and loading the registered dataset
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "tourism.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Registered dataset not found at: {DATA_PATH}"
    )

dataset = pd.read_csv(DATA_PATH)

if dataset.empty:
    raise ValueError("Dataset validation failed because the dataset is empty.")


# ------------------------------------------------------------
# 2. Defining the required business columns
# ------------------------------------------------------------

EXPECTED_BUSINESS_COLUMNS = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
]


# ------------------------------------------------------------
# 3. Identifying documented CSV index artefacts
# ------------------------------------------------------------

csv_index_columns = [
    column
    for column in dataset.columns
    if column.strip().lower().startswith("unnamed:")
]

business_columns = [
    column
    for column in dataset.columns
    if column not in csv_index_columns
]


# ------------------------------------------------------------
# 4. Validating the dataset schema
# ------------------------------------------------------------

missing_columns = sorted(
    set(EXPECTED_BUSINESS_COLUMNS) - set(business_columns)
)

unexpected_columns = sorted(
    set(business_columns) - set(EXPECTED_BUSINESS_COLUMNS)
)

if missing_columns:
    raise ValueError(
        f"Dataset validation failed. Missing columns: {missing_columns}"
    )

if unexpected_columns:
    raise ValueError(
        f"Dataset validation failed. Unexpected columns: {unexpected_columns}"
    )


# ------------------------------------------------------------
# 5. Validating the customer identifier
# ------------------------------------------------------------

missing_customer_ids = int(dataset["CustomerID"].isna().sum())
duplicate_customer_ids = int(dataset["CustomerID"].duplicated().sum())

if missing_customer_ids > 0:
    raise ValueError(
        f"CustomerID contains {missing_customer_ids} missing values."
    )

if duplicate_customer_ids > 0:
    raise ValueError(
        f"CustomerID contains {duplicate_customer_ids} duplicate values."
    )


# ------------------------------------------------------------
# 6. Validating the binary target variable
# ------------------------------------------------------------

if dataset["ProdTaken"].isna().any():
    raise ValueError("ProdTaken contains missing target values.")

observed_target_values = set(dataset["ProdTaken"].unique())
expected_target_values = {0, 1}

if observed_target_values != expected_target_values:
    raise ValueError(
        "ProdTaken must contain both binary classes 0 and 1. "
        f"Observed values: {sorted(observed_target_values)}"
    )


# ------------------------------------------------------------
# 7. Producing the registration summary
# ------------------------------------------------------------

duplicate_rows = int(dataset.duplicated().sum())
total_missing_values = int(dataset.isna().sum().sum())

target_summary = (
    dataset["ProdTaken"]
    .value_counts()
    .sort_index()
    .rename_axis("ProdTaken")
    .to_frame("RecordCount")
)

target_summary["Percentage"] = (
    target_summary["RecordCount"] / len(dataset) * 100
).round(2)

print("Dataset registration summary")
print("-" * 40)
print(f"Dataset path: {DATA_PATH}")
print(f"Number of rows: {dataset.shape[0]:,}")
print(f"Number of raw columns: {dataset.shape[1]}")
print(f"Number of business columns: {len(business_columns)}")
print(f"CSV index artefacts: {csv_index_columns}")
print(f"Duplicate rows: {duplicate_rows}")
print(f"Duplicate CustomerID values: {duplicate_customer_ids}")
print(f"Total missing values: {total_missing_values}")

print("\nTarget-class distribution:")
print(target_summary.to_string())

print("\nDataset validation status: PASSED")
