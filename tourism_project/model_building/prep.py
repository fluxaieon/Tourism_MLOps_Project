
# Importing libraries required for data preparation and reproducible splitting.
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------
# 1. Defining reproducible preparation settings
# ------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COLUMN = "ProdTaken"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "tourism.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"


# ------------------------------------------------------------
# 2. Loading the registered raw dataset
# ------------------------------------------------------------

if not RAW_DATA_PATH.exists():
    raise FileNotFoundError(
        f"Registered dataset not found at: {RAW_DATA_PATH}"
    )

dataset = pd.read_csv(RAW_DATA_PATH)

if dataset.empty:
    raise ValueError("Data preparation cannot continue with an empty dataset.")

if TARGET_COLUMN not in dataset.columns:
    raise ValueError(
        f"Required target column '{TARGET_COLUMN}' is unavailable."
    )

print(f"Raw dataset loaded from: {RAW_DATA_PATH}")
print(f"Raw dataset shape: {dataset.shape}")


# ------------------------------------------------------------
# 3. Removing non-predictive columns
# ------------------------------------------------------------

csv_index_columns = [
    column
    for column in dataset.columns
    if column.strip().lower().startswith("unnamed:")
]

columns_to_remove = csv_index_columns.copy()

if "CustomerID" in dataset.columns:
    columns_to_remove.append("CustomerID")

prepared_dataset = dataset.drop(columns=columns_to_remove)

print(f"Removed columns: {columns_to_remove}")


# ------------------------------------------------------------
# 4. Standardizing inconsistent categorical labels
# ------------------------------------------------------------

categorical_corrections = {
    "Gender": {
        "Fe Male": "Female",
    },
    "Occupation": {
        "Free Lancer": "Freelancer",
    },
    "MaritalStatus": {
        "Unmarried": "Single",
    },
}

print("\nCategorical corrections:")

for column, replacements in categorical_corrections.items():
    if column not in prepared_dataset.columns:
        raise ValueError(
            f"Expected categorical column '{column}' is unavailable."
        )

    for original_label, standardized_label in replacements.items():
        affected_records = int(
            prepared_dataset[column].eq(original_label).sum()
        )

        prepared_dataset[column] = prepared_dataset[column].replace(
            original_label,
            standardized_label,
        )

        print(
            f"{column}: '{original_label}' -> "
            f"'{standardized_label}' ({affected_records} records)"
        )


# ------------------------------------------------------------
# 5. Validating the prepared dataset
# ------------------------------------------------------------

remaining_index_columns = [
    column
    for column in prepared_dataset.columns
    if column.strip().lower().startswith("unnamed:")
]

if remaining_index_columns:
    raise ValueError(
        f"CSV index columns remain after preparation: "
        f"{remaining_index_columns}"
    )

if "CustomerID" in prepared_dataset.columns:
    raise ValueError("CustomerID remains after data preparation.")

if prepared_dataset.isna().any().any():
    raise ValueError(
        "Missing values remain after data preparation."
    )

print(f"\nPrepared dataset shape: {prepared_dataset.shape}")
print(
    f"Predictor columns: "
    f"{prepared_dataset.shape[1] - 1}"
)


# ------------------------------------------------------------
# 6. Creating stratified training and testing datasets
# ------------------------------------------------------------

train_dataset, test_dataset = train_test_split(
    prepared_dataset,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=prepared_dataset[TARGET_COLUMN],
)

if len(train_dataset) + len(test_dataset) != len(prepared_dataset):
    raise ValueError("The split datasets do not preserve all input records.")


# ------------------------------------------------------------
# 7. Saving the prepared datasets as workflow artefacts
# ------------------------------------------------------------

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

train_dataset.to_csv(TRAIN_DATA_PATH, index=False)
test_dataset.to_csv(TEST_DATA_PATH, index=False)

print(f"\nTraining dataset saved to: {TRAIN_DATA_PATH}")
print(f"Testing dataset saved to: {TEST_DATA_PATH}")
print(f"Training dataset shape: {train_dataset.shape}")
print(f"Testing dataset shape: {test_dataset.shape}")

print("\nTraining target distribution:")
print(
    train_dataset[TARGET_COLUMN]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
    .to_string()
)

print("\nTesting target distribution:")
print(
    test_dataset[TARGET_COLUMN]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
    .to_string()
)

print("\nData preparation status: PASSED")
