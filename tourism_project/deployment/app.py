
# Importing libraries required for the Streamlit prediction application.
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ------------------------------------------------------------
# 1. Configuring the application
# ------------------------------------------------------------

st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "tourism_model.joblib"
SCHEMA_PATH = APP_DIR / "feature_schema.json"


# ------------------------------------------------------------
# 2. Loading the deployment artefacts
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    """Load and cache the complete preprocessing and prediction pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Deployment model not found at: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_feature_schema():
    """Load and cache the feature configuration used by the input form."""
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Feature schema not found at: {SCHEMA_PATH}"
        )

    with SCHEMA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


try:
    model = load_model()
    feature_schema = load_feature_schema()
except Exception as error:
    st.error(f"Unable to load the deployment artefacts: {error}")
    st.stop()


# ------------------------------------------------------------
# 3. Displaying the application description
# ------------------------------------------------------------

st.title("Tourism Package Purchase Prediction")

st.write(
    "Estimating whether a customer is likely to purchase a tourism "
    "package using profile information available before customer contact."
)

st.info(
    "This prediction supports marketing prioritization and should be "
    "combined with appropriate business review."
)

with st.sidebar:
    st.header("Model Information")
    st.write("**Selected model:** Tuned XGBoost")
    st.write("**Input features:** 13 pre-contact attributes")
    st.write("**Test ROC-AUC:** 0.9037")
    st.write("**Positive-class recall:** 0.6981")
    st.caption(
        "Sales-interaction variables are excluded to prevent temporal leakage."
    )


# ------------------------------------------------------------
# 4. Collecting pre-contact customer information
# ------------------------------------------------------------

categorical_values = feature_schema["categorical_values"]
minimums = feature_schema["numerical_minimums"]
maximums = feature_schema["numerical_maximums"]
defaults = feature_schema["numerical_defaults"]

with st.form("customer_prediction_form"):
    st.subheader("Customer Profile")

    left_column, right_column = st.columns(2)

    with left_column:
        age = st.number_input(
            "Age",
            min_value=int(minimums["Age"]),
            max_value=int(maximums["Age"]),
            value=int(defaults["Age"]),
            step=1,
        )

        city_tier = st.selectbox(
            "City Tier",
            options=[1, 2, 3],
            index=0,
            help="Tier 1 represents the most developed city category.",
        )

        occupation = st.selectbox(
            "Occupation",
            options=categorical_values["Occupation"],
        )

        gender = st.selectbox(
            "Gender",
            options=categorical_values["Gender"],
        )

        persons_visiting = st.number_input(
            "Number of Persons Visiting",
            min_value=int(minimums["NumberOfPersonVisiting"]),
            max_value=int(maximums["NumberOfPersonVisiting"]),
            value=int(defaults["NumberOfPersonVisiting"]),
            step=1,
        )

        preferred_property_star = st.selectbox(
            "Preferred Property Rating",
            options=[
                int(value)
                for value in range(
                    int(minimums["PreferredPropertyStar"]),
                    int(maximums["PreferredPropertyStar"]) + 1,
                )
            ],
        )

        marital_status = st.selectbox(
            "Marital Status",
            options=categorical_values["MaritalStatus"],
        )

    with right_column:
        annual_trips = st.number_input(
            "Average Number of Trips per Year",
            min_value=int(minimums["NumberOfTrips"]),
            max_value=int(maximums["NumberOfTrips"]),
            value=int(defaults["NumberOfTrips"]),
            step=1,
        )

        passport_label = st.selectbox(
            "Valid Passport",
            options=["No", "Yes"],
        )
        passport = 1 if passport_label == "Yes" else 0

        car_label = st.selectbox(
            "Owns a Car",
            options=["No", "Yes"],
        )
        owns_car = 1 if car_label == "Yes" else 0

        children_visiting = st.number_input(
            "Number of Children Visiting",
            min_value=int(minimums["NumberOfChildrenVisiting"]),
            max_value=int(maximums["NumberOfChildrenVisiting"]),
            value=int(defaults["NumberOfChildrenVisiting"]),
            step=1,
        )

        designation = st.selectbox(
            "Designation",
            options=categorical_values["Designation"],
        )

        monthly_income = st.number_input(
            "Monthly Income",
            min_value=float(minimums["MonthlyIncome"]),
            max_value=float(maximums["MonthlyIncome"]),
            value=float(defaults["MonthlyIncome"]),
            step=500.0,
            format="%.2f",
        )

    submitted = st.form_submit_button(
        "Predict Purchase Likelihood",
        type="primary",
        use_container_width=True,
    )


# ------------------------------------------------------------
# 5. Generating and displaying the prediction
# ------------------------------------------------------------

if submitted:
    customer_data = pd.DataFrame(
        [
            {
                "Age": age,
                "CityTier": city_tier,
                "Occupation": occupation,
                "Gender": gender,
                "NumberOfPersonVisiting": persons_visiting,
                "PreferredPropertyStar": preferred_property_star,
                "MaritalStatus": marital_status,
                "NumberOfTrips": annual_trips,
                "Passport": passport,
                "OwnCar": owns_car,
                "NumberOfChildrenVisiting": children_visiting,
                "Designation": designation,
                "MonthlyIncome": monthly_income,
            }
        ],
        columns=feature_schema["features"],
    )

    prediction = int(model.predict(customer_data)[0])
    purchase_probability = float(
        model.predict_proba(customer_data)[0, 1]
    )

    st.divider()
    st.subheader("Prediction Result")

    probability_column, recommendation_column = st.columns(2)

    with probability_column:
        st.metric(
            "Purchase Probability",
            f"{purchase_probability:.1%}",
        )
        st.progress(purchase_probability)

    with recommendation_column:
        if prediction == 1:
            st.success(
                "Likely to purchase — prioritize this customer "
                "for marketing contact."
            )
        else:
            st.warning(
                "Less likely to purchase — assign a lower "
                "marketing-contact priority."
            )

    with st.expander("View model input"):
        st.dataframe(customer_data, use_container_width=True)
