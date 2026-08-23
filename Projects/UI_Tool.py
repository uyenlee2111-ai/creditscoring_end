import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---------------------------------------------------------
# Load the final model and calibration objects
# ---------------------------------------------------------
# The new notebook uses a selected tree-based model
# (XGBoost or Random Forest) plus a separate probability
# calibrator. TensorFlow/Keras is no longer required.

model = joblib.load("selected_credit_model.pkl")
calibrator = joblib.load("pd_calibrator.pkl")
pd_threshold = joblib.load("pd_threshold.pkl")
ead_threshold = joblib.load("ead_threshold.pkl")

st.title("Default Probability Predictor")

st.write(
    "Enter the customer's financial information to estimate the "
    "calibrated Probability of Default (PD) for the next month. "
    "The displayed PD is calibrated using the probability-calibration "
    "model developed in the credit-scoring notebook."
)

# ---------------------------------------------------------
# Input fields
# ---------------------------------------------------------
marriage = st.selectbox(
    "Marital Status (0 = Married, 1 = Single, 3 = Others)",
    options=[0, 1, 3]
)

sex = st.selectbox(
    "Sex (1 = Male, 2 = Female)",
    options=[1, 2]
)

education = st.selectbox(
    "Education level (1 = Graduate School, 2 = University, "
    "3 = High School, 4 = Others)",
    options=[1, 2, 3, 4]
)

age = st.slider("Age", min_value=18, max_value=100, value=50)

LIMIT_BAL = st.number_input(
    "Credit limit assigned to the customer",
    min_value=0.0,
    value=50000.0
)

AVG_Bill_amt = st.number_input(
    "Average bill amount over the past 6 months",
    min_value=0.0,
    value=50000.0
)

average_bill_trend = st.number_input(
    "Trend of average bill amount over the past 6 months (decimal)",
    value=0.0
)

average_payment = st.number_input(
    "Average payment amount over the past 6 months",
    min_value=0.0,
    value=5000.0
)

avg_payment_trend = st.number_input(
    "Trend of average payment amount over the past 6 months (decimal)",
    value=0.0
)

pay_0 = st.number_input(
    "First payment status (−2 = No bill, −1 = Fully paid, "
    "0 = Minimum paid, ≥1 = Months delayed)",
    value=0.0
)

pay_2 = st.number_input(
    "Second payment status (−2 = No bill, −1 = Fully paid, "
    "0 = Minimum paid, ≥1 = Months delayed)",
    value=0.0
)

pay_3 = st.number_input(
    "Third payment status (−2 = No bill, −1 = Fully paid, "
    "0 = Minimum paid, ≥1 = Months delayed)",
    value=0.0
)

pay_4 = st.number_input(
    "Fourth payment status (−2 = No bill, −1 = Fully paid, "
    "0 = Minimum paid, ≥1 = Months delayed)",
    value=0.0
)

pay_5 = st.number_input(
    "Fifth payment status (−2 = No bill, −1 = Fully paid, "
    "0 = Minimum paid, ≥1 = Months delayed)",
    value=0.0
)

pay_6 = st.number_input(
    "Sixth payment status (−2 = No bill, −1 = Fully paid, "
    "0 = Minimum paid, ≥1 = Months delayed)",
    value=0.0
)

# ---------------------------------------------------------
# Derived variables — must match the notebook
# ---------------------------------------------------------
avg_monthly_pay_to_bill = (
    average_payment / AVG_Bill_amt
    if AVG_Bill_amt > 0
    else 1.0
)

average_pay_to_bill_trend = (
    ((1 + avg_payment_trend) - (1 + average_bill_trend))
    / (1 + average_bill_trend)
    if (1 + average_bill_trend) != 0
    else 0.0
)

credit_utilization = (
    AVG_Bill_amt / LIMIT_BAL
    if LIMIT_BAL > 0
    else 0.0
)

late_payment_count = sum(
    p >= 1
    for p in [pay_0, pay_2, pay_3, pay_4, pay_5, pay_6]
)

# Consistent with the notebook's EAD definition:
# EAD = minimum(predicted next bill, available credit limit)
next_bill = (1 + average_bill_trend) * AVG_Bill_amt
EAD = min(next_bill, LIMIT_BAL) if next_bill > 0 else LIMIT_BAL

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if st.button("Predict"):

    X = pd.DataFrame([{
        "marriage": marriage,
        "sex": sex,
        "education": education,
        "age": age,
        "LIMIT_BAL": LIMIT_BAL,
        "AVG_Bill_amt": AVG_Bill_amt,
        "late_payment_count": late_payment_count,
        "average_bill_trend": average_bill_trend,
        "average_payment": average_payment,
        "avg_monthly_pay_to_bill": avg_monthly_pay_to_bill,
        "average_pay_to_bill_trend": average_pay_to_bill_trend,
        "credit_utilization": credit_utilization,
        "pay_0": pay_0,
        "pay_2": pay_2,
        "pay_3": pay_3,
        "pay_4": pay_4,
        "pay_5": pay_5,
        "pay_6": pay_6
    }])

    # selected_credit_model.pkl is the complete fitted pipeline,
    # so preprocessing is performed automatically here.
    raw_probability = model.predict_proba(X)[:, 1]

    # Apply the same logit transformation used by the notebook
    # before the logistic calibration model.
    eps = 1e-6
    raw_probability_clipped = np.clip(
        raw_probability,
        eps,
        1 - eps
    )

    raw_logit = np.log(
        raw_probability_clipped / (1 - raw_probability_clipped)
    )

    calibrated_probability = calibrator.predict_proba(
        raw_logit.reshape(-1, 1)
    )[:, 1]

    score = float(calibrated_probability[0])

    # -----------------------------------------------------
    # Risk classification
    # -----------------------------------------------------
    high_risk = (
        score >= pd_threshold
        or EAD >= ead_threshold
    )

    st.success(
        f"Calibrated Probability of Default: {score:.2%}"
    )

    st.caption(
        f"Model PD threshold: {pd_threshold:.2%} | "
        f"EAD threshold: {ead_threshold:,.2f}"
    )

    if high_risk:
        st.warning(
            "This customer is classified as High Risk. "
            "Consider reviewing the credit exposure and applying "
            "appropriate risk-management measures."
        )
    else:
        st.info(
            "This customer is classified as Low Risk. "
            "Standard credit terms may apply."
        )

    # -----------------------------------------------------
    # EAD information
    # -----------------------------------------------------
    st.write(f"**Estimated Exposure at Default (EAD):** € {EAD:,.2f}")

    if EAD >= ead_threshold:
        st.warning(
            "The estimated EAD is above the portfolio's "
            "95th-percentile exposure threshold."
        )

    # -----------------------------------------------------
    # Optional credit-limit recommendation
    # -----------------------------------------------------
    late_payment_month = sum(
        p for p in [pay_0, pay_2, pay_3, pay_4, pay_5, pay_6]
        if p >= 1
    )

    new_limit = LIMIT_BAL * (1 - 0.1 * late_payment_month)
    new_limit = max(new_limit, 0)

    if new_limit < LIMIT_BAL:
        st.info(
            f"Illustrative credit-limit adjustment based on "
            f"late-payment behavior: € {new_limit:,.0f} "
            f"(from € {LIMIT_BAL:,.0f})."
        )
