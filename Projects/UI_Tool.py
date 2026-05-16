import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential


# load trained model
# model = joblib.load("nn_model.pkl")
model = load_model("nn_model.keras")
preprocessing = joblib.load("preprocessing.pkl")

st.title("Default Probability Predictor")
st.write("Enter financial information of the customer below to estimate their default probability. 100% means certain default in next month, while 0% means no default risk.")

# input fields
marriage = st.selectbox("Marital Status (0 = Married, 1 = Single, 3 = Others)", options=[0, 1, 3])
sex = st.selectbox("Sex (1 = Male, 2 = Female)", options=[1, 2])
education = st.selectbox("Education level (1 = Graduate School, 2 = University, 3 = High School, 4 = Others)", options=[1, 2, 3, 4])
age = st.slider("Age", min_value=18, max_value=100, value=50)
LIMIT_BAL = st.number_input("Credit limit assigned to the customer (in currency units)")
AVG_Bill_amt = st.number_input("Average bill amount over the past 6 months.")
average_bill_trend = st.number_input("Trend of average bill amount over the past 6 months (in decimal)")
average_payment = st.number_input("Average payment amount over the past 6 months.")
avg_monthly_pay_to_bill = average_payment / AVG_Bill_amt if AVG_Bill_amt > 0 else 1
avg_payment_trend = st.number_input("Trend of average payment amount over the past 6 months (in decimal)")
pay_0 = st.number_input("First payment status (−2 = No bill, −1 = Fully paid, 0 = Minimum paid, ≥1 = Months delayed)")
pay_2 = st.number_input("Second payment status (−2 = No bill, −1 = Fully paid, 0 = Minimum paid, ≥1 = Months delayed)")
pay_3 = st.number_input("Third payment status (−2 = No bill, −1 = Fully paid, 0 = Minimum paid, ≥1 = Months delayed)")
pay_4 = st.number_input("Fourth payment status (−2 = No bill, −1 = Fully paid, 0 = Minimum paid, ≥1 = Months delayed)")
pay_5 = st.number_input("Fifth payment status (−2 = No bill, −1 = Fully paid, 0 = Minimum paid, ≥1 = Months delayed)")
pay_6 = st.number_input("Sixth payment status (−2 = No bill, −1 = Fully paid, 0 = Minimum paid, ≥1 = Months delayed)")
average_pay_to_bill_trend = ((1+avg_payment_trend) - (1+average_bill_trend))/(1+average_bill_trend) if AVG_Bill_amt > 0 else 1
credit_utilization = AVG_Bill_amt / LIMIT_BAL if LIMIT_BAL > 0 else 0
next_bill = average_bill_trend * AVG_Bill_amt
EAD = next_bill if next_bill > LIMIT_BAL else LIMIT_BAL
late_payment_count = sum([1 for p in [pay_0, pay_2, pay_3, pay_4, pay_5, pay_6] if p >= 1])

# prediction button
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

    X = preprocessing.transform(X)

    score = model.predict(X)[0][0] * 100  # convert to percentage
    next_bill = (1+average_bill_trend) * AVG_Bill_amt
    EAD = min(next_bill, LIMIT_BAL) if next_bill > 0 else LIMIT_BAL
    late_payment_month = sum([p for p in [pay_0, pay_2, pay_3, pay_4, pay_5, pay_6] if p >= 1]) 
    new_limit = LIMIT_BAL * (1 - 0.1 * late_payment_month)  # Reduce limit by 10% for each late payment month
    new_limit = max(new_limit, 0)  # Ensure limit doesn't go negative

    st.success(f"Predicted Default Probability: {score:.2f}%")

    if EAD > 200000 or score > 45:
        st.warning("This customer is classified as High Risk. Consider reducing credit limit and applying stricter credit terms.")
    else:
        st.info("This customer is classified as Low Risk. Standard credit terms may apply.")  
    if new_limit < LIMIT_BAL:
        st.info(f"Recommended new credit limit: € {new_limit:.0f} (reduced from  € {LIMIT_BAL:.0f} due to late payments)")

    