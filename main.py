import streamlit as st
import pandas as pd
import joblib
from src.rule_engine import RuleEngine

st.set_page_config(page_title="Loan Decision", layout="centered")

@st.cache_resource
def load_pipeline():
    return joblib.load("models/logistic_regression_model.joblib")

pipeline = load_pipeline()
engine = RuleEngine(pd_threshold=0.12) 


PURPOSE = {
    "Debt consolidation": "debt_consolidation",
    "Car purchase": "car",
    "Home improvement": "home_improvement",
    "Personal/other": "personal",
}
HOME = {
    "Renting": "rent",
    "Own outright": "own",
    "Mortgage": "mortgage",
    "Other": "other",
}
CHANNEL = {
    "Online application": "online",
    "In-branch": "branch",
}
REGION = {
    "North": "north",
    "East": "east",
    "South": "south",
    "West": "west",
}
TERM = {
    "12 months": 12,
    "24 months": 24,
    "36 months": 36,
    "48 months": 48,
    "60 months": 60,
}

EMPLOYMENT_LENGTH = {
    "< 1 year": 0,
    "1–2 years": 2,
    "3–5 years": 5,
    "6–10 years": 10,
    "11+ years": 11,
}

st.title("Loan Decision System")
st.subheader("Fill in the applicant details to get a loan decision, based on predicted probability of default (PD) and business rules.")

with st.form("applicant"):
    col1, col2 = st.columns(2)

    with col1:
        purpose_label = st.selectbox("Purpose for the loan?", list(PURPOSE.keys()))
        home_label = st.selectbox("Home ownership status?", list(HOME.keys()))
        channel_label = st.selectbox("Where are you applying from?", list(CHANNEL.keys()))
        region_label = st.selectbox("Which region are you from?", list(REGION.keys()))
        term_label = st.selectbox("Loan term", list(TERM.keys()))
        loan_amount = st.number_input("Loan amount (£)(100-1,000,000)", 1000, 1_000_000, 10_000, step=500)

    with col2:
        age = st.slider("Age", 18, 100, 32)
        annual_income = st.number_input("Annual income (£)", 0, 500_000, 30_000, step=1_000)
        employment_length_label = st.selectbox("Employment length", list(EMPLOYMENT_LENGTH.keys())
        )
        credit_score = st.slider("Credit score", 0, 999, 700, step=1)
        dti_percent = st.slider("Debt-to-income (%)", 0, 100, 20, step=1)
        interest_percent = st.slider("Interest rate (%)", 0.0, 100.0, 5.0, step=0.1)
        delinquencies_2y = st.number_input("Delinquencies (in the last 2 years)", 0, 20, 0, step=1)
        inquiries_6m = st.number_input("Credit inquiries (in the last 6 months)", 0, 20, 1, step=1)
        num_open_accounts = st.number_input("Open credit accounts", 0, 50, 5, step=1)

    submitted = st.form_submit_button("Get decision")


if submitted:
    applicant = {
        "purpose": PURPOSE[purpose_label],
        "home_ownership": HOME[home_label],
        "channel": CHANNEL[channel_label],
        "region": REGION[region_label],
        "loan_term_months": TERM[term_label],
        "age": age,
        "annual_income": annual_income,
        "employment_length": EMPLOYMENT_LENGTH[employment_length_label],
        "credit_score": credit_score,
        "debt_to_income": dti_percent / 100.0,
        "num_open_accounts": num_open_accounts,
        "delinquencies_2y": delinquencies_2y,
        "inquiries_6m": inquiries_6m,
        "loan_amount": loan_amount,
        "interest_rate": interest_percent / 100.0,  
    }

    df = pd.DataFrame([applicant])
    pd_value = float(pipeline.predict_proba(df)[:, 1][0])
    result = engine.apply_rules(applicant, pd_value)

    st.metric("Predicted PD", f"{pd_value:.2%}")
    st.subheader(f"Decision: {result['Decision']}")
    st.write("Reasons:")
    for r in result["Reasons"]:
        st.write(f"- {r}")

