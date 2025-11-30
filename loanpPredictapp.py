
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import sklearn
import pickle   
import requests

# --- Select the .pkl model ---
modelUrl = "https://raw.githubusercontent.com/tluis5312/BUS_458_Final_Case/2d2103fed3e60fd9be34e497f99400d2d88604c0/team_model.pkl"

# Retrieves model from URL
response = requests.get(modelUrl)

# Error handling
if response.status_code != 200:
    st.error("Failed to download model from GitHub.")
    st.stop()

try:
    model = pickle.loads(response.content)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Set global theme for app
globalCss = """
<style>
    body, p, div, label, input, span {
        color: black !important;
    }

    .title-main {
        background-color: #102770;
        padding: 15px;
        color: white;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 15px;
    }

    .title-sub {
        background-color: #3a80b5;
        padding: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
        border-radius: 10px;
        margin-bottom: 20px;
    }

    .stApp {
        background-color: #f5f5f5;
    }
</style>
"""
st.markdown(globalCss, unsafe_allow_html=True)

# Titles
st.markdown("<h1 class='title-main'>Personal Loan Approval Form</h1>", unsafe_allow_html=True)
st.markdown("<div class='title-sub'>Enter Applicant Information Below to Determine Loan Approval:</div>", unsafe_allow_html=True)

# ---Define categorical fields based on previously provided values---
reasonLevels = [
    'cover_an_unexpected_cost',
    'credit_card_refinancing',
    'home_improvement',
    'major_purchase',
    'other',
    'debt_conslidation'
]

employmentStatusLevels = ['full_time', 'part_time', 'unemployed']

employmentSectorLevels = [
    'consumer_discretionary', 'information_technology', 'energy',
    'consumer_staples', 'communication_services', 'materials', 'utilities',
    'real_estate', 'health_care', 'industrials', 'financials', 'Unknown'
]

everBkLevels = [0, 1]

lenderLevels = ['A', 'B', 'C']

# ---Numeric Input Fields---
# Monthly Gross Income
monthlyGrossIncome = st.number_input(
    "Monthly Gross Income ($):",
    min_value=-2559.0,
    max_value=14005.0,
    step=10.0
)
# Loan Amount
grantedLoanAmount = st.slider(
    "Granted Loan Amount ($):",
    min_value=1000,
    max_value=2000000,
    step=1000
)
# FICO Score
ficoScore = st.slider(
    "FICO Score:",
    min_value=300,
    max_value=850,
    step=1
)
# Monthly Housing Payment
monthlyHousingPayment = st.number_input(
    "Monthly Housing Payment ($):",
    min_value=150.0,
    max_value=4400.0,
    step=1
)

# ---Categorical Inputs---
# Reasons 
reason = st.selectbox("Reason for Loan", reasonLevels)
# Employment Status
employmentStatus = st.selectbox("Employment Status:", employmentStatusLevels)
# Employment Sector
employmentSector = st.selectbox("Employment Sector:", employmentSectorLevels)
# Bankruptcy
everBankruptOrForeclose = st.selectbox(
    "Ever Bankrupt or Foreclosed:",
    everBkLevels,
    format_func=lambda x: "Yes" if x == 1 else "No"
)
# Lender
lender = st.selectbox("Lender", lenderLevels)

# ---Build DataFrame---
inputDf = pd.DataFrame({
    # Numeric fields first, then categorical last
    "Monthly_Gross_Income": [monthlyGrossIncome],
    "Granted_Loan_Amount": [grantedLoanAmount],
    "FICO_score": [ficoScore],
    "Monthly_Housing_Payment": [monthlyHousingPayment],
    "Reason": [reason],
    "Employment_Status": [employmentStatus],
    "Employment_Sector": [employmentSector],
    "Ever_Bankrupt_or_Foreclose": [everBankruptOrForeclose],
    "Lender": [lender]
})

# One-hot encode categoricals
inputEncoded = pd.get_dummies(inputDf)

# Ensure all model columns exist
modelCols = model.feature_names_in_
for col in modelCols:
    if col not in inputEncoded:
        inputEncoded[col] = 0

# Reorder to match model training columns
inputEncoded = inputEncoded[modelCols]

# Stylize the sumbit button
buttonStyle = """
    <style>
        div.stButton > button:first-child {
            background-color: #d2fafc;
            color: #004b59;
            border-radius: 8px;
            border: 1px solid #66c2cc;
            padding: 10px 16px;
        }
        div.stButton > button:first-child:hover {
            background-color: #baf4f6;
            color: #003941;
        }
    </style>
"""
st.markdown(buttonStyle, unsafe_allow_html=True)

# Create prediction
if st.button("Evaluate Loan Application:"):
    prediction = model.predict(inputEncoded)[0]
    probability = model.predict_proba(inputEncoded)[0][1] if hasattr(model, "predict_proba") else None

    st.subheader("Prediction Result:", divider="blue")

    if prediction == 1:
        st.error(
            f"🚫 Loan likely to be **Denied / High Risk**\n\nProbability: **{probability:.2f}**"
        )
    else:
        st.success(
            f"💲 Loan likely to be **Approved / Low Risk**\n\nProbability: **{probability:.2f}**"
        )
