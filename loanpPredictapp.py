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

# Set global theme for app using CSS
minimalCss = """
<style>
    /* Make outer background soft gray */
    .stApp {
        background-color: #e9ecef;
    }

    /* Modern spacing for inputs */
    .block-container {
        padding-top: 2rem;
    }

    /* Make labels black (outside dropdowns) */
    label, .stSelectbox label, .stNumberInput label, .stSlider label {
        color: black !important;
        font-weight: 600;
    }
    
     /* Subtitle centered */
    .title-sub {
        text-align: center;
        font-weight: 600;
        margin-bottom: 1.5rem;
        color: black;
    }
    
    /* Closed dropdown box text */
    .stSelectbox > div[data-baseweb="select"] > div {
        color: white !important;
        background-color: #1f1f1f !important;  /* modern dark dropdown */
        border-radius: 6px;
        padding-left: 10px;
    }

    /* Default styling for dropdown options */
    .stSelectbox [role="listbox"] div {
        color: black !important;   /* dropdown options be black */
        background-color: white !important;
    }
</style>
"""
st.markdown(minimalCss, unsafe_allow_html=True)

# Main Title
st.markdown(
    "<h1 style='text-align:center; color:#FF4C4C; font-weight:900;'>Personal Loan Approval Form</h1>",
    unsafe_allow_html=True
)

# Subtitle
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
    value=0.0, 
    step=10.0
)

# Loan Amount
grantedLoanAmount = st.slider(
    "Granted Loan Amount ($):",
    min_value=1000.0,
    max_value=2000000.0,
    value=1000.0,
    step=1000.0
)

# FICO Score
ficoScore = st.slider(
    "FICO Score:",
    min_value=300,
    max_value=850,
    value=300,
    step=1
)

# Monthly Housing Payment
monthlyHousingPayment = st.number_input(
    "Monthly Housing Payment ($):",
    min_value=150.0,
    max_value=4400.0,
    value=150.0,
    step=1.0
)

# ---Categorical Inputs---
reason = st.selectbox("Reason for Loan", reasonLevels)
employmentStatus = st.selectbox("Employment Status:", employmentStatusLevels)
employmentSector = st.selectbox("Employment Sector:", employmentSectorLevels)

everBankruptOrForeclose = st.selectbox(
    "Ever Bankrupt or Foreclosed:",
    everBkLevels,
    format_func=lambda x: "Yes" if x == 1 else "No"
)

lender = st.selectbox("Lender", lenderLevels)

# ---Build DataFrame---
inputDf = pd.DataFrame({
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

# --- DO NOT ONE-HOT ENCODE (pipeline handles encoding internally) ---

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
    prediction = model.predict(inputDf)[0]
    probability = model.predict_proba(inputDf)[0][1] if hasattr(model, "predict_proba") else None

    st.subheader("Prediction Result:", divider="blue")
    
# Approval Message
    if prediction == 1:
        st.success(
            f"💲 Loan likely to be **APPROVED / Low Risk**\n\nProbability: **{probability:.2f}**"
    )
        
    else:
        st.error(
            f"🚫 Loan likely to be **DENIED / High Risk**\n\nProbability: **{probability:.2f}**"
        )
