import streamlit as st
import pandas as pd
import joblib
import json
import os

st.set_page_config(page_title='Employee Attrition Prediction', layout='centered')
st.title('🧠 Employee Attrition Prediction')

MODEL_FILE = 'rf_model.pkl'
FEATURE_FILE = 'feature_columns.json'
INFO_FILE = 'feature_info.json'

# Load model
model = joblib.load(MODEL_FILE) if os.path.exists(MODEL_FILE) else None
if model:
    st.success('✅ Model loaded successfully')
else:
    st.error('❌ Model file not found')

# Load feature list
feature_cols = json.load(open(FEATURE_FILE)) if os.path.exists(FEATURE_FILE) else None
if not feature_cols:
    st.error('❌ Feature list not found')

# Load typical (median) values for the features we do not ask for
feature_info = json.load(open(INFO_FILE)) if os.path.exists(INFO_FILE) else {}

# Input section: top 10 features
st.header('Enter Employee Details')
col1, col2 = st.columns(2)
with col1:
    MonthlyIncome = st.number_input('Monthly Income', min_value=1000, max_value=20000, value=5000, step=100)
    Age = st.slider('Age', 18, 60, 30)
    TotalWorkingYears = st.number_input('Total Working Years', min_value=0, max_value=40, value=8)
    OverTime = st.selectbox('OverTime', ['No', 'Yes'])
    DailyRate = st.number_input('Daily Rate', min_value=100, max_value=1500, value=800)

with col2:
    YearsAtCompany = st.number_input('Years At Company', min_value=0, max_value=40, value=5)
    HourlyRate = st.number_input('Hourly Rate', min_value=30, max_value=100, value=50)
    DistanceFromHome = st.number_input('Distance From Home', min_value=1, max_value=29, value=5)
    MonthlyRate = st.number_input('Monthly Rate', min_value=2000, max_value=27000, value=20000)
    NumCompaniesWorked = st.number_input('Num Companies Worked', min_value=0, max_value=9, value=2)

OverTime = 1 if OverTime == 'Yes' else 0

input_df = pd.DataFrame([{
    'MonthlyIncome': MonthlyIncome,
    'Age': Age,
    'TotalWorkingYears': TotalWorkingYears,
    'OverTime': OverTime,
    'DailyRate': DailyRate,
    'YearsAtCompany': YearsAtCompany,
    'HourlyRate': HourlyRate,
    'DistanceFromHome': DistanceFromHome,
    'MonthlyRate': MonthlyRate,
    'NumCompaniesWorked': NumCompaniesWorked
}])

st.subheader('Input Preview')
st.dataframe(input_df)

if st.button('🔍 Predict Attrition'):
    if model is None or feature_cols is None:
        st.error('Model or feature list not loaded.')
    else:
        X_pred = input_df.copy()
        for c in feature_cols:
            if c not in X_pred.columns:
                X_pred[c] = feature_info.get(c, {}).get('median', 0)   # typical value, else 0
        X_pred = X_pred[feature_cols]

        pred = model.predict(X_pred)[0]
        prob = model.predict_proba(X_pred)[0][1]

        if pred == 1:
            st.error(f'⚠️ High risk of attrition (Prob: {prob:.2f})')
        else:
            st.success(f'✅ Low risk of attrition (Prob: {prob:.2f})')
