import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👥",
    layout="wide"
)

# Load model and feature names
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(base_dir, 'models', 'attrition_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(base_dir, 'models', 'feature_names.pkl'), 'rb') as f:
        features = pickle.load(f)
    return model, features

model, feature_names = load_model()

# Title
st.title("👥 Employee Attrition Predictor")
st.markdown("**Predict whether an employee is likely to leave — and understand why.**")
st.markdown("---")

# Sidebar inputs
st.sidebar.header("Employee Profile")
st.sidebar.markdown("Fill in the employee details below:")

age = st.sidebar.slider("Age", 18, 60, 30)
monthly_income = st.sidebar.number_input("Monthly Income ($)", 1000, 20000, 5000, step=500)
job_level = st.sidebar.selectbox("Job Level", [1, 2, 3, 4, 5])
overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])
distance_from_home = st.sidebar.slider("Distance From Home (miles)", 1, 30, 5)
years_at_company = st.sidebar.slider("Years at Company", 0, 40, 3)
years_in_current_role = st.sidebar.slider("Years in Current Role", 0, 20, 2)
total_working_years = st.sidebar.slider("Total Working Years", 0, 40, 5)
num_companies_worked = st.sidebar.slider("Num Companies Worked", 0, 10, 2)
job_satisfaction = st.sidebar.selectbox("Job Satisfaction (1=Low, 4=High)", [1, 2, 3, 4])
work_life_balance = st.sidebar.selectbox("Work Life Balance (1=Bad, 4=Best)", [1, 2, 3, 4])
environment_satisfaction = st.sidebar.selectbox("Environment Satisfaction (1=Low, 4=High)", [1, 2, 3, 4])
department = st.sidebar.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
stock_option_level = st.sidebar.selectbox("Stock Option Level", [0, 1, 2, 3])

# Build input dataframe with all features
def build_input():
    # Create a base input with median values for all features
    input_dict = {col: [0] for col in feature_names}
    
    # Fill in user provided values
    input_dict['Age'] = [age]
    input_dict['MonthlyIncome'] = [monthly_income]
    input_dict['JobLevel'] = [job_level]
    input_dict['OverTime'] = [1 if overtime == "Yes" else 0]
    input_dict['DistanceFromHome'] = [distance_from_home]
    input_dict['YearsAtCompany'] = [years_at_company]
    input_dict['YearsInCurrentRole'] = [years_in_current_role]
    input_dict['TotalWorkingYears'] = [total_working_years]
    input_dict['NumCompaniesWorked'] = [num_companies_worked]
    input_dict['JobSatisfaction'] = [job_satisfaction]
    input_dict['WorkLifeBalance'] = [work_life_balance]
    input_dict['EnvironmentSatisfaction'] = [environment_satisfaction]
    input_dict['Department'] = [0 if department == "Human Resources" else 1 if department == "Research & Development" else 2]
    input_dict['MaritalStatus'] = [0 if marital_status == "Divorced" else 1 if marital_status == "Married" else 2]
    input_dict['StockOptionLevel'] = [stock_option_level]
    
    return pd.DataFrame(input_dict)

# Predict button
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    predict_btn = st.button("🔍 Predict Attrition Risk", use_container_width=True)

if predict_btn:
    input_df = build_input()
    prob = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]
    
    st.markdown("---")
    
    # Risk Score
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_color = "#e74c3c" if prob > 0.5 else "#f39c12" if prob > 0.3 else "#2ecc71"
        st.markdown(f"""
        <div style='background-color:{risk_color};padding:20px;border-radius:10px;text-align:center;'>
            <h2 style='color:white;margin:0;'>{'🔴 HIGH RISK' if prob > 0.5 else '🟡 MEDIUM RISK' if prob > 0.3 else '🟢 LOW RISK'}</h2>
            <h1 style='color:white;margin:0;'>{prob*100:.1f}%</h1>
            <p style='color:white;margin:0;'>Attrition Probability</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Monthly Income", f"${monthly_income:,}")
        st.metric("Years at Company", f"{years_at_company} years")
        st.metric("Job Level", f"Level {job_level}")
    
    with col3:
        st.metric("OverTime", overtime)
        st.metric("Job Satisfaction", f"{job_satisfaction}/4")
        st.metric("Work Life Balance", f"{work_life_balance}/4")
    
    # Key Risk Factors
    st.markdown("---")
    st.subheader("📊 Key Risk Factors")
    
    risk_factors = []
    if overtime == "Yes":
        risk_factors.append("⚠️ Employee is working overtime — 3x higher attrition risk")
    if job_level == 1:
        risk_factors.append("⚠️ Entry level position — highest attrition group (27%)")
    if monthly_income < 3000:
        risk_factors.append("⚠️ Low monthly income — below average for retained employees")
    if job_satisfaction <= 2:
        risk_factors.append("⚠️ Low job satisfaction score")
    if years_at_company <= 2:
        risk_factors.append("⚠️ Less than 2 years at company — critical retention window")
    if distance_from_home > 15:
        risk_factors.append("⚠️ High distance from home — top positive attrition correlator")
    if num_companies_worked > 4:
        risk_factors.append("⚠️ Worked at many companies — history of job hopping")
    if marital_status == "Single":
        risk_factors.append("⚠️ Single employees statistically leave more often")
    
    if risk_factors:
        for factor in risk_factors:
            st.warning(factor)
    else:
        st.success("✅ No major risk factors detected for this employee profile")
    
    # Business Impact
    st.markdown("---")
    st.subheader("💰 Estimated Business Impact")
    replacement_cost = monthly_income * 6
    st.error(f"Estimated cost to replace this employee: **${replacement_cost:,}**")
    st.caption("Industry standard: replacing an employee costs 6–9 months of their salary (SHRM, 2023)")

# Footer
st.markdown("---")
st.markdown("Built with scikit-learn, SHAP, and Streamlit | IBM HR Analytics Dataset | Nithin Kilari")