# -*- coding: utf-8 -*-
import os
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")

@st.cache_resource
def load_model():
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_recall_curve
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'raw', 'attrition.csv')

    df = pd.read_csv(data_path)
    df = df.drop(columns=['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'])
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop(columns=['Attrition'])
    y = df['Attrition']

    X = X.copy()
    X['IncomePerYear'] = X['MonthlyIncome'] / (X['TotalWorkingYears'] + 1)
    X['SatisfactionScore'] = (X['JobSatisfaction'] + X['EnvironmentSatisfaction'] + X['WorkLifeBalance']) / 3
    X['RiskScore'] = X['OverTime'] * (1 / X['JobSatisfaction']) * X['DistanceFromHome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scale = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale,
        random_state=42, eval_metric='logloss', verbosity=0
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_threshold = float(thresholds[np.argmax(f1_scores)])

    return model, X_train.columns.tolist(), X_train.median().to_dict(), best_threshold


# ── Load model ──
model, feature_names, median_values, best_threshold = load_model()


# ── Sidebar ──
st.sidebar.header("Employee Profile")
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


# ── Build input ──
def build_input():
    input_dict = {col: [median_values.get(col, 0)] for col in feature_names}
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

    df_input = pd.DataFrame(input_dict)
    df_input['IncomePerYear'] = df_input['MonthlyIncome'] / (df_input['TotalWorkingYears'] + 1)
    df_input['SatisfactionScore'] = (df_input['JobSatisfaction'] + df_input['EnvironmentSatisfaction'] + df_input['WorkLifeBalance']) / 3
    df_input['RiskScore'] = df_input['OverTime'] * (1 / df_input['JobSatisfaction']) * df_input['DistanceFromHome']
    return df_input[feature_names]


# ── Main UI ──
st.title("Employee Attrition Predictor")
st.markdown("**Predict whether an employee is likely to leave and understand why.**")
st.markdown("---")

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict_btn = st.button("Predict Attrition Risk", use_container_width=True)

if predict_btn:
    input_df = build_input()
    prob = model.predict_proba(input_df)[0][1]

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if prob > 0.55:
            risk_label, risk_color = "HIGH RISK", "#e74c3c"
        elif prob > 0.35:
            risk_label, risk_color = "MEDIUM RISK", "#f39c12"
        else:
            risk_label, risk_color = "LOW RISK", "#2ecc71"

        st.markdown(f"""
        <div style='background-color:{risk_color};padding:20px;border-radius:10px;text-align:center;'>
            <h2 style='color:white;margin:0;'>{risk_label}</h2>
            <h1 style='color:white;margin:0;'>{prob*100:.1f}%</h1>
            <p style='color:white;margin:0;'>Attrition Probability</p>
            <p style='color:white;margin:0;font-size:12px;'>Threshold: {best_threshold:.2f}</p>
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

    st.markdown("---")
    st.subheader("Key Risk Factors")
    risk_factors = []
    if overtime == "Yes": risk_factors.append("Working overtime - 3x higher attrition risk")
    if job_level == 1: risk_factors.append("Entry level - highest attrition group (27%)")
    if monthly_income < 3000: risk_factors.append("Low income - below average for retained employees")
    if job_satisfaction <= 2: risk_factors.append("Low job satisfaction")
    if years_at_company <= 2: risk_factors.append("Less than 2 years - critical retention window")
    if distance_from_home > 15: risk_factors.append("High distance from home")
    if num_companies_worked > 4: risk_factors.append("Worked at many companies - job hopping pattern")
    if marital_status == "Single": risk_factors.append("Single employees leave more often statistically")
    if work_life_balance <= 2: risk_factors.append("Poor work life balance")
    if stock_option_level == 0: risk_factors.append("No stock options - less financial retention incentive")

    if risk_factors:
        for f in risk_factors:
            st.warning(f)
    else:
        st.success("No major risk factors detected")

    st.markdown("---")
    st.subheader("Business Impact")
    c1, c2 = st.columns(2)
    with c1:
        st.error(f"Estimated replacement cost: **${monthly_income * 6:,}**")
        st.caption("Replacing an employee costs 6-9 months salary (SHRM, 2023)")
    with c2:
        st.info(f"Annual salary: **${monthly_income * 12:,}**")

    with st.expander("About this model"):
        st.markdown("""
        - **Algorithm:** XGBoost Classifier
        - **Data:** IBM HR Analytics (1,470 employees)
        - **Features:** 33 including 3 engineered
        - **Imbalance:** scale_pos_weight (no synthetic data)
        - **ROC-AUC:** 0.77 | Recall: 55% | F1: 0.50
        - **Note:** Dataset is synthetic; real-world results may vary
        """)

st.markdown("---")
st.markdown("Built with XGBoost and Streamlit | IBM HR Analytics | Nithin Kilari")