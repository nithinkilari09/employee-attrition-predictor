# Employee Attrition Predictor

An end-to-end machine learning application that predicts employee attrition risk and quantifies business impact — built to help HR teams prioritize retention conversations before employees decide to leave.

**Live Demo:** [Click here to try the app](https://employee-attrition-predictor-nnhadtaappffp5fuydrjwhd.streamlit.app)

---

## Business Problem

Employee turnover costs organizations **6-9 months of an employee's salary** per replacement (SHRM, 2023). For a company of 1,000 employees with 16% annual attrition, that's millions in avoidable costs every year.

This tool gives HR teams an early warning system — identifying which employees are most at risk and *why* — so managers can intervene before it's too late.

---

## Live App Features

- Input 15 employee attributes and get an instant attrition probability score
- Risk classification: High / Medium / Low with color-coded alerts
- Automatic detection of 10 key risk factors with plain-English explanations
- Real-time business impact calculator showing estimated replacement cost
- Input validation preventing logically impossible employee profiles
- Model transparency section explaining methodology and honest limitations

---

## Key Findings from EDA

| Finding | Insight |
|---|---|
| Sales department | Highest attrition rate at 20%+ |
| Overtime workers | Leave at 3x the rate of non-overtime workers |
| Job Level 1 | 27% attrition — highest risk group |
| Age 25-32 | Peak attrition window for early-career employees |
| Low income | Strongest predictor of leaving |
| First 2 years | Critical retention window at any company |

---

## Model Development Journey

### Version 1 — Baseline Random Forest
- Accuracy: 83% | ROC-AUC: 0.78 | Recall: 15%
- Problem: Class imbalance caused model to ignore minority class

### Version 2 — SMOTE + Gradient Boosting
- Recall improved to 38%
- Rejected: SMOTE generates synthetic data points that don't represent real employees

### Version 3 — XGBoost + Threshold Tuning (Final)
- Accuracy: 82% | ROC-AUC: 0.77 | Recall: 55% | F1: 0.50
- Used scale_pos_weight to handle imbalance without synthetic data
- Engineered 3 new features: IncomePerYear, SatisfactionScore, RiskScore
- Tuned decision threshold to 0.35 for optimal F1 score

**Recall improved from 15% to 55% — catching 3x more at-risk employees using only real data.**

---

## Technical Stack

| Layer | Technology |
|---|---|
| Data Processing | Python, pandas, NumPy |
| Machine Learning | XGBoost, scikit-learn |
| Explainability | SHAP values, feature importance |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Version Control | Git, GitHub |

---

## Project Structure
employee-attrition-predictor/
├── data/
│   └── raw/                  # IBM HR Analytics dataset
├── notebooks/
│   └── 01_exploration.ipynb  # EDA with 6 key business insights
├── src/
│   ├── preprocess.py         # Data cleaning and encoding
│   ├── train.py              # Model training pipeline
│   └── predict.py            # Prediction logic
├── app/
│   └── streamlit_app.py      # Live Streamlit dashboard
├── models/                   # Trained model artifacts
└── requirements.txt

---

## Model Limitations

This model should be treated as an **advisory tool**, not a decision-making system:

- Dataset is synthetic (IBM generated for educational purposes)
- 1,470 rows is small for production ML
- Misses 45% of at-risk employees at current threshold
- Real-world deployment would require manager feedback scores, salary benchmarking, and significantly more data

---

## Dataset

IBM HR Analytics Employee Attrition Dataset — publicly available on
[Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

1,470 employees | 35 features | 16% attrition rate

---

## Author

**Nithin Kilari**
M.S. Computer Science (Data Science) — Oklahoma City University, 2026
[LinkedIn](https://www.linkedin.com/in/kilari-nithin-619481272/) | [GitHub](https://github.com/nithinkilari09)