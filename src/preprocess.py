import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(filepath):
    # Load data
    df = pd.read_csv(filepath)
    
    # Drop useless columns
    df = df.drop(columns=['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'])
    
    # Encode target
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    
    # Separate categorical and numerical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Label encode categorical columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Separate features and target
    X = df.drop(columns=['Attrition'])
    y = df['Attrition']
    
    return X, y, df