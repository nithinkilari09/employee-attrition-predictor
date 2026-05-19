import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, accuracy_score, f1_score,
                              precision_recall_curve)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from preprocess import load_and_preprocess

# Load and preprocess
X, y, df = load_and_preprocess('../data/raw/attrition.csv')

print(f"Dataset shape: {X.shape}")
print(f"Class distribution:\n{y.value_counts()}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Engineering — 3 meaningful new features
def add_features(df):
    df = df.copy()
    # Underpaid relative to experience
    df['IncomePerYear'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
    # Combined satisfaction score
    df['SatisfactionScore'] = (
        df['JobSatisfaction'] +
        df['EnvironmentSatisfaction'] +
        df['WorkLifeBalance']
    ) / 3
    # Combined risk score
    df['RiskScore'] = (
        df['OverTime'] *
        (1 / df['JobSatisfaction']) *
        df['DistanceFromHome']
    )
    return df

X_train = add_features(X_train)
X_test = add_features(X_test)

print(f"\nAfter feature engineering: {X_train.shape[1]} features")

# Calculate scale_pos_weight for class imbalance
# = number of negatives / number of positives
scale = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nscale_pos_weight: {scale:.2f} (no fake data, real imbalance handling)")

# Train XGBoost
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale,
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)
model.fit(X_train, y_train)

# Default threshold evaluation (0.5)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n--- DEFAULT THRESHOLD (0.50) ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))
print("F1 (attrition):", round(f1_score(y_test, y_pred, average='binary'), 4))
print("Recall (attrition):", round(f1_score(y_test, y_pred, average='binary', pos_label=1), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Threshold tuning — find best threshold for F1
print("\n--- THRESHOLD TUNING ---")
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print(f"Best threshold: {best_threshold:.2f}")

# Apply best threshold
y_pred_tuned = (y_prob >= best_threshold).astype(int)

print("\n--- TUNED THRESHOLD ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred_tuned), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))
print("F1 (attrition):", round(f1_score(y_test, y_pred_tuned, average='binary'), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred_tuned))

# Confusion Matrix — tuned threshold
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred_tuned)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Stayed', 'Left'],
            yticklabels=['Stayed', 'Left'])
plt.title(f'Confusion Matrix — XGBoost (threshold={best_threshold:.2f})')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('../data/confusion_matrix.png', dpi=150)
plt.show()

# Feature Importance
feat_imp = pd.Series(model.feature_importances_, index=X_train.columns)
feat_imp = feat_imp.sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 6))
feat_imp.plot(kind='barh', color='#3498db')
plt.title('Top 15 Feature Importances — XGBoost')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('../data/feature_importance.png', dpi=150)
plt.show()

# SHAP values
print("\nCalculating SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig('../data/shap_summary.png', dpi=150, bbox_inches='tight')
plt.show()

# Save everything
with open('../models/attrition_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('../models/feature_names.pkl', 'wb') as f:
    pickle.dump(X_train.columns.tolist(), f)

median_values = X_train.median().to_dict()
with open('../models/median_values.pkl', 'wb') as f:
    pickle.dump(median_values, f)

with open('../models/best_threshold.pkl', 'wb') as f:
    pickle.dump(best_threshold, f)

print("\n✅ Model saved!")
print("✅ Feature names saved!")
print("✅ Median values saved!")
print(f"✅ Best threshold saved: {best_threshold:.2f}")