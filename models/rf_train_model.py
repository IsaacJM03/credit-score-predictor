"""
Random Forest Classifier - Loan Repayment Prediction
Dataset: 10,000 samples, 9 features
Target: Repayment_Status (0 = default, 1 = repaid)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, accuracy_score, roc_curve)
import pickle

# ── Load Data ──────────────────────────────────────────────────────────────────
df = pd.read_csv('data/comprehensive_loan_data.csv')
print(f"Dataset shape: {df.shape}")

# ── Feature Engineering ────────────────────────────────────────────────────────
le_emp = LabelEncoder()  # Employed, Self-Employed, Student, Unemployed
le_pay = LabelEncoder()  # Excellent, Fair, Good, Poor

df['Employment_Status_enc'] = le_emp.fit_transform(df['Employment_Status'])
df['Payment_History_enc']   = le_pay.fit_transform(df['Payment_History'])

features = [
    'Age', 'Monthly_Income', 'Credit_Score', 'Loan_Amount_Requested',
    'Employment_Status_enc', 'Existing_Debts', 'Number_of_Previous_Loans',
    'Payment_History_enc', 'Debt_to_Income_Ratio'
]

X = df[features]
y = df['Repayment_Status']

# ── Train/Test Split ───────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ── Model ──────────────────────────────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
rf.fit(X_train, y_train)

# ── Evaluation ─────────────────────────────────────────────────────────────────
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

print(f"\n{'='*50}")
print(f"Test Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC       : {roc_auc_score(y_test, y_prob):.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc')
print(f"\n5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Feature importances
print("\nFeature Importances:")
for feat, imp in sorted(zip(features, rf.feature_importances_), key=lambda x: -x[1]):
    bar = '█' * int(imp * 100)
    print(f"  {feat:<30} {imp:.4f}  {bar}")

# ── Save Model ─────────────────────────────────────────────────────────────────
with open('loan_rf_model.pkl', 'wb') as f:
    pickle.dump({'model': rf, 'encoders': {'employment': le_emp, 'payment': le_pay},
                 'features': features}, f)
print("\nModel saved to loan_rf_model.pkl")

# ── Predict New Sample ─────────────────────────────────────────────────────────
def predict_loan(age, monthly_income, credit_score, loan_amount,
                 employment_status, existing_debts, prev_loans,
                 payment_history, dti):
    """Predict repayment probability for a new applicant."""
    emp_enc = le_emp.transform([employment_status])[0]
    pay_enc = le_pay.transform([payment_history])[0]
    sample = pd.DataFrame([[age, monthly_income, credit_score, loan_amount,
                             emp_enc, existing_debts, prev_loans, pay_enc, dti]],
                           columns=features)
    prob = rf.predict_proba(sample)[0, 1]
    label = "Will Repay" if prob >= 0.5 else "Will Default"
    return label, prob

label, prob = predict_loan(35, 5500, 680, 15000, 'Employed', 8000, 2, 'Good', 0.35)
print(f"\nSample Prediction: {label} (probability: {prob:.3f})")