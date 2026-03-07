"""
Train Logistic Regression, Random Forest and a high-epoch MLPClassifier,
then persist models and preprocessing artifacts to the `models/` folder.

Usage:
    python save_and_store_models.py

This script mirrors the notebook preprocessing and hyperparameter grids.
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42
MODELS_DIR = 'models'
CSV_PATH = 'comprehensive_loan_data.csv'

os.makedirs(MODELS_DIR, exist_ok=True)

# --- Load and preprocess (same as notebook) ---------------------------------
print('Loading dataset...')
df = pd.read_csv(CSV_PATH)

payment_order = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
df['Payment_History'] = df['Payment_History'].map(payment_order)

df = pd.get_dummies(df, columns=['Employment_Status'], drop_first=True)

TARGET = 'Repayment_Status'
FEATURES = [c for c in df.columns if c != TARGET]
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save feature list and scaler
joblib.dump(FEATURES, os.path.join(MODELS_DIR, 'features_list.joblib'))
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))
print('Saved scaler and feature list.')

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# --- Logistic Regression (GridSearch) --------------------------------------
print('Training Logistic Regression (GridSearchCV)...')
lr_param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear'],
    'max_iter': [1000],
}

lr_grid = GridSearchCV(
    LogisticRegression(random_state=RANDOM_STATE),
    lr_param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0,
    refit=True,
)
# fit on scaled features
lr_grid.fit(X_train_scaled, y_train)
lr_best = lr_grid.best_estimator_
print('LR best params:', lr_grid.best_params_)
print('LR best CV ROC-AUC:', lr_grid.best_score_)

joblib.dump(lr_best, os.path.join(MODELS_DIR, 'logistic_regression_best.joblib'))
joblib.dump(lr_grid, os.path.join(MODELS_DIR, 'logistic_regression_gridsearch.joblib'))
print('Saved Logistic Regression model and gridsearch.')

# --- Random Forest (GridSearch) --------------------------------------------
print('Training Random Forest (GridSearchCV)...')
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1),
    rf_param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=1,
    verbose=0,
    refit=True,
)
# RF uses unscaled features
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_
print('RF best params:', rf_grid.best_params_)
print('RF best CV ROC-AUC:', rf_grid.best_score_)

joblib.dump(rf_best, os.path.join(MODELS_DIR, 'random_forest_best.joblib'))
joblib.dump(rf_grid, os.path.join(MODELS_DIR, 'random_forest_gridsearch.joblib'))
print('Saved Random Forest model and gridsearch.')

# --- High-epoch MLPClassifier ------------------------------------------------
print('Training MLPClassifier with high max_iter (this may take time)...')
# Use scaled data; set large max_iter to allow many epochs
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    alpha=1e-4,
    learning_rate='adaptive',
    max_iter=5000,
    early_stopping=True,
    n_iter_no_change=50,
    tol=1e-5,
    random_state=RANDOM_STATE,
    verbose=True,
)
mlp.fit(X_train_scaled, y_train)

# Evaluate quickly and save
y_prob_mlp = mlp.predict_proba(X_test_scaled)[:, 1]
mlp_auc = roc_auc_score(y_test, y_prob_mlp)
print('MLP Test ROC-AUC:', mlp_auc)
joblib.dump(mlp, os.path.join(MODELS_DIR, 'mlp_high_epoch.joblib'))
print('Saved MLPClassifier (high-epoch).')

# --- Save additional artifacts ----------------------------------------------
# We also save a small JSON-like metadata file
metadata = {
    'features': FEATURES,
    'target': TARGET,
    'random_state': RANDOM_STATE,
    'lr_best_params': lr_grid.best_params_,
    'rf_best_params': rf_grid.best_params_,
    'mlp_test_roc_auc': float(mlp_auc),
}
joblib.dump(metadata, os.path.join(MODELS_DIR, 'models_metadata.joblib'))

print('\nAll models and artifacts saved to the `models/` directory:')
for f in os.listdir(MODELS_DIR):
    print(' -', os.path.join(MODELS_DIR, f))

print('\nDone.')
