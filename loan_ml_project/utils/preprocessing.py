"""
utils/preprocessing.py
-----------------------
Data loading, cleaning, encoding, and scaling utilities for the loan ML pipeline.

This module handles all raw-data transformations so that every downstream model
receives a consistent, clean numeric feature matrix.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ---------------------------------------------------------------------------
# Column name constants (matches comprehensive_loan_data_45M.csv)
# ---------------------------------------------------------------------------
CAT_COLS = ["Employment_Status", "Payment_History"]
NUM_COLS = [
    "Age",
    "Monthly_Income",
    "Credit_Score",
    "Loan_Amount_Requested",
    "Existing_Debts",
    "Number_of_Previous_Loans",
    "Debt_to_Income_Ratio",
]
TARGET_COL = "Repayment_Status"

# Ordered mapping for Payment_History so it becomes a proper ordinal feature
PAYMENT_HISTORY_ORDER = {"Poor": 0, "Fair": 1, "Good": 2, "Excellent": 3}


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load the raw loan dataset from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to comprehensive_loan_data_45M.csv (or any compatible CSV).

    Returns
    -------
    pd.DataFrame
        Raw dataframe with all original columns.
    """
    df = pd.read_csv(csv_path)
    print(f"[preprocessing] Loaded {len(df):,} rows, {df.shape[1]} columns from '{csv_path}'.")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values.

    Strategy:
    - Numeric columns  → fill with column median (robust to outliers).
    - Categorical cols → fill with column mode (most common category).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with no missing values.
    """
    df = df.copy()

    missing_before = df.isnull().sum().sum()
    if missing_before == 0:
        print("[preprocessing] No missing values found.")
        return df

    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

    for col in df.select_dtypes(include=["object", "category"]).columns:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)

    print(f"[preprocessing] Imputed {missing_before} missing value(s).")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features.

    - Payment_History  : ordinal encoding (Poor=0 … Excellent=3).
    - Employment_Status: one-hot encoding (drop_first=True to avoid multicollinearity).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with categorical columns replaced by numeric representations.
    """
    df = df.copy()

    # Ordinal encode Payment_History
    if "Payment_History" in df.columns:
        df["Payment_History"] = df["Payment_History"].map(PAYMENT_HISTORY_ORDER)
        if df["Payment_History"].isnull().any():
            # Unseen categories → fall back to 1 (Fair)
            df["Payment_History"].fillna(1, inplace=True)

    # One-hot encode Employment_Status
    if "Employment_Status" in df.columns:
        df = pd.get_dummies(df, columns=["Employment_Status"], drop_first=True)

    print("[preprocessing] Categorical encoding complete.")
    return df


def scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray | None = None,
) -> tuple:
    """
    Fit a StandardScaler on the training set and transform both splits.

    Standardisation (zero mean, unit variance) is required by:
    - One-Class SVM (kernel-based, distance-sensitive)
    - Autoencoder (gradient descent stability)
    - K-Means / DBSCAN / Hierarchical clustering (distance metrics)

    Isolation Forest is tree-based and scale-invariant, but scaling does no harm.

    Parameters
    ----------
    X_train : np.ndarray  shape (n_train, n_features)
    X_test  : np.ndarray | None  shape (n_test, n_features)

    Returns
    -------
    (scaler, X_train_scaled [, X_test_scaled])
        If X_test is None, returns (scaler, X_train_scaled).
        If X_test is provided, returns (scaler, X_train_scaled, X_test_scaled).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return scaler, X_train_scaled, X_test_scaled

    return scaler, X_train_scaled


def preprocess_pipeline(
    csv_path: str,
    target_col: str = TARGET_COL,
    drop_target: bool = True,
) -> tuple:
    """
    End-to-end preprocessing: load → impute → encode → return features + target.

    Note: scaling is intentionally left to the caller so that train/test splits
    can be handled correctly (fit scaler only on train data).

    Parameters
    ----------
    csv_path    : str   Path to the CSV file.
    target_col  : str   Name of the target / label column.
    drop_target : bool  If True, the target column is removed from X.

    Returns
    -------
    (X, y, feature_names)
        X            : np.ndarray  shape (n_samples, n_features)
        y            : np.ndarray  shape (n_samples,)   – None if col absent
        feature_names: list[str]
    """
    df = load_data(csv_path)
    df = handle_missing_values(df)
    df = encode_categoricals(df)

    # Separate target
    if target_col in df.columns:
        y = df[target_col].values
        if drop_target:
            df = df.drop(columns=[target_col])
    else:
        y = None

    feature_names = df.columns.tolist()
    X = df.values.astype(np.float32)

    print(f"[preprocessing] Feature matrix shape: {X.shape}")
    return X, y, feature_names
