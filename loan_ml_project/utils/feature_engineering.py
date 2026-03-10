"""
utils/feature_engineering.py
------------------------------
Derive domain-specific features that improve fraud detection and borrower
segmentation beyond the raw columns.

Why engineered features matter
-------------------------------
Raw financial figures (income, loan amount, credit score) are less informative
in isolation.  Ratios and velocity metrics expose the *relationships* between
those values, which is where anomalous or risky behaviour tends to hide.

Feature catalogue
-----------------
debt_to_income_ratio       How much of monthly income is consumed by existing
                           debt obligations.  High values (>0.5) indicate stress.

loan_to_income_ratio       Size of the requested loan relative to monthly income.
                           Unusually large values may indicate fraud or inability
                           to repay.

repayment_consistency      Derived from Payment_History ordinal score and
                           repayment_delay_days (if present).  Measures how
                           reliably a borrower meets obligations.

application_velocity       Proxy for rapid repeated applications – a key fraud
                           signal.  Derived from Number_of_Previous_Loans divided
                           by borrower Age.

credit_utilization_score   Ratio of existing debts to credit score; reflects how
                           fully a borrower is leveraging their available credit.
"""

import numpy as np
import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append all engineered columns to *df* and return the extended DataFrame.

    This function is safe to call after preprocessing (categorical encoding
    already applied) because it only reads numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Pre-processed loan DataFrame (numeric columns only, no raw categoricals).

    Returns
    -------
    pd.DataFrame
        Original columns plus five new engineered columns.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # 1. debt_to_income_ratio
    # ------------------------------------------------------------------
    # Captures financial stress: borrowers spending >50 % of income on
    # existing debts are at high default risk and may commit fraud.
    if "Existing_Debts" in df.columns and "Monthly_Income" in df.columns:
        df["debt_to_income_ratio"] = (
            df["Existing_Debts"] / df["Monthly_Income"].replace(0, np.nan)
        ).fillna(0)
    elif "Debt_to_Income_Ratio" in df.columns:
        # Dataset already contains this ratio – keep a copy under the new name
        df["debt_to_income_ratio"] = df["Debt_to_Income_Ratio"]

    # ------------------------------------------------------------------
    # 2. loan_to_income_ratio
    # ------------------------------------------------------------------
    # A loan that is many multiples of monthly income is a red flag for
    # both fraud detection and risk segmentation.
    if "Loan_Amount_Requested" in df.columns and "Monthly_Income" in df.columns:
        df["loan_to_income_ratio"] = (
            df["Loan_Amount_Requested"] / df["Monthly_Income"].replace(0, np.nan)
        ).fillna(0)

    # ------------------------------------------------------------------
    # 3. repayment_consistency
    # ------------------------------------------------------------------
    # A simple proxy: ordinal payment history score normalised to [0, 1].
    # Higher value → more consistent repayer → lower fraud / default risk.
    if "Payment_History" in df.columns:
        max_ph = df["Payment_History"].max()
        if max_ph > 0:
            df["repayment_consistency"] = df["Payment_History"] / max_ph
        else:
            df["repayment_consistency"] = 0.0

    # ------------------------------------------------------------------
    # 4. application_velocity
    # ------------------------------------------------------------------
    # Number of previous loans per year of life is a coarse proxy for how
    # frequently a borrower has applied for credit.  Unusually high values
    # (relative to age) are a fraud signal: rapid repeated applications.
    if "Number_of_Previous_Loans" in df.columns and "Age" in df.columns:
        df["application_velocity"] = (
            df["Number_of_Previous_Loans"] / df["Age"].replace(0, np.nan)
        ).fillna(0)

    # ------------------------------------------------------------------
    # 5. credit_utilization_score
    # ------------------------------------------------------------------
    # Ratio of existing debts to credit score: high utilisation relative
    # to credit rating suggests the borrower is over-extended.
    if "Existing_Debts" in df.columns and "Credit_Score" in df.columns:
        df["credit_utilization_score"] = (
            df["Existing_Debts"] / df["Credit_Score"].replace(0, np.nan)
        ).fillna(0)

    new_cols = [
        "debt_to_income_ratio",
        "loan_to_income_ratio",
        "repayment_consistency",
        "application_velocity",
        "credit_utilization_score",
    ]
    available = [c for c in new_cols if c in df.columns]
    print(f"[feature_engineering] Added engineered features: {available}")
    return df


def get_feature_names_after_engineering(base_feature_names: list) -> list:
    """
    Return the expected column list after feature engineering, given a list of
    base feature names.  Useful for bookkeeping when saving models.
    """
    engineered = [
        "debt_to_income_ratio",
        "loan_to_income_ratio",
        "repayment_consistency",
        "application_velocity",
        "credit_utilization_score",
    ]
    # Avoid duplicates (debt_to_income_ratio may already exist as raw feature)
    return base_feature_names + [f for f in engineered if f not in base_feature_names]
