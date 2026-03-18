# Data Dictionary
## comprehensive_loan_data_45M.csv

**Project:** Credit Score Predictor  
**Path:** `/Users/mac/Documents/GitHub/Projects/credit-score-predictor/comprehensive_loan_data_45M.csv`  
**Last updated:** [DD Month YYYY]  
**Records:** [N] rows × [M] columns  

> ⚠ *Columns marked with `[CONFIRM]` require verification against actual dataset. Run `df.columns.tolist()` and `df.dtypes` in the notebook to populate.*

---

| # | Column Name | Data Type | Allowed Values / Units | Description | Example Value |
|---|------------|-----------|----------------------|-------------|---------------|
| 1 | `loan_id` | string/int | Unique alphanumeric ID | Unique identifier for each loan record | `LN-000123` |
| 2 | `loan_amount` | float | USD, > 0 | Requested loan principal amount | `15000.00` |
| 3 | `loan_term` | int | 12, 24, 36, 48, 60 (months) | Duration of the loan in months | `36` |
| 4 | `interest_rate` | float | %, 0–40 | Annual interest rate applied to the loan | `12.5` |
| 5 | `loan_purpose` | string/category | debt_consolidation, home_improvement, car, education, other | Stated purpose of the loan application | `debt_consolidation` |
| 6 | `annual_income` | float | USD, > 0 | Borrower's self-reported annual income | `55000.00` |
| 7 | `employment_length` | int/string | 0–10+ years | Length of current employment | `5` |
| 8 | `home_ownership` | string/category | RENT, OWN, MORTGAGE, OTHER | Borrower's housing status | `RENT` |
| 9 | `debt_to_income` | float | Ratio, 0–1 or % | Total monthly debt payments divided by gross monthly income | `0.28` |
| 10 | `credit_history_length` | int | Years, ≥ 0 | Length of borrower's credit history in years | `7` |
| 11 | `num_open_accounts` | int | ≥ 0 | Number of currently open credit accounts | `4` |
| 12 | `num_derogatory_marks` | int | ≥ 0 | Number of derogatory marks (late payments, collections, etc.) | `1` |
| 13 | `num_credit_inquiries` | int | ≥ 0 | Number of hard credit inquiries in last 12 months | `2` |
| 14 | `revolving_credit_balance` | float | USD, ≥ 0 | Current total revolving credit balance | `8200.00` |
| 15 | `revolving_utilization` | float | 0–1 (proportion) | Revolving credit used vs. total available | `0.45` |
| 16 | `total_credit_lines` | int | ≥ 0 | Total number of credit lines ever opened | `12` |
| 17 | `earliest_credit_line` | date/int | YYYY or year delta | Year of oldest credit account | `2010` |
| 18 | `state` | string | US state abbreviations | Borrower's state of residence | `CA` |
| 19 | `zip_code_prefix` | int | 3-digit prefix | First 3 digits of borrower's ZIP code (anonymized) | `902` |
| 20 | `application_type` | string/category | Individual, Joint | Whether application is individual or joint | `Individual` |
| 21 | `co_borrower_income` | float/null | USD or NaN | Co-borrower annual income (if joint application) | `42000.00` |
| 22 | `credit_score_band` | string/category | Poor, Fair, Good, Very Good, Exceptional | Binned credit score category (target variable option 1) | `Good` |
| 23 | `default_flag` | int | 0 = No default, 1 = Default | Binary indicator of loan default (target variable option 2) | `0` |
| 24 | `loan_status` | string/category | Fully Paid, Charged Off, Current, Late | Current status of the loan | `Fully Paid` |
| 25 | `issue_date` | date | YYYY-MM | Month and year loan was issued | `2021-06` |

---

## Notes

- **Target variable:** Use `default_flag` for binary classification or `credit_score_band` for multi-class classification. Confirm which is used in your model.
- **Missing values:** `co_borrower_income` is expected to be null for Individual applications. All other columns should have <5% missingness after preprocessing.
- **PII note:** This dataset is [synthetic / anonymized]. No real borrower names, SSNs, or exact addresses are present.
- **Column confirmation:** Run the following in the notebook to verify this dictionary against the actual data:

```python
import pandas as pd
df = pd.read_csv('data/comprehensive_loan_data_45M.csv')
print(df.dtypes)
print(df.isnull().sum())
print(df.describe())
```

---

*Suggested filename: `data_dictionary.md`*
