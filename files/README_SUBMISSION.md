# README – Submission Checklist
## Credit Score Predictor | Academic Project Submission

---

## Files to Upload (Google Form)

| Upload Field | Filename | Description |
|-------------|----------|-------------|
| **Paper Writeup** | `Paper_Writeup_IEEE.docx` | IEEE Access-format research paper including Abstract, Introduction, EDA, Results, and References. |
| **Similarity Index Report** | `Similarity_Report.pdf` | Turnitin/Urkund originality report with overall similarity %, settings applied, and author declaration. |
| **AI Content Evaluation Report** | `AI_Content_Evaluation.pdf` | Disclosure of AI tools used, prompts, extent of AI assistance by section, and verification steps. |
| **Project Write-up & EDA Slides** | `EDA_Slides.pptx` | 12-slide deck covering objectives, data overview, preprocessing, 25 EDA visualizations (10–12 highlighted), key findings, model results, and reproducibility instructions. |

---

## Repository Structure

```
credit-score-predictor/
├── data/
│   └── comprehensive_loan_data_5m.csv        # Raw dataset — do not modify
├── eda_visualizations.ipynb               # Main notebook (25 visualizations + model)
├── figures/                               # Exported PNG figures
│   ├── viz_01_loan_amount_hist.png
│   ├── viz_02_income_hist.png
│   ├── viz_03_dti_hist.png
│   ├── viz_04_credit_age_box.png
│   ├── viz_05_correlation_heatmap.png
│   ├── viz_06_scatter_dti_default.png
│   ├── viz_07_violin_income_band.png
│   ├── viz_08_pairplot.png
│   ├── viz_09_vif_table.png
│   ├── viz_10_class_distribution.png
│   ├── viz_11_default_by_income_quartile.png
│   ├── viz_12_default_by_loan_term.png
│   ├── viz_13_default_by_purpose.png
│   ├── viz_14_default_by_loan_term2.png
│   ├── viz_15_loan_purpose_bar.png
│   ├── viz_16_derog_scatter.png
│   ├── viz_17_open_accts_hist.png
│   ├── viz_18_geo_map.png
│   ├── viz_19_feat_importance.png
│   ├── viz_20_shap_summary.png
│   └── viz_21_25_[remaining].png
├── models/                                # Saved model artifacts (optional)
├── requirements.txt                       # Python dependencies
├── data_dictionary.md                     # Column-level data documentation
├── Paper_Writeup_IEEE.docx
├── Similarity_Report.pdf
├── AI_Content_Evaluation.pdf
├── EDA_Slides.pptx
└── README_SUBMISSION.md                   # This file
```

---

## Environment & Reproducibility

### Python Version
```
Python 3.9.x or 3.10.x (recommended: 3.10.12)
```

### Install Dependencies
```bash
# Using pip
pip install -r requirements.txt

# Or using conda
conda env create -f environment.yml
conda activate credit-scorer
```

### Minimum `requirements.txt`
```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
xgboost>=1.7.0
jupyter>=1.0.0
imbalanced-learn>=0.10.0
shap>=0.41.0
```

### How to Run the Notebook
```bash
# 1. Clone repo
git clone https://github.com/[your-username]/credit-score-predictor.git
cd credit-score-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify dataset path
# Dataset must be at: data/comprehensive_loan_data_5m.csv

# 4. Open and run notebook
jupyter notebook eda_visualizations.ipynb
# Select: Kernel → Restart & Run All
```

### Reproducibility Seeds
All random operations use `random_state=42`:
```python
train_test_split(..., random_state=42)
RandomForestClassifier(random_state=42)
XGBClassifier(seed=42)
np.random.seed(42)
```

---

## Data Provenance & Privacy

| Item | Status |
|------|--------|
| Dataset source | [Synthetic / Kaggle / Institutional — specify] |
| Personally Identifiable Information (PII) | ☐ None present / ☑ Anonymized |
| IRB / Ethics approval | ☐ Not required (synthetic) / ☐ Approved — Ref: [IRB#] |
| License | [CC0 / MIT / Proprietary — specify] |
| Data modifications | Cleaned version only; original preserved in `data/raw/` |

> **If using real borrower data:** Confirm all records are anonymized, no direct identifiers (name, SSN, address) are present, and institutional data use agreements have been signed.

---

## Submission Checklist

- [ ] `Paper_Writeup_IEEE.docx` — complete, spell-checked, all placeholders filled
- [ ] `Similarity_Report.pdf` — similarity ≤ [institutional threshold]%, signed declaration included
- [ ] `AI_Content_Evaluation.pdf` — all AI tools disclosed, prompts listed, signatures included
- [ ] `EDA_Slides.pptx` — 12 slides, 10+ figures embedded with captions, speaker notes added
- [ ] Notebook runs end-to-end without errors (`Kernel → Restart & Run All`)
- [ ] All 25 figures saved as PNG in `figures/` directory
- [ ] `data_dictionary.md` complete with all columns documented
- [ ] `requirements.txt` updated and tested on clean environment
- [ ] `random_state=42` confirmed in all relevant cells
- [ ] GitHub repository link is public (or access granted to evaluator)

---

*Suggested filename: `README_SUBMISSION.md`*
*Last updated: [DD Month YYYY] by [Author Name]*
