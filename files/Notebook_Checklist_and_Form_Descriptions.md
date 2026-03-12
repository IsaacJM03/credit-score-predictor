# Notebook Documentation Checklist & Google Form Copy-Paste Descriptions

---

## PART A: Notebook Documentation Checklist
### eda_visualizations.ipynb

Use this checklist to confirm the notebook is submission-ready before exporting.

---

### Required Cells Checklist

#### Cell Structure
- [ ] **Cell 1 – Header Comment**  
  Must include: project title, dataset path, authors, date, Python version, and random_state.  
  *Example:*
  ```python
  # ============================================================
  # PROJECT: Credit Score Prediction – EDA & Model Pipeline
  # DATASET: data/comprehensive_loan_data_45M.csv
  # NOTEBOOK: eda_visualizations.ipynb
  # AUTHORS: [Author Names]
  # INSTITUTION: [Your Institution]
  # DATE: [DD Month YYYY]
  # PYTHON: 3.10.x | random_state = 42 throughout
  # REPO: https://github.com/[username]/credit-score-predictor
  # ============================================================
  ```

- [ ] **Cell 2 – Imports**  
  All libraries imported at the top. Organized: stdlib → third-party → project modules.  
  Must include version assertions:
  ```python
  import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
  import sklearn, xgboost
  print(f"pandas {pd.__version__} | sklearn {sklearn.__version__}")
  ```

- [ ] **Cell 3 – Data Load**  
  Use relative path. Set and display random seed.
  ```python
  RANDOM_STATE = 42
  DATA_PATH = "data/comprehensive_loan_data_45M.csv"
  df = pd.read_csv(DATA_PATH)
  print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
  df.head()
  ```

- [ ] **Cell 4 – Data Inspection**  
  `df.info()`, `df.describe()`, `df.isnull().sum()` — all displayed and commented.

- [ ] **Cell 5 – Preprocessing**  
  Each preprocessing step has a comment explaining *why* (not just what):
  ```python
  # Drop rows with >30% missing values — avoids imputation bias on heavily sparse rows
  df = df[df.isnull().mean(axis=1) < 0.30]
  ```

---

### Visualization Cells (1–25)

For each of the 25 visualization cells, confirm:

- [ ] **Cell title / markdown header** present above cell (e.g., `## Visualization 1: Loan Amount Distribution`)
- [ ] **Figure title** set via `plt.title()` or `ax.set_title()`
- [ ] **Axis labels** set: `plt.xlabel()`, `plt.ylabel()` or equivalent
- [ ] **Caption** in markdown cell below the figure (1–2 sentences interpreting the finding)
- [ ] **Figure saved** with standardized filename:
  ```python
  plt.savefig("figures/viz_01_loan_amount_hist.png", dpi=150, bbox_inches="tight")
  plt.show()
  ```

#### Visualization Naming Convention
| Viz # | Filename Template |
|-------|------------------|
| 01 | `viz_01_loan_amount_hist.png` |
| 02 | `viz_02_income_hist.png` |
| 03 | `viz_03_dti_hist.png` |
| 04 | `viz_04_credit_age_box.png` |
| 05 | `viz_05_correlation_heatmap.png` |
| 06 | `viz_06_scatter_dti_default.png` |
| 07 | `viz_07_violin_income_band.png` |
| 08 | `viz_08_pairplot.png` |
| 09 | `viz_09_vif_table.png` |
| 10 | `viz_10_class_distribution.png` |
| 11–25 | `viz_[NN]_[descriptive_name].png` |

---

### Model Cells

- [ ] Model training cell includes `random_state=42` in constructor
- [ ] Hyperparameter values are stated explicitly (not left as defaults silently)
- [ ] Evaluation metrics computed and displayed: accuracy, F1, AUC-ROC, confusion matrix
- [ ] Feature importance or SHAP plot saved as PNG

---

### Final Cells

- [ ] **Environment export cell:**
  ```python
  import subprocess
  subprocess.run(["pip", "freeze"], capture_output=True, text=True).stdout
  # Copy output to requirements.txt
  ```
- [ ] Notebook runs `Kernel → Restart & Run All` without errors
- [ ] All figures present in `figures/` directory (confirm: `ls figures/ | wc -l` = 25+)

---

## PART B: Google Form Copy-Paste Descriptions

> *Paste these exactly into the corresponding upload field description boxes on the Google Form.*

---

### Upload Field 1: Paper Writeup (IEEE Access Format)

```
This document is the full research paper prepared in IEEE Access format (double-column, 
Times New Roman 10pt) describing the credit score prediction study. It includes: a 
150–250 word abstract with keywords, structured sections (Introduction, Related Work, 
Dataset, Methodology, Exploratory Data Analysis, Results, Discussion, Conclusion, Future 
Work, References in IEEE citation format), and an appendix with the data dictionary and 
repository link. The paper documents 25 EDA visualizations from eda_visualizations.ipynb 
applied to comprehensive_loan_data_45M.csv.
File: Paper_Writeup_IEEE.docx
```

---

### Upload Field 2: Similarity Index Report

```
This PDF contains the originality report generated via [Turnitin / Urkund / iThenticate] 
for the submitted paper writeup. The overall similarity index is [XX]% (quotes and 
bibliography excluded per institutional settings). The report includes a breakdown by 
source type (internet, publications, student papers), confirmation that all highlighted 
segments are properly cited or formulaic academic language, and a signed author 
declaration confirming original authorship.
File: Similarity_Report.pdf
```

---

### Upload Field 3: AI Content Evaluation Report

```
This PDF discloses all AI tools used during the preparation of this submission (including 
Claude by Anthropic and GitHub Copilot). It lists the exact prompts submitted to each 
tool, a section-by-section estimate of AI-generated content (overall ~[XX]%), the steps 
taken to verify and modify AI outputs, and citations for AI tools per IEEE format. Specific 
notebook cells assisted by AI are identified along with the validation methods applied. 
All submitted content reflects substantial author revision and intellectual contribution.
File: AI_Content_Evaluation.pdf
```

---

### Upload Field 4: Project Write-up & EDA Slides

```
This PowerPoint deck (12 slides) presents the credit score prediction project for academic 
review. It covers: project objectives, dataset overview ([N] records, [M] features from 
comprehensive_loan_data_45M.csv), preprocessing pipeline, 10–12 high-impact EDA visualizations 
from the 25 generated in eda_visualizations.ipynb (with embedded captions and speaker 
notes), key findings (class imbalance, top predictive features, distributional insights), 
model results table, and step-by-step reproducibility instructions. An executive summary 
slide provides a one-sentence problem statement, solution, and 3 key findings.
File: EDA_Slides.pptx
```

---

## PART C: Quick QA Checklist (10-Point Final Check)

Run through this checklist immediately before submitting the Google Form:

- [ ] **1. File formats correct:** `.docx` for paper, `.pdf` for reports, `.pptx` for slides
- [ ] **2. Filenames match exactly:** `Paper_Writeup_IEEE.docx`, `Similarity_Report.pdf`, `AI_Content_Evaluation.pdf`, `EDA_Slides.pptx`
- [ ] **3. Figures embedded with captions** in both slides and paper (not missing/broken links)
- [ ] **4. All references verified:** Every citation in the paper corresponds to a real, accessible source
- [ ] **5. Similarity threshold met:** Overall similarity ≤ [institutional threshold]% with quotes/bibliography excluded
- [ ] **6. AI disclosure complete:** All AI tools, prompts, and section-level estimates documented in `AI_Content_Evaluation.pdf`
- [ ] **7. Slides exported correctly:** PPTX opens without font substitution errors; all 12 slides present
- [ ] **8. Notebook runs end-to-end:** `Kernel → Restart & Run All` completes without errors or warnings
- [ ] **9. All 25 figures saved as PNG** in `figures/` directory with standardized naming convention
- [ ] **10. README_SUBMISSION.md included** in repository with reproducibility instructions and submission checklist signed off

---

*Suggested filename: `Notebook_Checklist_and_Form_Descriptions.md`*
