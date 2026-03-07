# Model Architecture & XAI Documentation
## Credit Score Predictor — Assignment Submission

**Course Assignment** | Deadline: Saturday 14 March 2026, 10:00 AM  
**Dataset:** `comprehensive_loan_data.csv` (N=400 records, 10 columns)  
**Task:** Binary Classification — `Repayment_Status` (0 = Default, 1 = Repaid)

---

## Table of Contents

1. [Dataset Summary](#1-dataset-summary)
2. [Preprocessing Pipeline](#2-preprocessing-pipeline)
3. [Model 1 — Logistic Regression](#3-model-1--logistic-regression)
   - Architecture
   - How it learns
   - Optimisation
   - Parameters & Hyperparameters
4. [Model 2 — Random Forest Classifier](#4-model-2--random-forest-classifier)
   - Architecture
   - How it learns
   - Optimisation
   - Parameters & Hyperparameters
5. [Explainable AI (XAI) Techniques](#5-explainable-ai-xai-techniques)
6. [Evaluation Results](#6-evaluation-results)
7. [Model Comparison & Selection](#7-model-comparison--selection)
8. [References](#8-references)

---

## 1. Dataset Summary

| Property | Value |
|----------|-------|
| Records | 400 rows |
| Features | 9 (after encoding: 12) |
| Target | `Repayment_Status` (binary: 0/1) |
| Missing values | None |
| Class balance | ~60% Repaid (1), ~40% Default (0) |

### Feature Roles

| Feature | Type | Preprocessing Applied |
|---------|------|-----------------------|
| `Age` | Numeric (continuous) | StandardScaler |
| `Monthly_Income` | Numeric (continuous) | StandardScaler |
| `Credit_Score` | Numeric (300–849) | StandardScaler |
| `Loan_Amount_Requested` | Numeric (continuous) | StandardScaler |
| `Employment_Status` | Nominal categorical | One-hot encoding (drop_first=True) |
| `Existing_Debts` | Numeric (continuous) | StandardScaler |
| `Number_of_Previous_Loans` | Numeric (discrete) | StandardScaler |
| `Payment_History` | **Ordinal** categorical | Ordinal encoding: Poor=0, Fair=1, Good=2, Excellent=3 |
| `Debt_to_Income_Ratio` | Numeric (0–1) | StandardScaler |

---

## 2. Preprocessing Pipeline

### 2.1 Ordinal vs. One-Hot Encoding

**Payment_History** was encoded ordinally because it carries inherent order information:  
`Poor (0) < Fair (1) < Good (2) < Excellent (3)`  
Using one-hot encoding here would discard this valuable structure.

**Employment_Status** was one-hot encoded because its categories (Employed, Self-Employed, Unemployed, Student) have **no natural order** — treating them as integers would imply a spurious ranking.

### 2.2 Train/Test Split Strategy

- **Split ratio:** 80% train / 20% test (320 / 80 samples)
- **Stratification:** `stratify=y` ensures the class ratio is preserved in both splits
- **Rationale:** With only 400 records, stratification is critical to prevent a test set with a very different class balance from misleading evaluation metrics

### 2.3 Feature Scaling

`StandardScaler` transforms each feature to zero mean, unit variance:
$$x_{\text{scaled}} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

**Why it matters for LR:** The regularisation penalty `C` (or `λ = 1/C`) is applied uniformly to all coefficients. Without scaling, features with large magnitudes (e.g., `Monthly_Income` in thousands of USD) would receive disproportionately small coefficients even if they are important — the model would under-regularise those features and over-regularise small-scale features.

**Scaler fit on train only:** To prevent **data leakage**, μ and σ are computed only on training data. The same parameters are then applied to transform the test set. Fitting the scaler on the full dataset would leak test set statistics into the training process.

**Random Forest does not require scaling:** Tree-based splits compare feature values against thresholds. Multiplying all values by a constant doesn't change which side of the threshold a sample falls on. Random Forest is therefore scale-invariant.

---

## 3. Model 1 — Logistic Regression

### 3.1 Architecture

Logistic Regression is a **generalized linear model** that estimates the probability of binary outcomes. Despite its name, it is a classifier, not a regression model.

The model learns a **weight vector** `w = [w₁, w₂, ..., wₙ]` and a **bias** `b` (intercept). For a given input vector `x`, the model computes the **log-odds (logit)**:

```
logit(p) = log(p / (1-p)) = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
         = wᵀx + b
```

This is then passed through the **sigmoid (logistic) function** to produce a probability in (0, 1):

```
P(y=1 | x) = σ(wᵀx + b) = 1 / (1 + exp(-(wᵀx + b)))
```

The **decision rule** is:
- If `P(y=1|x) ≥ 0.5` → predict class 1 (Repaid)
- If `P(y=1|x) < 0.5` → predict class 0 (Default)

The decision boundary is the set of points where `P(y=1|x) = 0.5`, which corresponds to `wᵀx + b = 0` — a **hyperplane** in feature space. This means LR is a **linear classifier**.

### 3.2 How the Model Learns — Loss Function

The model is trained by **maximum likelihood estimation**: find the weights that maximise the probability of observing the training labels given the inputs.

The **Binary Cross-Entropy loss** (negative log-likelihood) is:

```
L(w) = -(1/N) Σᵢ [ yᵢ·log(p̂ᵢ) + (1-yᵢ)·log(1-p̂ᵢ) ]
```

where `p̂ᵢ = σ(wᵀxᵢ + b)` and `N` is the number of training samples.

**Intuition:**
- When `yᵢ = 1` (Repaid), the loss is `-log(p̂ᵢ)`. If the model predicts close to 1, loss ≈ 0. If it predicts close to 0, loss → ∞.
- When `yᵢ = 0` (Default), the loss is `-log(1 - p̂ᵢ)`. The model is penalised for high confidence in the wrong direction.

### 3.3 Optimisation

**With L2 Regularisation (Ridge):**

The regularised objective is:
```
L_reg(w) = L(w) + λ/2 · ||w||₂²    where λ = 1/C
```

The gradient with respect to `wⱼ` is:
```
∂L_reg/∂wⱼ = (1/N)·Σᵢ(p̂ᵢ - yᵢ)·xᵢⱼ + λ·wⱼ
```

The `liblinear` solver uses **coordinate descent** — optimising one weight at a time while holding others fixed. This is efficient for small-to-medium datasets and supports both L1 and L2 penalties.

**With L1 Regularisation (Lasso):**

```
L_reg(w) = L(w) + λ·||w||₁    where λ = 1/C
```

L1 regularisation drives some coefficients **exactly to zero**, performing automatic feature selection. This is useful when many features are irrelevant or redundant.

**Effect of `C`:**
- **C → 0 (high λ):** Strong regularisation → coefficients → 0 → model underfits
- **C → ∞ (low λ):** Weak regularisation → model may overfit training data

### 3.4 Parameters & Hyperparameters Tuned

| Hyperparameter | Type | Description | Search Space | Selected Value |
|----------------|------|-------------|--------------|----------------|
| `C` | Continuous | Inverse regularisation strength. `C = 1/λ`. Controls the trade-off between fitting the data and keeping weights small. | `{0.001, 0.01, 0.1, 1, 10, 100}` | (see notebook output) |
| `penalty` | Categorical | Type of norm used for regularisation | `{l1, l2}` | (see notebook output) |
| `solver` | Categorical | Numerical optimisation algorithm. `liblinear` = coordinate descent, supports L1+L2 | `{liblinear}` | liblinear |
| `max_iter` | Integer | Maximum number of solver iterations to allow convergence | `{1000}` | 1000 |

**Fixed parameters (not tuned):**
- `random_state=42` — for reproducibility
- `fit_intercept=True` — always fit a bias term (default)

**Search strategy:** 5-fold stratified `GridSearchCV`, optimising **ROC-AUC** (preferred over accuracy for class-imbalanced datasets — AUC is threshold-independent and stable).

---

## 4. Model 2 — Random Forest Classifier

### 4.1 Architecture

Random Forest is a **bootstrap-aggregated (bagged) ensemble** of `T` binary decision trees. It belongs to the family of **ensemble methods** that combine many weak learners (individual trees) into a strong learner.

**Key design choices that distinguish it from a simple collection of trees:**
1. **Bootstrap sampling:** Each tree is trained on a random sample drawn *with replacement* from the training data (≈63% unique samples per tree). This introduces diversity between trees.
2. **Feature randomisation:** At each split node, only a random subset of `max_features` features are considered as split candidates. This further decorrelates the trees.

**Prediction rule (classification):**
```
ŷ = mode{ h₁(x), h₂(x), ..., hT(x) }
```

**Predicted probability:**
```
P̂(y=1|x) = (1/T) · Σₜ 𝟙[hₜ(x) = 1]
```

### 4.2 How Individual Trees Learn

Each internal tree is a **CART (Classification and Regression Tree)**. At each node, it selects the feature `j*` and threshold `θ*` that maximises the **Gini impurity reduction**:

```
ΔGini = Gini(D) - (|D_L|/|D|)·Gini(D_L) - (|D_R|/|D|)·Gini(D_R)
```

where the Gini impurity of a node is:
```
Gini(D) = 1 - Σₖ pₖ²
```
and `pₖ` is the fraction of class `k` samples in node `D`. A perfectly pure node has `Gini = 0`.

**Splitting rules:**
- Trees grow until they are pure (each leaf contains samples from one class) unless constrained by `max_depth`, `min_samples_split`, or `min_samples_leaf`.
- No gradient computation is involved — splitting is deterministic and greedy.

### 4.3 Why Ensembling Reduces Error — Bias-Variance Decomposition

For a predictor's expected error:
```
E[(y - ŷ)²] = Bias²[ŷ] + Var[ŷ] + Irreducible noise
```

| Model | Bias | Variance |
|-------|------|----------|
| Single deep tree | Low | **High** (overfits) |
| Random Forest (T trees) | Low | Low (≈ σ²/T for uncorrelated trees) |

**Why does variance decrease?** If trees were identical (perfectly correlated), averaging would not help: `Var[mean] = σ²`. By randomising features at each split, the trees become less correlated, so:
```
Var[ŷ_forest] ≈ ρ·σ²_tree + (1-ρ)/T · σ²_tree
```
where `ρ` is the average pairwise tree correlation. As `T → ∞` and `ρ → 0`, variance → 0.

### 4.4 Optimisation — What RF Optimises

Random Forest is **not trained end-to-end** via gradient descent. Instead:
- Each tree independently optimises local Gini impurity reduction
- No global loss surface is minimised
- `class_weight='balanced'` adjusts sample weights inversely proportional to class frequency, effectively penalising misclassification of the minority class more heavily

**Structural regularisation (the hyperparameters that prevent overfitting):**

| Hyperparameter | Role | Effect of increasing |
|----------------|------|----------------------|
| `n_estimators` | Number of trees | ↑ reduces variance, diminishing returns after ~200 |
| `max_depth` | Maximum tree depth | ↓ tree complexity, ↑ regularisation |
| `min_samples_split` | Min samples to split a node | ↑ simpler trees, ↑ regularisation |
| `min_samples_leaf` | Min samples at a leaf | ↑ smoother predictions, ↑ regularisation |
| `max_features` | Features at each split | ↓ tree correlation, ↑ diversity |

### 4.5 Parameters & Hyperparameters Tuned

| Hyperparameter | Description | Search Space | Selected Value |
|----------------|-------------|--------------|----------------|
| `n_estimators` | Number of trees `T` | `{100, 200, 300}` | (see notebook) |
| `max_depth` | Max depth of each tree; `None` = fully grown | `{None, 5, 10, 20}` | (see notebook) |
| `min_samples_split` | Minimum samples required at a node to allow a split | `{2, 5, 10}` | (see notebook) |
| `min_samples_leaf` | Minimum samples at any leaf node | `{1, 2, 4}` | (see notebook) |
| `max_features` | Number of features to consider at each split; `sqrt` is the standard for classification | `{sqrt, log2}` | (see notebook) |

**Fixed parameters:**
- `random_state=42` — reproducibility
- `class_weight='balanced'` — automatic minority-class upweighting
- `n_jobs=-1` — use all CPU cores for parallel tree training

---

## 5. Explainable AI (XAI) Techniques

Explainability is crucial in credit scoring — regulations (e.g., GDPR, ECOA) often require lenders to explain adverse decisions to applicants.

---

### 5.1 Logistic Regression Coefficients

**Type:** Model-specific, global  
**What it shows:** The **signed, magnitude-ordered coefficients** of the trained model on standardised features.

After standardisation, all features are on the same scale. Therefore, `|wⱼ|` directly indicates feature importance. The sign tells us the direction:
- `wⱼ > 0` → increasing feature `xⱼ` increases log-odds → **increases repayment probability**
- `wⱼ < 0` → increasing feature `xⱼ` decreases log-odds → **increases default probability**

**Limitation:** Coefficients only capture linear effects and cannot detect interaction terms.

---

### 5.2 Gini Feature Importance (Mean Decrease in Impurity, MDI)

**Type:** Model-specific to tree ensembles, global  
**What it shows:** For each feature `xⱼ`, the total reduction in Gini impurity across all splits on `xⱼ`, averaged across all trees:
```
Importance(xⱼ) = (1/T) · Σₜ Σ_{v: split on j} (|D_v|/|D|) · ΔGini(v)
```

**Limitations:**
- Biased towards high-cardinality numerical features (they have more candidate thresholds)
- Can assign high importance to features that are only correlated with the true driver

---

### 5.3 Permutation Feature Importance (PFI)

**Type:** Model-agnostic, global  
**What it shows:** For each feature `xⱼ`, the **decrease in ROC-AUC** when `xⱼ`'s values are randomly shuffled in the test set (breaking the feature-target association):
```
PI(xⱼ) = score_original - mean_k(score_shuffled_j(k))
```

Repeated 30 times to estimate variation (error bars in the plot).

**Advantages over MDI:**
- Evaluated on held-out test data → reflects actual generalisation
- Not biased by feature cardinality
- Works on any model (model-agnostic)

**Limitation:** May underestimate importance when features are correlated (shuffling one correlated feature can partially be compensated by the other).

---

### 5.4 SHAP — SHapley Additive exPlanations

**Type:** Both model-agnostic and model-specific efficient variants; global and local  
**Mathematical basis:** Rooted in **cooperative game theory** (Shapley 1953). Features are treated as "players" in a cooperative game, and the Shapley value assigns each player its **average marginal contribution** across all possible orderings of players.

**Formal definition:**
```
φⱼ(f, x) = Σ_{S ⊆ F\{j}} [|S|!(|F|-|S|-1)! / |F|!] · [f(x_{S∪{j}}) - f(x_S)]
```

where:
- `F` = full feature set
- `S` = a subset of features excluding `j`
- `f(x_S)` = model output with only features in `S` known (others marginalised)

**Key SHAP Properties (SHAP Axioms):**
1. **Efficiency:** `f(x) = φ₀ + Σⱼ φⱼ` — SHAP values sum exactly to the model output minus the baseline
2. **Symmetry:** If two features contribute equally in all subsets, they receive equal SHAP values
3. **Dummy:** A feature that never changes the model output receives `φⱼ = 0`
4. **Additivity:** The SHAP values of a forest = sum of SHAP values of individual trees

**SHAP Explainers used:**

| Explainer | Model | Algorithm | Complexity |
|-----------|-------|-----------|------------|
| `LinearExplainer` | Logistic Regression | Closed-form using covariance matrix | O(N·F²) |
| `TreeExplainer` | Random Forest | Polynomial-time exact algorithm (Lundberg 2020) | O(T·L·D²) |

**Visualisations generated:**

| Plot | What it shows |
|------|---------------|
| **Bar (mean \|SHAP\|)** | Global feature importance: average absolute SHAP value across all test samples |
| **Beeswarm** | Global + directional: each dot = one sample; x-axis = SHAP value; colour = feature value (red=high, blue=low) |
| **Waterfall** | Local explanation: step-by-step how each feature pushes a single prediction away from the baseline |
| **Force plot** | Local explanation: compact visual of feature contributions for one prediction |

**How to read the Beeswarm:**
- Features are ordered by mean |SHAP| (most important at top)
- Right-side dots (positive SHAP) → that feature increased `P(Repaid)` for that sample
- Left-side dots (negative SHAP) → that feature decreased `P(Repaid)` (increased default risk)
- Red colour (high feature value) on the right = "higher value → more likely to repay"
- Blue colour (low feature value) on the right = "lower value → more likely to repay"

---

## 6. Evaluation Results

Results are dynamically printed in the notebook. The metrics used are:

| Metric | Formula | Why Used |
|--------|---------|----------|
| **Accuracy** | `(TP+TN) / N` | Simple check; misleading if classes are imbalanced |
| **Precision (per class)** | `TP / (TP+FP)` | "When we predict Repaid, how often are we right?" |
| **Recall (per class)** | `TP / (TP+FN)` | "Of all actual Repaid, what fraction did we catch?" |
| **F1-Score** | `2·(P·R)/(P+R)` | Harmonic mean of P & R; good for imbalanced classes |
| **ROC-AUC** | Area under ROC curve | Threshold-independent; 0.5 = random, 1.0 = perfect |
| **Average Precision** | Area under Precision-Recall curve | More informative than ROC-AUC for imbalanced data |

**Confusion Matrix terminology:**
- **TP** (True Positive): Correctly predicted Repaid
- **TN** (True Negative): Correctly predicted Default
- **FP** (False Positive): Predicted Repaid, actually Defaulted (Type I error — costly to lender)
- **FN** (False Negative): Predicted Default, actually Repaid (Type II error — lost business opportunity)

---

## 7. Model Comparison & Selection

### Decision Framework

| Criterion | Logistic Regression | Random Forest |
|-----------|--------------------|-|
| Interpretability | ★★★ High — coefficients are the explanation | ★★ Medium — requires SHAP/PFI |
| Predictive power | ★★ Moderate — limited to linear boundary | ★★★ High — captures non-linear interactions |
| Regulatory compliance | ★★★ Easy to audit | ★★ Auditable with XAI tools |
| Training speed | Fast | Moderate |
| Sensitivity to outliers | High (depends on regularisation) | Low (robust — rank-based splits) |
| Feature scaling required | Yes | No |
| Handles feature interactions | No (unless polynomial features added) | Yes (implicitly via tree structure) |

### Recommendation

Both models should be reported. Their combination provides:
1. A **transparent baseline** (LR) that regulators, auditors, and applicants can understand
2. A **high-performance model** (RF) that can be used operationally, with SHAP for per-decision audits

If one model must be chosen, prefer the one with significantly better **CV ROC-AUC** (see notebook output), provided the performance gap justifies the complexity cost.

---

## 8. References

- Breiman, L. (2001). *Random Forests.* Machine Learning, 45, 5–32.
- Cox, D. R. (1958). *The regression analysis of binary sequences.* Journal of the Royal Statistical Society B, 20(2), 215–242.
- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS 2017.
- Lundberg, S. M., et al. (2020). *From local explanations to global understanding with explainable AI for trees.* Nature Machine Intelligence, 2(1), 56–67.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.* KDD 2016.
- Shapley, L. S. (1953). *A value for n-person games.* Contributions to the Theory of Games, 2, 307–317.
- Tibshirani, R. (1996). *Regression shrinkage and selection via the Lasso.* JRSS-B, 58(1), 267–288.

---

*Document prepared for course assignment — March 2026*  
*Companion to: `model_training.ipynb`*
