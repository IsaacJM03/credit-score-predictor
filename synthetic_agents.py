"""
Constraint-aware synthetic tabular data generation for credit-risk pipelines.

Primary mode uses a lightweight adversarial generator (GAN-style) trained on
real tabular records. Secondary mode uses a constrained bootstrap sampler.
Both modes pass through hard-constraint repair/filtering and soft-distribution
quality checks before release.
"""

from __future__ import annotations

import argparse
import os
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

DEFAULT_FEATURES = [
    "Age",
    "Monthly_Income",
    "Credit_Score",
    "Loan_Amount_Requested",
    "Employment_Status",
    "Existing_Debts",
    "Number_of_Previous_Loans",
    "Payment_History",
    "Debt_to_Income_Ratio",
]
DEFAULT_TARGET = "Repayment_Status"

PAYMENT_ORDER = {"Poor": 0, "Fair": 1, "Good": 2, "Excellent": 3}
EMPLOYMENT_ORDER = {"Employed": 0, "Self-Employed": 1, "Student": 2, "Unemployed": 3}

NUMERIC_COLUMNS = {
    "Age",
    "Monthly_Income",
    "Credit_Score",
    "Loan_Amount_Requested",
    "Existing_Debts",
    "Number_of_Previous_Loans",
    "Debt_to_Income_Ratio",
    DEFAULT_TARGET,
}


@dataclass
class GenerationReport:
    mode: str
    generated_rows: int
    accepted_rows: int
    repaired_values: int
    hard_rejects: int
    soft_score_mean: float


class _Generator(nn.Module):
    def __init__(self, noise_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _Discriminator(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TabularSyntheticAgent:
    """GAN-backed synthetic generator for mixed-type tabular data."""

    def __init__(self, feature_columns: List[str], target_column: str, seed: int = 42):
        self.feature_columns = list(feature_columns)
        self.target_column = target_column
        self.columns = self.feature_columns + [self.target_column]
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.categorical_levels: Dict[str, List[str]] = {}
        self.numeric_stats: Dict[str, Dict[str, float]] = {}
        self.col_ranges: Dict[str, Tuple[int, int]] = {}
        self.soft_reference: Dict[str, Dict[str, float]] = {}
        self.training_frame: Optional[pd.DataFrame] = None

        self.noise_dim = 32
        self.generator: Optional[_Generator] = None
        self.discriminator: Optional[_Discriminator] = None
        self.output_dim: Optional[int] = None

    def _set_seed(self) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def _prepare_schema(self, df: pd.DataFrame) -> None:
        self.numeric_cols = []
        self.categorical_cols = []
        self.categorical_levels = {}
        self.numeric_stats = {}
        self.col_ranges = {}

        cursor = 0
        for col in self.columns:
            if col in NUMERIC_COLUMNS:
                self.numeric_cols.append(col)
                vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
                mu = float(vals.mean())
                sigma = float(vals.std())
                sigma = sigma if sigma > 1e-8 else 1.0
                self.numeric_stats[col] = {
                    "mean": mu,
                    "std": sigma,
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                }
                self.col_ranges[col] = (cursor, cursor + 1)
                cursor += 1
            else:
                self.categorical_cols.append(col)
                levels = sorted(df[col].astype(str).fillna("Unknown").unique().tolist())
                if not levels:
                    levels = ["Unknown"]
                self.categorical_levels[col] = levels
                self.col_ranges[col] = (cursor, cursor + len(levels))
                cursor += len(levels)

        self.output_dim = cursor

    def _encode(self, df: pd.DataFrame) -> np.ndarray:
        rows = []
        for _, row in df.iterrows():
            encoded = []
            for col in self.columns:
                if col in self.numeric_cols:
                    value = float(row[col])
                    stats = self.numeric_stats[col]
                    encoded.append((value - stats["mean"]) / stats["std"])
                else:
                    vec = np.zeros(len(self.categorical_levels[col]), dtype=float)
                    try:
                        idx = self.categorical_levels[col].index(str(row[col]))
                    except ValueError:
                        idx = 0
                    vec[idx] = 1.0
                    encoded.extend(vec.tolist())
            rows.append(encoded)
        return np.asarray(rows, dtype=np.float32)

    def _decode(self, matrix: np.ndarray) -> pd.DataFrame:
        out = {}
        for col in self.columns:
            start, end = self.col_ranges[col]
            block = matrix[:, start:end]
            if col in self.numeric_cols:
                stats = self.numeric_stats[col]
                vals = block[:, 0] * stats["std"] + stats["mean"]
                out[col] = vals
            else:
                idx = block.argmax(axis=1)
                levels = self.categorical_levels[col]
                out[col] = [levels[int(i)] for i in idx]
        return pd.DataFrame(out)

    def _build_soft_reference(self, df: pd.DataFrame) -> None:
        ref = {}
        for col in self.numeric_cols:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
            ref[col] = {"mean": float(vals.mean()), "std": float(vals.std() + 1e-8)}
        for col in self.categorical_cols:
            vc = df[col].astype(str).value_counts(normalize=True)
            for lvl in self.categorical_levels[col]:
                ref[f"{col}::{lvl}"] = {"p": float(vc.get(lvl, 0.0))}
        self.soft_reference = ref

    def fit(
        self,
        df: pd.DataFrame,
        epochs: int = 40,
        batch_size: int = 128,
        cache_path: Optional[str] = None,
    ) -> "TabularSyntheticAgent":
        self._set_seed()
        train_df = df[self.columns].dropna().copy()
        if train_df.empty:
            raise ValueError("Training dataframe for synthetic agent is empty")
        self.training_frame = train_df

        if cache_path and os.path.exists(cache_path):
            payload = torch.load(cache_path, map_location="cpu")
            self.numeric_cols = payload["numeric_cols"]
            self.categorical_cols = payload["categorical_cols"]
            self.categorical_levels = payload["categorical_levels"]
            self.numeric_stats = payload["numeric_stats"]
            self.col_ranges = payload["col_ranges"]
            self.output_dim = payload["output_dim"]
            self.soft_reference = payload["soft_reference"]
            self.generator = _Generator(self.noise_dim, self.output_dim)
            self.generator.load_state_dict(payload["generator_state"])
            self.discriminator = _Discriminator(self.output_dim)
            self.discriminator.load_state_dict(payload["discriminator_state"])
            self.generator.eval()
            self.discriminator.eval()
            return self

        self._prepare_schema(train_df)
        self._build_soft_reference(train_df)
        encoded = self._encode(train_df)

        x_real = torch.tensor(encoded, dtype=torch.float32)
        n = x_real.shape[0]

        self.generator = _Generator(self.noise_dim, self.output_dim)
        self.discriminator = _Discriminator(self.output_dim)

        g_opt = torch.optim.Adam(self.generator.parameters(), lr=1e-3)
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=1e-3)
        bce = nn.BCELoss()

        for _ in range(max(1, epochs)):
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                real_batch = x_real[idx]
                bs = real_batch.shape[0]

                # Discriminator step
                z = torch.randn(bs, self.noise_dim)
                fake_batch = self.generator(z).detach()
                real_pred = self.discriminator(real_batch)
                fake_pred = self.discriminator(fake_batch)
                d_loss = bce(real_pred, torch.ones_like(real_pred)) + bce(fake_pred, torch.zeros_like(fake_pred))
                d_opt.zero_grad()
                d_loss.backward()
                d_opt.step()

                # Generator step
                z = torch.randn(bs, self.noise_dim)
                generated = self.generator(z)
                gen_pred = self.discriminator(generated)
                g_loss = bce(gen_pred, torch.ones_like(gen_pred))
                g_opt.zero_grad()
                g_loss.backward()
                g_opt.step()

        self.generator.eval()
        self.discriminator.eval()

        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            torch.save(
                {
                    "numeric_cols": self.numeric_cols,
                    "categorical_cols": self.categorical_cols,
                    "categorical_levels": self.categorical_levels,
                    "numeric_stats": self.numeric_stats,
                    "col_ranges": self.col_ranges,
                    "output_dim": self.output_dim,
                    "soft_reference": self.soft_reference,
                    "generator_state": self.generator.state_dict(),
                    "discriminator_state": self.discriminator.state_dict(),
                },
                cache_path,
            )

        return self

    def sample(self, n_samples: int) -> pd.DataFrame:
        if self.generator is None or self.output_dim is None:
            raise RuntimeError("TabularSyntheticAgent is not fitted")
        z = torch.randn(n_samples, self.noise_dim)
        with torch.no_grad():
            synthetic = self.generator(z).cpu().numpy()
        return self._decode(synthetic)

    def constrained_bootstrap(self, n_samples: int) -> pd.DataFrame:
        if self.training_frame is None:
            raise RuntimeError("TabularSyntheticAgent requires training data for fallback sampling")
        sampled = self.training_frame.sample(n=n_samples, replace=True, random_state=self.seed).reset_index(drop=True)
        for col in self.numeric_cols:
            vals = pd.to_numeric(sampled[col], errors="coerce").fillna(self.numeric_stats[col]["mean"]).astype(float)
            jitter = self.rng.normal(0.0, self.numeric_stats[col]["std"] * 0.08, size=n_samples)
            sampled[col] = vals + jitter
        return sampled

    def soft_score(self, df: pd.DataFrame) -> np.ndarray:
        if not self.soft_reference:
            return np.ones(len(df), dtype=float)

        numeric_scores = []
        for col in self.numeric_cols:
            ref = self.soft_reference[col]
            vals = pd.to_numeric(df[col], errors="coerce").fillna(ref["mean"]).astype(float).values
            z = np.abs(vals - ref["mean"]) / (ref["std"] + 1e-8)
            numeric_scores.append(np.exp(-0.15 * z))

        categorical_scores = []
        for col in self.categorical_cols:
            levels = self.categorical_levels[col]
            expected = np.array([self.soft_reference.get(f"{col}::{lvl}", {"p": 0.0})["p"] for lvl in levels]) + 1e-6
            expected = expected / expected.sum()
            expected = expected / expected.max()
            idx_map = {lvl: i for i, lvl in enumerate(levels)}
            observed = np.array([idx_map.get(str(v), 0) for v in df[col].astype(str)], dtype=np.int64)
            categorical_scores.append(expected[observed])

        parts = []
        if numeric_scores:
            parts.append(np.mean(np.vstack(numeric_scores), axis=0))
        if categorical_scores:
            parts.append(np.mean(np.vstack(categorical_scores), axis=0))

        if not parts:
            return np.ones(len(df), dtype=float)
        return np.mean(np.vstack(parts), axis=0)


def _apply_hard_constraints(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    fixed = df.copy()
    repaired = 0

    def _clip(col: str, low: float, high: float) -> None:
        nonlocal repaired
        vals = pd.to_numeric(fixed[col], errors="coerce").fillna(low)
        before = vals.copy()
        vals = vals.clip(low, high)
        repaired += int((before != vals).sum())
        fixed[col] = vals

    _clip("Age", 18, 80)
    _clip("Monthly_Income", 500, 100_000)
    _clip("Credit_Score", 300, 850)
    _clip("Loan_Amount_Requested", 500, 150_000)
    _clip("Existing_Debts", 0, 100_000)
    _clip("Number_of_Previous_Loans", 0, 30)
    _clip("Debt_to_Income_Ratio", 0.0, 1.0)

    fixed["Number_of_Previous_Loans"] = np.rint(fixed["Number_of_Previous_Loans"]).astype(int)
    fixed[DEFAULT_TARGET] = (pd.to_numeric(fixed[DEFAULT_TARGET], errors="coerce").fillna(0.0) >= 0.5).astype(int)

    fixed["Payment_History"] = fixed["Payment_History"].astype(str)
    fixed.loc[~fixed["Payment_History"].isin(PAYMENT_ORDER), "Payment_History"] = "Fair"

    fixed["Employment_Status"] = fixed["Employment_Status"].astype(str)
    fixed.loc[~fixed["Employment_Status"].isin(EMPLOYMENT_ORDER), "Employment_Status"] = "Employed"

    invalid = (
        (fixed["Monthly_Income"] <= 0)
        | (fixed["Loan_Amount_Requested"] <= 0)
        | (fixed["Debt_to_Income_Ratio"] < 0)
        | (fixed["Debt_to_Income_Ratio"] > 1)
    )
    rejects = int(invalid.sum())
    fixed = fixed[~invalid].reset_index(drop=True)
    return fixed, repaired, rejects


def _to_distill_schema(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    out = pd.DataFrame()
    out["Age"] = df["Age"].astype(float)
    out["Monthly_Income"] = df["Monthly_Income"].astype(float)
    out["Credit_Score"] = df["Credit_Score"].astype(float)
    out["Loan_Amount_Requested"] = df["Loan_Amount_Requested"].astype(float)
    out["Existing_Debts"] = df["Existing_Debts"].astype(float)

    out["Num_Previous_Loans"] = df["Number_of_Previous_Loans"].astype(int)
    out["Payment_History_Encoded"] = df["Payment_History"].map(PAYMENT_ORDER).fillna(1).astype(int)
    out["Employment_Status_Encoded"] = df["Employment_Status"].map(EMPLOYMENT_ORDER).fillna(0).astype(int)

    out["Debt_to_Income_Ratio"] = df["Debt_to_Income_Ratio"].astype(float).clip(0, 1)
    out["Loan_to_Income_Ratio"] = out["Loan_Amount_Requested"] / (out["Monthly_Income"] + 1.0)
    out["Repayment_Consistency"] = (out["Payment_History_Encoded"] + 1.0) / (out["Num_Previous_Loans"] + 1.0)

    base_velocity = (out["Num_Previous_Loans"] / (out["Age"] + 1.0)).clip(0, 1)
    out["Application_Velocity"] = np.clip(base_velocity + rng.normal(0, 0.05, len(out)), 0, 1)

    out["Credit_Utilization_Score"] = out["Existing_Debts"] / (out["Credit_Score"] + 1.0)
    out["Duration_Months"] = np.clip(
        (out["Loan_Amount_Requested"] / (out["Monthly_Income"] + 1.0)) * 12 + rng.normal(0, 3, len(out)),
        1,
        60,
    ).astype(int)
    out[DEFAULT_TARGET] = df[DEFAULT_TARGET].astype(int)
    return out


def _ensure_output_layout(output_root: str, run_name: str) -> Dict[str, str]:
    base = os.path.join(output_root, run_name)
    layout = {
        "base": base,
        "reports": os.path.join(base, "reports"),
        "visualizations": os.path.join(base, "visualizations"),
        "samples": os.path.join(base, "samples"),
    }
    for path in layout.values():
        os.makedirs(path, exist_ok=True)
    return layout


def _save_generation_visualizations(
    reference_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    output_dir: str,
    seed: int,
) -> List[str]:
    if not HAS_MATPLOTLIB:
        return []

    rng = np.random.default_rng(seed)
    ref = reference_df.copy()
    syn = synthetic_df.copy()
    if len(ref) > 200_000:
        ref = ref.sample(n=200_000, random_state=seed)
    if len(syn) > 200_000:
        syn = syn.sample(n=200_000, random_state=seed)

    saved = []
    plot_specs = [
        ("Monthly_Income", 80),
        ("Credit_Score", 80),
        ("Debt_to_Income_Ratio", 60),
    ]
    for column, bins in plot_specs:
        if column not in ref.columns or column not in syn.columns:
            continue
        rvals = pd.to_numeric(ref[column], errors="coerce").dropna().values
        svals = pd.to_numeric(syn[column], errors="coerce").dropna().values
        if len(rvals) == 0 or len(svals) == 0:
            continue

        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.hist(rvals, bins=bins, alpha=0.45, density=True, label="reference")
        ax.hist(svals, bins=bins, alpha=0.45, density=True, label="synthetic")
        ax.set_title(f"Distribution Comparison: {column}")
        ax.set_ylabel("Density")
        ax.legend()
        path = os.path.join(output_dir, f"dist_{column.lower()}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    if DEFAULT_TARGET in ref.columns and DEFAULT_TARGET in syn.columns:
        ref_rate = pd.to_numeric(ref[DEFAULT_TARGET], errors="coerce").fillna(0).astype(int).mean()
        syn_rate = pd.to_numeric(syn[DEFAULT_TARGET], errors="coerce").fillna(0).astype(int).mean()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["reference", "synthetic"], [ref_rate, syn_rate], color=["#457B9D", "#2A9D8F"])
        ax.set_ylim(0, 1)
        ax.set_title("Target Positive Rate")
        ax.set_ylabel("Repayment_Status mean")
        path = os.path.join(output_dir, "target_rate.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    return saved


def _resolve_parquet_backend() -> Tuple[Optional[str], Optional[Any], Optional[str]]:
    try:
        import pyarrow as pa  # type: ignore
        return "pyarrow", pa, None
    except ImportError:
        pass

    try:
        import fastparquet  # type: ignore
        return "fastparquet", fastparquet, None
    except ImportError:
        pass

    return None, None, "No parquet engine found. Install 'pyarrow' or 'fastparquet'."


def _resolve_reference_csv_path(preferred_path: str) -> Optional[str]:
    candidates = [
        preferred_path,
        "comprehensive_loan_data.csv",
        "loan_data.csv",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def _build_fresh_reference_frame(n_rows: int = 200_000, seed: int = 42) -> pd.DataFrame:
    """Create a standalone reference frame when no seed CSV is available."""
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(38, 11, n_rows), 18, 80)
    monthly_income = np.exp(rng.normal(8.7, 0.55, n_rows))
    monthly_income = np.clip(monthly_income, 500, 100_000)
    credit_score = np.clip(rng.normal(620, 95, n_rows), 300, 850)
    loan_amount = np.exp(rng.normal(9.0, 0.75, n_rows))
    loan_amount = np.clip(loan_amount, 500, 150_000)

    prev_loans = np.clip(rng.poisson(2.3, n_rows), 0, 30)
    dti = np.clip(rng.beta(2.2, 3.8, n_rows), 0.0, 1.0)
    existing_debts = np.clip(monthly_income * dti * rng.uniform(0.8, 1.3, n_rows), 0, 100_000)

    employment_values = np.array(list(EMPLOYMENT_ORDER.keys()))
    employment_probs = np.array([0.62, 0.17, 0.11, 0.10])
    employment = rng.choice(employment_values, size=n_rows, p=employment_probs)

    payment_values = np.array(list(PAYMENT_ORDER.keys()))
    payment_probs = np.array([0.18, 0.30, 0.34, 0.18])
    payment = rng.choice(payment_values, size=n_rows, p=payment_probs)

    # Simple risk model to seed class balance with realistic signal.
    pay_score = pd.Series(payment).map(PAYMENT_ORDER).values
    emp_penalty = pd.Series(employment).map({"Employed": 0.0, "Self-Employed": 0.2, "Student": 0.25, "Unemployed": 0.45}).values
    logit = (
        2.2 * dti
        + 0.55 * (loan_amount / (monthly_income + 1.0))
        + 0.22 * prev_loans
        + emp_penalty
        - 0.0065 * credit_score
        - 0.42 * pay_score
        + rng.normal(0, 0.35, n_rows)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    repayment_status = (rng.uniform(0, 1, n_rows) > prob).astype(int)

    return pd.DataFrame(
        {
            "Age": age,
            "Monthly_Income": monthly_income,
            "Credit_Score": credit_score,
            "Loan_Amount_Requested": loan_amount,
            "Employment_Status": employment,
            "Existing_Debts": existing_debts,
            "Number_of_Previous_Loans": prev_loans,
            "Payment_History": payment,
            "Debt_to_Income_Ratio": dti,
            DEFAULT_TARGET: repayment_status,
        }
    )


def generate_distill_synthetic_dataset(
    reference_df: pd.DataFrame,
    n_rows: int = 5_000_000,
    mode: str = "auto",
    seed: int = 42,
    epochs: int = 40,
    batch_size: int = 128,
    soft_tolerance: float = 0.25,
    cache_path: str = "models/synthetic/checkpoints/tabular_agent.pt",
    output_root: str = "results/synthetic",
    run_name: str = "synthetic_5m",
) -> Tuple[pd.DataFrame, GenerationReport]:
    mode = mode.lower().strip()
    if mode not in {"auto", "gan", "sampler"}:
        raise ValueError(f"Unsupported synthetic mode: {mode}")

    base_cols = [c for c in DEFAULT_FEATURES + [DEFAULT_TARGET] if c in reference_df.columns]
    missing = sorted(set(DEFAULT_FEATURES + [DEFAULT_TARGET]) - set(base_cols))
    if missing:
        raise ValueError(f"Reference dataframe missing required columns: {missing}")

    train_df = reference_df[base_cols].dropna().copy()
    layout = _ensure_output_layout(output_root=output_root, run_name=run_name)

    agent = TabularSyntheticAgent(DEFAULT_FEATURES, DEFAULT_TARGET, seed=seed)
    agent.fit(train_df, epochs=epochs, batch_size=batch_size, cache_path=cache_path)

    generated = 0
    accepted_frames = []
    accepted_count = 0
    repaired_total = 0
    hard_rejects = 0

    attempt_mode = "gan" if mode in {"auto", "gan"} else "sampler"
    for _ in range(16):
        remaining = max(0, n_rows - accepted_count)
        if remaining == 0:
            break
        need = int(min(1_000_000, max(10_000, remaining * 1.35)))
        if attempt_mode == "gan":
            batch = agent.sample(need)
        else:
            batch = agent.constrained_bootstrap(need)

        generated += len(batch)
        fixed, repaired, rejected = _apply_hard_constraints(batch)
        repaired_total += repaired
        hard_rejects += rejected

        if fixed.empty:
            continue

        scores = agent.soft_score(fixed)
        keep = scores >= (1.0 - soft_tolerance)
        kept = fixed.loc[keep]
        if not kept.empty:
            accepted_frames.append(kept)
            accepted_count += len(kept)

        if accepted_count >= n_rows:
            break

        if attempt_mode == "gan" and mode == "auto":
            attempt_mode = "sampler"

    if not accepted_frames:
        fallback = agent.constrained_bootstrap(n_rows)
        fixed, repaired, rejected = _apply_hard_constraints(fallback)
        repaired_total += repaired
        hard_rejects += rejected
        accepted_df = fixed.head(n_rows)
        scores = agent.soft_score(accepted_df)
        final_mode = "sampler"
    else:
        accepted_df = pd.concat(accepted_frames, ignore_index=True).head(n_rows)
        scores = agent.soft_score(accepted_df)
        final_mode = attempt_mode

    # Match target prevalence to reference to reduce GAN class-collapse artifacts.
    ref_rate = float(pd.to_numeric(train_df[DEFAULT_TARGET], errors="coerce").fillna(0.0).mean())
    ref_rate = min(max(ref_rate, 0.05), 0.95)
    raw_target = pd.to_numeric(accepted_df[DEFAULT_TARGET], errors="coerce").fillna(ref_rate).astype(float)
    if raw_target.nunique() <= 1:
        rng = np.random.default_rng(seed)
        accepted_df[DEFAULT_TARGET] = rng.binomial(1, ref_rate, size=len(accepted_df)).astype(int)
    else:
        threshold = float(np.quantile(raw_target, 1.0 - ref_rate))
        accepted_df[DEFAULT_TARGET] = (raw_target >= threshold).astype(int)

    distill_df = _to_distill_schema(accepted_df, seed=seed)

    report = GenerationReport(
        mode=final_mode,
        generated_rows=generated,
        accepted_rows=len(distill_df),
        repaired_values=repaired_total,
        hard_rejects=hard_rejects,
        soft_score_mean=float(np.mean(scores)) if len(scores) else 0.0,
    )

    sample_path = os.path.join(layout["samples"], "synthetic_sample_head.csv")
    distill_df.head(10_000).to_csv(sample_path, index=False)

    viz_paths = _save_generation_visualizations(
        reference_df=train_df,
        synthetic_df=accepted_df,
        output_dir=layout["visualizations"],
        seed=seed,
    )

    report_payload = {
        **asdict(report),
        "requested_rows": int(n_rows),
        "cache_path": cache_path,
        "sample_preview_path": sample_path,
        "visualizations": viz_paths,
    }
    report_path = os.path.join(layout["reports"], "generation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    return distill_df, report


def build_synthetic_dataset(
    reference_df: pd.DataFrame,
    output_csv_path: str,
    output_parquet_path: Optional[str] = None,
    n_rows: int = 5_000_000,
    mode: str = "auto",
    seed: int = 42,
    epochs: int = 40,
    batch_size: int = 128,
    soft_tolerance: float = 0.25,
    cache_path: str = "models/synthetic/checkpoints/tabular_agent_5m.pt",
    output_root: str = "results/synthetic",
    run_name: str = "dataset_5m",
    chunk_rows: int = 250_000,
    save_visualizations: bool = True,
) -> GenerationReport:
    """Build a large synthetic dataset as the primary deliverable.

    This function writes the generated dataset directly to CSV in chunks so it
    can scale to multimillion-row outputs without keeping the final dataset
    fully in memory.
    """
    mode = mode.lower().strip()
    if mode not in {"auto", "gan", "sampler"}:
        raise ValueError(f"Unsupported synthetic mode: {mode}")

    base_cols = [c for c in DEFAULT_FEATURES + [DEFAULT_TARGET] if c in reference_df.columns]
    missing = sorted(set(DEFAULT_FEATURES + [DEFAULT_TARGET]) - set(base_cols))
    if missing:
        raise ValueError(f"Reference dataframe missing required columns: {missing}")

    train_df = reference_df[base_cols].dropna().copy()
    layout = _ensure_output_layout(output_root=output_root, run_name=run_name)

    out_dir = os.path.dirname(output_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(output_csv_path):
        os.remove(output_csv_path)

    parquet_requested = output_parquet_path is not None
    parquet_written = False
    parquet_skip_reason: Optional[str] = None
    parquet_backend: Optional[str] = None
    parquet_module: Optional[Any] = None
    parquet_writer: Optional[Any] = None

    if parquet_requested and output_parquet_path:
        parquet_out_dir = os.path.dirname(output_parquet_path)
        if parquet_out_dir:
            os.makedirs(parquet_out_dir, exist_ok=True)
        if os.path.exists(output_parquet_path):
            os.remove(output_parquet_path)
        parquet_backend, parquet_module, parquet_skip_reason = _resolve_parquet_backend()
        if parquet_skip_reason:
            print(f"[WARN] Parquet export skipped: {parquet_skip_reason}")

    agent = TabularSyntheticAgent(DEFAULT_FEATURES, DEFAULT_TARGET, seed=seed)
    agent.fit(train_df, epochs=epochs, batch_size=batch_size, cache_path=cache_path)

    ref_rate = float(pd.to_numeric(train_df[DEFAULT_TARGET], errors="coerce").fillna(0.0).mean())
    ref_rate = min(max(ref_rate, 0.05), 0.95)

    generated = 0
    accepted_rows = 0
    repaired_total = 0
    hard_rejects = 0
    running_soft_scores = []
    sample_frames = []
    viz_frames = []

    attempt_mode = "gan" if mode in {"auto", "gan"} else "sampler"
    header_written = False

    for attempt in range(80):
        remaining = max(0, n_rows - accepted_rows)
        if remaining == 0:
            break

        need = int(min(1_000_000, max(20_000, remaining * 1.35)))
        if attempt_mode == "gan":
            batch = agent.sample(need)
        else:
            batch = agent.constrained_bootstrap(need)

        generated += len(batch)
        fixed, repaired, rejected = _apply_hard_constraints(batch)
        repaired_total += repaired
        hard_rejects += rejected

        if fixed.empty:
            continue

        scores = agent.soft_score(fixed)
        keep = scores >= (1.0 - soft_tolerance)
        kept = fixed.loc[keep].copy()

        if kept.empty:
            if attempt_mode == "gan" and mode == "auto":
                attempt_mode = "sampler"
            continue

        running_soft_scores.append(float(np.mean(scores[keep])))

        raw_target = pd.to_numeric(kept[DEFAULT_TARGET], errors="coerce").fillna(ref_rate).astype(float)
        if raw_target.nunique() <= 1:
            rng = np.random.default_rng(seed + attempt)
            kept[DEFAULT_TARGET] = rng.binomial(1, ref_rate, size=len(kept)).astype(int)
        else:
            threshold = float(np.quantile(raw_target, 1.0 - ref_rate))
            kept[DEFAULT_TARGET] = (raw_target >= threshold).astype(int)

        distill_chunk = _to_distill_schema(kept, seed=seed + attempt)
        take_n = min(len(distill_chunk), remaining, chunk_rows)
        if take_n <= 0:
            continue
        chunk_out = distill_chunk.head(take_n)

        chunk_out.to_csv(output_csv_path, mode="a", index=False, header=not header_written)
        header_written = True
        accepted_rows += int(take_n)

        if parquet_requested and output_parquet_path and parquet_skip_reason is None:
            try:
                if parquet_backend == "pyarrow":
                    import pyarrow.parquet as pq  # type: ignore

                    table = parquet_module.Table.from_pandas(chunk_out, preserve_index=False)  # type: ignore[attr-defined]
                    if parquet_writer is None:
                        parquet_writer = pq.ParquetWriter(output_parquet_path, table.schema, compression="snappy")
                    parquet_writer.write_table(table)
                    parquet_written = True
                elif parquet_backend == "fastparquet":
                    parquet_module.write(  # type: ignore[union-attr]
                        output_parquet_path,
                        chunk_out,
                        compression="SNAPPY",
                        write_index=False,
                        append=parquet_written,
                    )
                    parquet_written = True
            except Exception as exc:
                parquet_skip_reason = f"Parquet write failed: {exc}"
                print(f"[WARN] {parquet_skip_reason}")

        if sum(len(x) for x in sample_frames) < 10_000:
            need_sample = 10_000 - sum(len(x) for x in sample_frames)
            sample_frames.append(chunk_out.head(need_sample))

        if sum(len(x) for x in viz_frames) < 200_000:
            need_viz = 200_000 - sum(len(x) for x in viz_frames)
            viz_frames.append(kept.head(need_viz))

        if attempt_mode == "gan" and mode == "auto" and accepted_rows < n_rows:
            attempt_mode = "sampler"

    if accepted_rows < n_rows:
        raise RuntimeError(
            f"Synthetic dataset build incomplete: requested={n_rows}, generated={accepted_rows}. "
            "Consider increasing attempts or relaxing soft_tolerance."
        )

    if parquet_writer is not None:
        parquet_writer.close()

    if parquet_requested and not parquet_written and parquet_skip_reason is None:
        parquet_skip_reason = "No parquet rows were written."

    sample_df = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    sample_path = os.path.join(layout["samples"], "synthetic_dataset_head.csv")
    if not sample_df.empty:
        sample_df.to_csv(sample_path, index=False)

    viz_paths: List[str] = []
    if save_visualizations:
        viz_df = pd.concat(viz_frames, ignore_index=True) if viz_frames else train_df.copy()
        viz_paths = _save_generation_visualizations(
            reference_df=train_df,
            synthetic_df=viz_df,
            output_dir=layout["visualizations"],
            seed=seed,
        )

    report = GenerationReport(
        mode=mode,
        generated_rows=int(generated),
        accepted_rows=int(accepted_rows),
        repaired_values=int(repaired_total),
        hard_rejects=int(hard_rejects),
        soft_score_mean=float(np.mean(running_soft_scores)) if running_soft_scores else 0.0,
    )

    report_payload = {
        **asdict(report),
        "requested_rows": int(n_rows),
        "output_csv_path": output_csv_path,
        "output_parquet_path": output_parquet_path if parquet_requested else None,
        "parquet_written": bool(parquet_written),
        "parquet_skip_reason": parquet_skip_reason,
        "cache_path": cache_path,
        "sample_preview_path": sample_path,
        "visualizations": viz_paths,
        "chunk_rows": int(chunk_rows),
    }
    report_path = os.path.join(layout["reports"], "dataset_build_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dataset-first synthetic data builder for credit-risk pipelines."
    )
    parser.add_argument(
        "--reference-csv",
        default="comprehensive_loan_data.csv",
        help="Reference CSV used to train synthetic generator (defaults to comprehensive_loan_data.csv).",
    )
    parser.add_argument("--rows", type=int, default=5_000_000, help="Number of synthetic rows to generate.")
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "gan", "sampler"],
        help="Generation mode (auto: gan first, sampler fallback).",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Synthetic agent training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Synthetic agent training batch size.")
    parser.add_argument(
        "--soft-tolerance",
        type=float,
        default=0.25,
        help="Soft-constraint tolerance for row acceptance (0-1).",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path (default: results/synthetic/<run-name>/synthetic_dataset_<rows>.csv).",
    )
    parser.add_argument(
        "--output-parquet",
        default=None,
        help="Optional parquet path (default: results/synthetic/<run-name>/synthetic_dataset_<rows>.parquet).",
    )
    parser.add_argument("--chunk-rows", type=int, default=250_000, help="Rows written per chunk to CSV.")
    parser.add_argument("--run-name", default="dataset_5m", help="Output run folder name under results/synthetic.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--skip-visualizations",
        action="store_true",
        help="Skip distribution visualization export during dataset generation.",
    )
    args = parser.parse_args()

    reference_path = _resolve_reference_csv_path(args.reference_csv)

    output_csv = args.output_csv or os.path.join(
        "results", "synthetic", args.run_name, f"synthetic_dataset_{args.rows}.csv"
    )
    output_parquet = args.output_parquet or os.path.join(
        "results", "synthetic", args.run_name, f"synthetic_dataset_{args.rows}.parquet"
    )

    if reference_path:
        print(f"[DATASET] Loading reference data from: {reference_path}")
        reference_df = pd.read_csv(reference_path)
    else:
        print(
            "[DATASET] No seed CSV found; building fresh synthetic reference frame "
            "for standalone dataset generation."
        )
        reference_df = _build_fresh_reference_frame(seed=args.seed)

    report = build_synthetic_dataset(
        reference_df=reference_df,
        output_csv_path=output_csv,
        output_parquet_path=output_parquet,
        n_rows=args.rows,
        mode=args.mode,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        soft_tolerance=args.soft_tolerance,
        cache_path=f"models/synthetic/checkpoints/tabular_agent_{args.rows}.pt",
        output_root="results/synthetic",
        run_name=args.run_name,
        chunk_rows=args.chunk_rows,
        save_visualizations=not args.skip_visualizations,
    )
    print(f"[DATASET] Build complete: {asdict(report)}")
    print(f"[DATASET] CSV -> {output_csv}")
    print(f"[DATASET] Parquet -> {output_parquet}")
    report_path = os.path.join("results", "synthetic", args.run_name, "reports", "dataset_build_report.json")
    print(f"[DATASET] Report -> {report_path}")


if __name__ == "__main__":
    main()
