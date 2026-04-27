from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional
import time

from borrower_segmentation import run as run_segmentation
from classification import run as run_classification
from fraud_detection import run as run_fraud
from post_loan_monitoring import run as run_monitoring, run_lstm as run_post_loan_lstm
from survival_analysis import run as run_survival

from pipeline_utils import DEFAULT_DATASET_PATH, elapsed_seconds, log_message, save_json

SEED = 42


def run(dataset_path: Path | str = DEFAULT_DATASET_PATH, output_root: Path | str = Path("models"), seed: int = SEED):
    suite_start = time.perf_counter()
    dataset_path = Path(dataset_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_message("suite", f"Starting full training suite | dataset={dataset_path} | output_root={output_root}")

    summary: Dict[str, Dict] = {}

    step_start = time.perf_counter()
    log_message("suite", "Running classification training")
    summary["classification"] = run_classification(dataset_path=dataset_path, output_root=output_root / "classification", seed=seed)
    log_message("suite", f"Completed classification in {elapsed_seconds(step_start):.1f}s")

    step_start = time.perf_counter()
    log_message("suite", "Running post-loan monitoring training")
    summary["post_loan_monitoring"] = run_monitoring(
        dataset_path=dataset_path,
        output_root=output_root / "post_loan_monitoring",
        seed=seed,
        selected_models=["random_forest", "xgboost", "lightgbm", "weighted_soft_voting"],
    )
    log_message("suite", f"Completed post-loan monitoring in {elapsed_seconds(step_start):.1f}s")

    step_start = time.perf_counter()
    log_message("suite", "Running survival analysis training")
    summary["survival_analysis"] = run_survival(dataset_path=dataset_path, output_root=output_root / "survival_analysis", seed=seed)
    log_message("suite", f"Completed survival analysis in {elapsed_seconds(step_start):.1f}s")

    step_start = time.perf_counter()
    log_message("suite", "Running fraud detection training")
    summary["fraud_detection"] = run_fraud(dataset_path=dataset_path, output_root=output_root / "fraud_detection", seed=seed)
    log_message("suite", f"Completed fraud detection in {elapsed_seconds(step_start):.1f}s")

    step_start = time.perf_counter()
    log_message("suite", "Running borrower segmentation training")
    summary["borrower_segmentation"] = run_segmentation(dataset_path=dataset_path, output_root=output_root / "borrower_segmentation", seed=seed)
    log_message("suite", f"Completed borrower segmentation in {elapsed_seconds(step_start):.1f}s")

    step_start = time.perf_counter()
    log_message("suite", "Running LSTM training last")
    summary["post_loan_monitoring_lstm"] = run_post_loan_lstm(
        dataset_path=dataset_path,
        output_root=output_root / "post_loan_monitoring",
        seed=seed,
    )
    log_message("suite", f"Completed LSTM training in {elapsed_seconds(step_start):.1f}s")

    manifest_path = output_root / "suite_manifest.json"
    save_json({"dataset_path": str(dataset_path), "topics": summary}, manifest_path)
    log_message("suite", f"Training suite completed in {elapsed_seconds(suite_start):.1f}s | manifest={manifest_path}")
    return {"manifest_path": str(manifest_path), "topics": summary}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the full 5M model suite.")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output-root", default="models")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    run(dataset_path=args.dataset_path, output_root=args.output_root, seed=args.seed)


if __name__ == "__main__":
    main()