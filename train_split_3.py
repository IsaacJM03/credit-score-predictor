from __future__ import annotations

import argparse
from pathlib import Path

from borrower_segmentation import run as run_segmentation
from fraud_detection import run as run_fraud
from post_loan_monitoring import run as run_post_loan, run_lstm as run_post_loan_lstm
from pipeline_utils import DEFAULT_DATASET_PATH, log_message, save_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Split 3 training (6 models): remaining monitoring, anomaly, and clustering models.")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output-root", default="models/splits/split_3")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset_path)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    log_message("split_3", "Starting split 3 training")
    result = {
        "post_loan_monitoring": run_post_loan(
            dataset_path=dataset_path,
            output_root=output_root / "post_loan_monitoring",
            seed=args.seed,
            selected_models=["weighted_soft_voting"],
        ),
        "fraud_detection": run_fraud(
            dataset_path=dataset_path,
            output_root=output_root / "fraud_detection",
            seed=args.seed,
            selected_models=["one_class_svm", "autoencoder"],
        ),
        "borrower_segmentation": run_segmentation(
            dataset_path=dataset_path,
            output_root=output_root / "borrower_segmentation",
            seed=args.seed,
            selected_models=["dbscan", "agglomerative"],
        ),
    }

    log_message("split_3", "Running LSTM training last")
    result["post_loan_monitoring_lstm"] = run_post_loan_lstm(
        dataset_path=dataset_path,
        output_root=output_root / "post_loan_monitoring",
        seed=args.seed,
    )

    manifest_path = output_root / "split_3_manifest.json"
    save_json({"split": "split_3", "dataset_path": str(dataset_path), "topics": result}, manifest_path)
    log_message("split_3", f"Completed split 3 training | manifest={manifest_path}")


if __name__ == "__main__":
    main()