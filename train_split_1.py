from __future__ import annotations

import argparse
from pathlib import Path

from borrower_segmentation import run as run_segmentation
from classification import run as run_classification
from survival_analysis import run as run_survival
from pipeline_utils import DEFAULT_DATASET_PATH, log_message, save_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Split 1 training (6 models): classification core + survival + kmeans.")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output-root", default="models/splits/split_1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset_path)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    log_message("split_1", "Starting split 1 training")
    result = {
        "classification": run_classification(
            dataset_path=dataset_path,
            output_root=output_root / "classification",
            seed=args.seed,
            selected_models=["logistic_regression", "random_forest", "xgboost"],
        ),
        "survival_analysis": run_survival(
            dataset_path=dataset_path,
            output_root=output_root / "survival_analysis",
            seed=args.seed,
            selected_models=["kaplan_meier", "cox_ph"],
        ),
        "borrower_segmentation": run_segmentation(
            dataset_path=dataset_path,
            output_root=output_root / "borrower_segmentation",
            seed=args.seed,
            selected_models=["kmeans"],
        ),
    }

    manifest_path = output_root / "split_1_manifest.json"
    save_json({"split": "split_1", "dataset_path": str(dataset_path), "topics": result}, manifest_path)
    log_message("split_1", f"Completed split 1 training | manifest={manifest_path}")


if __name__ == "__main__":
    main()