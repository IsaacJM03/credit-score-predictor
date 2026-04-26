from __future__ import annotations

import argparse
from pathlib import Path

from main import run
from pipeline_utils import DEFAULT_DATASET_PATH


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run training for all topic-specific model pipelines.")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output-root", default="models")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    run(dataset_path=Path(args.dataset_path), output_root=Path(args.output_root), seed=args.seed)


if __name__ == "__main__":
    main()