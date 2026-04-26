from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from pipeline_utils import log_message, save_json


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Combine split training manifests into one merged manifest.")
    parser.add_argument("--split-1", default="models/splits/split_1/split_1_manifest.json")
    parser.add_argument("--split-2", default="models/splits/split_2/split_2_manifest.json")
    parser.add_argument("--split-3", default="models/splits/split_3/split_3_manifest.json")
    parser.add_argument("--output", default="models/splits/combined_manifest.json")
    args = parser.parse_args(argv)

    split_paths = [Path(args.split_1), Path(args.split_2), Path(args.split_3)]
    for path in split_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing split manifest: {path}")

    loaded = [_load_json(path) for path in split_paths]
    dataset_path = loaded[0].get("dataset_path", "")
    merged_topics: Dict[str, Dict] = {}
    split_sources: Dict[str, str] = {}

    for payload in loaded:
        split_name = payload.get("split", "unknown")
        topics = payload.get("topics", {})
        for topic_name, topic_result in topics.items():
            existing = merged_topics.setdefault(topic_name, {})
            existing_metrics = existing.setdefault("metrics", {})
            existing_artifacts = existing.setdefault("artifacts", [])
            existing_visualizations = existing.setdefault("visualizations", {})

            existing_metrics.update(topic_result.get("metrics", {}))
            existing_artifacts.extend(topic_result.get("artifacts", []))
            existing_visualizations.update(topic_result.get("visualizations", {}))
            if "features" in topic_result and "features" not in existing:
                existing["features"] = topic_result["features"]
            if "feature_count" in topic_result and "feature_count" not in existing:
                existing["feature_count"] = topic_result["feature_count"]
            if "visualization_dir" in topic_result:
                existing.setdefault("visualization_dirs", []).append(topic_result["visualization_dir"])
            split_sources.setdefault(topic_name, split_name)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_path": dataset_path,
        "split_manifests": [str(path) for path in split_paths],
        "topics": merged_topics,
        "topic_split_source": split_sources,
    }
    save_json(payload, output_path)
    log_message("combine", f"Combined manifest written to {output_path}")


if __name__ == "__main__":
    main()