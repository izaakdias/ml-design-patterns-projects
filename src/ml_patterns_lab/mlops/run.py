"""Command-line entry point for the repeatable maintenance-risk workflow."""

from __future__ import annotations

import json

from ml_patterns_lab.mlops.pipeline import run_pipeline


def main() -> None:
    version_path, metadata = run_pipeline()
    print(json.dumps({"artifact": str(version_path), "metadata": metadata}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
