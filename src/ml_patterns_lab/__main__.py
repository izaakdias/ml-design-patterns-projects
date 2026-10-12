"""Explicit local commands; optional HTTP dependencies load only when serving."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

from ml_patterns_lab import PROJECTS, __version__
from ml_patterns_lab.shared import artifact_root, promote_bundle


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m ml_patterns_lab")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--artifacts-dir", type=Path, help="Override MLDP_ARTIFACTS_DIR.")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train and register one or all projects.")
    train.add_argument("project", choices=(*PROJECTS, "all"))
    serve = commands.add_parser("serve", help="Serve a registered model.")
    serve.add_argument("project", choices=PROJECTS)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--model-version", help="Pin a registered run_id for this process.")
    monitor = commands.add_parser("monitor")
    monitor.add_argument("project", choices=("demand", "sensors"))
    listing = commands.add_parser("list", help="List registered metadata and current version.")
    listing.add_argument("project", choices=PROJECTS)
    promote = commands.add_parser("promote", help="Promote or roll back a verified version.")
    promote.add_argument("project", choices=PROJECTS)
    promote.add_argument("run_id")
    args = parser.parse_args(argv)
    if args.artifacts_dir:
        os.environ["MLDP_ARTIFACTS_DIR"] = str(args.artifacts_dir)

    if args.command == "train":
        projects = PROJECTS if args.project == "all" else (args.project,)
        for project in projects:
            module, function = (
                ("mlops.pipeline", "run_pipeline")
                if project == "mlops"
                else (f"{project}.train", "train")
            )
            path, metadata = getattr(
                importlib.import_module(f"ml_patterns_lab.{module}"), function
            )()
            print(
                json.dumps(
                    {"artifact": str(path), "metadata": metadata}, indent=2, ensure_ascii=False
                )
            )
    elif args.command == "serve":
        if not 1 <= args.port <= 65535:
            parser.error("--port must be between 1 and 65535.")
        if args.model_version:
            os.environ["MLDP_MODEL_VERSION"] = args.model_version
        try:
            import uvicorn
        except ImportError:
            parser.error('Serving requires installation with the "api" extra.')
        uvicorn.run(f"ml_patterns_lab.{args.project}.api:app", host=args.host, port=args.port)
    elif args.command == "monitor":
        importlib.import_module(f"ml_patterns_lab.{args.project}.monitor").main()
    elif args.command == "promote":
        promote_bundle(args.project, args.run_id)
        print(json.dumps({"project": args.project, "run_id": args.run_id}))
    else:
        root = artifact_root() / args.project
        latest = (
            json.loads((root / "latest.json").read_text())["run_id"]
            if (root / "latest.json").exists()
            else None
        )
        versions = [
            {**json.loads(path.read_text()), "current": path.parent.name == latest}
            for path in sorted(root.glob("*/metadata.json"))
        ]
        print(json.dumps(versions, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
