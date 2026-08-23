#!/usr/bin/env python3
"""Command-line entry point for the prospective HLCA external validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hlca_pipeline import (
    create_mapping_template,
    download_hlca,
    load_config,
    pipeline_status,
    prepare_markers,
    regenerate_figures,
    run_gemini,
    score_results,
)


DEFAULT_CONFIG = Path(__file__).with_name("config.toml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HLCA core external validation for LLM-scCurator",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="download the pinned HLCA core H5AD")
    download.add_argument("--force", action="store_true")
    download.add_argument("--skip-sha256", action="store_true")

    prepare = sub.add_parser("prepare", help="subsample cells and generate marker lists")
    prepare.add_argument("--overwrite", action="store_true")

    infer = sub.add_parser("infer", help="run repeated Gemini annotation calls")
    infer.add_argument("--dry-run", action="store_true")
    infer.add_argument("--limit", type=int, default=None)
    infer.add_argument("--overwrite", action="store_true")

    mapping = sub.add_parser("mapping", help="create the blinded prediction mapping table")
    mapping.add_argument("--overwrite", action="store_true")

    sub.add_parser("score", help="score reviewed predictions and make figures")
    figures = sub.add_parser(
        "figures",
        help="regenerate figures from frozen scores without inference or remapping",
    )
    figures.add_argument(
        "--skip-umap",
        action="store_true",
        help="regenerate only Fig. 3d and Fig. S6",
    )
    sub.add_parser("status", help="show completion status")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    if args.command == "download":
        path = download_hlca(cfg, force=args.force, compute_sha256=not args.skip_sha256)
        print(path)
    elif args.command == "prepare":
        manifest, markers = prepare_markers(cfg, overwrite=args.overwrite)
        print(manifest)
        print(markers)
    elif args.command == "infer":
        count = run_gemini(
            cfg,
            dry_run=args.dry_run,
            limit=args.limit,
            overwrite=args.overwrite,
        )
        print(f"Completed or reused {count} calls")
    elif args.command == "mapping":
        print(create_mapping_template(cfg, overwrite=args.overwrite))
    elif args.command == "score":
        for path in score_results(cfg):
            print(path)
    elif args.command == "figures":
        for path in regenerate_figures(cfg, include_scope=not args.skip_umap):
            print(path)
    elif args.command == "status":
        print(json.dumps(pipeline_status(cfg), indent=2))


if __name__ == "__main__":
    main()
