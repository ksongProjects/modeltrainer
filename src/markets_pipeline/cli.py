from __future__ import annotations

import argparse
import json
from typing import Any

from .pipeline import build_snapshots, evaluate_model, promote_model, register_findf_run, train_experts, train_fusion
from .settings import Settings


def _print(payload: Any) -> None:
    if isinstance(payload, str):
        print(payload)
        return
    print(json.dumps(payload, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="markets-trainer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_cmd = subparsers.add_parser("import-findf-run")
    import_cmd.add_argument("--job-id", required=True)

    snapshot_cmd = subparsers.add_parser("build-snapshots")
    snapshot_cmd.add_argument("--job-id", required=True)

    experts_cmd = subparsers.add_parser("train-experts")
    experts_cmd.add_argument("--snapshot-version", required=True)
    experts_cmd.add_argument("--horizon", choices=["1d", "5d", "10d"], required=True)

    fusion_cmd = subparsers.add_parser("train-fusion")
    fusion_cmd.add_argument("--snapshot-version", required=True)
    fusion_cmd.add_argument("--horizon", choices=["1d", "5d", "10d"], required=True)

    eval_cmd = subparsers.add_parser("evaluate-model")
    eval_cmd.add_argument("--model-version", required=True)

    promote_cmd = subparsers.add_parser("promote-model")
    promote_cmd.add_argument("--model-version", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = Settings.load()

    if args.command == "import-findf-run":
        _print({"manifest_path": register_findf_run(settings, args.job_id)})
        return
    if args.command == "build-snapshots":
        result = build_snapshots(settings, args.job_id)
        _print(
            {
                "snapshot_version": result.snapshot_version,
                "snapshot_path": str(result.snapshot_path),
                "metadata_path": str(result.metadata_path),
            }
        )
        return
    if args.command == "train-experts":
        _print(train_experts(settings, args.snapshot_version, args.horizon))
        return
    if args.command == "train-fusion":
        _print({"model_version": train_fusion(settings, args.snapshot_version, args.horizon)})
        return
    if args.command == "evaluate-model":
        _print({"report_path": evaluate_model(settings, args.model_version)})
        return
    if args.command == "promote-model":
        _print(promote_model(settings, args.model_version))
        return


if __name__ == "__main__":
    main()
