#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from actionrankwm.bundle_predictor import predict_model_bundle


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Predict a label-blind ActionRankWM model-input bundle"
    )
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--provider", required=True, choices=("openai", "compatible"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest = predict_model_bundle(
        args.bundle_dir, args.output_dir, args.provider, resume=args.resume
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "provider": manifest["provider"],
                "counts": manifest["counts"],
                "prediction_sha256": manifest["artifacts"]["predictions.jsonl"]["sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
