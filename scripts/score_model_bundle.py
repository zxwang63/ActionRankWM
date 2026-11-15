#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from actionrankwm.bundle_scoring import score_model_bundle


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Blind-score a label-isolated ActionRankWM prediction bundle"
    )
    parser.add_argument("--preflight-dir", required=True, type=Path)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--prediction-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = score_model_bundle(
        args.preflight_dir,
        args.bundle_dir,
        args.prediction_dir,
        args.config,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "provider": result["manifest"]["provider"],
                "counts": result["manifest"]["counts"],
                "diagnostic": result["diagnostic"],
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
