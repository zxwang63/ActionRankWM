#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from actionrankwm.bundle import export_model_bundle


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a label-blind model-input bundle from a passed ToolSandbox preflight"
    )
    parser.add_argument("--preflight-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = export_model_bundle(args.preflight_dir, args.config, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "requests": manifest["counts"]["requests"],
                "input_sha256": manifest["artifacts"]["model_inputs.jsonl"]["sha256"],
                "isolation": manifest["isolation"],
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
