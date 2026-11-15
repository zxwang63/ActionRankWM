#!/usr/bin/env bash
set -euo pipefail

SPLIT="${SPLIT:-e4b}"
CONFIG="${CONFIG:-configs/toolsandbox_external_${SPLIT}_direct.json}"
OUT="${OUT:-runs/${SERVED_NAME:-world-model}_${SPLIT}}"
TOOLSANDBOX_DIR="${TOOLSANDBOX_DIR:-third_party/ToolSandbox}"
PORT="${PORT:-8000}"
SERVED_NAME="${SERVED_NAME:-world-model}"

mkdir -p "$(dirname "$OUT")"
python scripts/preflight_toolsandbox.py --split "$SPLIT" --source-root "$TOOLSANDBOX_DIR" --run-dir "$OUT/preflight"
python scripts/export_model_bundle.py --preflight-dir "$OUT/preflight" --config "$CONFIG" --output-dir "$OUT/bundle"

SERVER_PID=""
if [[ -z "${WORLD_MODEL_BASE_URL:-}" ]]; then
  : "${MODEL_PATH:?set MODEL_PATH (local checkpoint) or WORLD_MODEL_BASE_URL (running endpoint)}"
  python scripts/serve_transformers.py --model "$MODEL_PATH" --served-model-name "$SERVED_NAME" \
      --port "$PORT" --max-output-tokens 1536 ${ADAPTER_DIR:+--adapter-dir "$ADAPTER_DIR"} &
  SERVER_PID=$!
  trap '[[ -n "$SERVER_PID" ]] && kill "$SERVER_PID"' EXIT
  export WORLD_MODEL_BASE_URL="http://127.0.0.1:${PORT}/v1"
  until curl -sf "$WORLD_MODEL_BASE_URL/models" >/dev/null; do sleep 10; done
fi
export WORLD_MODEL_MODEL="$SERVED_NAME"

python scripts/predict_model_bundle.py --bundle-dir "$OUT/bundle" --output-dir "$OUT/predictions" --provider compatible
python scripts/score_model_bundle.py --preflight-dir "$OUT/preflight" --bundle-dir "$OUT/bundle" \
    --prediction-dir "$OUT/predictions" --config "$CONFIG" --output-dir "$OUT/scored"
echo "Scored run written to $OUT/scored (scored_candidates.jsonl, task_decisions.jsonl, metrics.json)"
