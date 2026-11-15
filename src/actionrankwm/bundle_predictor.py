from __future__ import annotations

import json
import os
import platform
import socket
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Protocol

from actionrankwm import __version__
from actionrankwm.bundle import BUNDLE_SCHEMA, utc_now
from actionrankwm.io import (
    atomic_write_json,
    canonical_json,
    digest_file,
    digest_json,
    extract_first_json_object,
    iter_jsonl,
    read_json,
)


PREDICTION_SCHEMA = "actionrankwm.label_blind_predictions.v1"


class _Transport(Protocol):
    name: str
    model: str

    def complete(self, messages: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        ...


class _OpenAITransport:
    name = "openai"

    def __init__(self, model: str, config: Mapping[str, Any]) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The openai package is required for --provider openai") from exc
        self.model = model
        self._client = OpenAI(
            timeout=float(config.get("timeout_seconds", 180)),
            max_retries=int(config.get("max_retries", 1)),
        )
        self._temperature = config.get("temperature")
        self._max_tokens = int(config.get("max_tokens", 1536))

    def complete(self, messages: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self._max_tokens,
        }
        if self._temperature is not None:
            request["temperature"] = float(self._temperature)
        response = self._client.chat.completions.create(**request)
        raw = response.choices[0].message.content or ""
        usage: dict[str, Any] = {}
        if response.usage is not None:
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                "completion_tokens": getattr(response.usage, "completion_tokens", None),
                "total_tokens": getattr(response.usage, "total_tokens", None),
            }
        for output_key, attribute in (
            ("response_model", "model"),
            ("system_fingerprint", "system_fingerprint"),
            ("service_tier", "service_tier"),
        ):
            value = getattr(response, attribute, None)
            if value is not None:
                usage[output_key] = str(value)
        return raw, usage


class _CompatibleHTTPTransport:
    name = "openai-compatible-http"

    def __init__(self, model: str, config: Mapping[str, Any]) -> None:
        self.model = model
        base_url = os.environ.get("WORLD_MODEL_BASE_URL", "http://127.0.0.1:8000/v1")
        self._endpoint = base_url.rstrip("/") + "/chat/completions"
        self._api_key = os.environ.get("WORLD_MODEL_API_KEY", "local")
        self._timeout = float(config.get("timeout_seconds", 180))
        self._max_retries = int(config.get("max_retries", 1))
        self._temperature = config.get("temperature")
        self._max_tokens = int(config.get("max_tokens", 1536))

    def complete(self, messages: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self._max_tokens,
        }
        if self._temperature is not None:
            payload["temperature"] = float(self._temperature)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                request = urllib.request.Request(
                    self._endpoint, data=body, headers=headers, method="POST"
                )
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(request, timeout=self._timeout) as response:
                    decoded = json.loads(response.read().decode("utf-8"))
                raw = str(decoded["choices"][0]["message"]["content"] or "")
                usage_value = decoded.get("usage", {})
                usage = dict(usage_value) if isinstance(usage_value, Mapping) else {}
                for output_key, response_key in (
                    ("response_model", "model"),
                    ("system_fingerprint", "system_fingerprint"),
                    ("service_tier", "service_tier"),
                ):
                    value = decoded.get(response_key)
                    if value is not None:
                        usage[output_key] = str(value)
                return raw, usage
            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                KeyError,
                IndexError,
                TypeError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc
                if attempt < self._max_retries:
                    time.sleep(min(2.0, 0.25 * (2**attempt)))
        assert last_error is not None
        raise last_error


def _create_transport(provider: str, config: Mapping[str, Any]) -> _Transport:
    if provider == "openai":
        model = os.environ.get("OPENAI_MODEL", "").strip()
        if not model:
            raise RuntimeError("OPENAI_MODEL is not set")
        return _OpenAITransport(model, config)
    if provider == "compatible":
        model = os.environ.get("WORLD_MODEL_MODEL", "").strip()
        if not model:
            raise RuntimeError("WORLD_MODEL_MODEL is not set")
        return _CompatibleHTTPTransport(model, config)
    raise ValueError(f"Unsupported prediction provider: {provider}")


def _load_verified_inputs(bundle_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(bundle_dir / "manifest.json")
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise ValueError(f"Unsupported model-input bundle schema: {manifest.get('schema')}")
    artifacts = manifest.get("artifacts", {})
    for name in ("model_inputs.jsonl", "provider_config.json"):
        path = bundle_dir / name
        expected = str(artifacts[name]["sha256"])
        actual = digest_file(path)
        if actual != expected:
            raise ValueError(f"Bundle artifact hash mismatch for {name}: {actual} != {expected}")
    records = list(iter_jsonl(bundle_dir / "model_inputs.jsonl"))
    if len(records) != int(manifest["counts"]["requests"]):
        raise ValueError("Model-input record count does not match bundle manifest")
    seen: set[str] = set()
    for index, record in enumerate(records):
        if record.get("schema") != BUNDLE_SCHEMA:
            raise ValueError(f"Bad model-input schema at row {index}")
        join_id = str(record.get("join_id", ""))
        if len(join_id) != 64 or join_id in seen:
            raise ValueError(f"Invalid or duplicate opaque join ID at row {index}")
        seen.add(join_id)
        messages = record.get("messages")
        if not isinstance(messages, list) or digest_json(messages) != record.get("prompt_sha256"):
            raise ValueError(f"Prompt hash mismatch at row {index}")
    return manifest, records


def _predict_one(
    record: Mapping[str, Any], transport: _Transport
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        raw, usage = transport.complete(list(record["messages"]))
        latency = time.monotonic() - started
        try:
            parsed = extract_first_json_object(raw)
            error = None
        except Exception as exc:
            parsed = None
            error = f"parse:{type(exc).__name__}: {exc}"
        return {
            "schema": PREDICTION_SCHEMA,
            "join_id": record["join_id"],
            "prompt_sha256": record["prompt_sha256"],
            "provider": transport.name,
            "model": transport.model,
            "raw_text": raw,
            "parsed_observation": parsed,
            "parse_ok": parsed is not None and error is None,
            "latency_seconds": latency,
            "usage": usage,
            "error": error,
        }
    except Exception as exc:
        return {
            "schema": PREDICTION_SCHEMA,
            "join_id": record["join_id"],
            "prompt_sha256": record["prompt_sha256"],
            "provider": transport.name,
            "model": transport.model,
            "raw_text": "",
            "parsed_observation": None,
            "parse_ok": False,
            "latency_seconds": time.monotonic() - started,
            "usage": {},
            "error": f"{type(exc).__name__}: {exc}",
        }


def _resume_prefix(path: Path, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    completed = list(iter_jsonl(path))
    if len(completed) > len(inputs):
        raise ValueError("Partial prediction journal is longer than the input bundle")
    for index, prediction in enumerate(completed):
        if prediction.get("schema") != PREDICTION_SCHEMA:
            raise ValueError(f"Bad partial prediction schema at row {index}")
        if prediction.get("join_id") != inputs[index]["join_id"]:
            raise ValueError(f"Partial prediction journal is not an input prefix at row {index}")
        if prediction.get("prompt_sha256") != inputs[index]["prompt_sha256"]:
            raise ValueError(f"Partial prediction prompt hash mismatch at row {index}")
    return completed


def predict_model_bundle(
    bundle_dir: str | Path,
    output_dir: str | Path,
    provider: str,
    resume: bool = False,
) -> dict[str, Any]:
    bundle_dir = Path(bundle_dir).resolve()
    output_dir = Path(output_dir).resolve()
    bundle_manifest, inputs = _load_verified_inputs(bundle_dir)
    provider_config = read_json(bundle_dir / "provider_config.json")
    transport = _create_transport(provider, provider_config)
    run_identity = {
        "schema": PREDICTION_SCHEMA,
        "bundle_input_sha256": digest_file(bundle_dir / "model_inputs.jsonl"),
        "provider_config_sha256": digest_file(bundle_dir / "provider_config.json"),
        "provider": transport.name,
        "model": transport.model,
    }

    partial_path = output_dir / "predictions.partial.jsonl"
    final_path = output_dir / "predictions.jsonl"
    state_path = output_dir / "run_state.json"
    if output_dir.exists():
        if not resume:
            raise FileExistsError(f"Refusing to overwrite prediction run: {output_dir}")
        if final_path.exists():
            raise FileExistsError(f"Prediction run is already complete: {output_dir}")
        if not state_path.is_file() or read_json(state_path) != run_identity:
            raise ValueError("Resume identity does not match the existing prediction run")
    else:
        output_dir.mkdir(parents=True)
        atomic_write_json(state_path, run_identity)

    completed = _resume_prefix(partial_path, inputs)
    remaining = inputs[len(completed) :]
    workers = int(provider_config.get("concurrency", 1))
    if workers <= 0:
        raise ValueError("provider concurrency must be positive")
    started_at = utc_now()
    mode = "a" if partial_path.exists() else "x"
    with partial_path.open(mode, encoding="utf-8") as handle:
        def journal(prediction: dict[str, Any]) -> None:
            handle.write(canonical_json(prediction))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        if workers == 1:
            for record in remaining:
                journal(_predict_one(record, transport))
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for prediction in executor.map(
                    lambda record: _predict_one(record, transport), remaining
                ):
                    journal(prediction)

    predictions = _resume_prefix(partial_path, inputs)
    if len(predictions) != len(inputs):
        raise RuntimeError(f"Incomplete prediction journal: {len(predictions)} != {len(inputs)}")
    os.replace(partial_path, final_path)
    errors = sum(not bool(item.get("parse_ok")) for item in predictions)
    finished_at = utc_now()
    runtime: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
    }
    if os.environ.get("SLURM_JOB_ID"):
        runtime["scheduler"] = {
            "kind": "slurm",
            "job_id": os.environ["SLURM_JOB_ID"],
            "job_name": os.environ.get("SLURM_JOB_NAME"),
        }
    model_provenance = {
        key: os.environ[value]
        for key, value in (
            ("revision", "WORLD_MODEL_REVISION"),
            ("config_sha256", "WORLD_MODEL_CONFIG_SHA256"),
        )
        if os.environ.get(value)
    }
    manifest = {
        **run_identity,
        "package_version": __version__,
        "run_status": "complete",
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "resumed_completed_count": len(completed),
        "counts": {
            "requests": len(inputs),
            "predictions": len(predictions),
            "parse_or_provider_failures": errors,
        },
        "provider": {
            "name": transport.name,
            "model": transport.model,
            "credentials_recorded": False,
            "real_branch_exposure": False,
            "concurrency": workers,
            "model_provenance": model_provenance,
        },
        "isolation": bundle_manifest["isolation"],
        "runtime": runtime,
        "artifacts": {
            "predictions.jsonl": {
                "sha256": digest_file(final_path),
                "bytes": final_path.stat().st_size,
            },
            "run_state.json": {
                "sha256": digest_file(state_path),
                "bytes": state_path.stat().st_size,
            },
        },
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest
