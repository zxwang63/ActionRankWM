from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from actionrankwm.io import extract_first_json_object
from actionrankwm.providers.base import ProviderResult, WorldModelProvider
from actionrankwm.providers.prompts import build_messages
from actionrankwm.types import Action, BranchRecord, TaskSpec


class OpenAICompatibleProvider(WorldModelProvider):
    def __init__(
        self,
        model: str,
        config: dict[str, Any],
        base_url: str | None = None,
        api_key: str | None = None,
        name: str = "openai",
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The optional `openai` package is required for this provider") from exc
        client_arguments: dict[str, Any] = {
            "timeout": float(config.get("timeout_seconds", 180)),
            "max_retries": int(config.get("max_retries", 2)),
        }
        if base_url:
            client_arguments["base_url"] = base_url
        if api_key:
            client_arguments["api_key"] = api_key
        self._client = OpenAI(**client_arguments)
        self.model = model
        self.name = name
        self._temperature = config.get("temperature")
        self._max_tokens = int(config.get("max_tokens", 1600))
        self.prompt_style = str(config.get("prompt_style", "direct"))
        if self.prompt_style not in {"direct", "rule_trace"}:
            raise ValueError(f"Unsupported prompt style: {self.prompt_style}")

    @classmethod
    def from_environment(
        cls, config: dict[str, Any], local: bool = False
    ) -> "OpenAICompatibleProvider":
        if local:
            raise ValueError("Use LocalHTTPProvider for a local compatible endpoint")
        model = os.environ.get("OPENAI_MODEL")
        if not model:
            raise RuntimeError("OPENAI_MODEL is not set")
        return cls(model=model, config=config, name="openai")

    def predict(
        self,
        task: TaskSpec,
        action: Action,
        real_branch: BranchRecord | None = None,
    ) -> ProviderResult:
        del real_branch
        request: dict[str, Any] = {
            "model": self.model,
            "messages": build_messages(task, action, self.prompt_style),
            "max_tokens": self._max_tokens,
        }
        if self._temperature is not None:
            request["temperature"] = float(self._temperature)
        started = time.monotonic()
        try:
            response = self._client.chat.completions.create(**request)
            latency = time.monotonic() - started
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
            try:
                parsed = extract_first_json_object(raw)
            except Exception as exc:
                return ProviderResult(
                    raw_text=raw,
                    parsed_observation=None,
                    latency_seconds=latency,
                    usage=usage,
                    error=f"parse:{type(exc).__name__}: {exc}",
                )
            return ProviderResult(
                raw_text=raw,
                parsed_observation=parsed,
                latency_seconds=latency,
                usage=usage,
            )
        except Exception as exc:
            return ProviderResult(
                raw_text="",
                parsed_observation=None,
                latency_seconds=time.monotonic() - started,
                error=f"{type(exc).__name__}: {exc}",
            )


class LocalHTTPProvider(WorldModelProvider):
    """Dependency-free client for an on-node OpenAI-compatible server."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        config: dict[str, Any],
    ) -> None:
        self.model = model
        self.name = "openai-compatible-http"
        self._endpoint = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self._timeout = float(config.get("timeout_seconds", 180))
        self._max_retries = int(config.get("max_retries", 2))
        self._temperature = config.get("temperature")
        self._max_tokens = int(config.get("max_tokens", 1600))
        self.prompt_style = str(config.get("prompt_style", "direct"))
        if self.prompt_style not in {"direct", "rule_trace"}:
            raise ValueError(f"Unsupported prompt style: {self.prompt_style}")
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @classmethod
    def from_environment(cls, config: dict[str, Any]) -> "LocalHTTPProvider":
        return cls(
            model=os.environ.get("WORLD_MODEL_MODEL", "Qwen/Qwen-AgentWorld-35B-A3B"),
            base_url=os.environ.get("WORLD_MODEL_BASE_URL", "http://127.0.0.1:8000/v1"),
            api_key=os.environ.get("WORLD_MODEL_API_KEY", "local"),
            config=config,
        )

    def predict(
        self,
        task: TaskSpec,
        action: Action,
        real_branch: BranchRecord | None = None,
    ) -> ProviderResult:
        del real_branch
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": build_messages(task, action, self.prompt_style),
            "max_tokens": self._max_tokens,
        }
        if self._temperature is not None:
            payload["temperature"] = float(self._temperature)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        started = time.monotonic()
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                request = urllib.request.Request(
                    self._endpoint,
                    data=body,
                    headers=headers,
                    method="POST",
                )
                with self._opener.open(request, timeout=self._timeout) as response:
                    decoded = json.loads(response.read().decode("utf-8"))
                raw = str(decoded["choices"][0]["message"]["content"] or "")
                usage_value = decoded.get("usage", {})
                usage = dict(usage_value) if isinstance(usage_value, dict) else {}
                for output_key, response_key in (
                    ("response_model", "model"),
                    ("system_fingerprint", "system_fingerprint"),
                    ("service_tier", "service_tier"),
                ):
                    value = decoded.get(response_key)
                    if value is not None:
                        usage[output_key] = str(value)
                try:
                    parsed = extract_first_json_object(raw)
                except Exception as exc:
                    return ProviderResult(
                        raw_text=raw,
                        parsed_observation=None,
                        latency_seconds=time.monotonic() - started,
                        usage=usage,
                        error=f"parse:{type(exc).__name__}: {exc}",
                    )
                return ProviderResult(
                    raw_text=raw,
                    parsed_observation=parsed,
                    latency_seconds=time.monotonic() - started,
                    usage=usage,
                )
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
        return ProviderResult(
            raw_text="",
            parsed_observation=None,
            latency_seconds=time.monotonic() - started,
            error=f"{type(last_error).__name__}: {last_error}",
        )
