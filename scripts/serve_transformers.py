#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal offline OpenAI-compatible chat server for a local Hugging Face checkpoint (optionally with an ActionRankWM LoRA adapter)")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--served-model-name", default="transformers-local")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--adapter-dir", type=Path, default=None, help="optional adapter written by scripts/train_mr_sft_lora.py")
    return parser.parse_args()


class GenerationEngine:
    def __init__(
        self,
        model_path: Path,
        max_input_tokens: int,
        max_output_tokens: int,
        adapter_dir: Path | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not model_path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True
        )
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        self._model = AutoModelForCausalLM.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype="auto"
        ).to("cuda")
        if adapter_dir is not None:
            from actionrankwm.lora import load_adapter

            load_adapter(self._model, adapter_dir)
        self._model.eval()
        self._max_input_tokens = max_input_tokens
        self._max_output_tokens = max_output_tokens
        self._lock = threading.Lock()

    def generate(self, messages: list[dict[str, Any]], max_tokens: int, temperature: float) -> tuple[str, int, int]:
        try:
            rendered = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        encoded = self._tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_input_tokens,
        ).to(self._model.device)
        prompt_tokens = int(encoded.input_ids.shape[1])
        requested_tokens = min(max(1, max_tokens), self._max_output_tokens)
        generation_arguments: dict[str, Any] = {
            "max_new_tokens": requested_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "do_sample": temperature > 0.0,
        }
        if temperature > 0.0:
            generation_arguments["temperature"] = temperature
        with self._lock, self._torch.inference_mode():
            generated = self._model.generate(**encoded, **generation_arguments)
        output_ids = generated[0, prompt_tokens:]
        output = self._tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        output_tokens = int(output_ids.shape[0])
        return output, prompt_tokens, output_tokens


def make_handler(engine: GenerationEngine, served_model_name: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "ActionRankWMTransformers/0.1"

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
            if self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [{"id": served_model_name}]})
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
            if self.path != "/v1/chat/completions":
                self._json(404, {"error": {"message": "not found"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 4 * 1024 * 1024:
                    raise ValueError("invalid request size")
                request = json.loads(self.rfile.read(length))
                messages = request["messages"]
                if not isinstance(messages, list):
                    raise TypeError("messages must be a list")
                output, prompt_tokens, completion_tokens = engine.generate(
                    messages=messages,
                    max_tokens=int(request.get("max_tokens", 512)),
                    temperature=float(request.get("temperature", 0.0)),
                )
                self._json(
                    200,
                    {
                        "id": f"arwm-{time.time_ns()}",
                        "object": "chat.completion",
                        "model": served_model_name,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": output},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    },
                )
            except Exception as exc:
                self._json(
                    500,
                    {"error": {"type": type(exc).__name__, "message": str(exc)[:500]}},
                )

        def log_message(self, format: str, *args: object) -> None:
            print(
                json.dumps(
                    {
                        "time": self.log_date_time_string(),
                        "client": self.client_address[0],
                        "request": format % args,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    return Handler


def main() -> None:
    args = parse_args()
    engine = GenerationEngine(args.model, args.max_input_tokens, args.max_output_tokens, args.adapter_dir)
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(engine, args.served_model_name)
    )
    print(
        json.dumps(
            {
                "status": "ready",
                "host": args.host,
                "port": args.port,
                "served_model_name": args.served_model_name,
                "model_path": str(args.model.resolve()),
                "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
