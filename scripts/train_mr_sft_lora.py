#!/usr/bin/env python3
"""Train one matched-replay SFT (MR-SFT) LoRA adapter.

This is the paper's optimisation recipe without the internal run-authorisation
checks of the original cluster driver:

* rank-16 LoRA (alpha 32, no dropout) on the language-model linear layers
  (``actionrankwm.lora.inject_lora``), base weights frozen;
* loss = assistant-token cross-entropy on the positive (real-transition)
  response only, weighted by each record's ``sample_weight``;
* AdamW (lr 2e-4, betas 0.9/0.95, eps 1e-8, no weight decay), 16 records per
  update, 3% warm-up then cosine decay, gradient-norm clip 1.0;
* one epoch over the 360 released records (23 updates), each record
  forwarded twice per update with weight 0.5 (as in the original driver);
* deterministic per-seed shuffling; paper seeds 20260720, 20260721, 20260722.

Usage (needs a CUDA GPU, torch, transformers, safetensors):

    python scripts/train_mr_sft_lora.py \
        --model /path/to/Qwen3.5-4B --seed 20260720 \
        --records data/mr_sft_training/training_records.jsonl.gz \
        --output-dir outputs/mr_sft_qwen4b_seed20260720

Serve the adapter with ``scripts/serve_transformers.py --adapter-dir
<output-dir>/adapter`` and evaluate it like any other model (README).
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from statistics import fmean
from typing import Any

from actionrankwm.io import atomic_write_json, iter_jsonl, write_jsonl
from actionrankwm.lora import inject_lora, lora_parameters, save_adapter, trainable_parameter_summary
from actionrankwm.training import cosine_learning_rate, encode_response, positive_cross_entropy

GRADIENT_ACCUMULATION = 16
BASE_LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
MAX_TOKENS = 4096


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, type=Path, help="local Hugging Face checkpoint directory")
    parser.add_argument("--records", type=Path, default=Path("data/mr_sft_training/training_records.jsonl.gz"))
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("MR-SFT training requires a CUDA device")
    output_dir.mkdir(parents=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    records = list(iter_jsonl(args.records))
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(args.model), local_files_only=True, torch_dtype="auto").to("cuda")
    model.config.use_cache = False
    encoded = [
        {"record": r, "positive": encode_response(tokenizer, r["messages"], r["positive_response"], max_length=MAX_TOKENS)}
        for r in records
    ]

    specs = inject_lora(model, rank=16, alpha=32.0, dropout=0.0)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.train()
    optimizer = torch.optim.AdamW(lora_parameters(model), lr=BASE_LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)

    order = list(range(len(encoded)))
    random.Random(args.seed).shuffle(order)
    total_updates = math.ceil(len(order) / GRADIENT_ACCUMULATION)
    log: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    for update in range(total_updates):
        group = order[update * GRADIENT_ACCUMULATION : (update + 1) * GRADIENT_ACCUMULATION]
        lr = cosine_learning_rate(update, total_updates, base_learning_rate=BASE_LEARNING_RATE, warmup_ratio=WARMUP_RATIO)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = lr
        losses = []
        for position in group:
            item = encoded[position]
            weight = float(item["record"]["sample_weight"])
            for _ in range(2):
                batch = item["positive"]
                outputs = model(input_ids=batch.input_ids.to("cuda"), attention_mask=batch.attention_mask.to("cuda"))
                loss = positive_cross_entropy(outputs.logits, batch.labels.to("cuda"))
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss for record {item['record']['pair_id']}")
                (loss * (0.5 * weight / len(group))).backward()
                losses.append(float(loss.detach().cpu()))
                del outputs, loss
        norm = torch.nn.utils.clip_grad_norm_(lora_parameters(model), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        log.append({"update": update, "learning_rate": lr, "mean_loss": fmean(losses), "preclip_grad_norm": float(norm)})
        print(f"update {update + 1}/{total_updates} lr={lr:.2e} loss={fmean(losses):.4f}", flush=True)

    write_jsonl(output_dir / "training_log.jsonl", log)
    adapter_manifest = save_adapter(
        model,
        output_dir / "adapter",
        metadata={"condition": "matched_replay_sft", "seed": args.seed, "records": len(records), "base_model": args.model.name},
    )
    atomic_write_json(
        output_dir / "summary.json",
        {
            "seed": args.seed,
            "records": len(records),
            "optimizer_updates": total_updates,
            "lora_module_count": len(specs),
            "parameter_summary": trainable_parameter_summary(model),
            "adapter_manifest": adapter_manifest,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
