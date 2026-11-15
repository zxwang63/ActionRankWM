from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F


IGNORE_INDEX = -100


@dataclass(frozen=True)
class EncodedResponse:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    prompt_tokens: int
    target_tokens: int


def _apply_chat_template(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        result = tokenizer.apply_chat_template(
            list(messages), enable_thinking=False, **kwargs
        )
    except TypeError:
        result = tokenizer.apply_chat_template(list(messages), **kwargs)
    if isinstance(result, Mapping):
        if "input_ids" not in result:
            raise TypeError("Tokenized chat-template mapping has no input_ids")
        result = result["input_ids"]
    if isinstance(result, torch.Tensor):
        result = result.tolist()
    if result and isinstance(result[0], list):
        if len(result) != 1:
            raise ValueError("Expected exactly one tokenized conversation")
        result = result[0]
    if not isinstance(result, list) or not all(isinstance(item, int) for item in result):
        raise TypeError("Tokenizer chat template did not return a token-ID list")
    return result


def encode_response(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    response: str,
    *,
    max_length: int,
) -> EncodedResponse:
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    parsed = json.loads(response)
    if not isinstance(parsed, Mapping):
        raise ValueError("A world-model response must be one JSON object")
    prompt_ids = _apply_chat_template(
        tokenizer, messages, add_generation_prompt=True
    )
    complete_messages = [
        *[dict(message) for message in messages],
        {"role": "assistant", "content": response},
    ]
    full_ids = _apply_chat_template(
        tokenizer, complete_messages, add_generation_prompt=False
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Assistant training sequence does not extend the inference prompt")
    if len(full_ids) > max_length:
        raise ValueError(
            f"Training sequence requires {len(full_ids)} tokens, exceeding {max_length}"
        )
    target_tokens = len(full_ids) - len(prompt_ids)
    if target_tokens <= 0:
        raise ValueError("Assistant response produced no target tokens")
    input_ids = torch.tensor(full_ids, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, : len(prompt_ids)] = IGNORE_INDEX
    return EncodedResponse(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        prompt_tokens=len(prompt_ids),
        target_tokens=target_tokens,
    )


def sequence_log_probability(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    length_normalized: bool = True,
) -> torch.Tensor:
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("Expected logits [batch, sequence, vocab] and labels [batch, sequence]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("Logit/label sequence dimensions do not match")
    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(IGNORE_INDEX)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_log_probs = F.log_softmax(shifted_logits, dim=-1).gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    totals = (token_log_probs * mask).sum(dim=-1)
    counts = mask.sum(dim=-1)
    if torch.any(counts == 0):
        raise ValueError("Every sequence needs at least one supervised next token")
    return totals / counts if length_normalized else totals


def positive_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return -sequence_log_probability(logits, labels, length_normalized=True).mean()


def dccp_objective(
    positive_logp: torch.Tensor,
    negative_logp: torch.Tensor,
    reference_difference: torch.Tensor,
    positive_ce: torch.Tensor,
    *,
    beta: float,
    alpha: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if beta <= 0.0 or alpha < 0.0:
        raise ValueError("DCCP requires beta>0 and alpha>=0")
    if positive_logp.shape != negative_logp.shape:
        raise ValueError("Positive and negative log probabilities must align")
    if reference_difference.shape != positive_logp.shape:
        reference_difference = reference_difference.expand_as(positive_logp)
    policy_difference = positive_logp - negative_logp
    logits = beta * (policy_difference - reference_difference)
    preference = -F.logsigmoid(logits).mean()
    total = preference + alpha * positive_ce
    diagnostics = {
        "loss": total.detach(),
        "preference_loss": preference.detach(),
        "positive_ce": positive_ce.detach(),
        "policy_margin": policy_difference.detach().mean(),
        "reference_margin": reference_difference.detach().mean(),
        "preference_accuracy": (logits > 0).float().mean().detach(),
    }
    return total, diagnostics


def cosine_learning_rate(
    step: int,
    total_steps: int,
    *,
    base_learning_rate: float,
    warmup_ratio: float,
) -> float:
    if total_steps <= 0 or not 0 <= step < total_steps:
        raise ValueError("step must lie in [0, total_steps)")
    if base_learning_rate <= 0.0 or not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("Invalid learning-rate schedule parameters")
    warmup_steps = max(1, math.ceil(total_steps * warmup_ratio))
    if step < warmup_steps:
        return base_learning_rate * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps - 1, 1)
    return base_learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
