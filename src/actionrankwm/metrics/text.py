from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from actionrankwm.io import canonical_json


_TOKEN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
_MISSING = object()


def _f1(reference: Counter[Any], prediction: Counter[Any]) -> float:
    if not reference and not prediction:
        return 1.0
    overlap = sum((reference & prediction).values())
    precision = overlap / sum(prediction.values()) if prediction else 0.0
    recall = overlap / sum(reference.values()) if reference else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def token_f1(reference: Mapping[str, Any], prediction: Mapping[str, Any] | None) -> float:
    """Multiset token F1 over canonical JSON.

    This intentionally remains a surface metric. It is not allowed to inspect
    the latent task target or the downstream branch value.
    """

    if prediction is None:
        return 0.0
    reference_tokens = Counter(token.casefold() for token in _TOKEN.findall(canonical_json(reference)))
    prediction_tokens = Counter(token.casefold() for token in _TOKEN.findall(canonical_json(prediction)))
    return _f1(reference_tokens, prediction_tokens)


def _leaf_pairs(value: Any, path: tuple[str, ...] = ()) -> list[tuple[str, str]]:
    if isinstance(value, Mapping):
        if not value:
            return [(".".join(path), "{}")]
        pairs: list[tuple[str, str]] = []
        for key in sorted(value, key=str):
            pairs.extend(_leaf_pairs(value[key], path + (str(key),)))
        return pairs
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            return [(".".join(path), "[]")]
        pairs = []
        for index, item in enumerate(value):
            pairs.extend(_leaf_pairs(item, path + (str(index),)))
        return pairs
    return [(".".join(path), json.dumps(value, ensure_ascii=False, sort_keys=True))]


def structured_f1(reference: Mapping[str, Any], prediction: Mapping[str, Any] | None) -> float:
    """Exact leaf-path/value F1 for a structured observation."""

    if prediction is None:
        return 0.0
    return _f1(Counter(_leaf_pairs(reference)), Counter(_leaf_pairs(prediction)))


def _resolve_path(value: Any, path: str) -> Any:
    current = value
    for component in path.split(".") if path else []:
        if isinstance(current, Mapping):
            if component not in current:
                return _MISSING
            current = current[component]
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            try:
                current = current[int(component)]
            except (ValueError, IndexError):
                return _MISSING
        else:
            return _MISSING
    return current


def critical_field_accuracy(
    reference: Mapping[str, Any],
    prediction: Mapping[str, Any] | None,
    paths: Sequence[str],
) -> float:
    """Accuracy on predeclared critical paths that exist in the true observation.

    A path absent from the true observation is inapplicable and excluded. A
    missing prediction for an applicable path is incorrect.
    """

    if prediction is None:
        return 0.0
    applicable = 0
    correct = 0
    for path in paths:
        expected = _resolve_path(reference, path)
        if expected is _MISSING:
            continue
        applicable += 1
        actual = _resolve_path(prediction, path)
        if actual is not _MISSING and actual == expected:
            correct += 1
    return correct / applicable if applicable else 1.0
