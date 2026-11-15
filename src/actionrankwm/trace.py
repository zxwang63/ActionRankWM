from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from actionrankwm.types import Action, PredictionRecord, TaskSpec


def _visible_rules(task: TaskSpec) -> list[Mapping[str, Any]]:
    value = task.initial_state.get("environment_rules", [])
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _rule_matches(task: TaskSpec, action: Action, rule: Mapping[str, Any]) -> bool:
    if task.domain == "sqlite_mcp":
        if action.tool != "db.update":
            return False
        if str(action.arguments.get("table", "")) != str(rule.get("table", "")):
            return False
        updates_value = action.arguments.get("set", {})
        updates = updates_value if isinstance(updates_value, Mapping) else {}
        when_value = rule.get("when_set", {})
        when_set = when_value if isinstance(when_value, Mapping) else {}
        return all(updates.get(key) == value for key, value in when_set.items())
    if task.domain == "terminal_fs":
        return action.tool == "fs.write" and str(
            action.arguments.get("path", "")
        ) == str(rule.get("path", ""))
    return False


def audit_rule_trace(
    task: TaskSpec, prediction: PredictionRecord
) -> dict[str, Any]:
    rules = _visible_rules(task)
    action = next(
        (item for item in task.candidates if item.action_id == prediction.action_id),
        None,
    )
    expected_matches = (
        [_rule_matches(task, action, rule) for rule in rules]
        if action is not None
        else [False for _ in rules]
    )
    parsed = prediction.parsed_observation
    wrapper_valid = bool(
        isinstance(parsed, Mapping)
        and isinstance(parsed.get("direct_effect"), Mapping)
        and isinstance(parsed.get("rule_trace"), list)
        and isinstance(parsed.get("observation"), Mapping)
    )
    raw_trace = parsed.get("rule_trace") if isinstance(parsed, Mapping) else None
    trace = raw_trace if isinstance(raw_trace, list) else []
    entries = [item for item in trace if isinstance(item, Mapping)]
    indices = [
        item.get("rule_index")
        for item in entries
        if isinstance(item.get("rule_index"), int)
        and not isinstance(item.get("rule_index"), bool)
    ]
    expected_indices = list(range(len(rules)))
    valid_matched_fields = [
        item for item in entries if isinstance(item.get("matched"), bool)
    ]
    declared_match_count = sum(
        1 for item in valid_matched_fields if item.get("matched") is True
    )
    matched_with_effect_object = sum(
        1
        for item in valid_matched_fields
        if item.get("matched") is True
        and isinstance(item.get("derived_effect"), Mapping)
    )
    unmatched_with_empty_effect = sum(
        1
        for item in valid_matched_fields
        if item.get("matched") is False and item.get("derived_effect") == {}
    )
    correctness: list[bool] = []
    for item in valid_matched_fields:
        index = item.get("rule_index")
        if (
            isinstance(index, int)
            and not isinstance(index, bool)
            and 0 <= index < len(expected_matches)
        ):
            correctness.append(bool(item["matched"]) == expected_matches[index])
    return {
        "task_id": task.task_id,
        "domain": task.domain,
        "action_id": prediction.action_id,
        "parse_ok": prediction.parse_ok,
        "wrapper_valid": wrapper_valid,
        "direct_effect_present": bool(
            isinstance(parsed, Mapping)
            and isinstance(parsed.get("direct_effect"), Mapping)
        ),
        "observation_wrapper_present": bool(
            isinstance(parsed, Mapping)
            and isinstance(parsed.get("observation"), Mapping)
        ),
        "trace_present": isinstance(raw_trace, list),
        "expected_rule_count": len(rules),
        "observed_rule_count": len(trace),
        "mapping_entry_count": len(entries),
        "rule_count_covered": len(trace) == len(rules),
        "indices": indices,
        "index_coverage_valid": sorted(indices) == expected_indices
        and len(indices) == len(trace),
        "original_order_valid": indices == expected_indices
        and len(indices) == len(trace),
        "valid_matched_field_count": len(valid_matched_fields),
        "declared_match_count": declared_match_count,
        "expected_match_count": sum(expected_matches),
        "match_correct_count": sum(correctness),
        "match_comparable_count": len(correctness),
        "matched_with_effect_object_count": matched_with_effect_object,
        "unmatched_with_empty_effect_count": unmatched_with_empty_effect,
    }


def aggregate_trace_audits(
    audits: Sequence[Mapping[str, Any]], prompt_style: str
) -> dict[str, Any]:
    count = len(audits)

    def rate(numerator: int, denominator: int = count) -> float:
        return numerator / denominator if denominator else 0.0

    expected_entries = sum(int(item["expected_rule_count"]) for item in audits)
    observed_entries = sum(int(item["observed_rule_count"]) for item in audits)
    valid_matched = sum(int(item["valid_matched_field_count"]) for item in audits)
    declared_matches = sum(int(item["declared_match_count"]) for item in audits)
    expected_matches = sum(int(item["expected_match_count"]) for item in audits)
    comparable_matches = sum(int(item["match_comparable_count"]) for item in audits)
    correct_matches = sum(int(item["match_correct_count"]) for item in audits)
    matched_effects = sum(
        int(item["matched_with_effect_object_count"]) for item in audits
    )
    expected_unmatched = max(0, expected_entries - expected_matches)
    unmatched_empty = sum(
        int(item["unmatched_with_empty_effect_count"]) for item in audits
    )
    return {
        "prompt_style": prompt_style,
        "prediction_count": count,
        "wrapper_valid_rate": rate(sum(bool(item["wrapper_valid"]) for item in audits)),
        "direct_effect_present_rate": rate(
            sum(bool(item["direct_effect_present"]) for item in audits)
        ),
        "observation_wrapper_rate": rate(
            sum(bool(item["observation_wrapper_present"]) for item in audits)
        ),
        "trace_present_rate": rate(sum(bool(item["trace_present"]) for item in audits)),
        "rule_count_coverage_rate": rate(
            sum(bool(item["rule_count_covered"]) for item in audits)
        ),
        "index_coverage_valid_rate": rate(
            sum(bool(item["index_coverage_valid"]) for item in audits)
        ),
        "original_order_valid_rate": rate(
            sum(bool(item["original_order_valid"]) for item in audits)
        ),
        "expected_rule_entry_count": expected_entries,
        "observed_rule_entry_count": observed_entries,
        "valid_matched_field_rate": rate(valid_matched, observed_entries),
        "declared_match_rate": rate(declared_matches, valid_matched),
        "expected_match_rate": rate(expected_matches, expected_entries),
        "match_accuracy": rate(correct_matches, comparable_matches),
        "matched_effect_object_rate": rate(matched_effects, declared_matches),
        "unmatched_empty_effect_rate": rate(unmatched_empty, expected_unmatched),
    }
