from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


STRATA = (
    "all_candidates_clean",
    "all_real_best_clean",
    "any_real_best_failure",
    "any_candidate_failure",
    "all_relevant_candidates_clean",
    "clean_semantic_witness",
)
EVENTS = ("unsafe", "high_fidelity_unsafe")
FAILURE_COMPONENTS = ("parse_failure", "patch_repair", "replay_error")


def candidate_failure_flags(row: Mapping[str, Any]) -> dict[str, bool]:
    """Return the frozen strict engineering-failure decomposition."""

    required = (
        "task_id",
        "action_id",
        "parse_ok",
        "patch_repair",
        "prediction_error",
    )
    missing = [name for name in required if name not in row]
    if missing:
        raise ValueError(f"Candidate omits required fields: {missing}")
    if not isinstance(row["parse_ok"], bool):
        raise ValueError("Candidate parse_ok must be Boolean")
    flags = {
        "parse_failure": row["parse_ok"] is not True,
        "patch_repair": row["patch_repair"] is not None,
        "replay_error": row["prediction_error"] is not None,
    }
    flags["engineering_failure"] = any(flags.values())
    return flags


def _event_summary(rows: Sequence[Mapping[str, Any]], event: str) -> dict[str, Any]:
    selected = [row for row in rows if row[event] is True]
    denominator = len(selected)
    counts = {name: sum(bool(row[name]) for row in selected) for name in STRATA}
    return {
        "event_count": denominator,
        "counts": counts,
        "rates": {
            name: (count / denominator if denominator else None)
            for name, count in counts.items()
        },
    }


def _component_summary(
    candidates: Sequence[tuple[Mapping[str, Any], Mapping[str, bool]]],
) -> dict[str, Any]:
    counts = Counter()
    for _, flags in candidates:
        for component in (*FAILURE_COMPONENTS, "engineering_failure"):
            counts[component] += int(flags[component])
    total = len(candidates)
    return {
        "candidate_count": total,
        "counts": {
            component: counts[component]
            for component in (*FAILURE_COMPONENTS, "engineering_failure")
        },
        "rates": {
            component: (counts[component] / total if total else None)
            for component in (*FAILURE_COMPONENTS, "engineering_failure")
        },
    }


def _combine_component_summaries(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    total = sum(int(summary["candidate_count"]) for summary in summaries)
    counts = {
        component: sum(int(summary["counts"][component]) for summary in summaries)
        for component in (*FAILURE_COMPONENTS, "engineering_failure")
    }
    return {
        "candidate_count": total,
        "counts": counts,
        "rates": {
            component: (count / total if total else None)
            for component, count in counts.items()
        },
    }


def _task_component_summaries(
    *,
    by_action: Mapping[str, Mapping[str, Any]],
    flags: Mapping[str, Mapping[str, bool]],
    real_best: set[str],
    harmful_top: set[str],
    hf_harmful_top: set[str],
) -> dict[str, Any]:
    scopes = {
        "all_candidates": set(by_action),
        "real_best_candidates": real_best,
        "harmful_top_candidates": harmful_top,
        "high_fidelity_harmful_top_candidates": hf_harmful_top,
    }
    return {
        scope: _component_summary(
            [(by_action[action_id], flags[action_id]) for action_id in action_ids]
        )
        for scope, action_ids in scopes.items()
    }


def stratify_model_panel(
    *,
    model_key: str,
    decisions: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build task rows and strict failure strata for one complete model panel."""

    decision_by_task: dict[str, Mapping[str, Any]] = {}
    for decision in decisions:
        task_id = str(decision["task_id"])
        if task_id in decision_by_task:
            raise ValueError(f"Duplicate decision task: {model_key}/{task_id}")
        decision_by_task[task_id] = decision

    candidates_by_task: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for candidate in candidates:
        task_id = str(candidate["task_id"])
        action_id = str(candidate["action_id"])
        if action_id in candidates_by_task[task_id]:
            raise ValueError(f"Duplicate candidate: {model_key}/{task_id}/{action_id}")
        candidates_by_task[task_id][action_id] = candidate

    if set(decision_by_task) != set(candidates_by_task):
        raise ValueError(f"Decision/candidate task inventory mismatch for {model_key}")

    task_rows: list[dict[str, Any]] = []
    scoped_candidates: dict[str, list[tuple[Mapping[str, Any], Mapping[str, bool]]]] = {
        "all_candidates": [],
        "real_best_candidates": [],
        "harmful_top_candidates": [],
        "high_fidelity_harmful_top_candidates": [],
    }
    for task_id in sorted(decision_by_task):
        decision = decision_by_task[task_id]
        by_action = candidates_by_task[task_id]
        expected_count = int(decision.get("candidate_count", -1))
        if expected_count != len(by_action):
            raise ValueError(f"Candidate count mismatch for {model_key}/{task_id}")

        real_best = set(map(str, decision.get("real_best_action_ids", [])))
        harmful_top = set(
            map(str, decision.get("simulated_top_harmful_action_ids", []))
        )
        hf_harmful_top = set(
            map(
                str,
                decision.get(
                    "simulated_top_high_fidelity_harmful_action_ids", []
                ),
            )
        )
        referenced = real_best | harmful_top | hf_harmful_top
        if (
            not real_best
            or not referenced.issubset(by_action)
            or not hf_harmful_top.issubset(harmful_top)
        ):
            raise ValueError(f"Action-reference mismatch for {model_key}/{task_id}")

        flags = {
            action_id: candidate_failure_flags(candidate)
            for action_id, candidate in by_action.items()
        }
        clean = {
            action_id: not row["engineering_failure"]
            for action_id, row in flags.items()
        }
        relevant = real_best | harmful_top
        all_real_best_clean = all(clean[action_id] for action_id in real_best)
        unsafe = bool(decision.get("simulated_top_contains_harmful"))
        hf_unsafe = bool(
            decision.get("simulated_top_contains_high_fidelity_harmful")
        )
        if unsafe != bool(harmful_top) or hf_unsafe != bool(hf_harmful_top):
            raise ValueError(f"Unsafe-event mismatch for {model_key}/{task_id}")

        row = {
            "model_key": model_key,
            "task_id": task_id,
            "mechanism": str(
                decision.get("template_id")
                or decision.get("mechanism")
                or "unknown"
            ).removeprefix("toolsandbox."),
            "unsafe": unsafe,
            "high_fidelity_unsafe": hf_unsafe,
            "all_candidates_clean": all(clean.values()),
            "all_real_best_clean": all_real_best_clean,
            "any_real_best_failure": not all_real_best_clean,
            "any_candidate_failure": not all(clean.values()),
            "all_relevant_candidates_clean": all(
                clean[action_id] for action_id in relevant
            ),
            "clean_semantic_witness_unsafe": (
                unsafe
                and all_real_best_clean
                and any(clean[action_id] for action_id in harmful_top)
            ),
            "clean_semantic_witness_high_fidelity_unsafe": (
                hf_unsafe
                and all_real_best_clean
                and any(clean[action_id] for action_id in hf_harmful_top)
            ),
            "candidate_engineering_failure_count": sum(
                not value for value in clean.values()
            ),
            "real_best_engineering_failure_count": sum(
                not clean[action_id] for action_id in real_best
            ),
            "harmful_top_engineering_failure_count": sum(
                not clean[action_id] for action_id in harmful_top
            ),
            "high_fidelity_harmful_top_engineering_failure_count": sum(
                not clean[action_id] for action_id in hf_harmful_top
            ),
            "real_best_action_count": len(real_best),
            "harmful_top_action_count": len(harmful_top),
            "high_fidelity_harmful_top_action_count": len(hf_harmful_top),
            "failure_components": _task_component_summaries(
                by_action=by_action,
                flags=flags,
                real_best=real_best,
                harmful_top=harmful_top,
                hf_harmful_top=hf_harmful_top,
            ),
        }
        task_rows.append(row)

        for action_id, candidate in by_action.items():
            scoped_candidates["all_candidates"].append((candidate, flags[action_id]))
        for action_id in real_best:
            scoped_candidates["real_best_candidates"].append(
                (by_action[action_id], flags[action_id])
            )
        for action_id in harmful_top:
            scoped_candidates["harmful_top_candidates"].append(
                (by_action[action_id], flags[action_id])
            )
        for action_id in hf_harmful_top:
            scoped_candidates["high_fidelity_harmful_top_candidates"].append(
                (by_action[action_id], flags[action_id])
            )

    def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        materialized = []
        for row in rows:
            copied = dict(row)
            materialized.append(copied)
        result = {
            "model_task_count": len(materialized),
            "events": {},
            "failure_components": {
                scope: _combine_component_summaries(
                    [row["failure_components"][scope] for row in materialized]
                )
                for scope in (
                    "all_candidates",
                    "real_best_candidates",
                    "harmful_top_candidates",
                    "high_fidelity_harmful_top_candidates",
                )
            },
        }
        for event in EVENTS:
            event_rows = []
            witness_key = f"clean_semantic_witness_{event}"
            for row in materialized:
                copied = dict(row)
                copied["clean_semantic_witness"] = bool(row[witness_key])
                event_rows.append(copied)
            result["events"][event] = _event_summary(event_rows, event)
        return result

    by_mechanism = {
        mechanism: summarize(
            [row for row in task_rows if row["mechanism"] == mechanism]
        )
        for mechanism in sorted({row["mechanism"] for row in task_rows})
    }
    return {
        "model_key": model_key,
        "summary": summarize(task_rows),
        "by_mechanism": by_mechanism,
        "failure_components": {
            scope: _component_summary(rows) for scope, rows in scoped_candidates.items()
        },
        "task_rows": task_rows,
    }


def stratify_panels(
    panels: Mapping[
        str, tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]
    ],
) -> dict[str, Any]:
    """Aggregate complete model panels while retaining every model and task."""

    if not panels:
        raise ValueError("At least one model panel is required")
    per_model = {
        model_key: stratify_model_panel(
            model_key=model_key,
            decisions=decisions,
            candidates=candidates,
        )
        for model_key, (decisions, candidates) in sorted(panels.items())
    }
    all_rows = [
        row
        for model in per_model.values()
        for row in model["task_rows"]
    ]

    def pooled_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        result = {
            "model_task_count": len(rows),
            "events": {},
            "failure_components": {
                scope: _combine_component_summaries(
                    [row["failure_components"][scope] for row in rows]
                )
                for scope in (
                    "all_candidates",
                    "real_best_candidates",
                    "harmful_top_candidates",
                    "high_fidelity_harmful_top_candidates",
                )
            },
        }
        for event in EVENTS:
            prepared = []
            for row in rows:
                copied = dict(row)
                copied["clean_semantic_witness"] = bool(
                    row[f"clean_semantic_witness_{event}"]
                )
                prepared.append(copied)
            result["events"][event] = _event_summary(prepared, event)
        return result

    mechanisms = sorted({row["mechanism"] for row in all_rows})
    return {
        "pooled": pooled_summary(all_rows),
        "per_model": {
            key: {
                "summary": value["summary"],
                "by_mechanism": value["by_mechanism"],
                "failure_components": value["failure_components"],
            }
            for key, value in per_model.items()
        },
        "by_mechanism": {
            mechanism: pooled_summary(
                [row for row in all_rows if row["mechanism"] == mechanism]
            )
            for mechanism in mechanisms
        },
        "task_rows": sorted(
            all_rows, key=lambda row: (str(row["model_key"]), str(row["task_id"]))
        ),
    }
