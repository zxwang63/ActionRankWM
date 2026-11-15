from __future__ import annotations

import copy
import datetime as datetime_module
import importlib
import json
import uuid
from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

from actionrankwm.io import canonical_json
from actionrankwm import official_scenarios as v1


REPLAY_PROTOCOL = "official_scenario_two_message_replay_v11"


def _request_content(action: v1.OfficialAction) -> str:
    return f"{action.tool}(**{repr(action.arguments)})"


def _trace_list(context: Any, execution: Any) -> list[str]:
    frame = context.get_database(execution.DatabaseNamespace.SANDBOX)
    series = frame["tool_trace"][0]
    return [] if series is None else list(series.to_list())


def _clear_request_trace(context: Any, execution: Any) -> None:
    frame = context.get_database(execution.DatabaseNamespace.SANDBOX)
    context.update_database(
        execution.DatabaseNamespace.SANDBOX,
        frame.with_columns(execution.pl.lit(None).alias("tool_trace"))
        if hasattr(execution, "pl")
        else frame.with_columns(
            importlib.import_module("polars").lit(None).alias("tool_trace")
        ),
    )


def execute_official_plan_v11(
    scenario: Any,
    plan: v1.OfficialPlan,
    first_action_key: str,
) -> tuple[Any, list[dict[str, Any]]]:
    execution = importlib.import_module("tool_sandbox.common.execution_context")
    context = copy.deepcopy(scenario.starting_context)
    records: list[dict[str, Any]] = []
    events = v1._ordered_events(plan, first_action_key)
    tool_modules = [
        importlib.import_module(f"tool_sandbox.tools.{name}")
        for name in ("setting", "contact", "messaging", "reminder")
    ]

    for step_index, event in enumerate(events):
        if event.action is None:
            assert event.message is not None
            context.add_to_database(
                execution.DatabaseNamespace.SANDBOX,
                [
                    {
                        "sender": execution.RoleType[event.message["sender"]],
                        "recipient": execution.RoleType[event.message["recipient"]],
                        "content": str(event.message["content"]),
                    }
                ],
            )
            records.append(
                {
                    "event_key": event.key,
                    "kind": "message",
                    "sandbox_message_index": context.max_sandbox_message_index,
                    "ok": True,
                }
            )
            continue

        action = event.action
        context.add_to_database(
            execution.DatabaseNamespace.SANDBOX,
            [
                {
                    "sender": execution.RoleType.AGENT,
                    "recipient": execution.RoleType.EXECUTION_ENVIRONMENT,
                    "content": _request_content(action),
                }
            ],
        )
        request_index = context.max_sandbox_message_index
        function = context.name_to_tool.get(action.tool)
        error: str | None = None
        result: Any = None
        if function is None:
            error = f"UnknownTool:{action.tool}"
        else:
            deterministic_id = uuid.uuid5(
                uuid.NAMESPACE_URL,
                canonical_json(
                    {
                        "scenario": plan.scenario_name,
                        "first_action_key": first_action_key,
                        "step_index": step_index,
                        "action": action.to_dict(),
                        "replay": REPLAY_PROTOCOL,
                    }
                ),
            )
            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(datetime_module, "datetime", v1.FrozenDateTime)
                )
                for module in tool_modules:
                    if hasattr(module, "uuid4"):
                        stack.enter_context(
                            patch.object(module, "uuid4", lambda: deterministic_id)
                        )
                try:
                    with execution.new_context(context):
                        result = function(**copy.deepcopy(action.arguments))
                except Exception as exc:
                    error = f"{type(exc).__name__}:{exc}"

        traces = _trace_list(context, execution)
        if traces:
            _clear_request_trace(context, execution)
        if error is None and len(traces) != 1:
            raise RuntimeError(
                f"Expected exactly one successful trace for {action.key}, found {len(traces)}"
            )
        if error is not None and traces:
            raise RuntimeError(f"Failed tool unexpectedly emitted a trace: {action.key}")
        response: dict[str, Any] = {
            "sender": execution.RoleType.EXECUTION_ENVIRONMENT,
            "recipient": execution.RoleType.AGENT,
            "content": "" if error is None else error,
        }
        if traces:
            response["tool_trace"] = traces
        if error is not None:
            response["tool_call_exception"] = error
        context.add_to_database(execution.DatabaseNamespace.SANDBOX, [response])
        response_index = context.max_sandbox_message_index
        if response_index != request_index + 1:
            raise RuntimeError(
                f"Non-adjacent request/response indices: {request_index}, {response_index}"
            )
        request_frame = context.get_database(
            execution.DatabaseNamespace.SANDBOX,
            sandbox_message_index=request_index,
        )
        response_frame = context.get_database(
            execution.DatabaseNamespace.SANDBOX,
            sandbox_message_index=response_index,
        )
        if request_frame["tool_trace"][0] is not None:
            raise RuntimeError(f"Trace remained on agent request: {action.key}")
        response_trace = response_frame["tool_trace"][0]
        if (error is None) != (response_trace is not None):
            raise RuntimeError(f"Response trace/error mismatch: {action.key}")
        records.append(
            {
                "event_key": event.key,
                "kind": "action",
                "action": action.to_dict(),
                "request_index": request_index,
                "response_index": response_index,
                "ok": error is None,
                "result": v1._jsonable(result),
                "error": error,
                "trace_count": len(traces),
            }
        )
    return context, records


def evaluate_official_candidate_v11(
    scenario: Any,
    plan: v1.OfficialPlan,
    first_action_key: str,
) -> dict[str, Any]:
    context, execution_records = execute_official_plan_v11(
        scenario, plan, first_action_key
    )
    result = scenario.evaluation.evaluate(
        execution_context=context,
        max_turn_count=scenario.max_messages,
    )
    return {
        "replay_protocol": REPLAY_PROTOCOL,
        "first_action_key": first_action_key,
        "similarity": float(result.similarity),
        "milestone_similarity": float(result.milestone_similarity),
        "minefield_similarity": float(result.minefield_similarity),
        "milestone_mapping": v1._jsonable(dict(result.milestone_mapping)),
        "minefield_mapping": v1._jsonable(dict(result.minefield_mapping)),
        "turn_count": int(result.turn_count),
        "execution_records": execution_records,
        "context_digest": v1._context_digest(context),
    }


def audit_official_scenario_v11(name: str, family: str, scenario: Any) -> dict[str, Any]:
    plan = v1.extract_official_plan(name, family, scenario)
    branches: list[dict[str, Any]] = []
    nondeterminism: list[str] = []
    for action in plan.actions:
        first = evaluate_official_candidate_v11(scenario, plan, action.key)
        second = evaluate_official_candidate_v11(scenario, plan, action.key)
        stable_fields = (
            "similarity",
            "milestone_similarity",
            "minefield_similarity",
            "milestone_mapping",
            "minefield_mapping",
            "execution_records",
            "context_digest",
        )
        if any(first[field] != second[field] for field in stable_fields):
            nondeterminism.append(action.key)
        branches.append(first)
    values = [branch["similarity"] for branch in branches]
    levels = sorted({round(value, 12) for value in values})
    canonical_key = plan.actions[0].key if plan.actions else None
    canonical = next(
        (branch for branch in branches if branch["first_action_key"] == canonical_key),
        None,
    )
    gap = max(values) - min(values) if values else 0.0
    blockers = list(plan.blockers)
    if nondeterminism:
        blockers.append(f"nondeterministic_branches:{nondeterminism}")
    if canonical is None or abs(canonical["similarity"] - 1.0) > 1e-12:
        blockers.append("canonical_order_not_perfect")
    if len(levels) < 2:
        blockers.append("fewer_than_two_real_utility_levels")
    if gap < 0.20:
        blockers.append(f"real_utility_gap_below_0.20:{gap}")
    transaction_failures = [
        record
        for branch in branches
        for record in branch["execution_records"]
        if record["kind"] == "action"
        and (
            record["response_index"] != record["request_index"] + 1
            or (record["ok"] and record["trace_count"] != 1)
            or (not record["ok"] and record["trace_count"] != 0)
        )
    ]
    if transaction_failures:
        blockers.append(f"transaction_contract_failures:{len(transaction_failures)}")
    return {
        "scenario_name": name,
        "family": family,
        "replay_protocol": REPLAY_PROTOCOL,
        "plan": plan.to_dict(),
        "canonical_first_action_key": canonical_key,
        "branches": branches,
        "utility_levels": levels,
        "real_utility_gap": gap,
        "determinism_passed": not nondeterminism,
        "transaction_contract_passed": not transaction_failures,
        "passed": not blockers,
        "blockers": sorted(set(blockers)),
    }
