from __future__ import annotations

import copy
import datetime as datetime_module
import importlib
import json
import subprocess
import sys
import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from unittest.mock import patch

from actionrankwm.io import canonical_json, digest_json
from actionrankwm.toolsandbox_tasks import FROZEN_TIMESTAMP, UPSTREAM_COMMIT


FAMILY_FACTORIES = (
    ("single_tool_call", "tool_sandbox.scenarios.single_tool_call_scenarios", "named_single_tool_call_scenarios"),
    ("multiple_tool_call", "tool_sandbox.scenarios.multiple_tool_call_scenarios", "named_multiple_tool_call_scenarios"),
    ("multiple_user_turn", "tool_sandbox.scenarios.multiple_user_turn_scenarios", "named_multiple_user_turn_scenarios"),
    (
        "insufficient_information",
        "tool_sandbox.scenarios.insufficient_information_scenarios",
        "named_insufficient_information_scenarios",
    ),
)

LOCAL_TOOL_MODULES = frozenset(
    {
        "tool_sandbox.tools.setting",
        "tool_sandbox.tools.contact",
        "tool_sandbox.tools.messaging",
        "tool_sandbox.tools.reminder",
        "tool_sandbox.tools.user_tools",
    }
)

V0_NARROW_TOOLS = frozenset(
    {
        "end_conversation",
        "add_contact",
        "modify_contact",
        "remove_contact",
        "search_contacts",
        "search_messages",
        "send_message_with_phone_number",
        "set_low_battery_mode_status",
        "set_wifi_status",
        "get_wifi_status",
        "set_cellular_service_status",
        "get_cellular_service_status",
        "set_location_service_status",
        "get_location_service_status",
    }
)

PROTOTYPE_SCENARIOS = (
    "remove_contact_by_phone",
    "update_contact_relationship_with_relationship",
    "send_message_with_contact_content_cellular_off",
)

SETTING_SETTERS = {
    "low_battery_mode": "set_low_battery_mode_status",
    "wifi": "set_wifi_status",
    "cellular": "set_cellular_service_status",
    "location_service": "set_location_service_status",
}


class FrozenDateTime(datetime_module.datetime):
    @classmethod
    def now(cls, tz: datetime_module.tzinfo | None = None) -> "FrozenDateTime":
        return cls.fromtimestamp(FROZEN_TIMESTAMP, tz=tz)

    @classmethod
    def utcnow(cls) -> "FrozenDateTime":
        return cls.utcfromtimestamp(FROZEN_TIMESTAMP)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (datetime_module.date, datetime_module.datetime)):
        return value.isoformat()
    return str(value)


def _enum_text(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw)


def _callable_name(value: Any) -> str:
    while isinstance(value, partial):
        value = value.func
    module = getattr(value, "__module__", type(value).__module__)
    qualname = getattr(value, "__qualname__", getattr(value, "__name__", type(value).__name__))
    return f"{module}.{qualname}"


def _base_callable_name(value: Any) -> str:
    return _callable_name(value).rsplit(".", 1)[-1]


def _git_revision(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def runtime_preflight(source_root: str | Path) -> dict[str, Any]:
    source = Path(source_root).resolve()
    tool_sandbox = importlib.import_module("tool_sandbox")
    module_path = Path(str(tool_sandbox.__file__)).resolve()
    revision = _git_revision(source)
    if revision != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"ToolSandbox revision mismatch: expected {UPSTREAM_COMMIT}, found {revision}"
        )
    try:
        module_path.relative_to(source)
    except ValueError as exc:
        raise RuntimeError(
            f"ToolSandbox imported from {module_path}, outside frozen source {source}"
        ) from exc
    return {
        "python_executable": sys.executable,
        "module_path": str(module_path),
        "source_root": str(source),
        "revision": revision,
    }


def load_unaugmented_scenarios() -> dict[str, tuple[str, Any]]:
    discovery = importlib.import_module("tool_sandbox.common.tool_discovery")
    backend = discovery.ToolBackend.DEFAULT
    scenarios: dict[str, tuple[str, Any]] = {}
    with patch.object(datetime_module, "datetime", FrozenDateTime):
        for family, module_name, factory_name in FAMILY_FACTORIES:
            factory = getattr(importlib.import_module(module_name), factory_name)
            family_scenarios = factory(preferred_tool_backend=backend)
            overlap = set(scenarios) & set(family_scenarios)
            if overlap:
                raise ValueError(f"Duplicate official scenario names: {sorted(overlap)}")
            scenarios.update(
                {name: (family, scenario) for name, scenario in family_scenarios.items()}
            )
    return dict(sorted(scenarios.items()))


@lru_cache(maxsize=1)
def local_tool_names() -> frozenset[str]:
    inspect_module = importlib.import_module("inspect")
    tool_module = importlib.import_module("tool_sandbox.tools")
    discovery = importlib.import_module("tool_sandbox.common.tool_discovery")
    tools = discovery.find_tools_by_module(
        tool_module, preferred_tool_backend=discovery.ToolBackend.DEFAULT
    )
    return frozenset(
        name
        for name, function in tools.items()
        if inspect_module.getmodule(function) is not None
        and inspect_module.getmodule(function).__name__ in LOCAL_TOOL_MODULES
    )


def deterministic_topological_order(node_count: int, edges: Iterable[Sequence[int]]) -> list[int]:
    successors = {index: set() for index in range(node_count)}
    indegree = {index: 0 for index in range(node_count)}
    for raw_source, raw_target in edges:
        source, target = int(raw_source), int(raw_target)
        if source not in successors or target not in successors:
            raise ValueError(f"Milestone edge outside node inventory: {(source, target)}")
        if target not in successors[source]:
            successors[source].add(target)
            indegree[target] += 1
    ready = sorted(index for index, degree in indegree.items() if degree == 0)
    order: list[int] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for target in sorted(successors[node]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort()
    if len(order) != node_count:
        raise ValueError("Official milestone graph is cyclic")
    return order


@dataclass(frozen=True)
class OfficialAction:
    key: str
    milestone_index: int
    tool: str
    arguments: dict[str, Any]
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "milestone_index": self.milestone_index,
            "tool": self.tool,
            "arguments": copy.deepcopy(self.arguments),
            "source": self.source,
        }


@dataclass(frozen=True)
class OfficialEvent:
    key: str
    milestone_index: int
    kind: str
    action: OfficialAction | None = None
    message: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "milestone_index": self.milestone_index,
            "kind": self.kind,
            "action": None if self.action is None else self.action.to_dict(),
            "message": copy.deepcopy(self.message),
        }


@dataclass(frozen=True)
class OfficialPlan:
    scenario_name: str
    family: str
    events: tuple[OfficialEvent, ...]
    blockers: tuple[str, ...]
    milestone_order: tuple[int, ...]

    @property
    def actions(self) -> tuple[OfficialAction, ...]:
        return tuple(event.action for event in self.events if event.action is not None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "family": self.family,
            "milestone_order": list(self.milestone_order),
            "events": [event.to_dict() for event in self.events],
            "blockers": list(self.blockers),
        }


def _namespace_rows(context: Any, namespace_name: str) -> list[dict[str, Any]]:
    execution = importlib.import_module("tool_sandbox.common.execution_context")
    namespace = execution.DatabaseNamespace[namespace_name.upper()]
    rows = context.get_database(namespace).to_dicts()
    return [
        {str(key): _jsonable(value) for key, value in row.items() if key != "sandbox_message_index"}
        for row in rows
    ]


def _initial_symbolic_state(scenario: Any) -> dict[str, list[dict[str, Any]]]:
    return {
        namespace: _namespace_rows(scenario.starting_context, namespace)
        for namespace in ("setting", "contact", "messaging", "reminder")
    }


def _find_row(rows: list[dict[str, Any]], key: str, value: Any) -> dict[str, Any] | None:
    matches = [row for row in rows if row.get(key) == value]
    return matches[0] if len(matches) == 1 else None


def _apply_symbolic_action(state: dict[str, list[dict[str, Any]]], action: OfficialAction) -> None:
    tool, arguments = action.tool, action.arguments
    if tool in SETTING_SETTERS.values():
        field = next(field for field, setter in SETTING_SETTERS.items() if setter == tool)
        if state["setting"]:
            state["setting"][0][field] = bool(arguments["on"])
            if field == "low_battery_mode" and bool(arguments["on"]):
                for dependent in ("wifi", "cellular", "location_service"):
                    state["setting"][0][dependent] = False
        return
    if tool == "remove_contact":
        state["contact"] = [
            row for row in state["contact"] if row.get("person_id") != arguments.get("person_id")
        ]
        return
    if tool == "modify_contact":
        row = _find_row(state["contact"], "person_id", arguments.get("person_id"))
        if row is not None:
            row.update({key: value for key, value in arguments.items() if key != "person_id"})
        return
    if tool == "add_contact":
        state["contact"].append(copy.deepcopy(arguments))
        return
    if tool == "send_message_with_phone_number":
        state["messaging"].append(
            {
                "recipient_phone_number": arguments.get("phone_number"),
                "content": arguments.get("content"),
            }
        )


def _new_action(
    milestone_index: int,
    tool: str,
    arguments: Mapping[str, Any],
    source: str,
) -> OfficialAction:
    payload = {
        "milestone_index": milestone_index,
        "tool": str(tool),
        "arguments": _jsonable(dict(arguments)),
        "source": source,
    }
    key = f"m{milestone_index}:{digest_json(payload)[:16]}"
    return OfficialAction(
        key=key,
        milestone_index=milestone_index,
        tool=str(tool),
        arguments=dict(payload["arguments"]),
        source=source,
    )


def _trace_action(milestone_index: int, value: Any) -> tuple[OfficialAction | None, str | None]:
    try:
        trace = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        return None, f"m{milestone_index}:invalid_tool_trace:{type(exc).__name__}"
    if isinstance(trace, list):
        return None, f"m{milestone_index}:alternative_tool_trace_is_ambiguous"
    if not isinstance(trace, Mapping) or not isinstance(trace.get("arguments"), Mapping):
        return None, f"m{milestone_index}:malformed_tool_trace"
    if not isinstance(trace.get("tool_name"), str):
        return None, f"m{milestone_index}:missing_tool_name"
    return (
        _new_action(
            milestone_index,
            str(trace["tool_name"]),
            dict(trace["arguments"]),
            "official_tool_trace",
        ),
        None,
    )


def _state_actions(
    milestone_index: int,
    namespace: str,
    measure: str,
    targets: list[dict[str, Any]],
    reference_state: dict[str, list[dict[str, Any]]],
) -> tuple[list[OfficialAction], list[str]]:
    actions: list[OfficialAction] = []
    blockers: list[str] = []
    if namespace == "setting" and measure == "snapshot_similarity":
        if len(targets) != 1 or len(reference_state["setting"]) != 1:
            return [], [f"m{milestone_index}:setting_target_not_singleton"]
        before = reference_state["setting"][0]
        changed = [
            field
            for field in SETTING_SETTERS
            if field in targets[0] and targets[0][field] != before.get(field)
        ]
        unknown = set(targets[0]) - set(before) - set(SETTING_SETTERS)
        if unknown:
            blockers.append(f"m{milestone_index}:unknown_setting_fields:{sorted(unknown)}")
        for field in changed:
            actions.append(
                _new_action(
                    milestone_index,
                    SETTING_SETTERS[field],
                    {"on": bool(targets[0][field])},
                    f"official_setting_target:{field}",
                )
            )
        if not changed:
            blockers.append(f"m{milestone_index}:setting_target_has_no_invertible_change")
        return actions, blockers

    if namespace == "contact" and measure == "removal_similarity":
        for row in targets:
            person_id = row.get("person_id")
            if person_id is None or _find_row(reference_state["contact"], "person_id", person_id) is None:
                blockers.append(f"m{milestone_index}:contact_removal_not_unique")
                continue
            actions.append(
                _new_action(
                    milestone_index,
                    "remove_contact",
                    {"person_id": person_id},
                    "official_contact_removal",
                )
            )
        return actions, blockers

    if namespace == "contact" and measure == "update_similarity":
        allowed = {"name", "phone_number", "relationship", "is_self"}
        for target in targets:
            person_id = target.get("person_id")
            before = _find_row(reference_state["contact"], "person_id", person_id)
            if person_id is None or before is None:
                blockers.append(f"m{milestone_index}:contact_update_not_unique")
                continue
            changes = {
                key: value
                for key, value in target.items()
                if key in allowed and before.get(key) != value
            }
            if not changes:
                blockers.append(f"m{milestone_index}:contact_update_has_no_change:{person_id}")
                continue
            actions.append(
                _new_action(
                    milestone_index,
                    "modify_contact",
                    {"person_id": person_id, **changes},
                    "official_contact_update",
                )
            )
        return actions, blockers

    if namespace == "contact" and measure == "addition_similarity":
        for target in targets:
            arguments = {
                key: target[key]
                for key in ("name", "phone_number", "relationship", "is_self")
                if key in target
            }
            if "name" not in arguments or "phone_number" not in arguments:
                blockers.append(f"m{milestone_index}:contact_addition_missing_required_field")
                continue
            actions.append(
                _new_action(milestone_index, "add_contact", arguments, "official_contact_addition")
            )
        return actions, blockers

    if namespace == "messaging" and measure == "addition_similarity":
        for target in targets:
            phone = target.get("recipient_phone_number")
            content = target.get("content")
            if not isinstance(phone, str) or not isinstance(content, str):
                blockers.append(f"m{milestone_index}:message_addition_missing_required_field")
                continue
            actions.append(
                _new_action(
                    milestone_index,
                    "send_message_with_phone_number",
                    {"phone_number": phone, "content": content},
                    "official_message_addition",
                )
            )
        return actions, blockers

    return [], [f"m{milestone_index}:unsupported_state_inverse:{namespace}:{measure}"]


def extract_official_plan(name: str, family: str, scenario: Any) -> OfficialPlan:
    matcher = scenario.evaluation.milestone_matcher
    milestones = list(matcher.milestones)
    order = deterministic_topological_order(len(milestones), matcher.edge_list or [])
    current_state = _initial_symbolic_state(scenario)
    states_by_milestone: dict[int, dict[str, list[dict[str, Any]]]] = {}
    events: list[OfficialEvent] = []
    blockers: list[str] = []

    for milestone_index in order:
        milestone = milestones[milestone_index]
        milestone_actions: list[OfficialAction] = []
        milestone_messages: list[dict[str, Any]] = []
        for constraint_index, constraint in enumerate(milestone.snapshot_constraints):
            measure = _base_callable_name(constraint.snapshot_constraint)
            if measure == "guardrail_similarity":
                continue
            namespace = _enum_text(constraint.database_namespace).lower()
            target = constraint.target_dataframe
            if target is None:
                blockers.append(f"m{milestone_index}:c{constraint_index}:missing_target")
                continue
            target_rows = [_jsonable(row) for row in target.to_dicts()]
            if namespace == "sandbox":
                for row in target_rows:
                    if row.get("tool_trace") is not None:
                        if measure != "snapshot_similarity":
                            blockers.append(
                                f"m{milestone_index}:c{constraint_index}:dependent_tool_trace"
                            )
                            continue
                        action, blocker = _trace_action(milestone_index, row["tool_trace"])
                        if blocker:
                            blockers.append(blocker)
                        elif action is not None:
                            milestone_actions.append(action)
                    elif (
                        _enum_text(row.get("sender")) == "AGENT"
                        and _enum_text(row.get("recipient")) == "USER"
                        and isinstance(row.get("content"), str)
                        and measure == "snapshot_similarity"
                    ):
                        milestone_messages.append(
                            {
                                "sender": "AGENT",
                                "recipient": "USER",
                                "content": row["content"],
                            }
                        )
                    else:
                        blockers.append(
                            f"m{milestone_index}:c{constraint_index}:unsupported_sandbox_target:{measure}"
                        )
                continue

            reference_index = constraint.reference_milestone_node_index
            reference_state = (
                states_by_milestone.get(int(reference_index), current_state)
                if reference_index is not None
                else current_state
            )
            inferred, new_blockers = _state_actions(
                milestone_index,
                namespace,
                measure,
                target_rows,
                reference_state,
            )
            milestone_actions.extend(inferred)
            blockers.extend(f"c{constraint_index}:{item}" for item in new_blockers)

        unique_actions: dict[str, OfficialAction] = {}
        for action in milestone_actions:
            unique_actions.setdefault(canonical_json(action.to_dict()), action)
        milestone_actions = sorted(
            unique_actions.values(),
            key=lambda action: (action.tool, canonical_json(action.arguments), action.source),
        )
        for action in milestone_actions:
            _apply_symbolic_action(current_state, action)
            events.append(
                OfficialEvent(
                    key=f"action:{action.key}",
                    milestone_index=milestone_index,
                    kind="action",
                    action=action,
                )
            )
        for message_index, message in enumerate(milestone_messages):
            events.append(
                OfficialEvent(
                    key=f"message:m{milestone_index}:{message_index}",
                    milestone_index=milestone_index,
                    kind="message",
                    message=message,
                )
            )
        states_by_milestone[milestone_index] = copy.deepcopy(current_state)

    if family == "multiple_user_turn":
        blockers.append("requires_frozen_user_turn_schedule")
    if len([event for event in events if event.action is not None]) < 2:
        blockers.append("fewer_than_two_gold_actions")
    if scenario.evaluation.minefield_matcher.milestones:
        blockers.append("minefield_only_or_minefield_schedule_not_inverted")
    return OfficialPlan(
        scenario_name=name,
        family=family,
        events=tuple(events),
        blockers=tuple(sorted(set(blockers))),
        milestone_order=tuple(order),
    )


def scenario_inventory_row(name: str, family: str, scenario: Any) -> dict[str, Any]:
    allowed = sorted(set(scenario.starting_context.tool_allow_list or []))
    local = local_tool_names()
    plan = extract_official_plan(name, family, scenario)
    categories = sorted(_enum_text(category) for category in scenario.categories)
    matcher = scenario.evaluation.milestone_matcher
    minefield_matcher = scenario.evaluation.minefield_matcher
    summary = {
        "scenario_name": name,
        "template_group": name[:-4] if name.endswith("_alt") else name,
        "is_alt_view": name.endswith("_alt"),
        "family": family,
        "categories": categories,
        "tool_allow_list": allowed,
        "module_local": set(allowed).issubset(local),
        "v0_narrow_local": set(allowed).issubset(V0_NARROW_TOOLS),
        "milestone_count": len(matcher.milestones),
        "minefield_count": len(minefield_matcher.milestones),
        "milestone_edges": [list(edge) for edge in matcher.edge_list or []],
        "minefield_edges": [list(edge) for edge in minefield_matcher.edge_list or []],
        "plan": plan.to_dict(),
    }
    summary["inverse_ready"] = not plan.blockers
    summary["decision_candidate_count"] = len(plan.actions)
    summary["scenario_digest"] = digest_json(summary)
    return summary


def _ordered_events(plan: OfficialPlan, first_action_key: str) -> list[OfficialEvent]:
    selected = [
        event
        for event in plan.events
        if event.action is not None and event.action.key == first_action_key
    ]
    if len(selected) != 1:
        raise ValueError(f"Unknown or duplicate candidate key: {first_action_key}")
    return [selected[0], *[event for event in plan.events if event.key != selected[0].key]]


def _context_digest(context: Any) -> str:
    serialized = context.to_dict(serialize_console=False)
    serialized.pop("interactive_console", None)
    return digest_json(_jsonable(serialized))


def execute_official_plan(
    scenario: Any,
    plan: OfficialPlan,
    first_action_key: str,
) -> tuple[Any, list[dict[str, Any]]]:
    execution = importlib.import_module("tool_sandbox.common.execution_context")
    context = copy.deepcopy(scenario.starting_context)
    records: list[dict[str, Any]] = []
    events = _ordered_events(plan, first_action_key)
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
            records.append({"event_key": event.key, "kind": "message", "ok": True})
            continue

        action = event.action
        context.add_to_database(
            execution.DatabaseNamespace.SANDBOX,
            [
                {
                    "sender": execution.RoleType.EXECUTION_ENVIRONMENT,
                    "recipient": execution.RoleType.AGENT,
                    "content": "",
                }
            ],
        )
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
                    }
                ),
            )
            with ExitStack() as stack:
                stack.enter_context(patch.object(datetime_module, "datetime", FrozenDateTime))
                for module in tool_modules:
                    if hasattr(module, "uuid4"):
                        stack.enter_context(patch.object(module, "uuid4", lambda: deterministic_id))
                try:
                    with execution.new_context(context):
                        result = function(**copy.deepcopy(action.arguments))
                except Exception as exc:
                    error = f"{type(exc).__name__}:{exc}"
        records.append(
            {
                "event_key": event.key,
                "kind": "action",
                "action": action.to_dict(),
                "ok": error is None,
                "result": _jsonable(result),
                "error": error,
            }
        )
    return context, records


def evaluate_official_candidate(
    scenario: Any,
    plan: OfficialPlan,
    first_action_key: str,
) -> dict[str, Any]:
    context, execution_records = execute_official_plan(scenario, plan, first_action_key)
    result = scenario.evaluation.evaluate(
        execution_context=context,
        max_turn_count=scenario.max_messages,
    )
    return {
        "first_action_key": first_action_key,
        "similarity": float(result.similarity),
        "milestone_similarity": float(result.milestone_similarity),
        "minefield_similarity": float(result.minefield_similarity),
        "milestone_mapping": _jsonable(dict(result.milestone_mapping)),
        "minefield_mapping": _jsonable(dict(result.minefield_mapping)),
        "turn_count": int(result.turn_count),
        "execution_records": execution_records,
        "context_digest": _context_digest(context),
    }


def audit_prototype(name: str, family: str, scenario: Any) -> dict[str, Any]:
    plan = extract_official_plan(name, family, scenario)
    branches: list[dict[str, Any]] = []
    nondeterminism: list[str] = []
    for action in plan.actions:
        first = evaluate_official_candidate(scenario, plan, action.key)
        second = evaluate_official_candidate(scenario, plan, action.key)
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
    return {
        "scenario_name": name,
        "family": family,
        "plan": plan.to_dict(),
        "canonical_first_action_key": canonical_key,
        "branches": branches,
        "utility_levels": levels,
        "real_utility_gap": gap,
        "determinism_passed": not nondeterminism,
        "passed": not blockers,
        "blockers": sorted(set(blockers)),
    }
