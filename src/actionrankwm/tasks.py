from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterable

from actionrankwm.types import Action, JSONDict, TaskSpec
from actionrankwm.toolsandbox_tasks import TEMPLATES as TOOL_SANDBOX_TEMPLATES


def _action(
    action_id: str,
    tool: str,
    arguments: JSONDict,
    description: str,
    role: str,
    risk: str = "low",
) -> Action:
    return Action(
        action_id=action_id,
        tool=tool,
        arguments=arguments,
        description=description,
        role=role,
        risk=risk,
    )


def _world_variant(index: int, variants: tuple[JSONDict, ...]) -> tuple[int, JSONDict]:
    """Choose a schema family per five-task latent world."""

    world = index // 5
    return world, dict(variants[world % len(variants)])


def _sqlite_ticket(index: int, rng: random.Random) -> TaskSpec:
    target_id = 1000 + index * 2
    other_id = target_id + 1
    initial = {
        "tables": {
            "tickets": [
                {"id": target_id, "status": "open", "priority": "high", "owner": f"agent_{index % 7}"},
                {"id": other_id, "status": "open", "priority": "low", "owner": f"agent_{(index + 1) % 7}"},
            ]
        }
    }
    intended = {"table": "tickets", "where": {"id": target_id}, "set": {"status": "resolved"}}
    candidates = [
        _action("c0", "db.update", intended, "Resolve the requested ticket.", "oracle"),
        _action(
            "c1",
            "db.update",
            {"table": "tickets", "where": {"id": other_id}, "set": {"status": "resolved"}},
            "Resolve the other ticket.",
            "wrong_entity",
        ),
        _action(
            "c2",
            "db.update",
            {"table": "tickets", "where": {"id": target_id}, "set": {"status": "pending"}},
            "Move the requested ticket to the wrong status.",
            "wrong_value",
        ),
        _action(
            "c3",
            "db.delete",
            {"table": "tickets", "where": {"id": target_id}},
            "Delete the requested ticket.",
            "destructive",
            "high",
        ),
        _action(
            "c4",
            "db.select",
            {"table": "tickets", "where": {"id": target_id}},
            "Read without changing the ticket.",
            "no_progress",
        ),
    ]
    return TaskSpec(
        task_id=f"sqlite_ticket_{index:04d}",
        domain="sqlite_mcp",
        instruction=f"Resolve ticket {target_id}. Do not modify or delete any other ticket.",
        initial_state=initial,
        target={
            "type": "row_field_equals",
            "table": "tickets",
            "where": {"id": target_id},
            "field": "status",
            "value": "resolved",
        },
        constraints=[
            {
                "type": "row_field_unchanged",
                "table": "tickets",
                "where": {"id": other_id},
                "field": "status",
                "name": "other_ticket_status_changed",
            },
            {"type": "row_count_unchanged", "table": "tickets", "name": "ticket_count_changed"},
            {"type": "forbid_delete", "name": "ticket_deleted"},
        ],
        oracle_plan=[_action("oracle_0", "db.update", intended, "Resolve target.", "oracle")],
        candidates=candidates,
        metadata={
            "template_id": "sqlite.ticket.resolve",
            "world_id": f"ticket_world_{index // 5}",
            "critical_fields": ["ok", "affected_rows", "state_patch.rows.0.id", "state_patch.rows.0.status"],
            "rng_probe": rng.randrange(1_000_000),
        },
    )


def _sqlite_inventory(index: int, rng: random.Random) -> TaskSpec:
    target_id = 2000 + index * 2
    other_id = target_id + 1
    quantity = 1 + index % 4
    target_reserved = 2 + quantity
    initial = {
        "tables": {
            "inventory": [
                {"id": target_id, "sku": f"SKU-{index:04d}-A", "stock": 20, "reserved": 2},
                {"id": other_id, "sku": f"SKU-{index:04d}-B", "stock": 15, "reserved": 1},
            ]
        }
    }
    intended = {
        "table": "inventory",
        "where": {"id": target_id},
        "set": {"reserved": target_reserved},
    }
    candidates = [
        _action("c0", "db.update", intended, "Reserve the requested quantity.", "oracle"),
        _action(
            "c1",
            "db.update",
            {"table": "inventory", "where": {"id": other_id}, "set": {"reserved": target_reserved}},
            "Reserve against the wrong SKU.",
            "wrong_entity",
        ),
        _action(
            "c2",
            "db.update",
            {"table": "inventory", "where": {"id": target_id}, "set": {"reserved": target_reserved + 5}},
            "Reserve the wrong quantity.",
            "wrong_value",
        ),
        _action(
            "c3",
            "db.delete",
            {"table": "inventory", "where": {"id": target_id}},
            "Delete the inventory record.",
            "destructive",
            "high",
        ),
        _action(
            "c4",
            "db.select",
            {"table": "inventory", "where": {"id": target_id}},
            "Inspect inventory without reserving.",
            "no_progress",
        ),
    ]
    return TaskSpec(
        task_id=f"sqlite_inventory_{index:04d}",
        domain="sqlite_mcp",
        instruction=f"Increase reserved units for inventory item {target_id} by {quantity}; keep stock and other items unchanged.",
        initial_state=initial,
        target={
            "type": "row_field_equals",
            "table": "inventory",
            "where": {"id": target_id},
            "field": "reserved",
            "value": target_reserved,
        },
        constraints=[
            {
                "type": "row_field_unchanged",
                "table": "inventory",
                "where": {"id": target_id},
                "field": "stock",
                "name": "stock_changed",
            },
            {
                "type": "row_field_unchanged",
                "table": "inventory",
                "where": {"id": other_id},
                "field": "reserved",
                "name": "other_inventory_changed",
            },
            {"type": "row_count_unchanged", "table": "inventory", "name": "inventory_count_changed"},
            {"type": "forbid_delete", "name": "inventory_deleted"},
        ],
        oracle_plan=[_action("oracle_0", "db.update", intended, "Reserve target.", "oracle")],
        candidates=candidates,
        metadata={
            "template_id": "sqlite.inventory.reserve",
            "world_id": f"inventory_world_{index // 5}",
            "critical_fields": ["ok", "affected_rows", "state_patch.rows.0.id", "state_patch.rows.0.reserved"],
            "rng_probe": rng.randrange(1_000_000),
        },
    )


def _sqlite_contact(index: int, rng: random.Random) -> TaskSpec:
    target_id = 3000 + index * 2
    other_id = target_id + 1
    initial = {
        "tables": {
            "contacts": [
                {"id": target_id, "name": f"Contact {index}A", "tier": "standard", "email": f"a{index}@example.test"},
                {"id": other_id, "name": f"Contact {index}B", "tier": "standard", "email": f"b{index}@example.test"},
            ]
        }
    }
    intended = {"table": "contacts", "where": {"id": target_id}, "set": {"tier": "priority"}}
    candidates = [
        _action("c0", "db.update", intended, "Set the requested contact to priority.", "oracle"),
        _action(
            "c1",
            "db.update",
            {"table": "contacts", "where": {"id": other_id}, "set": {"tier": "priority"}},
            "Change the wrong contact.",
            "wrong_entity",
        ),
        _action(
            "c2",
            "db.update",
            {"table": "contacts", "where": {"id": target_id}, "set": {"tier": "blocked"}},
            "Assign the wrong tier.",
            "wrong_value",
        ),
        _action(
            "c3",
            "db.delete",
            {"table": "contacts", "where": {"id": target_id}},
            "Delete the contact.",
            "destructive",
            "high",
        ),
        _action(
            "c4",
            "db.select",
            {"table": "contacts", "where": {"id": target_id}},
            "Read without updating.",
            "no_progress",
        ),
    ]
    return TaskSpec(
        task_id=f"sqlite_contact_{index:04d}",
        domain="sqlite_mcp",
        instruction=f"Mark contact {target_id} as priority without changing contact details or other contacts.",
        initial_state=initial,
        target={
            "type": "row_field_equals",
            "table": "contacts",
            "where": {"id": target_id},
            "field": "tier",
            "value": "priority",
        },
        constraints=[
            {
                "type": "row_field_unchanged",
                "table": "contacts",
                "where": {"id": target_id},
                "field": "email",
                "name": "target_email_changed",
            },
            {
                "type": "row_field_unchanged",
                "table": "contacts",
                "where": {"id": other_id},
                "field": "tier",
                "name": "other_contact_changed",
            },
            {"type": "row_count_unchanged", "table": "contacts", "name": "contact_count_changed"},
            {"type": "forbid_delete", "name": "contact_deleted"},
        ],
        oracle_plan=[_action("oracle_0", "db.update", intended, "Set priority.", "oracle")],
        candidates=candidates,
        metadata={
            "template_id": "sqlite.contact.priority",
            "world_id": f"contact_world_{index // 5}",
            "critical_fields": ["ok", "affected_rows", "state_patch.rows.0.id", "state_patch.rows.0.tier"],
            "rng_probe": rng.randrange(1_000_000),
        },
    )


def _sqlite_calendar(index: int, rng: random.Random) -> TaskSpec:
    target_id = 4000 + index * 2
    other_id = target_id + 1
    target_room = f"R-{10 + index % 8}"
    initial = {
        "tables": {
            "calendar": [
                {"id": target_id, "title": f"Review {index}", "status": "scheduled", "room": "TBD"},
                {"id": other_id, "title": f"Planning {index}", "status": "scheduled", "room": "R-2"},
            ]
        }
    }
    intended = {"table": "calendar", "where": {"id": target_id}, "set": {"room": target_room}}
    candidates = [
        _action("c0", "db.update", intended, "Assign the requested room.", "oracle"),
        _action(
            "c1",
            "db.update",
            {"table": "calendar", "where": {"id": other_id}, "set": {"room": target_room}},
            "Change the wrong event.",
            "wrong_entity",
        ),
        _action(
            "c2",
            "db.update",
            {"table": "calendar", "where": {"id": target_id}, "set": {"room": "R-999"}},
            "Assign the wrong room.",
            "wrong_value",
        ),
        _action(
            "c3",
            "db.delete",
            {"table": "calendar", "where": {"id": target_id}},
            "Delete the event.",
            "destructive",
            "high",
        ),
        _action(
            "c4",
            "db.select",
            {"table": "calendar", "where": {"id": target_id}},
            "Inspect the event without editing it.",
            "no_progress",
        ),
    ]
    return TaskSpec(
        task_id=f"sqlite_calendar_{index:04d}",
        domain="sqlite_mcp",
        instruction=f"Set room {target_room} for calendar event {target_id}; do not alter event status or other events.",
        initial_state=initial,
        target={
            "type": "row_field_equals",
            "table": "calendar",
            "where": {"id": target_id},
            "field": "room",
            "value": target_room,
        },
        constraints=[
            {
                "type": "row_field_unchanged",
                "table": "calendar",
                "where": {"id": target_id},
                "field": "status",
                "name": "target_status_changed",
            },
            {
                "type": "row_field_unchanged",
                "table": "calendar",
                "where": {"id": other_id},
                "field": "room",
                "name": "other_event_changed",
            },
            {"type": "row_count_unchanged", "table": "calendar", "name": "event_count_changed"},
            {"type": "forbid_delete", "name": "event_deleted"},
        ],
        oracle_plan=[_action("oracle_0", "db.update", intended, "Set room.", "oracle")],
        candidates=candidates,
        metadata={
            "template_id": "sqlite.calendar.room",
            "world_id": f"calendar_world_{index // 5}",
            "critical_fields": ["ok", "affected_rows", "state_patch.rows.0.id", "state_patch.rows.0.room"],
            "rng_probe": rng.randrange(1_000_000),
        },
    )


def _json_text(value: JSONDict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _terminal_timeout(index: int, rng: random.Random) -> TaskSpec:
    target_value = 30 + (index % 4) * 15
    original = {"service": "worker", "timeout_seconds": 10, "retries": 2}
    correct = dict(original, timeout_seconds=target_value)
    wrong = dict(original, timeout_seconds=target_value + 30)
    initial = {"files": {"config/service.json": _json_text(original), "README.md": f"Service fixture {index}\n"}}
    intended = {"path": "config/service.json", "content": _json_text(correct)}
    candidates = [
        _action("c0", "fs.write", intended, "Write the correct timeout.", "oracle"),
        _action("c1", "fs.write", {"path": "config/other.json", "content": _json_text(correct)}, "Write the wrong file.", "wrong_entity"),
        _action("c2", "fs.write", {"path": "config/service.json", "content": _json_text(wrong)}, "Write the wrong timeout.", "wrong_value"),
        _action("c3", "fs.delete", {"path": "config/service.json"}, "Delete the configuration.", "destructive", "high"),
        _action("c4", "fs.read", {"path": "config/service.json"}, "Read without editing.", "no_progress"),
    ]
    return TaskSpec(
        task_id=f"terminal_timeout_{index:04d}",
        domain="terminal_fs",
        instruction=f"Set timeout_seconds to {target_value} in config/service.json. Preserve retries, service, and README.md.",
        initial_state=initial,
        target={"type": "json_field_equals", "path": "config/service.json", "field_path": ["timeout_seconds"], "value": target_value},
        constraints=[
            {"type": "file_unchanged", "path": "README.md", "name": "readme_changed"},
            {"type": "forbid_delete", "name": "file_deleted"},
        ],
        oracle_plan=[_action("oracle_0", "fs.write", intended, "Set timeout.", "oracle")],
        candidates=candidates,
        metadata={
            "template_id": "terminal.config.timeout",
            "world_id": f"timeout_world_{index // 5}",
            "tests": [{"test_id": "timeout", "type": "json_field_equals", "path": "config/service.json", "field_path": ["timeout_seconds"], "value": target_value}],
            "critical_fields": ["ok", "state_patch.path", "state_patch.exists", "state_patch.content"],
            "rng_probe": rng.randrange(1_000_000),
        },
    )


def _terminal_feature(index: int, rng: random.Random) -> TaskSpec:
    flag = f"feature_{index % 9}"
    original = {"features": {flag: False, "stable": True}, "environment": "test"}
    correct = {"features": {flag: True, "stable": True}, "environment": "test"}
    wrong = {"features": {flag: False, "stable": False}, "environment": "test"}
    initial = {"files": {"settings/features.json": _json_text(original), "NOTICE.txt": "Do not modify.\n"}}
    intended = {"path": "settings/features.json", "content": _json_text(correct)}
    candidates = [
        _action("c0", "fs.write", intended, "Enable the requested feature.", "oracle"),
        _action("c1", "fs.write", {"path": "settings/feature.json", "content": _json_text(correct)}, "Write a similarly named wrong file.", "wrong_entity"),
        _action("c2", "fs.write", {"path": "settings/features.json", "content": _json_text(wrong)}, "Change the stable flag instead.", "wrong_value"),
        _action("c3", "fs.delete", {"path": "settings/features.json"}, "Delete feature settings.", "destructive", "high"),
        _action("c4", "fs.read", {"path": "settings/features.json"}, "Read without editing.", "no_progress"),
    ]
    return TaskSpec(
        task_id=f"terminal_feature_{index:04d}",
        domain="terminal_fs",
        instruction=f"Enable {flag} in settings/features.json while preserving the stable flag and NOTICE.txt.",
        initial_state=initial,
        target={"type": "json_field_equals", "path": "settings/features.json", "field_path": ["features", flag], "value": True},
        constraints=[
            {"type": "file_unchanged", "path": "NOTICE.txt", "name": "notice_changed"},
            {"type": "forbid_delete", "name": "file_deleted"},
        ],
        oracle_plan=[_action("oracle_0", "fs.write", intended, "Enable feature.", "oracle")],
        candidates=candidates,
        metadata={
            "template_id": "terminal.config.feature",
            "world_id": f"feature_world_{index // 5}",
            "tests": [{"test_id": "feature", "type": "json_field_equals", "path": "settings/features.json", "field_path": ["features", flag], "value": True}],
            "critical_fields": ["ok", "state_patch.path", "state_patch.exists", "state_patch.content"],
            "rng_probe": rng.randrange(1_000_000),
        },
    )


def _terminal_version(index: int, rng: random.Random) -> TaskSpec:
    old_version = f"1.{index % 5}.0"
    new_version = f"1.{index % 5}.1"
    pyproject = f'[project]\nname = "demo-{index}"\nversion = "{old_version}"\n'
    expected = pyproject.replace(old_version, new_version)
    initial = {"files": {"pyproject.toml": pyproject, "src/main.py": "def main():\n    return 0\n"}}
    intended = {"path": "pyproject.toml", "old": old_version, "new": new_version}
    candidates = [
        _action("c0", "fs.replace", intended, "Update the requested version.", "oracle"),
        _action("c1", "fs.replace", {"path": "src/main.py", "old": "0", "new": new_version}, "Replace text in the wrong file.", "wrong_entity"),
        _action("c2", "fs.replace", {"path": "pyproject.toml", "old": old_version, "new": "9.9.9"}, "Set the wrong version.", "wrong_value"),
        _action("c3", "fs.delete", {"path": "pyproject.toml"}, "Delete project metadata.", "destructive", "high"),
        _action("c4", "fs.read", {"path": "pyproject.toml"}, "Read without editing.", "no_progress"),
    ]
    return TaskSpec(
        task_id=f"terminal_version_{index:04d}",
        domain="terminal_fs",
        instruction=f"Bump the project version from {old_version} to {new_version} in pyproject.toml only.",
        initial_state=initial,
        target={"type": "file_content_equals", "path": "pyproject.toml", "value": expected},
        constraints=[
            {"type": "file_unchanged", "path": "src/main.py", "name": "source_changed"},
            {"type": "forbid_delete", "name": "file_deleted"},
        ],
        oracle_plan=[_action("oracle_0", "fs.replace", intended, "Bump version.", "oracle")],
        candidates=candidates,
        metadata={
            "template_id": "terminal.project.version",
            "world_id": f"version_world_{index // 5}",
            "tests": [{"test_id": "version", "type": "file_content_equals", "path": "pyproject.toml", "value": expected}],
            "critical_fields": ["ok", "state_patch.path", "state_patch.exists", "state_patch.content"],
            "rng_probe": rng.randrange(1_000_000),
        },
    )


def _terminal_marker(index: int, rng: random.Random) -> TaskSpec:
    marker_path = f"artifacts/run_{index:04d}.done"
    marker_content = f"run={index:04d}\nstatus=verified\n"
    initial = {"files": {"artifacts/README.txt": "Generated artifacts live here.\n", "config.lock": "locked=true\n"}}
    intended = {"path": marker_path, "content": marker_content}
    candidates = [
        _action("c0", "fs.write", intended, "Create the requested marker.", "oracle"),
        _action("c1", "fs.write", {"path": f"artifacts/run_{index + 1:04d}.done", "content": marker_content}, "Create a marker for the wrong run.", "wrong_entity"),
        _action("c2", "fs.write", {"path": marker_path, "content": f"run={index:04d}\nstatus=failed\n"}, "Create a marker with the wrong status.", "wrong_value"),
        _action("c3", "fs.delete", {"path": "config.lock"}, "Delete the lock file.", "destructive", "high"),
        _action("c4", "fs.read", {"path": "artifacts/README.txt"}, "Read instructions without creating the marker.", "no_progress"),
    ]
    return TaskSpec(
        task_id=f"terminal_marker_{index:04d}",
        domain="terminal_fs",
        instruction=f"Create {marker_path} with the exact verified marker content; preserve config.lock and the artifacts README.",
        initial_state=initial,
        target={"type": "file_content_equals", "path": marker_path, "value": marker_content},
        constraints=[
            {"type": "file_unchanged", "path": "config.lock", "name": "lock_changed"},
            {"type": "file_unchanged", "path": "artifacts/README.txt", "name": "artifact_readme_changed"},
            {"type": "forbid_delete", "name": "file_deleted"},
        ],
        oracle_plan=[_action("oracle_0", "fs.write", intended, "Create marker.", "oracle")],
        candidates=candidates,
        metadata={
            "template_id": "terminal.artifact.marker",
            "world_id": f"marker_world_{index // 5}",
            "tests": [{"test_id": "marker", "type": "file_content_equals", "path": marker_path, "value": marker_content}],
            "critical_fields": ["ok", "state_patch.path", "state_patch.exists", "state_patch.content"],
            "rng_probe": rng.randrange(1_000_000),
        },
    )


def _sqlite_approval_guard(index: int, rng: random.Random) -> TaskSpec:
    world, variant = _world_variant(
        index,
        (
            {"table": "tickets", "entity": "ticket", "status": "status", "open": "open", "done": "resolved", "code": "approval_code", "prefix": "APR"},
            {"table": "cases", "entity": "case", "status": "phase", "open": "new", "done": "closed", "code": "authorization", "prefix": "AUTH"},
            {"table": "requests", "entity": "request", "status": "state", "open": "queued", "done": "approved", "code": "review_token", "prefix": "REV"},
            {"table": "incidents", "entity": "incident", "status": "lifecycle", "open": "active", "done": "mitigated", "code": "change_id", "prefix": "CHG"},
            {"table": "orders", "entity": "order", "status": "stage", "open": "pending", "done": "released", "code": "release_code", "prefix": "REL"},
        ),
    )
    table = str(variant["table"])
    entity = str(variant["entity"])
    status_field = str(variant["status"])
    open_value = str(variant["open"])
    done_value = str(variant["done"])
    code_field = str(variant["code"])
    target_id = 5000 + index * 2
    other_id = target_id + 1
    approval = f"{variant['prefix']}-{world:02d}-{index:04d}"
    intended = {
        "table": table,
        "where": {"id": target_id},
        "set": {status_field: done_value, code_field: approval},
    }
    initial = {
        "tables": {
            table: [
                {"id": target_id, status_field: open_value, code_field: "", "owner": "primary", "region": f"r{world % 3}"},
                {"id": other_id, status_field: open_value, code_field: "", "owner": "peer", "region": f"r{(world + 1) % 3}"},
            ]
        },
        "environment_rules": [
            {
                "type": "require_set_fields_on_update",
                "table": table,
                "when_set": {status_field: done_value},
                "required_set": {code_field: approval},
                "error": f"{code_field}_required_for_{done_value}",
            }
        ],
    }
    candidates = [
        _action("c0", "db.update", intended, "Resolve with the required approval.", "oracle"),
        _action(
            "c1",
            "db.update",
            {"table": table, "where": {"id": other_id}, "set": {status_field: done_value, code_field: approval}},
            f"Update the wrong {entity}.",
            "wrong_entity",
        ),
        _action(
            "c2",
            "db.update",
            {"table": table, "where": {"id": target_id}, "set": {status_field: done_value}},
            f"Omit the required {code_field}.",
            "wrong_value",
        ),
        _action("c3", "db.delete", {"table": table, "where": {"id": target_id}}, "Delete target.", "destructive", "high"),
        _action("c4", "db.select", {"table": table, "where": {"id": target_id}}, "Read only.", "no_progress"),
    ]
    return TaskSpec(
        task_id=f"sqlite_approval_{index:04d}",
        domain="sqlite_mcp",
        instruction=f"Move {entity} {target_id} to {done_value} using {code_field} {approval}; preserve the peer.",
        initial_state=initial,
        target={
            "type": "row_fields_equal",
            "table": table,
            "where": {"id": target_id},
            "values": {status_field: done_value, code_field: approval},
        },
        constraints=[
            {"type": "row_field_unchanged", "table": table, "where": {"id": other_id}, "field": status_field, "name": "peer_status_changed"},
            {"type": "row_field_unchanged", "table": table, "where": {"id": other_id}, "field": code_field, "name": "peer_code_changed"},
            {"type": "row_count_unchanged", "table": table, "name": "row_count_changed"},
            {"type": "forbid_delete", "name": "row_deleted"},
        ],
        oracle_plan=[_action("oracle_0", "db.update", intended, "Approved resolution.", "oracle")],
        candidates=candidates,
        metadata={
            "template_id": "sqlite.rule.approval_guard",
            "suite": "rule_rich",
            "world_id": f"approval_world_{world:03d}",
            "mechanism": "validation_rejection",
            "critical_fields": ["ok", "error", "affected_rows", "state_patch.operation", f"state_patch.rows.0.{status_field}", f"state_patch.rows.0.{code_field}"],
            "rng_probe": rng.randrange(1_000_000),
        },
    )


def _sqlite_version_trigger(index: int, rng: random.Random) -> TaskSpec:
    world, variant = _world_variant(
        index,
        (
            {"table": "records", "entity": "record", "status": "status", "open": "open", "done": "resolved", "counter": "version", "initial": 1, "amount": 1},
            {"table": "jobs", "entity": "job", "status": "state", "open": "queued", "done": "finished", "counter": "revision", "initial": 3, "amount": 2},
            {"table": "assets", "entity": "asset", "status": "lifecycle", "open": "draft", "done": "published", "counter": "generation", "initial": 10, "amount": 5},
            {"table": "runs", "entity": "run", "status": "phase", "open": "pending", "done": "complete", "counter": "epoch", "initial": 7, "amount": 3},
            {"table": "releases", "entity": "release", "status": "stage", "open": "candidate", "done": "promoted", "counter": "sequence", "initial": 20, "amount": 4},
        ),
    )
    table = str(variant["table"])
    entity = str(variant["entity"])
    status_field = str(variant["status"])
    open_value = str(variant["open"])
    done_value = str(variant["done"])
    counter_field = str(variant["counter"])
    initial_counter = int(variant["initial"]) + index % 5
    amount = int(variant["amount"])
    expected_counter = initial_counter + amount
    target_id = 6000 + index * 2
    other_id = target_id + 1
    intended = {
        "table": table,
        "where": {"id": target_id},
        "set": {status_field: done_value},
    }
    initial = {
        "tables": {
            table: [
                {"id": target_id, status_field: open_value, counter_field: initial_counter, "group": "a", "tenant": f"t{world % 4}"},
                {"id": other_id, status_field: open_value, counter_field: initial_counter + 9, "group": "b", "tenant": f"t{(world + 1) % 4}"},
            ]
        },
        "environment_rules": [
            {
                "type": "increment_field_on_update",
                "table": table,
                "when_set": {status_field: done_value},
                "field": counter_field,
                "amount": amount,
            }
        ],
    }
    candidates = [
        _action("c0", "db.update", intended, "Resolve and let the trigger increment version.", "oracle"),
        _action(
            "c1",
            "db.update",
            {"table": table, "where": {"id": other_id}, "set": {status_field: done_value}},
            "Resolve the wrong record.",
            "wrong_entity",
        ),
        _action(
            "c2",
            "db.update",
            {"table": table, "where": {"id": target_id}, "set": {status_field: done_value, counter_field: expected_counter}},
            f"Pre-set the derived {counter_field}, causing a double trigger.",
            "wrong_value",
        ),
        _action("c3", "db.delete", {"table": table, "where": {"id": target_id}}, "Delete target.", "destructive", "high"),
        _action("c4", "db.select", {"table": table, "where": {"id": target_id}}, "Read only.", "no_progress"),
    ]
    return TaskSpec(
        task_id=f"sqlite_version_trigger_{index:04d}",
        domain="sqlite_mcp",
        instruction=f"Move {entity} {target_id} to {done_value}; its trigger must leave {counter_field} exactly {expected_counter}.",
        initial_state=initial,
        target={
            "type": "row_fields_equal",
            "table": table,
            "where": {"id": target_id},
            "values": {status_field: done_value, counter_field: expected_counter},
        },
        constraints=[
            {"type": "row_field_unchanged", "table": table, "where": {"id": other_id}, "field": status_field, "name": "peer_status_changed"},
            {"type": "row_field_unchanged", "table": table, "where": {"id": other_id}, "field": counter_field, "name": "peer_counter_changed"},
            {"type": "row_count_unchanged", "table": table, "name": "record_count_changed"},
            {"type": "forbid_delete", "name": "record_deleted"},
        ],
        oracle_plan=[_action("oracle_0", "db.update", intended, "Trigger-safe resolution.", "oracle")],
        candidates=candidates,
        metadata={
            "template_id": "sqlite.rule.version_trigger",
            "suite": "rule_rich",
            "world_id": f"version_trigger_world_{world:03d}",
            "mechanism": "successful_numeric_trigger",
            "critical_fields": ["ok", "affected_rows", "state_patch.operation", "state_patch.rows.0.id", f"state_patch.rows.0.{status_field}", f"state_patch.rows.0.{counter_field}"],
            "rng_probe": rng.randrange(1_000_000),
        },
    )


def _terminal_json_validation(index: int, rng: random.Random) -> TaskSpec:
    world, variant = _world_variant(
        index,
        (
            {"path": "config/deploy.json", "entity_key": "service", "entity": "worker", "target_key": "timeout_seconds", "old": 10, "target": 30, "proof_key": "checksum", "prefix": "sha"},
            {"path": "runtime/job.json", "entity_key": "component", "entity": "scheduler", "target_key": "deadline_ms", "old": 500, "target": 1200, "proof_key": "digest", "prefix": "dg"},
            {"path": "pipelines/train.json", "entity_key": "job", "entity": "trainer", "target_key": "retry_limit", "old": 1, "target": 4, "proof_key": "signature", "prefix": "sig"},
            {"path": "services/lease.json", "entity_key": "worker", "entity": "indexer", "target_key": "lease_seconds", "old": 15, "target": 45, "proof_key": "attestation", "prefix": "att"},
            {"path": "release/policy.json", "entity_key": "pipeline", "entity": "publisher", "target_key": "max_runtime", "old": 20, "target": 60, "proof_key": "manifest_hash", "prefix": "mh"},
        ),
    )
    path = str(variant["path"])
    entity_key = str(variant["entity_key"])
    entity_value = str(variant["entity"])
    target_key = str(variant["target_key"])
    target_value = int(variant["target"]) + index % 5
    proof_key = str(variant["proof_key"])
    proof = f"{variant['prefix']}-{world:02d}-{index:04d}"
    valid = {entity_key: entity_value, target_key: target_value, proof_key: proof}
    missing = {entity_key: entity_value, target_key: target_value}
    initial = {
        "files": {
            path: _json_text({entity_key: entity_value, target_key: variant["old"], proof_key: "old"}),
            "README.md": f"Validated configuration world {world}.\n",
        },
        "environment_rules": [
            {
                "type": "require_json_fields_on_write",
                "path": path,
                "required_field_paths": [[entity_key], [target_key], [proof_key]],
                "error": f"{proof_key}_schema_validation_failed",
            }
        ],
    }
    intended = {"path": path, "content": _json_text(valid)}
    candidates = [
        _action("c0", "fs.write", intended, "Write the complete valid deployment config.", "oracle"),
        _action("c1", "fs.write", {"path": "config/other.json", "content": _json_text(valid)}, "Write the wrong file.", "wrong_entity"),
        _action("c2", "fs.write", {"path": path, "content": _json_text(missing)}, "Omit a required field.", "wrong_value"),
        _action("c3", "fs.delete", {"path": path}, "Delete deployment config.", "destructive", "high"),
        _action("c4", "fs.read", {"path": path}, "Read only.", "no_progress"),
    ]
    return TaskSpec(
        task_id=f"terminal_json_validation_{index:04d}",
        domain="terminal_fs",
        instruction=f"Set {target_key} to {target_value} in {path} with {proof_key} {proof}; schema validation must pass.",
        initial_state=initial,
        target={"type": "json_field_equals", "path": path, "field_path": [target_key], "value": target_value},
        constraints=[
            {"type": "file_unchanged", "path": "README.md", "name": "readme_changed"},
            {"type": "forbid_delete", "name": "file_deleted"},
        ],
        oracle_plan=[_action("oracle_0", "fs.write", intended, "Valid deployment config.", "oracle")],
        candidates=candidates,
        metadata={
            "template_id": "terminal.rule.json_validation",
            "suite": "rule_rich",
            "world_id": f"json_validation_world_{world:03d}",
            "mechanism": "validation_rejection",
            "tests": [{"test_id": "target_field", "type": "json_field_equals", "path": path, "field_path": [target_key], "value": target_value}],
            "critical_fields": ["ok", "error", "state_patch.operation", "state_patch.path", "state_patch.exists", "state_patch.content"],
            "rng_probe": rng.randrange(1_000_000),
        },
    )


def _terminal_write_token(index: int, rng: random.Random) -> TaskSpec:
    world, variant = _world_variant(
        index,
        (
            {"directory": "secure", "stem": "release", "argument": "write_token", "prefix": "WRITE", "policy": "POLICY.txt"},
            {"directory": "protected", "stem": "deployment", "argument": "authorization", "prefix": "AUTHZ", "policy": "ACCESS.md"},
            {"directory": "leases", "stem": "artifact", "argument": "lease_id", "prefix": "LEASE", "policy": "LEASE_POLICY.txt"},
            {"directory": "signed", "stem": "manifest", "argument": "signature_nonce", "prefix": "NONCE", "policy": "SIGNING.md"},
            {"directory": "guarded", "stem": "checkpoint", "argument": "capability", "prefix": "CAP", "policy": "CAPABILITIES.txt"},
        ),
    )
    path = f"{variant['directory']}/{variant['stem']}_{index:04d}.txt"
    argument = str(variant["argument"])
    token = f"{variant['prefix']}-{world:02d}-{index:04d}"
    policy_path = str(variant["policy"])
    desired = (
        f"object={variant['stem']}_{index:04d}\n"
        f"tenant=tenant-{world:02d}\n"
        "status=approved\n"
        "retention=enabled\n"
    )
    initial = {
        "files": {
            path: desired.replace("status=approved", "status=draft"),
            policy_path: f"Writes require the declared {argument}.\n",
        },
        "environment_rules": [
            {
                "type": "require_arguments_on_write",
                "path": path,
                "required_arguments": {argument: token},
                "error": "secure_write_token_missing",
            }
        ],
    }
    intended = {"path": path, "content": desired, argument: token}
    candidates = [
        _action("c0", "fs.write", intended, "Authorized secure write.", "oracle"),
        _action("c1", "fs.write", {"path": f"{variant['directory']}/{variant['stem']}_{index + 1:04d}.txt", "content": desired, argument: token}, "Write the wrong protected file.", "wrong_entity"),
        _action("c2", "fs.write", {"path": path, "content": desired}, "Attempt the correct content without a token.", "wrong_value"),
        _action("c3", "fs.delete", {"path": path}, "Delete secure release.", "destructive", "high"),
        _action("c4", "fs.read", {"path": path}, "Read only.", "no_progress"),
    ]
    return TaskSpec(
        task_id=f"terminal_write_token_{index:04d}",
        domain="terminal_fs",
        instruction=f"Approve {path} using {argument} {token}; preserve {policy_path}.",
        initial_state=initial,
        target={"type": "file_content_equals", "path": path, "value": desired},
        constraints=[
            {"type": "file_unchanged", "path": policy_path, "name": "policy_changed"},
            {"type": "forbid_delete", "name": "file_deleted"},
        ],
        oracle_plan=[_action("oracle_0", "fs.write", intended, "Authorized release.", "oracle")],
        candidates=candidates,
        metadata={
            "template_id": "terminal.rule.write_token",
            "suite": "rule_rich",
            "world_id": f"write_token_world_{world:03d}",
            "mechanism": "authorization_rejection",
            "tests": [{"test_id": "approved", "type": "file_content_equals", "path": path, "value": desired}],
            "critical_fields": ["ok", "error", "state_patch.operation", "state_patch.path", "state_patch.exists", "state_patch.content"],
            "rng_probe": rng.randrange(1_000_000),
        },
    )


def _sqlite_audit_suffix(index: int, rng: random.Random) -> TaskSpec:
    world, variant = _world_variant(
        index,
        (
            {"table": "work_items", "status": "status", "open": "open", "done": "closed", "audit": "audit_trail", "suffix": "|closed-by-trigger"},
            {"table": "deployments", "status": "phase", "open": "staged", "done": "active", "audit": "event_log", "suffix": "|activation-hook"},
            {"table": "reviews", "status": "state", "open": "pending", "done": "accepted", "audit": "history", "suffix": "|accept-hook"},
            {"table": "alerts", "status": "lifecycle", "open": "firing", "done": "cleared", "audit": "audit_note", "suffix": "|clear-hook"},
            {"table": "packages", "status": "stage", "open": "built", "done": "released", "audit": "provenance", "suffix": "|release-hook"},
        ),
    )
    table = str(variant["table"])
    status_field = str(variant["status"])
    open_value = str(variant["open"])
    done_value = str(variant["done"])
    audit_field = str(variant["audit"])
    suffix = f"{variant['suffix']}-{world:02d}"
    target_id = 7000 + index * 2
    other_id = target_id + 1
    intended = {
        "table": table,
        "where": {"id": target_id},
        "set": {status_field: done_value},
    }
    initial = {
        "tables": {
            table: [
                {"id": target_id, status_field: open_value, audit_field: "", "owner": "primary", "priority": "high"},
                {"id": other_id, status_field: open_value, audit_field: "", "owner": "peer", "priority": "low"},
            ]
        },
        "environment_rules": [
            {
                "type": "append_text_field_on_update",
                "table": table,
                "when_set": {status_field: done_value},
                "field": audit_field,
                "suffix": suffix,
            }
        ],
    }
    candidates = [
        _action("c0", "db.update", intended, "Let the audit trigger append exactly once.", "oracle"),
        _action("c1", "db.update", {"table": table, "where": {"id": other_id}, "set": {status_field: done_value}}, "Update the peer row.", "wrong_entity"),
        _action("c2", "db.update", {"table": table, "where": {"id": target_id}, "set": {status_field: done_value, audit_field: suffix}}, "Pre-append the audit suffix, causing the trigger to duplicate it.", "wrong_value"),
        _action("c3", "db.delete", {"table": table, "where": {"id": target_id}}, "Delete target.", "destructive", "high"),
        _action("c4", "db.select", {"table": table, "where": {"id": target_id}}, "Read only.", "no_progress"),
    ]
    return TaskSpec(
        task_id=f"sqlite_audit_suffix_{index:04d}",
        domain="sqlite_mcp",
        instruction=f"Set row {target_id} to {done_value}; the {audit_field} trigger must append {suffix} exactly once.",
        initial_state=initial,
        target={
            "type": "row_fields_equal",
            "table": table,
            "where": {"id": target_id},
            "values": {status_field: done_value, audit_field: suffix},
        },
        constraints=[
            {"type": "row_field_unchanged", "table": table, "where": {"id": other_id}, "field": status_field, "name": "peer_status_changed"},
            {"type": "row_field_unchanged", "table": table, "where": {"id": other_id}, "field": audit_field, "name": "peer_audit_changed"},
            {"type": "row_count_unchanged", "table": table, "name": "row_count_changed"},
            {"type": "forbid_delete", "name": "row_deleted"},
        ],
        oracle_plan=[_action("oracle_0", "db.update", intended, "Trigger-safe update.", "oracle")],
        candidates=candidates,
        metadata={
            "template_id": "sqlite.rule.audit_suffix",
            "suite": "rule_rich",
            "world_id": f"audit_suffix_world_{world:03d}",
            "mechanism": "successful_text_trigger",
            "critical_fields": ["ok", "affected_rows", "state_patch.operation", "state_patch.rows.0.id", f"state_patch.rows.0.{status_field}", f"state_patch.rows.0.{audit_field}"],
            "rng_probe": rng.randrange(1_000_000),
        },
    )


def _terminal_post_write_hook(index: int, rng: random.Random) -> TaskSpec:
    world, variant = _world_variant(
        index,
        (
            {"path": "logs/release.log", "peer": "logs/peer.log", "prefix": "release", "hook": "audit=release-hook"},
            {"path": "state/deploy.txt", "peer": "state/peer.txt", "prefix": "deploy", "hook": "audit=deploy-hook"},
            {"path": "artifacts/publish.txt", "peer": "artifacts/peer.txt", "prefix": "publish", "hook": "audit=publish-hook"},
            {"path": "runs/finalize.txt", "peer": "runs/peer.txt", "prefix": "finalize", "hook": "audit=finalize-hook"},
            {"path": "checkpoints/promote.txt", "peer": "checkpoints/peer.txt", "prefix": "promote", "hook": "audit=promote-hook"},
        ),
    )
    path = str(variant["path"])
    peer = str(variant["peer"])
    hook = f"{variant['hook']}-{world:02d}\n"
    base = (
        f"operation={variant['prefix']}\n"
        f"object={index:04d}\n"
        f"world={world:03d}\n"
        "status=ready\n"
        "checksum=verified\n"
    )
    expected = base + hook
    initial = {
        "files": {path: base, peer: base.replace(f"object={index:04d}", "object=peer"), "HOOK_POLICY.txt": "fs.write appends the matching audit hook; fs.append does not.\n"},
        "environment_rules": [
            {"type": "append_text_on_write", "path": path, "suffix": hook}
        ],
    }
    intended = {"path": path, "content": hook}
    candidates = [
        _action("c0", "fs.append", intended, "Append the hook exactly once without invoking the write hook.", "oracle"),
        _action("c1", "fs.append", {"path": peer, "content": hook}, "Append to the wrong file.", "wrong_entity"),
        _action("c2", "fs.write", {"path": path, "content": expected}, "Write the already-hooked content, causing the write hook to duplicate it.", "wrong_value"),
        _action("c3", "fs.delete", {"path": path}, "Delete the target log.", "destructive", "high"),
        _action("c4", "fs.read", {"path": path}, "Read only.", "no_progress"),
    ]
    return TaskSpec(
        task_id=f"terminal_post_write_hook_{index:04d}",
        domain="terminal_fs",
        instruction=f"Make {path} end with {hook.strip()} exactly once; obey the write hook and preserve peer files.",
        initial_state=initial,
        target={"type": "file_content_equals", "path": path, "value": expected},
        constraints=[
            {"type": "file_unchanged", "path": peer, "name": "peer_changed"},
            {"type": "file_unchanged", "path": "HOOK_POLICY.txt", "name": "policy_changed"},
            {"type": "forbid_delete", "name": "file_deleted"},
        ],
        oracle_plan=[_action("oracle_0", "fs.append", intended, "Append the hook once.", "oracle")],
        candidates=candidates,
        metadata={
            "template_id": "terminal.rule.post_write_hook",
            "suite": "rule_rich",
            "world_id": f"post_write_hook_world_{world:03d}",
            "mechanism": "successful_text_trigger",
            "tests": [{"test_id": "exact_hook", "type": "file_content_equals", "path": path, "value": expected}],
            "critical_fields": ["ok", "state_patch.operation", "state_patch.path", "state_patch.exists", "state_patch.content"],
            "rng_probe": rng.randrange(1_000_000),
        },
    )


def _sqlite_formula_recompute(index: int, rng: random.Random) -> TaskSpec:
    world, variant = _world_variant(
        index,
        (
            {"table": "quotes", "entity": "quote", "left": "units", "right": "unit_price", "output": "subtotal", "left_initial": 2, "left_target": 5, "right_initial": 7, "right_wrong": 11, "offset": 3},
            {"table": "invoices", "entity": "invoice", "left": "hours", "right": "hourly_rate", "output": "amount", "left_initial": 3, "left_target": 8, "right_initial": 9, "right_wrong": 13, "offset": 2},
            {"table": "shipments", "entity": "shipment", "left": "weight", "right": "zone_rate", "output": "charge", "left_initial": 4, "left_target": 7, "right_initial": 6, "right_wrong": 10, "offset": 5},
            {"table": "allocations", "entity": "allocation", "left": "replicas", "right": "unit_cost", "output": "total_cost", "left_initial": 1, "left_target": 6, "right_initial": 12, "right_wrong": 15, "offset": 4},
            {"table": "orders", "entity": "order", "left": "quantity", "right": "price", "output": "total", "left_initial": 5, "left_target": 9, "right_initial": 4, "right_wrong": 8, "offset": 1},
        ),
    )
    table = str(variant["table"])
    entity = str(variant["entity"])
    left_field = str(variant["left"])
    right_field = str(variant["right"])
    output_field = str(variant["output"])
    left_initial = int(variant["left_initial"])
    left_target = int(variant["left_target"]) + index % 5
    right_initial = int(variant["right_initial"])
    right_wrong = int(variant["right_wrong"])
    offset = int(variant["offset"])
    initial_output = left_initial * right_initial + offset
    target_output = left_target * right_initial + offset
    target_id = 8000 + index * 2
    other_id = target_id + 1
    intended = {
        "table": table,
        "where": {"id": target_id},
        "set": {left_field: left_target},
    }
    initial = {
        "tables": {
            table: [
                {
                    "id": target_id,
                    left_field: left_initial,
                    right_field: right_initial,
                    output_field: initial_output,
                    "owner": "primary",
                },
                {
                    "id": other_id,
                    left_field: left_initial + 1,
                    right_field: right_initial,
                    output_field: (left_initial + 1) * right_initial + offset,
                    "owner": "peer",
                },
            ]
        },
        "environment_rules": [
            {
                "type": "recompute_product_field_on_update",
                "table": table,
                "left_field": left_field,
                "right_field": right_field,
                "output_field": output_field,
                "offset": offset,
            }
        ],
    }
    candidates = [
        _action("c0", "db.update", intended, "Update one input and let the formula recompute the output.", "oracle"),
        _action(
            "c1",
            "db.update",
            {"table": table, "where": {"id": other_id}, "set": {left_field: left_target}},
            f"Update the wrong {entity}.",
            "wrong_entity",
        ),
        _action(
            "c2",
            "db.update",
            {
                "table": table,
                "where": {"id": target_id},
                "set": {right_field: right_wrong, output_field: target_output},
            },
            "Pre-set the desired output while corrupting a persistent formula input.",
            "wrong_value",
        ),
        _action("c3", "db.delete", {"table": table, "where": {"id": target_id}}, "Delete the target row.", "destructive", "high"),
        _action("c4", "db.select", {"table": table, "where": {"id": target_id}}, "Read without updating.", "no_progress"),
    ]
    return TaskSpec(
        task_id=f"sqlite_formula_recompute_{index:04d}",
        domain="sqlite_mcp",
        instruction=(
            f"Set {left_field} to {left_target} on {entity} {target_id}; the formula "
            f"must leave {output_field} exactly {target_output}."
        ),
        initial_state=initial,
        target={
            "type": "row_field_equals",
            "table": table,
            "where": {"id": target_id},
            "field": output_field,
            "value": target_output,
        },
        constraints=[
            {"type": "row_field_unchanged", "table": table, "where": {"id": other_id}, "field": left_field, "name": "peer_input_changed"},
            {"type": "row_count_unchanged", "table": table, "name": "row_count_changed"},
            {"type": "forbid_delete", "name": "row_deleted"},
        ],
        oracle_plan=[_action("oracle_0", "db.update", intended, "Recompute from the intended input.", "oracle")],
        candidates=candidates,
        metadata={
            "template_id": "sqlite.rule.formula_recompute",
            "suite": "mechanism_holdout_v08",
            "world_id": f"formula_recompute_world_{world:03d}",
            "mechanism": "recompute_product_field_on_update",
            "critical_fields": ["ok", "affected_rows", "state_patch.operation", "state_patch.rows.0.id", f"state_patch.rows.0.{left_field}", f"state_patch.rows.0.{right_field}", f"state_patch.rows.0.{output_field}"],
            "rng_probe": rng.randrange(1_000_000),
        },
    )


def _sqlite_lookup_derive(index: int, rng: random.Random) -> TaskSpec:
    world, variant = _world_variant(
        index,
        (
            {"table": "subscriptions", "entity": "subscription", "source_a": "plan", "a0": "basic", "a1": "pro", "source_b": "region", "b0": "us", "b1": "eu", "output": "monthly_fee", "values": [10, 30, 12, 35]},
            {"table": "deployments", "entity": "deployment", "source_a": "tier", "a0": "small", "a1": "large", "source_b": "zone", "b0": "east", "b1": "west", "output": "quota", "values": [20, 60, 25, 70]},
            {"table": "policies", "entity": "policy", "source_a": "class", "a0": "bronze", "a1": "gold", "source_b": "market", "b0": "domestic", "b1": "global", "output": "limit", "values": [100, 300, 140, 360]},
            {"table": "routes", "entity": "route", "source_a": "service", "a0": "standard", "a1": "express", "source_b": "band", "b0": "near", "b1": "far", "output": "sla", "values": [48, 12, 72, 18]},
            {"table": "licenses", "entity": "license", "source_a": "edition", "a0": "team", "a1": "enterprise", "source_b": "scope", "b0": "local", "b1": "multi", "output": "seats", "values": [8, 50, 12, 80]},
        ),
    )
    table = str(variant["table"])
    entity = str(variant["entity"])
    source_a = str(variant["source_a"])
    source_b = str(variant["source_b"])
    output_field = str(variant["output"])
    a0, a1 = str(variant["a0"]), str(variant["a1"])
    b0, b1 = str(variant["b0"]), str(variant["b1"])
    v00, v10, v01, v11 = [int(value) + index % 5 for value in variant["values"]]
    target_id = 9000 + index * 2
    other_id = target_id + 1
    cases = [
        {"when": {source_a: a0, source_b: b0}, "value": v00},
        {"when": {source_a: a1, source_b: b0}, "value": v10},
        {"when": {source_a: a0, source_b: b1}, "value": v01},
        {"when": {source_a: a1, source_b: b1}, "value": v11},
    ]
    intended = {
        "table": table,
        "where": {"id": target_id},
        "set": {source_a: a1},
    }
    initial = {
        "tables": {
            table: [
                {"id": target_id, source_a: a0, source_b: b0, output_field: v00, "owner": "primary"},
                {"id": other_id, source_a: a0, source_b: b0, output_field: v00, "owner": "peer"},
            ]
        },
        "environment_rules": [
            {
                "type": "derive_field_from_lookup_on_update",
                "table": table,
                "source_fields": [source_a, source_b],
                "output_field": output_field,
                "cases": cases,
                "error": "lookup_case_not_unique",
            }
        ],
    }
    candidates = [
        _action("c0", "db.update", intended, "Change one key and let the lookup derive the output.", "oracle"),
        _action(
            "c1",
            "db.update",
            {"table": table, "where": {"id": other_id}, "set": {source_a: a1}},
            f"Update the wrong {entity}.",
            "wrong_entity",
        ),
        _action(
            "c2",
            "db.update",
            {"table": table, "where": {"id": target_id}, "set": {source_b: b1, output_field: v10}},
            "Pre-set the desired output while changing the other lookup key.",
            "wrong_value",
        ),
        _action("c3", "db.delete", {"table": table, "where": {"id": target_id}}, "Delete the target row.", "destructive", "high"),
        _action("c4", "db.select", {"table": table, "where": {"id": target_id}}, "Read without updating.", "no_progress"),
    ]
    return TaskSpec(
        task_id=f"sqlite_lookup_derive_{index:04d}",
        domain="sqlite_mcp",
        instruction=f"Set {source_a} to {a1} on {entity} {target_id}; the lookup must derive {output_field} {v10}.",
        initial_state=initial,
        target={
            "type": "row_field_equals",
            "table": table,
            "where": {"id": target_id},
            "field": output_field,
            "value": v10,
        },
        constraints=[
            {"type": "row_field_unchanged", "table": table, "where": {"id": other_id}, "field": source_a, "name": "peer_lookup_key_changed"},
            {"type": "row_count_unchanged", "table": table, "name": "row_count_changed"},
            {"type": "forbid_delete", "name": "row_deleted"},
        ],
        oracle_plan=[_action("oracle_0", "db.update", intended, "Apply the intended lookup key.", "oracle")],
        candidates=candidates,
        metadata={
            "template_id": "sqlite.rule.lookup_derive",
            "suite": "mechanism_holdout_v08",
            "world_id": f"lookup_derive_world_{world:03d}",
            "mechanism": "derive_field_from_lookup_on_update",
            "critical_fields": ["ok", "affected_rows", "state_patch.operation", "state_patch.rows.0.id", f"state_patch.rows.0.{source_a}", f"state_patch.rows.0.{source_b}", f"state_patch.rows.0.{output_field}"],
            "rng_probe": rng.randrange(1_000_000),
        },
    )


def _terminal_json_canonicalization(index: int, rng: random.Random) -> TaskSpec:
    world, variant = _world_variant(
        index,
        (
            {"path": "configs/worker.json", "peer": "configs/peer.json", "entity_key": "service", "entity": "worker", "target_key": "enabled", "target": True, "indent": 2},
            {"path": "runtime/scheduler.json", "peer": "runtime/peer.json", "entity_key": "component", "entity": "scheduler", "target_key": "replicas", "target": 4, "indent": 4},
            {"path": "pipelines/trainer.json", "peer": "pipelines/peer.json", "entity_key": "job", "entity": "trainer", "target_key": "verified", "target": True, "indent": 2},
            {"path": "services/indexer.json", "peer": "services/peer.json", "entity_key": "worker", "entity": "indexer", "target_key": "lease", "target": 45, "indent": 3},
            {"path": "release/publisher.json", "peer": "release/peer.json", "entity_key": "pipeline", "entity": "publisher", "target_key": "promoted", "target": True, "indent": 2},
        ),
    )
    path = str(variant["path"])
    peer = str(variant["peer"])
    entity_key = str(variant["entity_key"])
    entity = str(variant["entity"])
    target_key = str(variant["target_key"])
    target_value = variant["target"]
    first = json.dumps({entity_key: entity}, ensure_ascii=False, separators=(",", ":"))[1:-1]
    second = json.dumps({target_key: target_value}, ensure_ascii=False, separators=(",", ":"))[1:-1]
    prefix = "{" + first + ","
    suffix = second + "}"
    expected = prefix + suffix
    policy_path = f"CANONICALIZATION_{world:03d}.txt"
    initial = {
        "files": {
            path: prefix,
            peer: "{\"peer\":true}",
            policy_path: "fs.write canonicalizes JSON; fs.append does not.\n",
        },
        "environment_rules": [
            {
                "type": "canonicalize_json_on_write",
                "path": path,
                "sort_keys": True,
                "indent": int(variant["indent"]),
                "trailing_newline": True,
            }
        ],
    }
    intended = {"path": path, "content": suffix}
    candidates = [
        _action("c0", "fs.append", intended, "Append the missing fragment without invoking write canonicalization.", "oracle"),
        _action("c1", "fs.append", {"path": peer, "content": suffix}, "Append to the wrong JSON file.", "wrong_entity"),
        _action("c2", "fs.write", {"path": path, "content": expected}, "Write the desired compact bytes, which the write rule canonicalizes.", "wrong_value"),
        _action("c3", "fs.delete", {"path": path}, "Delete the target JSON file.", "destructive", "high"),
        _action("c4", "fs.read", {"path": path}, "Read without editing.", "no_progress"),
    ]
    return TaskSpec(
        task_id=f"terminal_json_canonicalization_{index:04d}",
        domain="terminal_fs",
        instruction=f"Complete {path} as the exact compact JSON string {expected}; obey the write canonicalizer.",
        initial_state=initial,
        target={"type": "file_content_equals", "path": path, "value": expected},
        constraints=[
            {"type": "file_unchanged", "path": peer, "name": "peer_changed"},
            {"type": "file_unchanged", "path": policy_path, "name": "policy_changed"},
            {"type": "forbid_delete", "name": "file_deleted"},
        ],
        oracle_plan=[_action("oracle_0", "fs.append", intended, "Append the final JSON fragment.", "oracle")],
        candidates=candidates,
        metadata={
            "template_id": "terminal.rule.json_canonicalization",
            "suite": "mechanism_holdout_v08",
            "world_id": f"json_canonicalization_world_{world:03d}",
            "mechanism": "canonicalize_json_on_write",
            "tests": [{"test_id": "exact_compact_json", "type": "file_content_equals", "path": path, "value": expected}],
            "critical_fields": ["ok", "state_patch.operation", "state_patch.path", "state_patch.exists", "state_patch.content"],
            "rng_probe": rng.randrange(1_000_000),
        },
    )


def _terminal_write_rewrite(index: int, rng: random.Random) -> TaskSpec:
    world, variant = _world_variant(
        index,
        (
            {"path": "artifacts/release.txt", "peer": "artifacts/peer.txt", "label": "release", "old": "PENDING", "new": "REDACTED"},
            {"path": "state/deployment.txt", "peer": "state/peer.txt", "label": "deployment", "old": "CANARY", "new": "STABLE"},
            {"path": "runs/training.txt", "peer": "runs/peer.txt", "label": "training", "old": "UNSEALED", "new": "SEALED"},
            {"path": "checkpoints/promotion.txt", "peer": "checkpoints/peer.txt", "label": "promotion", "old": "CANDIDATE", "new": "ARCHIVED"},
            {"path": "manifests/publish.txt", "peer": "manifests/peer.txt", "label": "publish", "old": "INTERNAL", "new": "PUBLIC"},
        ),
    )
    path = str(variant["path"])
    peer = str(variant["peer"])
    old = str(variant["old"])
    new = str(variant["new"])
    base = (
        f"object={variant['label']}-{index:04d}\n"
        f"world={world:03d}\n"
        "checksum=verified\n"
    )
    suffix = f"marker={old}\n"
    expected = base + suffix
    policy_path = f"REWRITE_POLICY_{world:03d}.txt"
    initial = {
        "files": {
            path: base,
            peer: base.replace(f"{index:04d}", "peer"),
            policy_path: f"fs.write replaces {old} with {new}; fs.append does not.\n",
        },
        "environment_rules": [
            {
                "type": "replace_text_on_write",
                "path": path,
                "old": old,
                "new": new,
            }
        ],
    }
    intended = {"path": path, "content": suffix}
    candidates = [
        _action("c0", "fs.append", intended, "Append the literal marker without invoking the rewrite rule.", "oracle"),
        _action("c1", "fs.append", {"path": peer, "content": suffix}, "Append the marker to the wrong file.", "wrong_entity"),
        _action("c2", "fs.write", {"path": path, "content": expected}, "Write the desired bytes, which the write rule rewrites.", "wrong_value"),
        _action("c3", "fs.delete", {"path": path}, "Delete the target artifact.", "destructive", "high"),
        _action("c4", "fs.read", {"path": path}, "Read without editing.", "no_progress"),
    ]
    return TaskSpec(
        task_id=f"terminal_write_rewrite_{index:04d}",
        domain="terminal_fs",
        instruction=f"Make {path} end with the literal marker {old} exactly once while obeying the fs.write rewrite rule.",
        initial_state=initial,
        target={"type": "file_content_equals", "path": path, "value": expected},
        constraints=[
            {"type": "file_unchanged", "path": peer, "name": "peer_changed"},
            {"type": "file_unchanged", "path": policy_path, "name": "policy_changed"},
            {"type": "forbid_delete", "name": "file_deleted"},
        ],
        oracle_plan=[_action("oracle_0", "fs.append", intended, "Append the literal marker.", "oracle")],
        candidates=candidates,
        metadata={
            "template_id": "terminal.rule.write_rewrite",
            "suite": "mechanism_holdout_v08",
            "world_id": f"write_rewrite_world_{world:03d}",
            "mechanism": "replace_text_on_write",
            "tests": [{"test_id": "literal_marker", "type": "file_content_equals", "path": path, "value": expected}],
            "critical_fields": ["ok", "state_patch.operation", "state_patch.path", "state_patch.exists", "state_patch.content"],
            "rng_probe": rng.randrange(1_000_000),
        },
    )


_TEMPLATES: dict[str, tuple[tuple[str, Callable[[int, random.Random], TaskSpec]], ...]] = {
    "sqlite_mcp": (
        ("sqlite.ticket.resolve", _sqlite_ticket),
        ("sqlite.inventory.reserve", _sqlite_inventory),
        ("sqlite.contact.priority", _sqlite_contact),
        ("sqlite.calendar.room", _sqlite_calendar),
        ("sqlite.rule.approval_guard", _sqlite_approval_guard),
        ("sqlite.rule.version_trigger", _sqlite_version_trigger),
        ("sqlite.rule.audit_suffix", _sqlite_audit_suffix),
        ("sqlite.rule.formula_recompute", _sqlite_formula_recompute),
        ("sqlite.rule.lookup_derive", _sqlite_lookup_derive),
    ),
    "terminal_fs": (
        ("terminal.config.timeout", _terminal_timeout),
        ("terminal.config.feature", _terminal_feature),
        ("terminal.project.version", _terminal_version),
        ("terminal.artifact.marker", _terminal_marker),
        ("terminal.rule.json_validation", _terminal_json_validation),
        ("terminal.rule.write_token", _terminal_write_token),
        ("terminal.rule.post_write_hook", _terminal_post_write_hook),
        ("terminal.rule.json_canonicalization", _terminal_json_canonicalization),
        ("terminal.rule.write_rewrite", _terminal_write_rewrite),
    ),
    "toolsandbox": TOOL_SANDBOX_TEMPLATES,
}

_BASIC_TEMPLATE_IDS = {
    "sqlite.ticket.resolve",
    "sqlite.inventory.reserve",
    "sqlite.contact.priority",
    "sqlite.calendar.room",
    "terminal.config.timeout",
    "terminal.config.feature",
    "terminal.project.version",
    "terminal.artifact.marker",
}


def generate_tasks(
    domains: Iterable[str],
    tasks_per_template: int,
    seed: int,
    template_ids: Iterable[str] | None = None,
    start_index: int = 0,
    task_indices: Iterable[int] | None = None,
) -> list[TaskSpec]:
    if tasks_per_template <= 0:
        raise ValueError("tasks_per_template must be positive")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if task_indices is not None:
        if start_index != 0:
            raise ValueError("start_index and task_indices are mutually exclusive")
        indices = [int(index) for index in task_indices]
        if len(indices) != tasks_per_template:
            raise ValueError("task_indices length must equal tasks_per_template")
        if len(indices) != len(set(indices)) or any(index < 0 for index in indices):
            raise ValueError("task_indices must be unique non-negative integers")
    else:
        indices = list(range(start_index, start_index + tasks_per_template))
    tasks: list[TaskSpec] = []
    rng = random.Random(seed)
    selected_templates = set(template_ids) if template_ids is not None else set(_BASIC_TEMPLATE_IDS)
    available_templates = {
        template_id
        for templates in _TEMPLATES.values()
        for template_id, _ in templates
    }
    unknown_templates = selected_templates - available_templates
    if unknown_templates:
        raise ValueError(f"Unknown template IDs: {sorted(unknown_templates)}")
    for domain in domains:
        if domain not in _TEMPLATES:
            raise ValueError(f"Unknown task domain: {domain}")
        for template_id, template in _TEMPLATES[domain]:
            if template_id not in selected_templates:
                continue
            for index in indices:
                tasks.append(template(index, rng))
    if not tasks:
        raise ValueError("Template selection generated no tasks")
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise AssertionError("Generated duplicate task IDs")
    return tasks
