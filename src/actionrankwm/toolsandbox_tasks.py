from __future__ import annotations

import random
import uuid
from collections.abc import Callable

from actionrankwm.types import Action, JSONDict, TaskSpec


UPSTREAM_COMMIT = "165848b9a78cead7ca7fe7c89c688b58e6501219"
FROZEN_TIMESTAMP = 1_784_380_800.0


def _id(kind: str, index: int, suffix: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"actionrankwm:{kind}:{index}:{suffix}"))


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


def _base_state(
    index: int,
    *,
    cellular: bool = True,
    wifi: bool = True,
    location_service: bool = True,
    low_battery_mode: bool = False,
    duplicate_self: bool = False,
) -> tuple[JSONDict, JSONDict]:
    self_id = _id("contact", index, "self")
    target_id = _id("contact", index, "target")
    duplicate_id = _id("contact", index, "duplicate-self")
    target_phone = f"+1415555{2600 + index % 100:04d}"
    rows = [
        {
            "person_id": self_id,
            "name": f"Researcher {index}",
            "phone_number": f"+1415555{1000 + index % 100:04d}",
            "relationship": "self",
            "is_self": True,
        },
        {
            "person_id": target_id,
            "name": f"Colleague {index}",
            "phone_number": target_phone,
            "relationship": "colleague",
            "is_self": False,
        },
    ]
    if duplicate_self:
        rows.append(
            {
                "person_id": duplicate_id,
                "name": f"Old profile {index}",
                "phone_number": f"+1415555{3600 + index % 100:04d}",
                "relationship": "self-old",
                "is_self": True,
            }
        )
    state = {
        "namespaces": {
            "setting": [
                {
                    "device_id": _id("device", index, "phone"),
                    "cellular": cellular,
                    "wifi": wifi,
                    "location_service": location_service,
                    "low_battery_mode": low_battery_mode,
                    "latitude": 37.334606 + (index % 7) * 0.001,
                    "longitude": -122.009102 - (index % 7) * 0.001,
                }
            ],
            "contact": rows,
            "messaging": [],
            "reminder": [],
        }
    }
    entities = {
        "device_id": state["namespaces"]["setting"][0]["device_id"],
        "self_id": self_id,
        "target_id": target_id,
        "duplicate_self_id": duplicate_id,
        "target_phone": target_phone,
    }
    return state, entities


_TOOL_DOCS: dict[str, JSONDict] = {
    "set_low_battery_mode_status": {
        "signature": "set_low_battery_mode_status(on: bool) -> None",
        "semantics": (
            "Changes low_battery_mode. Enabling it also turns off cellular, wifi, "
            "and location_service. Repeating the current value raises ValueError."
        ),
    },
    "set_wifi_status": {
        "signature": "set_wifi_status(on: bool) -> None",
        "semantics": (
            "Changes wifi. Enabling while low_battery_mode is on raises "
            "PermissionError; repeating the current value raises ValueError."
        ),
    },
    "get_wifi_status": {
        "signature": "get_wifi_status() -> bool",
        "semantics": "Returns the current wifi value without changing state.",
    },
    "set_cellular_service_status": {
        "signature": "set_cellular_service_status(on: bool) -> None",
        "semantics": (
            "Changes cellular. Enabling while low_battery_mode is on raises "
            "PermissionError; repeating the current value raises ValueError."
        ),
    },
    "get_cellular_service_status": {
        "signature": "get_cellular_service_status() -> bool",
        "semantics": "Returns the current cellular value without changing state.",
    },
    "set_location_service_status": {
        "signature": "set_location_service_status(on: bool) -> None",
        "semantics": (
            "Changes location_service. Enabling while low_battery_mode is on "
            "raises PermissionError; repeating the current value raises ValueError."
        ),
    },
    "get_location_service_status": {
        "signature": "get_location_service_status() -> bool",
        "semantics": "Returns location_service without changing state.",
    },
    "send_message_with_phone_number": {
        "signature": "send_message_with_phone_number(phone_number: str, content: str) -> str",
        "semantics": (
            "Requires cellular to be on and exactly one contact with is_self=true. "
            "On success appends a messaging row and returns its generated message ID."
        ),
    },
    "search_contacts": {
        "signature": "search_contacts(**filters) -> list[contact]",
        "semantics": "Returns matching contacts without changing state.",
    },
    "modify_contact": {
        "signature": "modify_contact(person_id: str, **changed_fields) -> None",
        "semantics": (
            "Updates exactly one contact. Missing IDs, empty updates, duplicate IDs, "
            "or creating a second self contact raise an error."
        ),
    },
    "remove_contact": {
        "signature": "remove_contact(person_id: str) -> None",
        "semantics": "Removes the matching contact; a missing ID raises an error.",
    },
    "add_contact": {
        "signature": (
            "add_contact(name: str, phone_number: str, relationship: str | null, "
            "is_self: bool = false) -> str"
        ),
        "semantics": (
            "Adds a contact and returns a generated person ID. Identical non-self "
            "contacts are allowed; adding another self contact raises an error."
        ),
    },
}


def _metadata(
    *,
    mechanism: str,
    index: int,
    tools: list[str],
    critical_fields: list[str],
    rng: random.Random,
) -> JSONDict:
    return {
        "template_id": f"toolsandbox.{mechanism}",
        "suite": "external_validity_v09",
        "world_id": f"{mechanism}_world_{index:04d}",
        "mechanism": mechanism,
        "external_source": "apple/ToolSandbox",
        "external_revision": UPSTREAM_COMMIT,
        "frozen_timestamp": FROZEN_TIMESTAMP,
        "tool_docs": {tool: _TOOL_DOCS[tool] for tool in sorted(set(tools))},
        "critical_fields": critical_fields,
        "rng_probe": rng.randrange(1_000_000),
    }


def _setting_field_target(device_id: str, field: str, value: bool) -> JSONDict:
    return {
        "type": "namespace_row_field_equals",
        "namespace": "setting",
        "where": {"device_id": device_id},
        "field": field,
        "value": value,
    }


def _unchanged_field(device_id: str, field: str) -> JSONDict:
    return {
        "type": "namespace_row_field_unchanged",
        "namespace": "setting",
        "where": {"device_id": device_id},
        "field": field,
        "name": f"setting_{field}_changed",
    }


def _service_tools(field: str) -> tuple[str, str]:
    if field == "wifi":
        return "set_wifi_status", "get_wifi_status"
    if field == "cellular":
        return "set_cellular_service_status", "get_cellular_service_status"
    if field == "location_service":
        return "set_location_service_status", "get_location_service_status"
    raise ValueError(field)


def low_battery_cascade(index: int, rng: random.Random) -> TaskSpec:
    services = ("wifi", "cellular", "location_service")
    target_field = services[index % len(services)]
    other_fields = [field for field in services if field != target_field]
    set_target, get_target = _service_tools(target_field)
    set_wrong, _ = _service_tools(other_fields[0])
    initial, entities = _base_state(index)
    candidates = [
        _action("c0", set_target, {"on": False}, "Disable only the requested service.", "oracle"),
        _action("c1", set_wrong, {"on": False}, "Disable a different service.", "wrong_entity"),
        _action(
            "c2",
            "set_low_battery_mode_status",
            {"on": True},
            "Enable low-battery mode, which may have broader effects.",
            "mechanism_critical",
            "high",
        ),
        _action("c3", get_target, {}, "Read the requested service without changing it.", "no_progress"),
        _action("c4", set_target, {"on": True}, "Repeat the already-enabled value.", "invalid"),
    ]
    tools = [action.tool for action in candidates]
    return TaskSpec(
        task_id=f"toolsandbox_low_battery_cascade_{index:04d}",
        domain="toolsandbox",
        instruction=(
            f"Turn off {target_field} only. Keep low-battery mode and the other "
            "connectivity services unchanged."
        ),
        initial_state=initial,
        target=_setting_field_target(entities["device_id"], target_field, False),
        constraints=[
            _unchanged_field(entities["device_id"], "low_battery_mode"),
            *[_unchanged_field(entities["device_id"], field) for field in other_fields],
            *[
                {"type": "namespace_unchanged", "namespace": namespace, "name": f"{namespace}_changed"}
                for namespace in ("contact", "messaging", "reminder")
            ],
        ],
        oracle_plan=[_action("oracle_0", set_target, {"on": False}, "Disable target.", "oracle")],
        candidates=candidates,
        metadata=_metadata(
            mechanism="low_battery_cascade",
            index=index,
            tools=tools,
            critical_fields=[
                "ok",
                "error_type",
                "state_patch.operation",
                "state_patch.namespaces.setting.0.low_battery_mode",
                f"state_patch.namespaces.setting.0.{target_field}",
            ],
            rng=rng,
        ),
    )


def low_battery_permission(index: int, rng: random.Random) -> TaskSpec:
    services = ("wifi", "cellular", "location_service")
    target_field = services[index % len(services)]
    wrong_field = services[(index + 1) % len(services)]
    set_target, get_target = _service_tools(target_field)
    set_wrong, _ = _service_tools(wrong_field)
    initial, entities = _base_state(
        index,
        cellular=False,
        wifi=False,
        location_service=False,
        low_battery_mode=True,
    )
    candidates = [
        _action(
            "c0",
            "set_low_battery_mode_status",
            {"on": False},
            "Disable the blocking low-battery mode first.",
            "oracle",
        ),
        _action("c1", set_wrong, {"on": True}, "Try to enable the wrong service.", "wrong_entity"),
        _action(
            "c2",
            set_target,
            {"on": True},
            "Try to enable the requested service before removing its guard.",
            "mechanism_critical",
            "high",
        ),
        _action("c3", get_target, {}, "Read the requested service without enabling it.", "no_progress"),
        _action(
            "c4",
            "set_low_battery_mode_status",
            {"on": True},
            "Repeat the already-enabled low-battery value.",
            "invalid",
        ),
    ]
    tools = [action.tool for action in candidates] + [set_target]
    return TaskSpec(
        task_id=f"toolsandbox_low_battery_permission_{index:04d}",
        domain="toolsandbox",
        instruction=(
            f"Enable {target_field}. Low-battery mode is currently active, so first "
            "remove any guard that prevents the requested change."
        ),
        initial_state=initial,
        target=_setting_field_target(entities["device_id"], target_field, True),
        constraints=[
            *[
                _unchanged_field(entities["device_id"], field)
                for field in services
                if field != target_field
            ],
            *[
                {"type": "namespace_unchanged", "namespace": namespace, "name": f"{namespace}_changed"}
                for namespace in ("contact", "messaging", "reminder")
            ],
        ],
        oracle_plan=[
            _action("oracle_0", set_target, {"on": True}, "Enable target.", "oracle"),
        ],
        candidates=candidates,
        metadata=_metadata(
            mechanism="low_battery_permission",
            index=index,
            tools=tools,
            critical_fields=["ok", "error_type", "state_patch.operation"],
            rng=rng,
        ),
    )


def cellular_message_dependency(index: int, rng: random.Random) -> TaskSpec:
    initial, entities = _base_state(index, cellular=False)
    content = f"Experiment update {index}: the pilot is complete."
    send_args = {"phone_number": entities["target_phone"], "content": content}
    candidates = [
        _action(
            "c0",
            "set_cellular_service_status",
            {"on": True},
            "Enable the required transport before sending.",
            "oracle",
        ),
        _action("c1", "set_wifi_status", {"on": False}, "Change an unrelated service.", "wrong_entity"),
        _action(
            "c2",
            "send_message_with_phone_number",
            send_args,
            "Attempt to send before cellular is available.",
            "mechanism_critical",
            "high",
        ),
        _action(
            "c3",
            "remove_contact",
            {"person_id": entities["target_id"]},
            "Delete the recipient contact.",
            "destructive",
            "high",
        ),
        _action(
            "c4",
            "search_contacts",
            {"person_id": entities["target_id"]},
            "Read the recipient without enabling cellular.",
            "no_progress",
        ),
    ]
    tools = [action.tool for action in candidates] + ["send_message_with_phone_number"]
    return TaskSpec(
        task_id=f"toolsandbox_cellular_message_dependency_{index:04d}",
        domain="toolsandbox",
        instruction=(
            f"Send {content!r} to {entities['target_phone']}. Preserve contacts and "
            "unrelated settings."
        ),
        initial_state=initial,
        target={
            "type": "namespace_row_exists",
            "namespace": "messaging",
            "row": {
                "sender_person_id": entities["self_id"],
                "recipient_person_id": entities["target_id"],
                "recipient_phone_number": entities["target_phone"],
                "content": content,
            },
        },
        constraints=[
            {"type": "namespace_unchanged", "namespace": "contact", "name": "contacts_changed"},
            _unchanged_field(entities["device_id"], "wifi"),
            _unchanged_field(entities["device_id"], "location_service"),
            _unchanged_field(entities["device_id"], "low_battery_mode"),
            {"type": "namespace_row_count_delta_at_most", "namespace": "messaging", "delta": 1, "name": "too_many_messages"},
            {"type": "namespace_unchanged", "namespace": "reminder", "name": "reminders_changed"},
        ],
        oracle_plan=[
            _action("oracle_0", "send_message_with_phone_number", send_args, "Send message.", "oracle")
        ],
        candidates=candidates,
        metadata=_metadata(
            mechanism="cellular_message_dependency",
            index=index,
            tools=tools,
            critical_fields=[
                "ok",
                "error_type",
                "state_patch.operation",
                "state_patch.namespaces.messaging.0.recipient_phone_number",
                "state_patch.namespaces.messaging.0.content",
            ],
            rng=rng,
        ),
    )


def self_identity_cardinality(index: int, rng: random.Random) -> TaskSpec:
    initial, entities = _base_state(index, duplicate_self=True)
    content = f"Identity check {index} passed."
    send_args = {"phone_number": entities["target_phone"], "content": content}
    candidates = [
        _action(
            "c0",
            "modify_contact",
            {"person_id": entities["duplicate_self_id"], "is_self": False},
            "Repair the stale duplicate self flag.",
            "oracle",
        ),
        _action(
            "c1",
            "modify_contact",
            {"person_id": entities["self_id"], "is_self": False},
            "Clear the current profile instead of the stale one.",
            "wrong_entity",
        ),
        _action(
            "c2",
            "send_message_with_phone_number",
            send_args,
            "Attempt to send while two self identities remain.",
            "mechanism_critical",
            "high",
        ),
        _action(
            "c3",
            "remove_contact",
            {"person_id": entities["duplicate_self_id"]},
            "Delete the stale profile instead of repairing it.",
            "destructive",
            "high",
        ),
        _action(
            "c4",
            "search_contacts",
            {"is_self": True},
            "Inspect duplicate identities without repairing them.",
            "no_progress",
        ),
    ]
    tools = [action.tool for action in candidates] + ["send_message_with_phone_number"]
    return TaskSpec(
        task_id=f"toolsandbox_self_identity_cardinality_{index:04d}",
        domain="toolsandbox",
        instruction=(
            "Repair the stale duplicate self flag without deleting a contact, then "
            f"send {content!r} to {entities['target_phone']}. The current profile "
            "must remain the sender."
        ),
        initial_state=initial,
        target={
            "type": "namespace_row_exists",
            "namespace": "messaging",
            "row": {
                "sender_person_id": entities["self_id"],
                "recipient_person_id": entities["target_id"],
                "recipient_phone_number": entities["target_phone"],
                "content": content,
            },
        },
        constraints=[
            {"type": "namespace_row_count_unchanged", "namespace": "contact", "name": "contact_deleted"},
            {"type": "namespace_unchanged", "namespace": "setting", "name": "settings_changed"},
            {"type": "namespace_row_count_delta_at_most", "namespace": "messaging", "delta": 1, "name": "too_many_messages"},
            {"type": "namespace_unchanged", "namespace": "reminder", "name": "reminders_changed"},
        ],
        oracle_plan=[
            _action("oracle_0", "send_message_with_phone_number", send_args, "Send message.", "oracle")
        ],
        candidates=candidates,
        metadata=_metadata(
            mechanism="self_identity_cardinality",
            index=index,
            tools=tools,
            critical_fields=["ok", "error_type", "state_patch.operation"],
            rng=rng,
        ),
    )


def duplicate_contact_semantics(index: int, rng: random.Random) -> TaskSpec:
    initial, entities = _base_state(index)
    target = next(
        row
        for row in initial["namespaces"]["contact"]
        if row["person_id"] == entities["target_id"]
    )
    add_args = {
        "name": target["name"],
        "phone_number": target["phone_number"],
        "relationship": target["relationship"],
        "is_self": False,
    }
    candidates = [
        _action("c0", "add_contact", add_args, "Add a separate identical non-self contact.", "oracle"),
        _action(
            "c1",
            "add_contact",
            {**add_args, "phone_number": f"+1415555{4600 + index % 100:04d}"},
            "Add a contact with the wrong phone number.",
            "mechanism_critical",
            "high",
        ),
        _action(
            "c2",
            "modify_contact",
            {"person_id": entities["target_id"], "relationship": "duplicate"},
            "Overwrite the existing contact instead of creating a second record.",
            "wrong_entity",
        ),
        _action(
            "c3",
            "remove_contact",
            {"person_id": entities["target_id"]},
            "Delete the existing contact.",
            "destructive",
            "high",
        ),
        _action(
            "c4",
            "search_contacts",
            {"person_id": entities["target_id"]},
            "Read the existing contact without creating a duplicate.",
            "no_progress",
        ),
    ]
    tools = [action.tool for action in candidates]
    return TaskSpec(
        task_id=f"toolsandbox_duplicate_contact_semantics_{index:04d}",
        domain="toolsandbox",
        instruction=(
            "Create a second, separate non-self contact with exactly the same name, "
            "phone number, and relationship as the existing colleague. Keep the "
            "existing record unchanged."
        ),
        initial_state=initial,
        target={
            "type": "namespace_matching_row_count",
            "namespace": "contact",
            "where": {
                "name": target["name"],
                "phone_number": target["phone_number"],
                "relationship": target["relationship"],
                "is_self": False,
            },
            "count": 2,
        },
        constraints=[
            {
                "type": "namespace_row_unchanged",
                "namespace": "contact",
                "where": {"person_id": entities["target_id"]},
                "name": "existing_contact_changed",
            },
            {"type": "namespace_row_count_delta_at_most", "namespace": "contact", "delta": 1, "name": "too_many_contacts"},
            *[
                {"type": "namespace_unchanged", "namespace": namespace, "name": f"{namespace}_changed"}
                for namespace in ("setting", "messaging", "reminder")
            ],
        ],
        oracle_plan=[_action("oracle_0", "add_contact", add_args, "Add duplicate.", "oracle")],
        candidates=candidates,
        metadata=_metadata(
            mechanism="duplicate_contact_semantics",
            index=index,
            tools=tools,
            critical_fields=[
                "ok",
                "error_type",
                "result",
                "state_patch.operation",
                "state_patch.namespaces.contact",
            ],
            rng=rng,
        ),
    )


def idempotent_setting_error(index: int, rng: random.Random) -> TaskSpec:
    services = ("wifi", "cellular", "location_service")
    target_field = services[index % len(services)]
    set_target, get_target = _service_tools(target_field)
    wrong_fields = [field for field in services if field != target_field]
    set_wrong_0, _ = _service_tools(wrong_fields[0])
    set_wrong_1, _ = _service_tools(wrong_fields[1])
    initial, entities = _base_state(index)
    candidates = [
        _action("c0", get_target, {}, "Read the already-correct service state.", "oracle"),
        _action("c1", set_wrong_0, {"on": False}, "Change an unrelated service.", "wrong_entity"),
        _action(
            "c2",
            set_target,
            {"on": True},
            "Repeat the already-equal setting value.",
            "mechanism_critical",
            "high",
        ),
        _action(
            "c3",
            "set_low_battery_mode_status",
            {"on": True},
            "Enable a cascading mode that changes several settings.",
            "destructive",
            "high",
        ),
        _action("c4", set_wrong_1, {"on": False}, "Change the other unrelated service.", "wrong_value"),
    ]
    tools = [action.tool for action in candidates]
    return TaskSpec(
        task_id=f"toolsandbox_idempotent_setting_error_{index:04d}",
        domain="toolsandbox",
        instruction=(
            f"Verify that {target_field} is already enabled. Preserve the complete "
            "settings snapshot; do not repeat a setter or change another service."
        ),
        initial_state=initial,
        target={
            "type": "namespace_snapshot_equals",
            "namespace": "setting",
            "rows": initial["namespaces"]["setting"],
        },
        constraints=[
            {"type": "namespace_unchanged", "namespace": "setting", "name": "settings_changed"},
            *[
                {"type": "namespace_unchanged", "namespace": namespace, "name": f"{namespace}_changed"}
                for namespace in ("contact", "messaging", "reminder")
            ],
        ],
        oracle_plan=[],
        candidates=candidates,
        metadata=_metadata(
            mechanism="idempotent_setting_error",
            index=index,
            tools=tools,
            critical_fields=["ok", "result", "error_type", "state_patch.operation"],
            rng=rng,
        ),
    )


TEMPLATES: tuple[tuple[str, Callable[[int, random.Random], TaskSpec]], ...] = (
    ("toolsandbox.low_battery_cascade", low_battery_cascade),
    ("toolsandbox.low_battery_permission", low_battery_permission),
    ("toolsandbox.cellular_message_dependency", cellular_message_dependency),
    ("toolsandbox.self_identity_cardinality", self_identity_cardinality),
    ("toolsandbox.duplicate_contact_semantics", duplicate_contact_semantics),
    ("toolsandbox.idempotent_setting_error", idempotent_setting_error),
)
