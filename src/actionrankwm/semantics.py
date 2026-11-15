from __future__ import annotations

import copy
import json
from typing import Any, Mapping

from actionrankwm.types import JSONDict, TaskSpec


def _canonical_rows(rows: Any) -> list[JSONDict]:
    if not isinstance(rows, list):
        raise ValueError("namespace rows must be a list")
    normalized = [dict(row) for row in rows if isinstance(row, Mapping)]
    if len(normalized) != len(rows):
        raise ValueError("every namespace row must be an object")
    return sorted(normalized, key=lambda row: json.dumps(row, sort_keys=True, default=str))


def _matching_rows(rows: Any, where: Mapping[str, Any]) -> list[JSONDict]:
    return [
        row
        for row in _canonical_rows(rows)
        if all(row.get(key) == expected for key, expected in where.items())
    ]


def normalize_observation(value: Mapping[str, Any] | None) -> JSONDict | None:
    if value is None:
        return None
    current: Mapping[str, Any] = value
    if isinstance(current.get("environment_observation"), Mapping):
        current = current["environment_observation"]
    elif isinstance(current.get("observation"), Mapping):
        current = current["observation"]
    return dict(current)


def canonicalize_predicted_observation(
    task: TaskSpec, value: Mapping[str, Any] | None
) -> tuple[JSONDict | None, str | None]:
    """Canonicalize supported prediction shorthand without reading real outcomes.

    The repair may use only the task's initial state and fields emitted by the
    model. It is recorded separately so formatting robustness cannot masquerade
    as exact structured fidelity.
    """

    observation = normalize_observation(value)
    if observation is None:
        return None, None
    canonical = copy.deepcopy(observation)
    patch = canonical.get("state_patch")
    if not isinstance(patch, Mapping):
        return canonical, None
    mutable_patch = dict(patch)
    canonical["state_patch"] = mutable_patch
    if (
        task.domain == "sqlite_mcp"
        and mutable_patch.get("operation") == "update"
        and "rows" not in mutable_patch
        and isinstance(mutable_patch.get("set"), Mapping)
    ):
        table = str(mutable_patch.get("table", ""))
        where_value = mutable_patch.get("where", {})
        where = dict(where_value) if isinstance(where_value, Mapping) else {}
        updates = dict(mutable_patch["set"])
        source_rows = task.initial_state.get("tables", {}).get(table, [])
        resulting_rows = []
        for row in source_rows:
            if all(row.get(key) == expected for key, expected in where.items()):
                resulting = copy.deepcopy(row)
                resulting.update(updates)
                resulting_rows.append(resulting)
        mutable_patch["rows"] = resulting_rows
        return canonical, "sqlite_set_where_to_complete_rows"
    if (
        task.domain == "terminal_fs"
        and isinstance(mutable_patch.get("path"), str)
        and bool(mutable_patch.get("path"))
        and ("domain" not in mutable_patch or "operation" not in mutable_patch)
    ):
        repaired_patch = dict(mutable_patch)
        if "domain" not in repaired_patch:
            repaired_patch["domain"] = "terminal_fs"
        if "operation" not in repaired_patch:
            if repaired_patch.get("exists") is False:
                repaired_patch["operation"] = "delete"
            elif "content" in repaired_patch and repaired_patch.get("exists", True) is not False:
                repaired_patch["operation"] = "write"
            else:
                return canonical, None
        canonical["state_patch"] = repaired_patch
        return canonical, "terminal_single_file_fields_to_patch"
    if (
        task.domain == "terminal_fs"
        and "operation" not in mutable_patch
        and isinstance(mutable_patch.get("files"), Mapping)
    ):
        predicted_files: dict[str, str] = {}
        for path, raw_value in mutable_patch["files"].items():
            if isinstance(raw_value, Mapping):
                if raw_value.get("exists", True) is False:
                    continue
                if "content" not in raw_value:
                    return canonical, None
                predicted_files[str(path)] = str(raw_value["content"])
            else:
                predicted_files[str(path)] = str(raw_value)
        initial_files = {
            str(path): str(content)
            for path, content in task.initial_state.get("files", {}).items()
        }
        changed_paths = sorted(
            path
            for path in set(initial_files) | set(predicted_files)
            if initial_files.get(path) != predicted_files.get(path)
        )
        if len(changed_paths) != 1:
            return canonical, None
        changed_path = changed_paths[0]
        if changed_path in predicted_files:
            canonical["state_patch"] = {
                "domain": "terminal_fs",
                "operation": "write",
                "path": changed_path,
                "exists": True,
                "content": predicted_files[changed_path],
            }
        else:
            canonical["state_patch"] = {
                "domain": "terminal_fs",
                "operation": "delete",
                "path": changed_path,
                "exists": False,
                "deleted_content": initial_files[changed_path],
            }
        return canonical, "terminal_full_files_to_single_patch"
    return canonical, None


def apply_state_patch(initial_state: JSONDict, patch: Mapping[str, Any]) -> JSONDict:
    state = copy.deepcopy(initial_state)
    domain = patch.get("domain")
    operation = patch.get("operation")
    if domain == "sqlite_mcp":
        if operation in {"none", "select", "test"}:
            return state
        table = str(patch.get("table", ""))
        tables = state.setdefault("tables", {})
        rows = list(tables.get(table, []))
        where = dict(patch.get("where", {}))
        if operation == "update":
            if "rows" not in patch:
                raise ValueError("sqlite update patch requires explicit complete rows")
            predicted_rows = patch["rows"]
            if not isinstance(predicted_rows, list):
                raise ValueError("sqlite update patch rows must be a list")
            for predicted in predicted_rows:
                if not isinstance(predicted, Mapping):
                    continue
                matched = False
                for index, row in enumerate(rows):
                    if "id" in predicted:
                        same = row.get("id") == predicted.get("id")
                    else:
                        same = bool(where) and all(
                            row.get(key) == value for key, value in where.items()
                        )
                    if same:
                        rows[index] = dict(predicted)
                        matched = True
                        break
                if not matched and predicted:
                    rows.append(dict(predicted))
            tables[table] = sorted(rows, key=lambda row: (str(row.get("id", "")), json.dumps(row, sort_keys=True)))
            return state
        if operation == "delete":
            if where:
                rows = [row for row in rows if not all(row.get(key) == value for key, value in where.items())]
            else:
                deleted = patch.get("deleted_rows", [])
                deleted_ids = {row.get("id") for row in deleted if isinstance(row, Mapping)}
                rows = [row for row in rows if row.get("id") not in deleted_ids]
            tables[table] = rows
            return state
        if operation == "insert":
            predicted_rows = patch.get("rows", [])
            if isinstance(predicted_rows, list):
                rows.extend(dict(row) for row in predicted_rows if isinstance(row, Mapping))
            tables[table] = rows
            return state
        raise ValueError(f"Unsupported sqlite patch operation: {operation}")

    if domain == "terminal_fs":
        if operation in {"none", "read", "test"}:
            return state
        files = state.setdefault("files", {})
        path = str(patch.get("path", ""))
        if not path:
            raise ValueError("terminal patch requires path")
        if operation in {"write", "replace", "append"}:
            if patch.get("exists", True) is False:
                files.pop(path, None)
            elif "content" in patch:
                files[path] = str(patch["content"])
            else:
                raise ValueError("terminal mutation patch requires content")
            return state
        if operation == "delete":
            files.pop(path, None)
            return state
        raise ValueError(f"Unsupported terminal patch operation: {operation}")

    if domain == "toolsandbox":
        if operation == "none":
            return state
        if operation != "replace_namespaces":
            raise ValueError(f"Unsupported ToolSandbox patch operation: {operation}")
        replacement = patch.get("namespaces")
        if not isinstance(replacement, Mapping):
            raise ValueError("ToolSandbox replacement patch requires namespaces")
        namespaces = state.setdefault("namespaces", {})
        for namespace, rows in replacement.items():
            namespaces[str(namespace)] = _canonical_rows(rows)
        return state

    raise ValueError(f"Unsupported or missing patch domain: {domain}")


def target_success_for_state(task: TaskSpec, state: JSONDict) -> bool:
    target = task.target
    target_type = target.get("type")
    if task.domain == "sqlite_mcp":
        rows = state.get("tables", {}).get(str(target.get("table", "")), [])
        where = dict(target.get("where", {}))
        matches = [row for row in rows if all(row.get(key) == value for key, value in where.items())]
        if target_type == "row_field_equals":
            return bool(matches) and all(row.get(target.get("field")) == target.get("value") for row in matches)
        if target_type == "row_fields_equal":
            expected = dict(target.get("values", {}))
            return bool(matches) and all(
                all(row.get(field) == value for field, value in expected.items())
                for row in matches
            )
        if target_type == "row_exists":
            expected = dict(target.get("row", {}))
            return any(all(row.get(key) == value for key, value in expected.items()) for row in rows)
        raise ValueError(f"Unsupported sqlite target type: {target_type}")

    if task.domain == "terminal_fs":
        files = state.get("files", {})
        path = str(target.get("path", ""))
        if target_type == "file_content_equals":
            return files.get(path) == target.get("value")
        if target_type == "file_contains":
            return path in files and str(target.get("value")) in str(files[path])
        if target_type == "file_exists":
            return path in files
        if target_type == "json_field_equals":
            try:
                current: Any = json.loads(str(files[path]))
                for key in target.get("field_path", []):
                    current = current[key]
                return current == target.get("value")
            except (KeyError, TypeError, json.JSONDecodeError):
                return False
        if target_type == "tests_pass":
            return all(_terminal_test_passes(test, files) for test in task.metadata.get("tests", []))
        raise ValueError(f"Unsupported terminal target type: {target_type}")

    if task.domain == "toolsandbox":
        namespace = str(target.get("namespace", ""))
        rows = state.get("namespaces", {}).get(namespace, [])
        if target_type == "namespace_row_field_equals":
            matches = _matching_rows(rows, dict(target.get("where", {})))
            return bool(matches) and all(
                row.get(target.get("field")) == target.get("value") for row in matches
            )
        if target_type == "namespace_row_exists":
            expected = dict(target.get("row", {}))
            return bool(_matching_rows(rows, expected))
        if target_type == "namespace_matching_row_count":
            matches = _matching_rows(rows, dict(target.get("where", {})))
            return len(matches) == int(target.get("count", -1))
        if target_type == "namespace_snapshot_equals":
            return _canonical_rows(rows) == _canonical_rows(target.get("rows", []))
        raise ValueError(f"Unsupported ToolSandbox target type: {target_type}")

    raise ValueError(f"Unsupported domain: {task.domain}")


def _terminal_test_passes(test: Mapping[str, Any], files: Mapping[str, Any]) -> bool:
    test_type = test.get("type")
    path = str(test.get("path", ""))
    content = files.get(path)
    if test_type == "file_content_equals":
        return content == test.get("value")
    if test_type == "file_contains":
        return isinstance(content, str) and str(test.get("value")) in content
    if test_type == "file_exists":
        return path in files
    if test_type == "json_field_equals":
        try:
            current: Any = json.loads(str(content))
            for key in test.get("field_path", []):
                current = current[key]
            return current == test.get("value")
        except (json.JSONDecodeError, KeyError, TypeError):
            return False
    raise ValueError(f"Unsupported terminal test: {test_type}")


def constraint_violations_for_state(
    task: TaskSpec, state: JSONDict, patch: Mapping[str, Any] | None = None
) -> list[str]:
    initial = task.initial_state
    violations: list[str] = []
    for constraint in task.constraints:
        constraint_type = constraint.get("type")
        name = str(constraint.get("name", constraint_type))
        if task.domain == "sqlite_mcp":
            initial_tables = initial.get("tables", {})
            current_tables = state.get("tables", {})
            if constraint_type == "row_field_unchanged":
                table = str(constraint["table"])
                where = dict(constraint["where"])
                field = str(constraint["field"])
                original = [row for row in initial_tables.get(table, []) if all(row.get(k) == v for k, v in where.items())]
                current = [row for row in current_tables.get(table, []) if all(row.get(k) == v for k, v in where.items())]
                if not original or not current or original[0].get(field) != current[0].get(field):
                    violations.append(name)
            elif constraint_type == "row_count_unchanged":
                table = str(constraint["table"])
                if len(initial_tables.get(table, [])) != len(current_tables.get(table, [])):
                    violations.append(name)
            elif constraint_type == "forbid_delete":
                if patch and patch.get("operation") == "delete":
                    violations.append(name)
            else:
                raise ValueError(f"Unsupported sqlite constraint: {constraint_type}")
        elif task.domain == "terminal_fs":
            initial_files = initial.get("files", {})
            current_files = state.get("files", {})
            if constraint_type == "file_unchanged":
                path = str(constraint["path"])
                if initial_files.get(path) != current_files.get(path):
                    violations.append(name)
            elif constraint_type == "forbid_delete":
                if patch and patch.get("operation") == "delete":
                    violations.append(name)
            else:
                raise ValueError(f"Unsupported terminal constraint: {constraint_type}")
        elif task.domain == "toolsandbox":
            namespace = str(constraint.get("namespace", ""))
            initial_rows = initial.get("namespaces", {}).get(namespace, [])
            current_rows = state.get("namespaces", {}).get(namespace, [])
            if constraint_type == "namespace_unchanged":
                if _canonical_rows(initial_rows) != _canonical_rows(current_rows):
                    violations.append(name)
            elif constraint_type == "namespace_row_count_unchanged":
                if len(initial_rows) != len(current_rows):
                    violations.append(name)
            elif constraint_type == "namespace_row_count_delta":
                if len(current_rows) - len(initial_rows) != int(constraint.get("delta", 0)):
                    violations.append(name)
            elif constraint_type == "namespace_row_count_delta_at_most":
                if len(current_rows) - len(initial_rows) > int(constraint.get("delta", 0)):
                    violations.append(name)
            elif constraint_type == "namespace_row_field_unchanged":
                where = dict(constraint.get("where", {}))
                field = str(constraint.get("field", ""))
                original = _matching_rows(initial_rows, where)
                current = _matching_rows(current_rows, where)
                if (
                    len(original) != 1
                    or len(current) != 1
                    or original[0].get(field) != current[0].get(field)
                ):
                    violations.append(name)
            elif constraint_type == "namespace_row_unchanged":
                where = dict(constraint.get("where", {}))
                original = _matching_rows(initial_rows, where)
                current = _matching_rows(current_rows, where)
                if original != current or len(original) != 1:
                    violations.append(name)
            else:
                raise ValueError(f"Unsupported ToolSandbox constraint: {constraint_type}")
        else:
            raise ValueError(f"Unsupported domain: {task.domain}")
    return sorted(set(violations))
