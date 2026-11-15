from __future__ import annotations

import re
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Iterable

from actionrankwm.envs.base import BaseEnvironment
from actionrankwm.types import Action, JSONDict, StepResult, TaskSpec


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _identifier(value: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"Unsafe SQL identifier: {value!r}")
    return f'"{value}"'


def _sqlite_type(values: Iterable[Any]) -> str:
    non_null = [value for value in values if value is not None]
    if non_null and all(isinstance(value, (bool, int)) for value in non_null):
        return "INTEGER"
    if non_null and all(isinstance(value, (int, float)) for value in non_null):
        return "REAL"
    return "TEXT"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class SQLiteMCPEnvironment(BaseEnvironment):
    def __init__(self, task: TaskSpec, state: JSONDict | None = None) -> None:
        super().__init__(task)
        self._tmpdir = Path(tempfile.mkdtemp(prefix="actionrankwm_sqlite_"))
        self._db_path = self._tmpdir / "world.sqlite3"
        self._deleted = 0
        self._initial = task.initial_state
        self._initialize(state or task.initial_state)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self, state: JSONDict) -> None:
        tables = state.get("tables", {})
        if not isinstance(tables, dict):
            raise ValueError("sqlite_mcp state requires a tables object")
        with self._connect() as connection:
            for table, raw_rows in tables.items():
                rows = list(raw_rows)
                if not rows:
                    continue
                columns = sorted({str(key) for row in rows for key in row})
                definitions = []
                for column in columns:
                    column_type = _sqlite_type(row.get(column) for row in rows)
                    suffix = " PRIMARY KEY" if column == "id" else ""
                    definitions.append(f"{_identifier(column)} {column_type}{suffix}")
                connection.execute(
                    f"CREATE TABLE {_identifier(str(table))} ({', '.join(definitions)})"
                )
                placeholders = ", ".join("?" for _ in columns)
                column_sql = ", ".join(_identifier(column) for column in columns)
                for row in rows:
                    connection.execute(
                        f"INSERT INTO {_identifier(str(table))} ({column_sql}) VALUES ({placeholders})",
                        [row.get(column) for column in columns],
                    )

    def _table_columns(self, connection: sqlite3.Connection, table: str) -> set[str]:
        rows = connection.execute(f"PRAGMA table_info({_identifier(table)})").fetchall()
        if not rows:
            raise ValueError(f"Unknown table: {table}")
        return {str(row["name"]) for row in rows}

    def _select_rows(
        self, connection: sqlite3.Connection, table: str, where: JSONDict | None = None
    ) -> list[JSONDict]:
        columns = self._table_columns(connection, table)
        where = where or {}
        unknown = set(where) - columns
        if unknown:
            raise ValueError(f"Unknown where columns for {table}: {sorted(unknown)}")
        sql = f"SELECT * FROM {_identifier(table)}"
        values: list[Any] = []
        if where:
            clauses = []
            for key in sorted(where):
                clauses.append(f"{_identifier(key)} = ?")
                values.append(where[key])
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id" if "id" in columns else ""
        return [dict(row) for row in connection.execute(sql, values).fetchall()]

    def state_dict(self) -> JSONDict:
        with self._connect() as connection:
            table_rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            tables: JSONDict = {}
            for table_row in table_rows:
                table = str(table_row["name"])
                tables[table] = self._select_rows(connection, table)
        return {"tables": tables}

    def clone(self) -> "SQLiteMCPEnvironment":
        return SQLiteMCPEnvironment(self.task, state=self.state_dict())

    def _target_success_for_state(self, state: JSONDict) -> bool:
        target = self.task.target
        tables = state.get("tables", {})
        table = str(target.get("table", ""))
        rows = list(tables.get(table, []))
        target_type = target.get("type")
        where = dict(target.get("where", {}))
        matches = [row for row in rows if all(row.get(key) == value for key, value in where.items())]
        if target_type == "row_field_equals":
            return bool(matches) and all(
                row.get(target.get("field")) == target.get("value") for row in matches
            )
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

    def target_success(self) -> bool:
        return self._target_success_for_state(self.state_dict())

    def _environment_rules(self, table: str) -> list[JSONDict]:
        rules = self.task.initial_state.get("environment_rules", [])
        if not isinstance(rules, list):
            raise ValueError("environment_rules must be a list")
        return [
            dict(rule)
            for rule in rules
            if isinstance(rule, dict) and str(rule.get("table", "")) == table
        ]

    def _validate_update_rules(self, table: str, updates: JSONDict) -> None:
        known = {
            "require_set_fields_on_update",
            "increment_field_on_update",
            "append_text_field_on_update",
            "recompute_product_field_on_update",
            "derive_field_from_lookup_on_update",
        }
        for rule in self._environment_rules(table):
            rule_type = str(rule.get("type", ""))
            if rule_type not in known:
                raise ValueError(f"Unsupported SQLite environment rule: {rule_type}")
            if rule_type != "require_set_fields_on_update":
                continue
            when_set = dict(rule.get("when_set", {}))
            if not all(updates.get(key) == value for key, value in when_set.items()):
                continue
            required_set = dict(rule.get("required_set", {}))
            if not all(updates.get(key) == value for key, value in required_set.items()):
                raise ValueError(str(rule.get("error", "required_update_fields_missing")))

    def _apply_update_rules(
        self,
        connection: sqlite3.Connection,
        table: str,
        where: JSONDict,
        updates: JSONDict,
    ) -> None:
        for rule in self._environment_rules(table):
            when_set = dict(rule.get("when_set", {}))
            if not all(updates.get(key) == value for key, value in when_set.items()):
                continue
            rows = self._select_rows(connection, table, where)
            rule_type = rule.get("type")
            if rule_type == "increment_field_on_update":
                field = str(rule["field"])
                amount = rule.get("amount", 1)
                for row in rows:
                    current = row.get(field)
                    if not isinstance(current, (int, float)) or isinstance(current, bool):
                        raise ValueError(f"Cannot increment non-numeric field {field}")
                    if "id" not in row:
                        raise ValueError("increment_field_on_update requires row id")
                    connection.execute(
                        f"UPDATE {_identifier(table)} SET {_identifier(field)} = ? WHERE id = ?",
                        [current + amount, row["id"]],
                    )
            elif rule_type == "append_text_field_on_update":
                field = str(rule["field"])
                suffix = str(rule.get("suffix", ""))
                if not suffix:
                    raise ValueError("append_text_field_on_update requires a non-empty suffix")
                for row in rows:
                    current = row.get(field)
                    if not isinstance(current, str):
                        raise ValueError(f"Cannot append to non-text field {field}")
                    if "id" not in row:
                        raise ValueError("append_text_field_on_update requires row id")
                    connection.execute(
                        f"UPDATE {_identifier(table)} SET {_identifier(field)} = ? WHERE id = ?",
                        [current + suffix, row["id"]],
                    )
            elif rule_type == "recompute_product_field_on_update":
                left_field = str(rule.get("left_field", ""))
                right_field = str(rule.get("right_field", ""))
                output_field = str(rule.get("output_field", ""))
                fields = {left_field, right_field, output_field}
                columns = self._table_columns(connection, table)
                if "" in fields or not fields <= columns:
                    raise ValueError(
                        "recompute_product_field_on_update references unknown fields"
                    )
                offset = rule.get("offset", 0)
                if not _is_number(offset):
                    raise ValueError(
                        "recompute_product_field_on_update requires numeric offset"
                    )
                for row in rows:
                    left = row.get(left_field)
                    right = row.get(right_field)
                    if not _is_number(left) or not _is_number(right):
                        raise ValueError(
                            "recompute_product_field_on_update requires numeric inputs"
                        )
                    if "id" not in row:
                        raise ValueError(
                            "recompute_product_field_on_update requires row id"
                        )
                    connection.execute(
                        f"UPDATE {_identifier(table)} SET {_identifier(output_field)} = ? WHERE id = ?",
                        [left * right + offset, row["id"]],
                    )
            elif rule_type == "derive_field_from_lookup_on_update":
                raw_source_fields = rule.get("source_fields", [])
                if not isinstance(raw_source_fields, list):
                    raise ValueError(
                        "derive_field_from_lookup_on_update requires source_fields"
                    )
                source_fields = [str(field) for field in raw_source_fields]
                output_field = str(rule.get("output_field", ""))
                columns = self._table_columns(connection, table)
                if (
                    not source_fields
                    or len(source_fields) != len(set(source_fields))
                    or not output_field
                    or not set(source_fields + [output_field]) <= columns
                ):
                    raise ValueError(
                        "derive_field_from_lookup_on_update references invalid fields"
                    )
                cases = rule.get("cases", [])
                if not isinstance(cases, list) or not cases:
                    raise ValueError(
                        "derive_field_from_lookup_on_update requires lookup cases"
                    )
                normalized_cases: list[tuple[JSONDict, Any]] = []
                for case in cases:
                    if not isinstance(case, dict) or "value" not in case:
                        raise ValueError(
                            "derive_field_from_lookup_on_update has an invalid case"
                        )
                    conditions = case.get("when")
                    if not isinstance(conditions, dict) or set(conditions) != set(
                        source_fields
                    ):
                        raise ValueError(
                            "derive_field_from_lookup_on_update cases must specify every source field"
                        )
                    normalized_cases.append((dict(conditions), case["value"]))
                for row in rows:
                    matched_values = [
                        value
                        for conditions, value in normalized_cases
                        if all(row.get(field) == conditions[field] for field in source_fields)
                    ]
                    if len(matched_values) != 1:
                        raise ValueError(
                            str(rule.get("error", "lookup_derivation_not_unique"))
                        )
                    if "id" not in row:
                        raise ValueError(
                            "derive_field_from_lookup_on_update requires row id"
                        )
                    connection.execute(
                        f"UPDATE {_identifier(table)} SET {_identifier(output_field)} = ? WHERE id = ?",
                        [matched_values[0], row["id"]],
                    )

    def violations(self) -> list[str]:
        current = self.state_dict()
        initial_tables = self._initial.get("tables", {})
        current_tables = current.get("tables", {})
        violations: list[str] = []
        for constraint in self.task.constraints:
            constraint_type = constraint.get("type")
            if constraint_type == "row_field_unchanged":
                table = str(constraint["table"])
                where = dict(constraint["where"])
                field = str(constraint["field"])
                initial_rows = [
                    row
                    for row in initial_tables.get(table, [])
                    if all(row.get(key) == value for key, value in where.items())
                ]
                current_rows = [
                    row
                    for row in current_tables.get(table, [])
                    if all(row.get(key) == value for key, value in where.items())
                ]
                if not initial_rows or not current_rows or initial_rows[0].get(field) != current_rows[0].get(field):
                    violations.append(str(constraint.get("name", "row_field_changed")))
            elif constraint_type == "row_count_unchanged":
                table = str(constraint["table"])
                if len(initial_tables.get(table, [])) != len(current_tables.get(table, [])):
                    violations.append(str(constraint.get("name", "row_count_changed")))
            elif constraint_type == "forbid_delete" and self._deleted:
                violations.append(str(constraint.get("name", "delete_forbidden")))
            else:
                if constraint_type not in {
                    "row_field_unchanged",
                    "row_count_unchanged",
                    "forbid_delete",
                }:
                    raise ValueError(f"Unsupported sqlite constraint: {constraint_type}")
        return violations

    def apply(self, action: Action) -> StepResult:
        before = self.state_dict()
        ok = True
        error: str | None = None
        patch: JSONDict = {"domain": "sqlite_mcp", "operation": "none"}
        observation: JSONDict
        try:
            arguments = action.arguments
            table = str(arguments.get("table", ""))
            with self._connect() as connection:
                if action.tool == "db.select":
                    rows = self._select_rows(connection, table, dict(arguments.get("where", {})))
                    patch = {
                        "domain": "sqlite_mcp",
                        "operation": "select",
                        "table": table,
                        "rows": rows,
                    }
                    observation = {"ok": True, "affected_rows": 0, "rows": rows, "state_patch": patch}
                elif action.tool == "db.update":
                    columns = self._table_columns(connection, table)
                    where = dict(arguments.get("where", {}))
                    updates = dict(arguments.get("set", {}))
                    if not where or not updates:
                        raise ValueError("db.update requires non-empty where and set")
                    unknown = (set(where) | set(updates)) - columns
                    if unknown:
                        raise ValueError(f"Unknown columns for {table}: {sorted(unknown)}")
                    self._validate_update_rules(table, updates)
                    table_before = self._select_rows(connection, table)
                    set_sql = ", ".join(f"{_identifier(key)} = ?" for key in sorted(updates))
                    where_sql = " AND ".join(f"{_identifier(key)} = ?" for key in sorted(where))
                    values = [updates[key] for key in sorted(updates)] + [where[key] for key in sorted(where)]
                    cursor = connection.execute(
                        f"UPDATE {_identifier(table)} SET {set_sql} WHERE {where_sql}", values
                    )
                    primary_affected_rows = int(cursor.rowcount)
                    self._apply_update_rules(connection, table, where, updates)
                    table_after = self._select_rows(connection, table)
                    before_by_id = {row.get("id"): row for row in table_before}
                    rows = [
                        row for row in table_after if before_by_id.get(row.get("id")) != row
                    ]
                    patch = {
                        "domain": "sqlite_mcp",
                        "operation": "update",
                        "table": table,
                        "where": where,
                        "rows": rows,
                    }
                    observation = {
                        "ok": True,
                        "affected_rows": primary_affected_rows,
                        "rows": rows,
                        "state_patch": patch,
                    }
                elif action.tool == "db.delete":
                    where = dict(arguments.get("where", {}))
                    if not where:
                        raise ValueError("db.delete requires non-empty where")
                    deleted_rows = self._select_rows(connection, table, where)
                    columns = self._table_columns(connection, table)
                    unknown = set(where) - columns
                    if unknown:
                        raise ValueError(f"Unknown columns for {table}: {sorted(unknown)}")
                    where_sql = " AND ".join(f"{_identifier(key)} = ?" for key in sorted(where))
                    cursor = connection.execute(
                        f"DELETE FROM {_identifier(table)} WHERE {where_sql}",
                        [where[key] for key in sorted(where)],
                    )
                    self._deleted += max(0, int(cursor.rowcount))
                    patch = {
                        "domain": "sqlite_mcp",
                        "operation": "delete",
                        "table": table,
                        "where": where,
                        "deleted_rows": deleted_rows,
                    }
                    observation = {
                        "ok": True,
                        "affected_rows": int(cursor.rowcount),
                        "deleted_rows": deleted_rows,
                        "state_patch": patch,
                    }
                else:
                    raise ValueError(f"Unsupported sqlite tool: {action.tool}")
        except Exception as exc:
            ok = False
            error = f"{type(exc).__name__}: {exc}"
            observation = {"ok": False, "error": error, "state_patch": patch}
        after = self.state_dict()
        return StepResult(
            ok=ok,
            observation=observation,
            state_before=before,
            state_after=after,
            state_patch=patch,
            target_success=self.target_success(),
            violations=self.violations(),
            error=error,
        )

    def close(self) -> None:
        shutil.rmtree(self._tmpdir, ignore_errors=True)
