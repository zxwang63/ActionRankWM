from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from actionrankwm.envs.base import BaseEnvironment
from actionrankwm.types import Action, JSONDict, StepResult, TaskSpec


class TerminalFSEnvironment(BaseEnvironment):
    def __init__(self, task: TaskSpec, state: JSONDict | None = None) -> None:
        super().__init__(task)
        self._tmpdir = Path(tempfile.mkdtemp(prefix="actionrankwm_fs_"))
        self._root = self._tmpdir / "workspace"
        self._root.mkdir()
        self._deleted = 0
        self._initial = task.initial_state
        self._initialize(state or task.initial_state)

    def _safe_path(self, relative: str) -> Path:
        candidate = (self._root / relative).resolve()
        root = self._root.resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError(f"Path escapes workspace: {relative}")
        return candidate

    def _initialize(self, state: JSONDict) -> None:
        files = state.get("files", {})
        if not isinstance(files, dict):
            raise ValueError("terminal_fs state requires a files object")
        for relative, content in files.items():
            path = self._safe_path(str(relative))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(content), encoding="utf-8")

    def state_dict(self) -> JSONDict:
        files: JSONDict = {}
        for path in sorted(self._root.rglob("*")):
            if path.is_file():
                files[str(path.relative_to(self._root))] = path.read_text(encoding="utf-8")
        return {"files": files}

    def clone(self) -> "TerminalFSEnvironment":
        return TerminalFSEnvironment(self.task, state=self.state_dict())

    def _test_results_for_state(self, state: JSONDict) -> list[JSONDict]:
        files = state.get("files", {})
        results: list[JSONDict] = []
        for index, test in enumerate(self.task.metadata.get("tests", [])):
            test_type = test.get("type")
            path = str(test.get("path", ""))
            content = files.get(path)
            passed = False
            if test_type == "file_content_equals":
                passed = content == test.get("value")
            elif test_type == "file_contains":
                passed = isinstance(content, str) and str(test.get("value")) in content
            elif test_type == "file_exists":
                passed = path in files
            elif test_type == "json_field_equals":
                try:
                    parsed = json.loads(str(content))
                    current: Any = parsed
                    for key in test.get("field_path", []):
                        current = current[key]
                    passed = current == test.get("value")
                except (json.JSONDecodeError, KeyError, TypeError):
                    passed = False
            else:
                raise ValueError(f"Unsupported terminal test: {test_type}")
            results.append({"test_id": test.get("test_id", f"test_{index}"), "passed": passed})
        return results

    def _target_success_for_state(self, state: JSONDict) -> bool:
        target = self.task.target
        files = state.get("files", {})
        target_type = target.get("type")
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
            results = self._test_results_for_state(state)
            return bool(results) and all(result["passed"] for result in results)
        raise ValueError(f"Unsupported terminal target type: {target_type}")

    def target_success(self) -> bool:
        return self._target_success_for_state(self.state_dict())

    def _environment_rules(self, relative: str) -> list[JSONDict]:
        rules = self.task.initial_state.get("environment_rules", [])
        if not isinstance(rules, list):
            raise ValueError("environment_rules must be a list")
        return [
            dict(rule)
            for rule in rules
            if isinstance(rule, dict) and str(rule.get("path", "")) == relative
        ]

    def _apply_write_rules(
        self, relative: str, arguments: JSONDict, content: str
    ) -> str:
        known = {
            "require_json_fields_on_write",
            "require_arguments_on_write",
            "canonicalize_json_on_write",
            "append_text_on_write",
            "replace_text_on_write",
        }
        parsed_json: Any | None = None
        for rule in self._environment_rules(relative):
            rule_type = str(rule.get("type", ""))
            if rule_type not in known:
                raise ValueError(f"Unsupported terminal environment rule: {rule_type}")
            if rule_type == "require_arguments_on_write":
                required = dict(rule.get("required_arguments", {}))
                if not all(arguments.get(key) == value for key, value in required.items()):
                    raise ValueError(str(rule.get("error", "required_write_arguments_missing")))
            elif rule_type == "require_json_fields_on_write":
                try:
                    parsed_json = json.loads(content) if parsed_json is None else parsed_json
                except json.JSONDecodeError as exc:
                    raise ValueError(str(rule.get("error", "invalid_json"))) from exc
                for field_path in rule.get("required_field_paths", []):
                    current: Any = parsed_json
                    try:
                        for component in field_path:
                            current = current[component]
                    except (KeyError, TypeError):
                        raise ValueError(
                            str(rule.get("error", "required_json_fields_missing"))
                        ) from None
            elif rule_type == "canonicalize_json_on_write":
                try:
                    parsed_json = json.loads(content) if parsed_json is None else parsed_json
                except json.JSONDecodeError as exc:
                    raise ValueError(str(rule.get("error", "invalid_json"))) from exc
                content = json.dumps(
                    parsed_json,
                    ensure_ascii=False,
                    sort_keys=bool(rule.get("sort_keys", True)),
                    indent=int(rule.get("indent", 2)),
                )
                if rule.get("trailing_newline", True):
                    content += "\n"
            elif rule_type == "append_text_on_write":
                suffix = str(rule.get("suffix", ""))
                if not suffix:
                    raise ValueError("append_text_on_write requires a non-empty suffix")
                content += suffix
            elif rule_type == "replace_text_on_write":
                old = str(rule.get("old", ""))
                new = str(rule.get("new", ""))
                if not old:
                    raise ValueError(
                        "replace_text_on_write requires non-empty old text"
                    )
                content = content.replace(old, new)
        return content

    def violations(self) -> list[str]:
        current = self.state_dict()
        initial_files = self._initial.get("files", {})
        current_files = current.get("files", {})
        violations: list[str] = []
        for constraint in self.task.constraints:
            constraint_type = constraint.get("type")
            if constraint_type == "file_unchanged":
                path = str(constraint["path"])
                if initial_files.get(path) != current_files.get(path):
                    violations.append(str(constraint.get("name", f"file_changed:{path}")))
            elif constraint_type == "forbid_delete" and self._deleted:
                violations.append(str(constraint.get("name", "delete_forbidden")))
            else:
                if constraint_type not in {"file_unchanged", "forbid_delete"}:
                    raise ValueError(f"Unsupported terminal constraint: {constraint_type}")
        return violations

    def apply(self, action: Action) -> StepResult:
        before = self.state_dict()
        ok = True
        error: str | None = None
        patch: JSONDict = {"domain": "terminal_fs", "operation": "none"}
        try:
            arguments = action.arguments
            relative = str(arguments.get("path", ""))
            path = self._safe_path(relative) if relative else self._root
            if action.tool == "fs.read":
                content = path.read_text(encoding="utf-8")
                patch = {
                    "domain": "terminal_fs",
                    "operation": "read",
                    "path": relative,
                    "exists": True,
                    "content": content,
                }
                observation = {"ok": True, "path": relative, "content": content, "state_patch": patch}
            elif action.tool == "fs.write":
                content = str(arguments.get("content", ""))
                content = self._apply_write_rules(relative, arguments, content)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                patch = {
                    "domain": "terminal_fs",
                    "operation": "write",
                    "path": relative,
                    "exists": True,
                    "content": content,
                }
                observation = {"ok": True, "path": relative, "bytes": len(content.encode()), "state_patch": patch}
            elif action.tool == "fs.append":
                appended = str(arguments.get("content", ""))
                if not appended:
                    raise ValueError("fs.append requires non-empty content")
                content = path.read_text(encoding="utf-8") + appended
                path.write_text(content, encoding="utf-8")
                patch = {
                    "domain": "terminal_fs",
                    "operation": "append",
                    "path": relative,
                    "exists": True,
                    "content": content,
                    "appended": appended,
                }
                observation = {
                    "ok": True,
                    "path": relative,
                    "bytes_appended": len(appended.encode()),
                    "state_patch": patch,
                }
            elif action.tool == "fs.replace":
                old = str(arguments.get("old", ""))
                new = str(arguments.get("new", ""))
                if not old:
                    raise ValueError("fs.replace requires non-empty old text")
                content = path.read_text(encoding="utf-8")
                replacements = content.count(old)
                if replacements == 0:
                    raise ValueError("old text not found")
                content = content.replace(old, new)
                path.write_text(content, encoding="utf-8")
                patch = {
                    "domain": "terminal_fs",
                    "operation": "replace",
                    "path": relative,
                    "exists": True,
                    "content": content,
                    "replacements": replacements,
                }
                observation = {
                    "ok": True,
                    "path": relative,
                    "replacements": replacements,
                    "state_patch": patch,
                }
            elif action.tool == "fs.delete":
                previous = path.read_text(encoding="utf-8")
                path.unlink()
                self._deleted += 1
                patch = {
                    "domain": "terminal_fs",
                    "operation": "delete",
                    "path": relative,
                    "exists": False,
                    "deleted_content": previous,
                }
                observation = {"ok": True, "path": relative, "deleted": True, "state_patch": patch}
            elif action.tool == "tests.run":
                results = self._test_results_for_state(self.state_dict())
                patch = {"domain": "terminal_fs", "operation": "test", "results": results}
                observation = {"ok": True, "results": results, "state_patch": patch}
            else:
                raise ValueError(f"Unsupported terminal tool: {action.tool}")
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
