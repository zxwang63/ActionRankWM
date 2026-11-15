from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from actionrankwm.types import Action, JSONDict, StepResult, TaskSpec


class BaseEnvironment(ABC):
    def __init__(self, task: TaskSpec) -> None:
        self.task = task

    @abstractmethod
    def clone(self) -> "BaseEnvironment":
        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> JSONDict:
        raise NotImplementedError

    @abstractmethod
    def apply(self, action: Action) -> StepResult:
        raise NotImplementedError

    @abstractmethod
    def target_success(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def violations(self) -> list[str]:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def __enter__(self) -> "BaseEnvironment":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def make_environment(task: TaskSpec) -> BaseEnvironment:
    if task.domain == "sqlite_mcp":
        from actionrankwm.envs.sqlite_mcp import SQLiteMCPEnvironment

        return SQLiteMCPEnvironment(task)
    if task.domain == "terminal_fs":
        from actionrankwm.envs.terminal_fs import TerminalFSEnvironment

        return TerminalFSEnvironment(task)
    if task.domain == "toolsandbox":
        from actionrankwm.envs.toolsandbox import ToolSandboxEnvironment

        return ToolSandboxEnvironment(task)
    raise ValueError(f"Unsupported domain: {task.domain}")
