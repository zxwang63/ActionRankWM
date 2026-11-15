from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from actionrankwm.types import Action, JSONDict, TaskSpec

if TYPE_CHECKING:
    from actionrankwm.types import BranchRecord


@dataclass
class ProviderResult:
    raw_text: str
    parsed_observation: JSONDict | None
    latency_seconds: float = 0.0
    usage: JSONDict = field(default_factory=dict)
    error: str | None = None


class WorldModelProvider(ABC):
    name: str
    model: str

    @abstractmethod
    def predict(
        self,
        task: TaskSpec,
        action: Action,
        real_branch: "BranchRecord | None" = None,
    ) -> ProviderResult:
        raise NotImplementedError


def create_provider(name: str, config: dict[str, Any]) -> WorldModelProvider:
    if name.startswith("replay-"):
        from actionrankwm.providers.replay import ReplayProvider

        return ReplayProvider(mode=name.removeprefix("replay-"))
    if name == "openai":
        from actionrankwm.providers.openai_compatible import OpenAICompatibleProvider

        return OpenAICompatibleProvider.from_environment(config=config, local=False)
    if name == "compatible":
        from actionrankwm.providers.openai_compatible import LocalHTTPProvider

        return LocalHTTPProvider.from_environment(config=config)
    raise ValueError(f"Unknown provider: {name}")
