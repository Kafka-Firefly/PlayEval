"""Common game environment types and interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeAlias

Action: TypeAlias = int


@dataclass(frozen=True)
class Observation:
    """An immutable view of a game's current state."""

    board: tuple[tuple[int, ...], ...]
    current_player: int
    legal_actions: tuple[Action, ...]
    done: bool = False
    winner: int | None = None


@dataclass(frozen=True)
class StepResult:
    """Outcome of one action; reward is from the acting player's perspective."""

    observation: Observation
    done: bool
    winner: int | None
    reward: float


class GameEnv(ABC):
    """Abstract interface for deterministic or seeded turn-based games."""

    @abstractmethod
    def reset(self, seed: int | None = None) -> Observation:
        """Start a new game, optionally using a reproducibility seed."""

    @abstractmethod
    def legal_actions(self) -> list[Action]:
        """Return the actions available to the current player."""

    @abstractmethod
    def step(self, action: Action) -> StepResult:
        """Apply an action or raise an exception if it is illegal."""

    @abstractmethod
    def render_text(self) -> str:
        """Describe the current position in plain text for an agent."""

    @abstractmethod
    def state_dict(self) -> dict[str, object]:
        """Return a JSON-serializable snapshot of the complete game state."""
