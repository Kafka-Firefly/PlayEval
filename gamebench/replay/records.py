"""Structured match records and JSON serialization."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TurnRecord:
    """Details for one action within a game."""

    turn: int
    player: int
    agent: str
    observation: dict[str, Any]
    action: int | str | float | bool | None
    elapsed_seconds: float
    next_observation: dict[str, Any]


@dataclass(frozen=True)
class GameRecord:
    """Replayable metadata and turns for a single game."""

    game_index: int
    seed: int
    player_agents: dict[str, str]
    winner: int | None
    turns: tuple[TurnRecord, ...]


@dataclass(frozen=True)
class MatchResult:
    """Aggregate result and replay records for a set of games."""

    seed: int
    agent_a: str
    agent_b: str
    games: tuple[GameRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert the match to JSON-compatible values."""
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        """Write the complete match and all turn details to a JSON file."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )


def observation_to_dict(observation: Any) -> dict[str, Any]:
    """Serialize a dataclass observation or a JSON-compatible mapping."""
    if hasattr(observation, "__dataclass_fields__"):
        return asdict(observation)
    if isinstance(observation, dict):
        return observation
    raise TypeError("Observation must be a dataclass or dict to be recorded.")
