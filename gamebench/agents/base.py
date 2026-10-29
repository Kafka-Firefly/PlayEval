"""Shared agent interface."""

from abc import ABC, abstractmethod

from gamebench.envs.base import Action, Observation


class Agent(ABC):
    """An agent that selects one legal action from an observation."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable display name used in match results."""

    @abstractmethod
    def select_action(self, observation: Observation) -> Action:
        """Choose an action from the observation's legal actions."""

    def reset(self, seed: int | None = None) -> None:
        """Reset per-game agent state, using a seed when applicable."""
