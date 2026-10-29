"""Built-in agents for GameBench environments."""

from gamebench.agents.base import Agent
from gamebench.agents.connect_four import RandomAgent, RuleAgent

__all__ = ["Agent", "RandomAgent", "RuleAgent"]
