"""Random and tactical agents for Connect Four."""

import random

from gamebench.agents.base import Agent
from gamebench.envs.base import Action, Observation
from gamebench.envs.connect_four import COLS, ROWS


class RandomAgent(Agent):
    """Select uniformly from the legal actions."""

    def __init__(self, name: str = "random", seed: int | None = None) -> None:
        self._name = name
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return self._name

    def reset(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select_action(self, observation: Observation) -> Action:
        if not observation.legal_actions:
            raise ValueError("No legal actions are available.")
        return self._rng.choice(observation.legal_actions)


class RuleAgent(Agent):
    """Win immediately when possible, otherwise block an immediate loss."""

    def __init__(self, name: str = "rule", seed: int | None = None) -> None:
        self._name = name
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return self._name

    def reset(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select_action(self, observation: Observation) -> Action:
        legal = observation.legal_actions
        if not legal:
            raise ValueError("No legal actions are available.")
        player = observation.current_player
        for action in legal:
            if _would_win(observation.board, action, player):
                return action
        opponent = 3 - player
        for action in legal:
            if _would_win(observation.board, action, opponent):
                return action
        return self._rng.choice(legal)


def _would_win(board: tuple[tuple[int, ...], ...], column: int, player: int) -> bool:
    """Check whether dropping a piece in a column creates four in a row."""
    mutable = [list(row) for row in board]
    for row in range(ROWS - 1, -1, -1):
        if mutable[row][column] == 0:
            mutable[row][column] = player
            return _has_four(mutable, row, column, player)
    return False


def _has_four(board: list[list[int]], row: int, column: int, player: int) -> bool:
    """Check the four line directions through a recently placed piece."""
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for direction in (-1, 1):
            r, c = row + dr * direction, column + dc * direction
            while 0 <= r < ROWS and 0 <= c < COLS and board[r][c] == player:
                count += 1
                r += dr * direction
                c += dc * direction
        if count >= 4:
            return True
    return False
