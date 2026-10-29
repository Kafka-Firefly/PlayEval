"""Connect Four (four-in-a-row) environment."""

from gamebench.envs.base import Action, GameEnv, Observation, StepResult

ROWS = 6
COLS = 7


class IllegalActionError(ValueError):
    """Raised when a move is not legal in the current position."""


class ConnectFour(GameEnv):
    """Two-player Connect Four, with players numbered 1 and 2."""

    def __init__(self) -> None:
        self._board: list[list[int]] = []
        self._current_player = 1
        self._done = False
        self._winner: int | None = None
        self._moves = 0
        self._seed: int | None = None
        self.reset()

    def reset(self, seed: int | None = None) -> Observation:
        """Clear the board and start with player 1."""
        self._board = [[0 for _ in range(COLS)] for _ in range(ROWS)]
        self._current_player = 1
        self._done = False
        self._winner = None
        self._moves = 0
        self._seed = seed
        return self._observation()

    def legal_actions(self) -> list[Action]:
        """Return non-full column indices, or no actions after the game ends."""
        if self._done:
            return []
        return [column for column in range(COLS) if self._board[0][column] == 0]

    def step(self, action: Action) -> StepResult:
        """Drop a piece into a column and evaluate the resulting position."""
        if self._done:
            raise IllegalActionError("Cannot move: the game is already over.")
        if not isinstance(action, int) or isinstance(action, bool):
            raise IllegalActionError(f"Column must be an integer from 0 to {COLS - 1}; got {action!r}.")
        if action < 0 or action >= COLS:
            raise IllegalActionError(f"Column {action} is out of range; choose 0 to {COLS - 1}.")
        if action not in self.legal_actions():
            raise IllegalActionError(f"Column {action} is full; choose a non-full column.")

        for row in range(ROWS - 1, -1, -1):
            if self._board[row][action] == 0:
                self._board[row][action] = self._current_player
                break

        actor = self._current_player
        self._moves += 1
        if self._has_four(row, action, actor):
            self._done = True
            self._winner = actor
            reward = 1.0
        elif self._moves == ROWS * COLS:
            self._done = True
            reward = 0.0
        else:
            reward = 0.0
            self._current_player = 3 - actor
        return StepResult(self._observation(), self._done, self._winner, reward)

    def render_text(self) -> str:
        """Render a compact board and the current player's legal choices."""
        rows = [" ".join("." if cell == 0 else str(cell) for cell in row) for row in self._board]
        rows.append("0 1 2 3 4 5 6")
        if self._done:
            status = f"Game over; winner: {self._winner if self._winner is not None else 'draw'}."
        else:
            status = f"Player {self._current_player} to move. Legal columns: {self.legal_actions()}"
        return "\n".join(rows + [status])

    def state_dict(self) -> dict[str, object]:
        """Return the board and metadata using JSON-compatible values."""
        return {
            "game": "connect_four",
            "board": [row[:] for row in self._board],
            "current_player": self._current_player,
            "done": self._done,
            "winner": self._winner,
            "moves": self._moves,
            "seed": self._seed,
        }

    def _observation(self) -> Observation:
        return Observation(
            board=tuple(tuple(row) for row in self._board),
            current_player=self._current_player,
            legal_actions=tuple(self.legal_actions()),
            done=self._done,
            winner=self._winner,
        )

    def _has_four(self, row: int, column: int, player: int) -> bool:
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            for direction in (-1, 1):
                r, c = row + dr * direction, column + dc * direction
                while 0 <= r < ROWS and 0 <= c < COLS and self._board[r][c] == player:
                    count += 1
                    r += dr * direction
                    c += dc * direction
            if count >= 4:
                return True
        return False
