"""Tests for the Connect Four environment."""

import pytest

from gamebench.envs import ConnectFour, IllegalActionError


def play_columns(env: ConnectFour, columns: list[int]):
    result = None
    for column in columns:
        result = env.step(column)
        if result.done:
            break
    assert result is not None
    return result


@pytest.mark.parametrize(
    "columns",
    [
        [0, 0, 1, 1, 2, 2, 3],  # horizontal
        [0, 1, 0, 1, 0, 1, 0],  # vertical
        [0, 1, 1, 2, 4, 2, 2, 3, 4, 3, 5, 3, 3],  # rising diagonal
        [6, 5, 5, 4, 2, 4, 4, 3, 2, 3, 1, 3, 3],  # falling diagonal
    ],
)
def test_all_four_win_directions(columns: list[int]) -> None:
    result = play_columns(ConnectFour(), columns)
    assert result.done
    assert result.winner == 1
    assert result.reward == 1.0


def test_draw() -> None:
    env = ConnectFour()
    draw_sequence = [
        2, 5, 5, 5, 6, 6, 0, 2, 3, 1, 1, 6, 3, 1, 3, 0, 6, 3, 4, 1, 0,
        1, 1, 4, 5, 4, 4, 5, 6, 2, 2, 6, 2, 0, 2, 3, 3, 4, 5, 4, 0, 0,
    ]
    result = play_columns(env, draw_sequence)
    assert result.done
    assert result.winner is None
    assert result.reward == 0.0
    assert env.legal_actions() == []


def test_illegal_actions_raise_clear_exception() -> None:
    env = ConnectFour()
    for action in (-1, 7, "2", True):
        with pytest.raises(IllegalActionError):
            env.step(action)  # type: ignore[arg-type]
    for _ in range(6):
        env.step(0)
        if env.state_dict()["done"]:
            break
        env.step(1)
    with pytest.raises(IllegalActionError, match="full"):
        env.step(0)


def test_same_seed_resets_to_same_state() -> None:
    first = ConnectFour()
    second = ConnectFour()
    assert first.reset(seed=42) == second.reset(seed=42)
    assert first.state_dict() == second.state_dict()
