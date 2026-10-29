"""Tests for built-in agents and match recording."""

import json

from gamebench.agents import RandomAgent, RuleAgent
from gamebench.envs import ConnectFour
from gamebench.runner import run_match


def test_rule_agent_takes_win_and_blocks_loss() -> None:
    env = ConnectFour()
    for action in [0, 6, 1, 6, 2, 5]:
        observation = env.step(action).observation
    assert RuleAgent().select_action(observation) == 3

    env = ConnectFour()
    for action in [0, 6, 1, 6, 2, 5, 4]:
        observation = env.step(action).observation
    assert observation.current_player == 2
    assert RuleAgent().select_action(observation) == 3


def test_rule_agent_beats_random_well_over_half_the_games() -> None:
    result = run_match(ConnectFour, RuleAgent(), RandomAgent(), n_games=60, seed=1234)
    wins = sum(
        game.winner is not None
        and game.player_agents[str(game.winner)] == "rule"
        for game in result.games
    )
    assert wins / len(result.games) > 0.5


def test_runner_alternates_seats_and_records_json(tmp_path) -> None:
    result = run_match(ConnectFour, RandomAgent(), RuleAgent(), n_games=2, seed=9)
    assert result.games[0].player_agents == {"1": "random", "2": "rule"}
    assert result.games[1].player_agents == {"1": "rule", "2": "random"}
    assert all(game.turns for game in result.games)
    assert all(turn.elapsed_seconds >= 0 for game in result.games for turn in game.turns)
    path = tmp_path / "match.json"
    result.save_json(path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["seed"] == 9
    assert saved["agent_a"] == "random"
    assert len(saved["games"]) == 2
    assert "observation" in saved["games"][0]["turns"][0]
    assert "action" in saved["games"][0]["turns"][0]


def test_same_seed_reproduces_game_actions() -> None:
    def actions():
        match = run_match(ConnectFour, RandomAgent(), RandomAgent(), n_games=4, seed=57)
        return [[turn.action for turn in game.turns] for game in match.games]

    assert actions() == actions()
