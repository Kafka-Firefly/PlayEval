"""Match scheduling and replay capture."""

import random
from time import perf_counter
from typing import TypeVar

from gamebench.agents.base import Agent
from gamebench.envs.base import GameEnv
from gamebench.replay.records import GameRecord, MatchResult, TurnRecord, observation_to_dict

EnvT = TypeVar("EnvT", bound=GameEnv)


def run_match(
    env_cls: type[EnvT],
    agent_a: Agent,
    agent_b: Agent,
    n_games: int,
    seed: int,
) -> MatchResult:
    """Run games with alternating seats and capture every observed turn."""
    if n_games <= 0:
        raise ValueError("n_games must be greater than zero.")
    seed_rng = random.Random(seed)
    game_records: list[GameRecord] = []
    for game_index in range(n_games):
        game_seed = seed_rng.randrange(0, 2**32)
        env = env_cls()
        observation = env.reset(seed=game_seed)
        player_agents = {1: agent_a, 2: agent_b} if game_index % 2 == 0 else {1: agent_b, 2: agent_a}
        player_agents[1].reset(_agent_seed(game_seed, 1))
        player_agents[2].reset(_agent_seed(game_seed, 2))
        turns: list[TurnRecord] = []
        while not observation.done:
            player = observation.current_player
            agent = player_agents[player]
            before = observation_to_dict(observation)
            started = perf_counter()
            action = agent.select_action(observation)
            elapsed = perf_counter() - started
            result = env.step(action)
            turns.append(
                TurnRecord(
                    turn=len(turns) + 1,
                    player=player,
                    agent=agent.name,
                    observation=before,
                    action=action,
                    elapsed_seconds=elapsed,
                    next_observation=observation_to_dict(result.observation),
                )
            )
            observation = result.observation
        game_records.append(
            GameRecord(
                game_index=game_index,
                seed=game_seed,
                player_agents={str(player): agent.name for player, agent in player_agents.items()},
                winner=observation.winner,
                turns=tuple(turns),
            )
        )
    return MatchResult(seed=seed, agent_a=agent_a.name, agent_b=agent_b.name, games=tuple(game_records))


def _agent_seed(game_seed: int, player: int) -> int:
    """Derive independent deterministic seeds for each seat."""
    return (game_seed * 31 + player) % (2**32)
