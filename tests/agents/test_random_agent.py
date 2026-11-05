"""Tests de l'agent aléatoire (étape 1 de la phase 2).

Chaque test décrit un comportement attendu. Lance-les avec :

    pytest tests/agents/test_random_agent.py
"""

from __future__ import annotations

import numpy as np
import pytest

from gm2048.agents import Agent
from gm2048.agents.baselines import RandomAgent
from gm2048.engine import Action, Game, new_board

ALL_LEGAL = np.array([True, True, True, True])


def mask(*actions: Action) -> np.ndarray:
    """Masque où seuls les coups donnés sont légaux."""
    result = np.zeros(4, dtype=np.bool_)
    for action in actions:
        result[action] = True
    return result


def test_respects_the_agent_contract() -> None:
    agent = RandomAgent(seed=0)
    assert isinstance(agent, Agent)
    assert agent.name == "random"


def test_returns_an_action() -> None:
    action = RandomAgent(seed=0).act(new_board(), ALL_LEGAL)
    assert isinstance(action, Action)


def test_plays_the_only_legal_move() -> None:
    agent = RandomAgent(seed=0)
    for _ in range(50):
        assert agent.act(new_board(), mask(Action.DOWN)) is Action.DOWN


def test_only_plays_legal_moves_and_uses_all_of_them() -> None:
    agent = RandomAgent(seed=0)
    played = {agent.act(new_board(), mask(Action.UP, Action.LEFT)) for _ in range(200)}
    assert played == {Action.UP, Action.LEFT}


def test_every_move_is_played_about_equally() -> None:
    agent = RandomAgent(seed=1)
    actions = [agent.act(new_board(), ALL_LEGAL) for _ in range(4_000)]
    for action in Action:
        assert 800 < actions.count(action) < 1_200


def test_same_seed_plays_the_same_moves() -> None:
    first, second = RandomAgent(seed=42), RandomAgent(seed=42)
    moves_first = [first.act(new_board(), ALL_LEGAL) for _ in range(50)]
    moves_second = [second.act(new_board(), ALL_LEGAL) for _ in range(50)]
    assert moves_first == moves_second


def test_different_seeds_play_different_moves() -> None:
    first, second = RandomAgent(seed=1), RandomAgent(seed=2)
    moves_first = [first.act(new_board(), ALL_LEGAL) for _ in range(50)]
    moves_second = [second.act(new_board(), ALL_LEGAL) for _ in range(50)]
    assert moves_first != moves_second


def test_raises_when_no_move_is_legal() -> None:
    with pytest.raises(ValueError):
        RandomAgent(seed=0).act(new_board(), mask())


@pytest.mark.parametrize("seed", range(10))
def test_plays_full_games_without_illegal_moves(seed: int) -> None:
    game = Game(seed=seed)
    agent = RandomAgent(seed=seed)
    while not game.done:
        result = game.step(agent.act(game.board, game.legal_actions()))
        assert result.moved, "l'agent a joué un coup qui ne déplace aucune tuile"
    assert game.moves > 0
