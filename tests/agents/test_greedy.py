"""Tests de l'agent glouton (étape 3 de la phase 2).

Chaque test décrit un comportement attendu. Lance-les avec :

    pytest tests/agents/test_greedy.py

Les plateaux sont écrits avec les valeurs des tuiles (0 = case vide), comme on
les voit à l'écran ; ``from_values`` les convertit en exposants.
"""

from __future__ import annotations

import numpy as np
import pytest

from gm2048.agents import Agent
from gm2048.agents.baselines import GreedyAgent, RandomAgent
from gm2048.engine import Action, Game, from_values, slide
from gm2048.evaluation import evaluate

HORIZONTAL = {Action.LEFT, Action.RIGHT}
VERTICAL = {Action.UP, Action.DOWN}

# Aucune fusion possible, et les quatre coups sont légaux : tous valent 0 point.
NO_MERGE = from_values(
    [
        [0, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
)

# Une seule fusion possible, horizontale (2 + 2) : seuls LEFT et RIGHT rapportent des points.
ONE_HORIZONTAL_MERGE = from_values(
    [
        [2, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [8, 0, 0, 0],
    ]
)

# Fusion horizontale 2 + 2 = 4 points, fusion verticale 4 + 4 = 8 points.
SMALL_HORIZONTAL_BIG_VERTICAL = from_values(
    [
        [2, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 4],
        [0, 0, 0, 4],
    ]
)

# Horizontal : deux fusions dans la même ligne, 2 + 2 et 4 + 4, soit 4 + 8 = 12 points.
# Vertical : une seule fusion 4 + 4 = 8 points. La plus grosse fusion vaut 8 des deux
# côtés : il faut additionner toutes les fusions du coup pour voir que l'horizontal gagne.
TWO_MERGES_BEAT_ONE = from_values(
    [
        [2, 2, 4, 4],
        [0, 0, 0, 0],
        [4, 0, 0, 0],
        [4, 0, 0, 0],
    ]
)

# Seul DOWN est légal : la ligne du haut est pleine, sans fusion, et rien n'est en dessous.
ONLY_DOWN = from_values(
    [
        [2, 4, 8, 16],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
)

ALL_LEGAL = np.array([True, True, True, True])


def mask(*actions: Action) -> np.ndarray:
    """Masque où seuls les coups donnés sont légaux."""
    result = np.zeros(4, dtype=np.bool_)
    for action in actions:
        result[action] = True
    return result


def played_moves(board: np.ndarray, legal_mask: np.ndarray, n: int = 200) -> set[Action]:
    """Les coups joués par un même agent sur ``n`` appels avec le même plateau."""
    agent = GreedyAgent(seed=0)
    return {agent.act(board, legal_mask) for _ in range(n)}


# --- Le contrat ---------------------------------------------------------------


def test_respects_the_agent_contract() -> None:
    agent = GreedyAgent(seed=0)
    assert isinstance(agent, Agent)
    assert agent.name == "greedy"


def test_returns_an_action() -> None:
    action = GreedyAgent(seed=0).act(NO_MERGE, ALL_LEGAL)
    assert isinstance(action, Action)


def test_raises_when_no_move_is_legal() -> None:
    with pytest.raises(ValueError):
        GreedyAgent(seed=0).act(NO_MERGE, mask())


def test_does_not_modify_the_board() -> None:
    board = SMALL_HORIZONTAL_BIG_VERTICAL.copy()
    GreedyAgent(seed=0).act(board, ALL_LEGAL)
    np.testing.assert_array_equal(board, SMALL_HORIZONTAL_BIG_VERTICAL)


# --- Choisir le coup qui rapporte le plus ------------------------------------


def test_plays_the_only_legal_move() -> None:
    assert played_moves(ONLY_DOWN, mask(Action.DOWN), n=20) == {Action.DOWN}


def test_plays_the_only_merge() -> None:
    # Les quatre coups sont légaux, mais seuls LEFT et RIGHT fusionnent les deux 2.
    assert played_moves(ONE_HORIZONTAL_MERGE, ALL_LEGAL) <= HORIZONTAL


def test_prefers_the_biggest_merge() -> None:
    assert played_moves(SMALL_HORIZONTAL_BIG_VERTICAL, ALL_LEGAL) <= VERTICAL


def test_adds_up_every_merge_of_the_move() -> None:
    assert played_moves(TWO_MERGES_BEAT_ONE, ALL_LEGAL) <= HORIZONTAL


def test_only_considers_moves_allowed_by_the_mask() -> None:
    # Le meilleur coup (vertical) est exclu par le masque : il faut se contenter de l'horizontal.
    assert played_moves(SMALL_HORIZONTAL_BIG_VERTICAL, mask(*HORIZONTAL)) <= HORIZONTAL


# --- Départager les égalités --------------------------------------------------


def test_breaks_ties_at_random_among_the_best_moves() -> None:
    # UP et DOWN rapportent tous les deux 8 points : l'agent doit jouer les deux, jamais un autre.
    # Question bonus : pourquoi LEFT et RIGHT rapportent-ils toujours autant de points ?
    assert played_moves(SMALL_HORIZONTAL_BIG_VERTICAL, ALL_LEGAL) == VERTICAL


def test_plays_at_random_when_no_move_scores() -> None:
    agent = GreedyAgent(seed=1)
    actions = [agent.act(NO_MERGE, ALL_LEGAL) for _ in range(4_000)]
    for action in Action:
        assert 800 < actions.count(action) < 1_200


def test_same_seed_plays_the_same_moves() -> None:
    first, second = GreedyAgent(seed=42), GreedyAgent(seed=42)
    moves_first = [first.act(NO_MERGE, ALL_LEGAL) for _ in range(50)]
    moves_second = [second.act(NO_MERGE, ALL_LEGAL) for _ in range(50)]
    assert moves_first == moves_second


def test_different_seeds_play_different_moves() -> None:
    first, second = GreedyAgent(seed=1), GreedyAgent(seed=2)
    moves_first = [first.act(NO_MERGE, ALL_LEGAL) for _ in range(50)]
    moves_second = [second.act(NO_MERGE, ALL_LEGAL) for _ in range(50)]
    assert moves_first != moves_second


# --- Sur de vraies parties ----------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_always_plays_a_move_with_the_best_reward(seed: int) -> None:
    game = Game(seed=seed)
    agent = GreedyAgent(seed=seed)
    while not game.done:
        board, legal_mask = game.board, game.legal_actions()
        best = max(slide(board, action).reward for action in Action if legal_mask[action])
        action = agent.act(board, legal_mask)
        assert legal_mask[action], "l'agent a joué un coup illégal"
        assert slide(board, action).reward == best
        game.step(action)
    assert game.moves > 0


def test_scores_more_than_the_random_agent() -> None:
    greedy = evaluate(GreedyAgent(seed=0), n_games=20)
    baseline = evaluate(RandomAgent(seed=0), n_games=20)
    assert greedy.mean_score > baseline.mean_score


def test_is_available_in_the_command_line() -> None:
    # Pour lancer `gm2048 eval --agent greedy`, ajoute l'agent au dictionnaire AGENTS de cli.py.
    from gm2048.cli import AGENTS

    assert isinstance(AGENTS["greedy"](0), GreedyAgent)
