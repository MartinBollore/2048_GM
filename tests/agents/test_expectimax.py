"""Tests de l'agent expectimax (étape 5 de la phase 2).

Chaque test décrit un comportement attendu. Lance-les avec :

    pytest tests/agents/test_expectimax.py

Les valeurs attendues sont calculées à la main dans les commentaires, avec des
poids très simples pour que le calcul reste faisable :

- ``EMPTY_ONLY`` : la note d'un plateau est son nombre de cases vides ;
- ``CORNER_ONLY`` : la note vaut l'exposant de la plus grosse tuile si elle est
  dans un coin, 0 sinon ;
- ``NO_WEIGHTS`` : la note vaut toujours 0.

Refais au moins un calcul toi-même sur papier : c'est le meilleur moyen de
comprendre l'arbre.
"""

from __future__ import annotations

import numpy as np
import pytest

from gm2048.agents import Agent
from gm2048.agents.baselines import ExpectimaxAgent, HeuristicWeights
from gm2048.agents.baselines.expectimax import GAME_OVER_VALUE
from gm2048.agents.baselines.heuristic import DEFAULT_WEIGHTS, evaluate_board
from gm2048.engine import Action, Game, from_values, legal_actions

HORIZONTAL = {Action.LEFT, Action.RIGHT}
ALL_LEGAL = np.array([True, True, True, True])

EMPTY_ONLY = HeuristicWeights(empty=1.0, monotonicity=0.0, smoothness=0.0, corner=0.0)
CORNER_ONLY = HeuristicWeights(empty=0.0, monotonicity=0.0, smoothness=0.0, corner=1.0)
NO_WEIGHTS = HeuristicWeights(empty=0.0, monotonicity=0.0, smoothness=0.0, corner=0.0)


def mask(*actions: Action) -> np.ndarray:
    """Masque où seuls les coups donnés sont légaux."""
    result = np.zeros(4, dtype=np.bool_)
    for action in actions:
        result[action] = True
    return result


def agent(depth: int = 1, weights: HeuristicWeights = EMPTY_ONLY, seed: int = 0) -> ExpectimaxAgent:
    return ExpectimaxAgent(seed=seed, depth=depth, weights=weights)


# Plein, sans aucune fusion possible : la partie est perdue.
GAME_OVER = from_values(
    [
        [2, 4, 2, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [4, 2, 4, 2],
    ]
)

# Une seule tuile 2, dans un coin.
ONE_TILE_IN_A_CORNER = from_values(
    [
        [2, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
)

# Un afterstate avec une seule case vide, en bas à droite, à côté d'un 4 et d'un 8.
# - S'il y apparaît un 2 : plateau plein sans fusion, partie perdue.
# - S'il y apparaît un 4 : il peut fusionner avec le 4 de gauche, la partie continue.
ONE_HOLE = from_values(
    [
        [2, 4, 2, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 8],
        [4, 2, 4, 0],
    ]
)

# ONE_HOLE après l'apparition d'un 4 : seuls LEFT et RIGHT sont légaux, et rapportent
# tous les deux 8 points (4 + 4). Mais ils ne se valent pas :
# - LEFT donne la dernière ligne 4 2 8 _ : la case vide est entre un 8 et un 8,
#   aucune tuile qui apparaît (2 ou 4) ne peut fusionner : partie perdue à coup sûr ;
# - RIGHT donne la dernière ligne _ 4 2 8 : la colonne de droite contient alors
#   deux 8 l'un sur l'autre, la partie continue quoi qu'il apparaisse.
TRAP = from_values(
    [
        [2, 4, 2, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 8],
        [4, 2, 4, 4],
    ]
)

# Deux 2 côte à côte en haut à gauche.
TWO_TWOS = from_values(
    [
        [2, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
)

# Une seule tuile au centre : les quatre coups sont symétriques.
SYMMETRIC = from_values(
    [
        [0, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
)

# Un plateau quelconque, ni vide ni perdu.
SOME_BOARD = from_values(
    [
        [64, 2, 8, 0],
        [4, 0, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 0],
    ]
)


# --- Le contrat ---------------------------------------------------------------------


def test_respects_the_agent_contract() -> None:
    expectimax = ExpectimaxAgent(seed=0)
    assert isinstance(expectimax, Agent)
    assert expectimax.name == "expectimax"


def test_default_depth_and_weights() -> None:
    expectimax = ExpectimaxAgent(seed=0)
    assert expectimax.depth == 2
    assert expectimax.weights == DEFAULT_WEIGHTS


def test_keeps_the_given_depth_and_weights() -> None:
    expectimax = ExpectimaxAgent(seed=0, depth=3, weights=CORNER_ONLY)
    assert expectimax.depth == 3
    assert expectimax.weights == CORNER_ONLY


@pytest.mark.parametrize("depth", [0, -1])
def test_refuses_a_depth_below_one(depth: int) -> None:
    with pytest.raises(ValueError):
        ExpectimaxAgent(seed=0, depth=depth)


def test_starts_with_an_empty_cache() -> None:
    assert ExpectimaxAgent(seed=0).cache == {}


def test_returns_an_action() -> None:
    assert isinstance(agent().act(SYMMETRIC, ALL_LEGAL), Action)


def test_raises_when_no_move_is_legal() -> None:
    with pytest.raises(ValueError):
        agent().act(SYMMETRIC, mask())


# --- Les feuilles : profondeur 0 et partie perdue -----------------------------------


def test_at_depth_zero_the_value_is_the_board_score() -> None:
    expectimax = agent(weights=DEFAULT_WEIGHTS)
    assert expectimax._max_value(SOME_BOARD, 0) == pytest.approx(
        evaluate_board(SOME_BOARD, DEFAULT_WEIGHTS)
    )


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_a_lost_game_is_worth_game_over_value_at_any_depth(depth: int) -> None:
    assert agent()._max_value(GAME_OVER, depth) == GAME_OVER_VALUE


# --- Les nœuds hasard ---------------------------------------------------------------


def test_chance_value_weights_the_two_and_the_four() -> None:
    # Une seule case vide : un 2 (90 %) fait perdre, un 4 (10 %) donne un plateau plein,
    # donc 0 case vide. 0.9 * GAME_OVER_VALUE + 0.1 * 0.
    assert agent()._chance_value(ONE_HOLE, 1) == pytest.approx(0.9 * GAME_OVER_VALUE)


def test_chance_value_averages_over_the_empty_cells() -> None:
    # 15 cases vides. Note CORNER_ONLY du plateau obtenu :
    # - un 2 (exposant 1), où qu'il tombe : la plus grosse tuile vaut 1 et le 2 du coin
    #   en fait partie, note 1 ;
    # - un 4 (exposant 2) : il devient la plus grosse tuile. Note 2 s'il tombe dans
    #   l'un des 3 coins libres, 0 sur les 12 autres cases.
    # 0.9 * 1 + 0.1 * (3 * 2 + 12 * 0) / 15 = 0.9 + 0.04 = 0.94
    expectimax = agent(weights=CORNER_ONLY)
    assert expectimax._chance_value(ONE_TILE_IN_A_CORNER, 1) == pytest.approx(0.94)


def test_chance_value_does_not_modify_the_afterstate() -> None:
    afterstate = ONE_TILE_IN_A_CORNER.copy()
    agent(weights=CORNER_ONLY)._chance_value(afterstate, 1)
    np.testing.assert_array_equal(afterstate, ONE_TILE_IN_A_CORNER)


# --- Les nœuds max ------------------------------------------------------------------


def test_max_value_at_depth_one() -> None:
    # LEFT ou RIGHT : 4 points, il reste 15 cases vides. Après l'apparition d'une tuile,
    # il en reste 14 : valeur 4 + 14 = 18.
    # DOWN : 0 point, 14 cases vides, puis 13 : valeur 13. UP ne bouge rien, il est illégal.
    assert agent()._max_value(TWO_TWOS, 1) == pytest.approx(18.0)


def test_max_value_sees_the_trap() -> None:
    # LEFT : 8 + GAME_OVER_VALUE. RIGHT : 8 + 0 (plateau plein mais vivant, 0 case vide).
    assert agent()._max_value(TRAP, 1) == pytest.approx(8.0)


def test_chance_value_at_depth_two() -> None:
    # Un 2 (90 %) fait perdre. Un 4 (10 %) donne TRAP, qui vaut 8 à la profondeur 1.
    # 0.9 * GAME_OVER_VALUE + 0.1 * 8
    assert agent()._chance_value(ONE_HOLE, 2) == pytest.approx(0.9 * GAME_OVER_VALUE + 0.8)


def test_max_value_at_depth_two() -> None:
    # LEFT : 4 points, afterstate avec un 4 en haut à gauche et 15 cases vides.
    #   Pour chaque tuile qui apparaît, on calcule la valeur à la profondeur 1 :
    #   - un 4 sur la ligne ou la colonne du premier 4 (6 cases) : on peut les fusionner,
    #     8 points + 14 cases vides après la tuile suivante = 22 ;
    #   - sinon (un 2 n'importe où, ou un 4 sur les 9 autres cases) : aucune fusion,
    #     0 point + 13 cases vides = 13.
    #   Valeur : 4 + 0.9 * 13 + 0.1 * (6 * 22 + 9 * 13) / 15 = 4 + 11.7 + 1.66 = 17.36.
    # RIGHT : pareil par symétrie, 17.36.
    # DOWN : 0 point, les deux 2 restent côte à côte en bas. Quelle que soit la tuile qui
    #   apparaît, on fusionne les deux 2 au coup suivant : 4 points + 13 cases vides = 17.
    #   Valeur : 0 + 17 = 17.
    assert agent()._max_value(TWO_TWOS, 2) == pytest.approx(17.36)


# --- Le cache -----------------------------------------------------------------------


def test_the_cache_is_filled_by_the_search() -> None:
    expectimax = agent(depth=2)
    expectimax.act(TWO_TWOS, legal_actions(TWO_TWOS))
    assert expectimax.cache
    for (board_bytes, depth), value in expectimax.cache.items():
        assert isinstance(board_bytes, bytes)
        assert 0 <= depth <= 2
        assert isinstance(value, float)


def test_the_cache_holds_max_values() -> None:
    expectimax = agent(depth=1)
    value = expectimax._max_value(TRAP, 1)
    assert expectimax.cache[(TRAP.tobytes(), 1)] == value


def test_the_cache_is_used() -> None:
    expectimax = agent(depth=1)
    expectimax.cache[(TRAP.tobytes(), 1)] = 123.0
    assert expectimax._max_value(TRAP, 1) == 123.0


def test_act_starts_with_a_fresh_cache() -> None:
    expectimax = agent(depth=1)
    expectimax.cache[(b"un vieux plateau", 1)] = 0.0
    expectimax.act(TWO_TWOS, legal_actions(TWO_TWOS))
    assert (b"un vieux plateau", 1) not in expectimax.cache


# --- L'agent ------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(10))
def test_avoids_the_trap(seed: int) -> None:
    # Le glouton et l'heuristique tireraient à pile ou face entre LEFT et RIGHT.
    expectimax = ExpectimaxAgent(seed=seed, depth=1)
    assert expectimax.act(TRAP, legal_actions(TRAP)) is Action.RIGHT


def test_only_considers_moves_allowed_by_the_mask() -> None:
    # LEFT mène à la défaite, mais c'est le seul coup autorisé.
    assert agent().act(TRAP, mask(Action.LEFT)) is Action.LEFT


def test_plays_the_best_move_at_depth_two() -> None:
    expectimax = agent(depth=2)
    assert {expectimax.act(TWO_TWOS, legal_actions(TWO_TWOS)) for _ in range(5)} <= HORIZONTAL


def test_does_not_modify_the_board() -> None:
    board = TWO_TWOS.copy()
    agent(depth=2).act(board, legal_actions(board))
    np.testing.assert_array_equal(board, TWO_TWOS)


def test_breaks_ties_at_random() -> None:
    # Avec des poids nuls, aucun coup ne rapporte rien : les quatre se valent.
    expectimax = agent(weights=NO_WEIGHTS)
    assert {expectimax.act(SYMMETRIC, ALL_LEGAL) for _ in range(60)} == set(Action)


def test_same_seed_plays_the_same_moves() -> None:
    first, second = agent(weights=NO_WEIGHTS, seed=42), agent(weights=NO_WEIGHTS, seed=42)
    moves_first = [first.act(SYMMETRIC, ALL_LEGAL) for _ in range(20)]
    moves_second = [second.act(SYMMETRIC, ALL_LEGAL) for _ in range(20)]
    assert moves_first == moves_second


def test_plays_legal_moves_in_a_real_game() -> None:
    game = Game(seed=0)
    expectimax = ExpectimaxAgent(seed=0, depth=1)
    for _ in range(30):
        result = game.step(expectimax.act(game.board, game.legal_actions()))
        assert result.moved, "l'agent a joué un coup qui ne déplace aucune tuile"


def test_is_available_in_the_command_line() -> None:
    # Pour lancer `gm2048 eval --agent expectimax`, ajoute l'agent au dictionnaire AGENTS.
    from gm2048.cli import AGENTS

    assert isinstance(AGENTS["expectimax"](0), ExpectimaxAgent)
