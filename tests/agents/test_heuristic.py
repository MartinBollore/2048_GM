"""Tests de l'agent heuristique (étape 4 de la phase 2).

Chaque test décrit un comportement attendu. Lance-les avec :

    pytest tests/agents/test_heuristic.py

Pour avancer critère par critère, filtre avec ``-k`` :

    pytest tests/agents/test_heuristic.py -k empty

Les plateaux sont écrits avec les valeurs des tuiles (0 = case vide), comme on
les voit à l'écran ; ``from_values`` les convertit en exposants. Dans les
commentaires, les calculs sont faits sur les exposants.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from gm2048.agents import Agent
from gm2048.agents.baselines import GreedyAgent, HeuristicAgent, HeuristicWeights
from gm2048.agents.baselines.heuristic import (
    DEFAULT_WEIGHTS,
    corner_score,
    empty_score,
    evaluate_board,
    monotonicity_score,
    smoothness_score,
)
from gm2048.engine import Action, Game, from_values, new_board, slide
from gm2048.evaluation import evaluate

HORIZONTAL = {Action.LEFT, Action.RIGHT}
VERTICAL = {Action.UP, Action.DOWN}
ALL_LEGAL = np.array([True, True, True, True])
NO_WEIGHTS = HeuristicWeights(empty=0.0, monotonicity=0.0, smoothness=0.0, corner=0.0)


def mask(*actions: Action) -> np.ndarray:
    """Masque où seuls les coups donnés sont légaux."""
    result = np.zeros(4, dtype=np.bool_)
    for action in actions:
        result[action] = True
    return result


def corner_only(weight: float) -> HeuristicWeights:
    """Poids qui ne regardent que le critère du coin."""
    return HeuristicWeights(empty=0.0, monotonicity=0.0, smoothness=0.0, corner=weight)


# Rangé : chaque ligne décroît de gauche à droite, chaque colonne de haut en bas.
SORTED = from_values(
    [
        [64, 32, 16, 8],
        [32, 16, 8, 4],
        [16, 8, 4, 2],
        [8, 4, 2, 0],
    ]
)

# Les mêmes tuiles, mélangées.
SHUFFLED = from_values(
    [
        [2, 64, 4, 32],
        [16, 0, 8, 8],
        [4, 32, 2, 16],
        [8, 4, 16, 2],
    ]
)

# Une seule ligne en désordre : exposants 1 3 2 0, écarts +2 -1 -2.
# 2 de montée, 3 de descente : pénalité 2. Les colonnes (une tuile puis du vide) sont rangées.
ONE_MESSY_ROW = from_values(
    [
        [2, 8, 4, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
)

# Deux lignes rangées en sens opposés (exposants 1 2 3 4 et 4 3 2 1) : pénalité 0 chacune.
# Colonnes : 1 4 0 0 → +3 -4 → 3 ; 2 3 0 0 → +1 -3 → 1 ; 3 2 0 0 → 0 ; 4 1 0 0 → 0.
# Total : 4. On calcule bien une pénalité par ligne, pas une seule pour tout le plateau.
OPPOSITE_ROWS = from_values(
    [
        [2, 4, 8, 16],
        [16, 8, 4, 2],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
)

# Paires de voisines non vides : (2, 8) sur la ligne du haut, écart |1 - 3| = 2,
# et (2, 32) dans la colonne de gauche, écart |1 - 5| = 4. Le 4 n'a que des cases vides
# autour de lui. Total : 6.
ROUGH = from_values(
    [
        [2, 8, 0, 4],
        [32, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
)

# Un plateau où les quatre critères sont non nuls :
# 12 cases vides ; monotonie -2 (ligne du haut 6 1 3 0 : 2 de montée, 8 de descente) ;
# régularité -11 (|6 - 1| + |1 - 3| sur la ligne, |6 - 2| dans la colonne) ; coin : 64, donc 6.
RICH = from_values(
    [
        [64, 2, 8, 0],
        [4, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
)

# Dilemme entre points immédiats et note du plateau :
# - UP ou DOWN fusionne les deux 4 (8 points), mais le 64 quitte la ligne du haut sans
#   arriver dans un coin (UP le laisse en place, DOWN l'envoie en bas au milieu) ;
# - LEFT ou RIGHT ne rapporte aucun point, mais pousse le 64 dans un coin (exposant 6).
DILEMMA = from_values(
    [
        [0, 64, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 4],
        [0, 0, 0, 4],
    ]
)

# Une seule tuile au centre : les quatre coups donnent des plateaux symétriques, de même note.
SYMMETRIC = from_values(
    [
        [0, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
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


def played_moves(
    board: np.ndarray, legal_mask: np.ndarray, weights: HeuristicWeights, n: int = 200
) -> set[Action]:
    """Les coups joués par un même agent sur ``n`` appels avec le même plateau."""
    agent = HeuristicAgent(seed=0, weights=weights)
    return {agent.act(board, legal_mask) for _ in range(n)}


# --- Critère 1 : les cases vides --------------------------------------------------


def test_empty_score_of_an_empty_board() -> None:
    assert empty_score(new_board()) == 16.0


def test_empty_score_of_a_full_board() -> None:
    assert empty_score(SORTED + 1) == 0.0


def test_empty_score_counts_the_empty_cells() -> None:
    assert empty_score(ROUGH) == 12.0


# --- Critère 2 : la monotonie -------------------------------------------------------


def test_monotonicity_of_an_empty_board_is_zero() -> None:
    assert monotonicity_score(new_board()) == 0.0


def test_monotonicity_of_a_sorted_board_is_zero() -> None:
    assert monotonicity_score(SORTED) == 0.0


@pytest.mark.parametrize("flip", [np.fliplr, np.flipud, np.rot90])
def test_monotonicity_does_not_care_about_the_direction(
    flip: Callable[[np.ndarray], np.ndarray],
) -> None:
    assert monotonicity_score(np.ascontiguousarray(flip(SORTED))) == 0.0


def test_monotonicity_penalizes_a_messy_row() -> None:
    assert monotonicity_score(ONE_MESSY_ROW) == -2.0


def test_monotonicity_also_looks_at_columns() -> None:
    assert monotonicity_score(np.ascontiguousarray(ONE_MESSY_ROW.T)) == -2.0


def test_monotonicity_computes_one_penalty_per_line() -> None:
    assert monotonicity_score(OPPOSITE_ROWS) == -4.0


def test_a_sorted_board_is_more_monotonic_than_a_shuffled_one() -> None:
    assert monotonicity_score(SORTED) > monotonicity_score(SHUFFLED)


# --- Critère 3 : la régularité ------------------------------------------------------


def test_smoothness_of_an_empty_board_is_zero() -> None:
    assert smoothness_score(new_board()) == 0.0


def test_smoothness_of_identical_neighbours_is_zero() -> None:
    assert smoothness_score(np.ones((4, 4), dtype=np.uint8)) == 0.0


def test_smoothness_ignores_empty_cells() -> None:
    board = from_values(
        [
            [2, 0, 64, 0],
            [0, 0, 0, 0],
            [1024, 0, 0, 8],
            [0, 0, 0, 0],
        ]
    )
    assert smoothness_score(board) == 0.0


def test_smoothness_adds_up_the_gaps_between_neighbours() -> None:
    assert smoothness_score(ROUGH) == -6.0


def test_a_sorted_board_is_smoother_than_a_shuffled_one() -> None:
    assert smoothness_score(SORTED) > smoothness_score(SHUFFLED)


# --- Critère 4 : le coin ------------------------------------------------------------


def test_corner_score_of_an_empty_board_is_zero() -> None:
    assert corner_score(new_board()) == 0.0


@pytest.mark.parametrize("corner", [(0, 0), (0, 3), (3, 0), (3, 3)])
def test_corner_score_is_the_exponent_of_the_biggest_tile_in_a_corner(
    corner: tuple[int, int],
) -> None:
    board = from_values(
        [
            [0, 0, 0, 0],
            [0, 2, 8, 0],
            [0, 4, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    board[corner] = 6  # une tuile 64
    assert corner_score(board) == 6.0


@pytest.mark.parametrize("position", [(0, 1), (1, 1), (2, 3)])
def test_corner_score_is_zero_when_the_biggest_tile_is_not_in_a_corner(
    position: tuple[int, int],
) -> None:
    board = from_values(
        [
            [2, 0, 0, 4],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [8, 0, 0, 2],
        ]
    )
    board[position] = 6  # une tuile 64, hors des coins qui ne contiennent que des petites tuiles
    assert corner_score(board) == 0.0


def test_corner_score_when_the_biggest_tile_appears_twice() -> None:
    board = from_values(
        [
            [0, 0, 0, 0],
            [0, 64, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 64],
        ]
    )
    assert corner_score(board) == 6.0


# --- La note du plateau -------------------------------------------------------------


def test_evaluate_board_with_zero_weights_is_zero() -> None:
    assert evaluate_board(RICH, NO_WEIGHTS) == 0.0


@pytest.mark.parametrize(
    ("weights", "expected"),
    [
        (HeuristicWeights(empty=1.0, monotonicity=0.0, smoothness=0.0, corner=0.0), 12.0),
        (HeuristicWeights(empty=0.0, monotonicity=1.0, smoothness=0.0, corner=0.0), -2.0),
        (HeuristicWeights(empty=0.0, monotonicity=0.0, smoothness=1.0, corner=0.0), -11.0),
        (HeuristicWeights(empty=0.0, monotonicity=0.0, smoothness=0.0, corner=1.0), 6.0),
    ],
)
def test_evaluate_board_uses_each_weight_for_its_criterion(
    weights: HeuristicWeights, expected: float
) -> None:
    assert evaluate_board(RICH, weights) == pytest.approx(expected)


def test_evaluate_board_adds_up_the_weighted_criteria() -> None:
    weights = HeuristicWeights(empty=2.0, monotonicity=3.0, smoothness=0.5, corner=4.0)
    # 2 * 12 + 3 * (-2) + 0.5 * (-11) + 4 * 6
    assert evaluate_board(RICH, weights) == pytest.approx(36.5)


# --- L'agent : le contrat -----------------------------------------------------------


def test_respects_the_agent_contract() -> None:
    agent = HeuristicAgent(seed=0)
    assert isinstance(agent, Agent)
    assert agent.name == "heuristic"


def test_uses_the_default_weights_unless_told_otherwise() -> None:
    assert HeuristicAgent(seed=0).weights == DEFAULT_WEIGHTS
    custom = corner_only(3.0)
    assert HeuristicAgent(seed=0, weights=custom).weights == custom


def test_returns_an_action() -> None:
    assert isinstance(HeuristicAgent(seed=0).act(SYMMETRIC, ALL_LEGAL), Action)


def test_raises_when_no_move_is_legal() -> None:
    with pytest.raises(ValueError):
        HeuristicAgent(seed=0).act(SYMMETRIC, mask())


def test_does_not_modify_the_board() -> None:
    board = DILEMMA.copy()
    HeuristicAgent(seed=0).act(board, ALL_LEGAL)
    np.testing.assert_array_equal(board, DILEMMA)


def test_plays_the_only_legal_move() -> None:
    assert played_moves(ONLY_DOWN, mask(Action.DOWN), DEFAULT_WEIGHTS, n=20) == {Action.DOWN}


# --- L'agent : points gagnés + note du plateau --------------------------------------


def test_with_zero_weights_it_plays_like_the_greedy_agent() -> None:
    # Note nulle partout : seuls comptent les 8 points de la fusion verticale.
    assert played_moves(DILEMMA, ALL_LEGAL, NO_WEIGHTS) == VERTICAL


def test_a_small_corner_bonus_does_not_beat_the_points() -> None:
    # LEFT/RIGHT : 0 point + 1 * 6 = 6. UP/DOWN : 8 points + 0 = 8.
    assert played_moves(DILEMMA, ALL_LEGAL, corner_only(1.0)) == VERTICAL


def test_a_big_corner_bonus_beats_the_points() -> None:
    # LEFT/RIGHT : 0 point + 2 * 6 = 12. UP/DOWN : 8 points + 0 = 8.
    assert played_moves(DILEMMA, ALL_LEGAL, corner_only(2.0)) == HORIZONTAL


def test_only_considers_moves_allowed_by_the_mask() -> None:
    assert played_moves(DILEMMA, mask(*VERTICAL), corner_only(2.0)) == VERTICAL


# --- L'agent : les égalités ---------------------------------------------------------


def test_breaks_ties_at_random() -> None:
    assert played_moves(SYMMETRIC, ALL_LEGAL, DEFAULT_WEIGHTS) == set(Action)


def test_same_seed_plays_the_same_moves() -> None:
    first, second = HeuristicAgent(seed=42), HeuristicAgent(seed=42)
    moves_first = [first.act(SYMMETRIC, ALL_LEGAL) for _ in range(50)]
    moves_second = [second.act(SYMMETRIC, ALL_LEGAL) for _ in range(50)]
    assert moves_first == moves_second


def test_different_seeds_play_different_moves() -> None:
    first, second = HeuristicAgent(seed=1), HeuristicAgent(seed=2)
    moves_first = [first.act(SYMMETRIC, ALL_LEGAL) for _ in range(50)]
    moves_second = [second.act(SYMMETRIC, ALL_LEGAL) for _ in range(50)]
    assert moves_first != moves_second


# --- Sur de vraies parties ----------------------------------------------------------


@pytest.mark.parametrize("seed", range(2))
def test_always_plays_a_move_with_the_best_value(seed: int) -> None:
    game = Game(seed=seed)
    agent = HeuristicAgent(seed=seed)

    def value(board: np.ndarray, action: Action) -> float:
        result = slide(board, action)
        return result.reward + evaluate_board(result.afterstate, DEFAULT_WEIGHTS)

    for _ in range(150):
        if game.done:
            break
        board, legal_mask = game.board, game.legal_actions()
        best = max(value(board, action) for action in Action if legal_mask[action])
        action = agent.act(board, legal_mask)
        assert legal_mask[action], "l'agent a joué un coup illégal"
        assert value(board, action) == pytest.approx(best)
        game.step(action)


def test_scores_more_than_the_greedy_agent() -> None:
    heuristic = evaluate(HeuristicAgent(seed=0), n_games=3)
    greedy = evaluate(GreedyAgent(seed=0), n_games=3)
    assert heuristic.mean_score > greedy.mean_score


def test_is_available_in_the_command_line() -> None:
    # Pour lancer `gm2048 eval --agent heuristic`, ajoute l'agent au dictionnaire AGENTS de cli.py.
    from gm2048.cli import AGENTS

    assert isinstance(AGENTS["heuristic"](0), HeuristicAgent)
