"""Tests de l'apprentissage TD(0) (étape 4 de la phase 3).

Chaque test décrit un comportement attendu. Lance-les avec :

    pytest tests/training/test_td.py

Le dernier test entraîne un vrai réseau pendant 30 parties : il prend quelques
secondes.
"""

from __future__ import annotations

import dataclasses
import itertools
import statistics
from collections.abc import Sequence

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from gm2048.agents.ntuple import NTupleNetwork
from gm2048.engine import Action, ActionMask, Board
from gm2048.engine.bitboard import legal_actions_bits, slide_bits, to_bitboard
from gm2048.evaluation import GameResult, play_game
from gm2048.training import Move, best_move, play_and_learn, td_update
from gm2048.training import td as td_module

boards = arrays(np.uint8, (4, 4), elements=st.integers(min_value=0, max_value=11))


def bitboard(rows: Sequence[Sequence[int]]) -> int:
    """Le bitboard d'un plateau écrit en exposants, ligne par ligne."""
    return to_bitboard(np.array(rows, dtype=np.uint8))


def random_network(seed: int = 0) -> NTupleNetwork:
    """Un réseau aux poids tirés au hasard, pour que les valeurs ne soient pas toutes 0."""
    net = NTupleNetwork()
    rng = np.random.default_rng(seed)
    for w in net.weights:
        w[:] = rng.normal(size=w.size).astype(np.float32)
    return net


def net_features(bits: int) -> list[tuple[int, int]]:
    """Les poids que lit un réseau aux tuples par défaut pour ce plateau."""
    return NTupleNetwork().features(bits)


def copy_weights(net: NTupleNetwork) -> list[np.ndarray]:
    return [w.copy() for w in net.weights]


def same_weights(a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> bool:
    return all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


# Un plateau aux 16 exposants différents : ses 40 poids lus sont tous différents
# (vérifié à l'étape 3), donc update(PREVIOUS, d) fait monter sa note exactement de d.
PREVIOUS = bitboard([[3, 9, 0, 14], [7, 1, 12, 5], [10, 15, 4, 8], [2, 6, 13, 11]])
NEXT = bitboard([[1, 2, 0, 0], [3, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
GAME_OVER = bitboard([[1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 1]])


# --- Move -----------------------------------------------------------------------


def test_move_is_frozen() -> None:
    move = Move(action=Action.LEFT, afterstate=NEXT, reward=4)
    with pytest.raises(dataclasses.FrozenInstanceError):
        move.reward = 8  # type: ignore[misc]


# --- best_move ------------------------------------------------------------------


def test_best_move_is_none_when_no_move_is_legal() -> None:
    assert best_move(NTupleNetwork(), GAME_OVER) is None


@given(boards)
@settings(max_examples=200)
def test_best_move_returns_a_legal_move_and_its_slide(board: Board) -> None:
    bits = to_bitboard(board)
    move = best_move(random_network(), bits)
    if not legal_actions_bits(bits).any():
        assert move is None
        return
    assert move is not None
    assert isinstance(move.action, Action)
    assert legal_actions_bits(bits)[move.action]
    assert (move.afterstate, move.reward) == slide_bits(bits, move.action)


def test_best_move_with_a_new_network_takes_the_most_points() -> None:
    # Haut ne bouge rien. Bas ne rapporte rien. Gauche et droite fusionnent les
    # deux 2 (4 points) : à égalité, droite passe avant gauche dans l'ordre de Action.
    bits = bitboard([[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    move = best_move(NTupleNetwork(), bits)
    assert move == Move(Action.RIGHT, *slide_bits(bits, Action.RIGHT))
    assert move.reward == 4


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        # Une tuile au centre : les 4 coups sont légaux et valent 0, haut gagne.
        ([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], Action.UP),
        # Une tuile en haut à gauche : haut et gauche sont illégaux, droite gagne.
        ([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], Action.RIGHT),
        # Une tuile en haut à droite : seuls bas et gauche sont légaux, bas gagne.
        ([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], Action.DOWN),
    ],
)
def test_best_move_breaks_ties_in_action_order(rows: list[list[int]], expected: Action) -> None:
    move = best_move(NTupleNetwork(), bitboard(rows))
    assert move is not None
    assert move.action == expected


def test_best_move_follows_the_network_value() -> None:
    bits = bitboard([[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    down = slide_bits(bits, Action.DOWN)[0]
    right = slide_bits(bits, Action.RIGHT)[0]
    left = slide_bits(bits, Action.LEFT)[0]
    # Un poids que lit l'afterstate de bas, mais pas celles de droite et de gauche :
    # on le rend très positif, et bas devient le coup le plus prometteur.
    only_down = set(net_features(down)) - set(net_features(right)) - set(net_features(left))
    t, i = min(only_down)
    net = NTupleNetwork()
    net.weights[t][i] = 100.0
    assert net.value(down) > 4 + max(net.value(right), net.value(left))  # le décor du test
    move = best_move(net, bits)
    assert move is not None
    assert move.action == Action.DOWN


def test_best_move_never_plays_an_illegal_move_even_if_it_looks_good() -> None:
    # Haut et gauche ne bougent rien : leur « afterstate » serait le plateau lui-même.
    bits = bitboard([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    net = NTupleNetwork()
    net.update(bits, 1000.0)
    move = best_move(net, bits)
    assert move is not None
    assert move.action in (Action.RIGHT, Action.DOWN)


@given(boards)
@settings(max_examples=200)
def test_best_move_maximises_reward_plus_value(board: Board) -> None:
    net = random_network()
    bits = to_bitboard(board)
    expected: Action | None = None
    best = 0.0
    for action in Action:
        afterstate, reward = slide_bits(bits, action)
        if afterstate != bits and (expected is None or reward + net.value(afterstate) > best):
            expected, best = action, reward + net.value(afterstate)
    move = best_move(net, bits)
    assert (move.action if move else None) == expected


def test_best_move_does_not_change_the_network() -> None:
    net = random_network()
    before = copy_weights(net)
    best_move(net, PREVIOUS)
    assert same_weights(before, net.weights)


# --- td_update ------------------------------------------------------------------


def test_td_update_by_hand() -> None:
    # Réseau neuf : V vaut 0 partout. Cible = 8 + 0 = 8, delta = 8 - 0 = 8.
    # Avec alpha = 0.5, la note de PREVIOUS monte de 0.5 * 8 = 4.
    net = NTupleNetwork()
    delta = td_update(net, PREVIOUS, Move(Action.LEFT, NEXT, 8), alpha=0.5)
    assert delta == pytest.approx(8.0)
    assert net.value(PREVIOUS) == pytest.approx(4.0, rel=1e-5)


def test_td_update_compares_target_and_prediction() -> None:
    net = random_network()
    v_previous, v_next = net.value(PREVIOUS), net.value(NEXT)
    delta = td_update(net, PREVIOUS, Move(Action.LEFT, NEXT, 16), alpha=0.1)
    assert delta == pytest.approx(16 + v_next - v_previous, rel=1e-5, abs=1e-4)
    assert net.value(PREVIOUS) == pytest.approx(v_previous + 0.1 * delta, rel=1e-5, abs=1e-4)


def test_td_update_targets_zero_at_the_end_of_the_game() -> None:
    net = random_network()
    v_previous = net.value(PREVIOUS)
    delta = td_update(net, PREVIOUS, None, alpha=1.0)
    assert delta == pytest.approx(-v_previous, rel=1e-5, abs=1e-4)
    assert net.value(PREVIOUS) == pytest.approx(0.0, abs=1e-4)


def test_td_update_with_alpha_zero_changes_nothing() -> None:
    net = random_network()
    before = copy_weights(net)
    delta = td_update(net, PREVIOUS, Move(Action.LEFT, NEXT, 8), alpha=0.0)
    assert delta != 0.0
    assert same_weights(before, net.weights)


def test_td_update_only_corrects_the_previous_afterstate() -> None:
    net = random_network()
    before = copy_weights(net)
    td_update(net, PREVIOUS, Move(Action.LEFT, NEXT, 8), alpha=0.1)
    changed = {
        (t, int(i)) for t, w in enumerate(net.weights) for i in np.flatnonzero(w != before[t])
    }
    assert changed
    assert changed <= set(net.features(PREVIOUS))


# --- play_and_learn -------------------------------------------------------------


class NetworkAgent:
    """Un agent qui joue best_move sur le moteur NumPy, sans rien apprendre."""

    name = "network"

    def __init__(self, net: NTupleNetwork) -> None:
        self.net = net

    def act(self, board: Board, legal_mask: ActionMask) -> Action:
        move = best_move(self.net, to_bitboard(board))
        assert move is not None
        return move.action


def test_play_and_learn_returns_a_game_result() -> None:
    result = play_and_learn(NTupleNetwork(), alpha=0.1, seed=0)
    assert isinstance(result, GameResult)
    assert result.seed == 0
    assert result.moves > 0
    assert result.score > 0
    assert result.max_tile >= 8
    assert result.max_tile & (result.max_tile - 1) == 0  # une puissance de 2
    assert result.seconds > 0


@pytest.mark.parametrize("seed", range(5))
def test_play_and_learn_plays_the_same_game_as_the_numpy_engine(seed: int) -> None:
    # Avec alpha = 0, rien n'est appris : la partie doit être exactement celle que
    # joue le même réseau dans Game, le moteur de la phase 1 (même score, même
    # tuile max, même nombre de coups ; la durée n'est pas comparée).
    net = random_network(seed)
    assert play_and_learn(net, alpha=0.0, seed=seed) == play_game(NetworkAgent(net), seed)


def test_play_and_learn_with_alpha_zero_keeps_the_weights() -> None:
    net = random_network()
    before = copy_weights(net)
    play_and_learn(net, alpha=0.0, seed=1)
    assert same_weights(before, net.weights)


def test_play_and_learn_changes_the_weights_when_alpha_is_positive() -> None:
    net = NTupleNetwork()
    play_and_learn(net, alpha=0.1, seed=1)
    assert any(w.any() for w in net.weights)


def test_play_and_learn_is_reproducible() -> None:
    a, b = NTupleNetwork(), NTupleNetwork()
    assert play_and_learn(a, alpha=0.1, seed=7) == play_and_learn(b, alpha=0.1, seed=7)
    assert same_weights(a.weights, b.weights)


def test_play_and_learn_corrects_each_afterstate_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, Move | None, float]] = []
    real_td_update = td_module.td_update

    def recording_td_update(
        net: NTupleNetwork, afterstate: int, next_move: Move | None, alpha: float
    ) -> float:
        calls.append((afterstate, next_move, alpha))
        return real_td_update(net, afterstate, next_move, alpha)

    monkeypatch.setattr(td_module, "td_update", recording_td_update)
    result = play_and_learn(NTupleNetwork(), alpha=0.25, seed=3)

    assert len(calls) == result.moves  # une correction par coup joué
    assert all(alpha == 0.25 for _, _, alpha in calls)
    # Chaque correction vise l'afterstate jouée juste avant le coup qu'elle reçoit.
    for (_, next_move, _), (afterstate, _, _) in itertools.pairwise(calls):
        assert next_move is not None
        assert next_move.afterstate == afterstate
    # La dernière correction est celle de fin de partie, avec une cible de 0.
    assert calls[-1][1] is None


def test_a_short_training_improves_the_play() -> None:
    def mean_score(net: NTupleNetwork) -> float:
        # alpha = 0 : on mesure le niveau sans continuer à apprendre.
        return statistics.mean(play_and_learn(net, 0.0, 10_000 + s).score for s in range(20))

    net = NTupleNetwork()
    before = mean_score(net)
    for seed in range(30):
        play_and_learn(net, alpha=0.1, seed=seed)
    assert mean_score(net) > 1.2 * before
