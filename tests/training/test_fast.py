"""Tests de l'apprentissage compilé avec Numba (étape Numba de la phase 3).

Chaque fonction compilée est comparée à ta version Python, déjà validée : mêmes
plateaux, mêmes notes, mêmes poids, mêmes parties. Lance-les avec :

    pytest tests/training/test_fast.py

Le premier lancement compile les fonctions, ce qui prend quelques secondes.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from gm2048.agents.ntuple import NTupleNetwork, symmetries
from gm2048.engine.bitboard import (
    BITBOARD_LIMIT,
    slide_bits,
    spawn_bits,
    to_bitboard,
    transpose,
)
from gm2048.evaluation import GameResult
from gm2048.training import best_move, play_and_learn
from gm2048.training.fast import (
    best_move_fast,
    network_arrays,
    play_and_learn_numba,
    slide_fast,
    spawn_fast,
    transpose_fast,
    update_fast,
    value_fast,
)

# Tous les bitboards possibles, y compris ceux dont le bit de poids fort est à 1
# (une case 15 avec un exposant de 8 ou plus) : c'est là que les types comptent.
bitboards = st.integers(min_value=0, max_value=BITBOARD_LIMIT - 1)
# Des plateaux réalistes, avec des fusions possibles.
boards = arrays(np.uint8, (4, 4), elements=st.integers(min_value=0, max_value=11))
actions = st.integers(min_value=0, max_value=3)

# La compilation du premier appel dépasse le délai par défaut d'Hypothesis.
compiled = settings(deadline=None, max_examples=200)

GAME_OVER = to_bitboard(
    np.array([[1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 1]], dtype=np.uint8)
)
MIXED_TUPLES = ((0, 1, 2, 3), (0, 1, 4), (5, 6, 9, 10), (15,))


def random_network(seed: int = 0, tuples: Sequence[tuple[int, ...]] | None = None) -> NTupleNetwork:
    """Un réseau aux poids tirés au hasard, pour que les valeurs ne soient pas toutes 0."""
    net = NTupleNetwork() if tuples is None else NTupleNetwork(tuples)
    rng = np.random.default_rng(seed)
    for w in net.weights:
        w[:] = rng.normal(size=w.size).astype(np.float32)
    return net


def same_weights(a: NTupleNetwork, b: NTupleNetwork) -> bool:
    return all(np.array_equal(x, y) for x, y in zip(a.weights, b.weights, strict=True))


# --- Le moteur ------------------------------------------------------------------


@given(bitboards)
@compiled
def test_transpose_fast_matches_transpose(bits: int) -> None:
    assert int(transpose_fast(np.uint64(bits))) == transpose(bits)


@given(bitboards, actions)
@compiled
def test_slide_fast_matches_slide_bits_on_any_bitboard(bits: int, action: int) -> None:
    afterstate, reward = slide_fast(np.uint64(bits), action)
    assert (int(afterstate), int(reward)) == slide_bits(bits, action)


@given(boards, actions)
@compiled
def test_slide_fast_matches_slide_bits_on_real_boards(board: np.ndarray, action: int) -> None:
    bits = to_bitboard(board)
    afterstate, reward = slide_fast(np.uint64(bits), action)
    assert (int(afterstate), int(reward)) == slide_bits(bits, action)


@pytest.mark.parametrize("seed", range(20))
def test_spawn_fast_draws_like_spawn_bits(seed: int) -> None:
    # Les deux générateurs partent de la même graine : chaque tuile doit apparaître
    # au même endroit, avec la même valeur, et les générateurs rester synchronisés.
    rng_python = np.random.default_rng(seed)
    rng_numba = np.random.default_rng(seed)
    bits = 0
    for _ in range(12):
        expected = spawn_bits(bits, rng_python)
        bits_numba = spawn_fast(np.uint64(bits), rng_numba)
        assert int(bits_numba) == expected
        bits = expected
    assert rng_python.random() == rng_numba.random()


def test_spawn_fast_keeps_a_big_tile_in_the_last_cell() -> None:
    # Une tuile 2048 dans la case 15 : le bit de poids fort du bitboard est à 1.
    rng_python = np.random.default_rng(7)
    rng_numba = np.random.default_rng(7)
    bits = 11 << 60
    for _ in range(10):
        expected = spawn_bits(bits, rng_python)
        assert int(spawn_fast(np.uint64(bits), rng_numba)) == expected
        bits = expected


def test_spawn_fast_raises_on_a_full_board() -> None:
    with pytest.raises(ValueError):
        spawn_fast(np.uint64(GAME_OVER), np.random.default_rng(0))


# --- Le réseau ------------------------------------------------------------------


def test_network_arrays_shares_the_weights_of_the_network() -> None:
    net = NTupleNetwork()
    weights, _ = network_arrays(net)
    assert isinstance(weights, tuple)
    assert len(weights) == len(net.weights)
    # Les mêmes tableaux, pas des copies : sinon le réseau n'apprendrait rien.
    assert all(w is v for w, v in zip(weights, net.weights, strict=True))


@pytest.mark.parametrize("tuples", [None, MIXED_TUPLES])
def test_network_arrays_lists_the_cells_read_for_each_symmetry(
    tuples: tuple[tuple[int, ...], ...] | None,
) -> None:
    net = NTupleNetwork() if tuples is None else NTupleNetwork(tuples)
    _, cells = network_arrays(net)
    assert isinstance(cells, tuple)
    assert len(cells) == len(net.tuples)
    syms = symmetries()
    for t, tup in enumerate(net.tuples):
        assert cells[t].dtype == np.int64
        assert cells[t].shape == (8, len(tup))
        expected = [[syms[s][c] for c in tup] for s in range(8)]
        assert cells[t].tolist() == expected


@given(bitboards)
@compiled
def test_value_fast_matches_value_exactly(bits: int) -> None:
    net = random_network()
    weights, cells = network_arrays(net)
    assert value_fast(weights, cells, np.uint64(bits)) == net.value(bits)


@given(bitboards)
@compiled
def test_value_fast_works_with_tuples_of_different_lengths(bits: int) -> None:
    net = random_network(tuples=MIXED_TUPLES)
    weights, cells = network_arrays(net)
    assert value_fast(weights, cells, np.uint64(bits)) == net.value(bits)


@pytest.mark.parametrize("tuples", [None, MIXED_TUPLES])
def test_update_fast_matches_update_exactly(tuples: tuple[tuple[int, ...], ...] | None) -> None:
    python_net, numba_net = random_network(1, tuples), random_network(1, tuples)
    weights, cells = network_arrays(numba_net)
    rng = np.random.default_rng(2)
    for _ in range(300):
        bits = int(rng.integers(0, 2**63)) | (int(rng.integers(0, 2)) << 63)
        delta = float(rng.normal(scale=50.0))
        python_net.update(bits, delta)
        update_fast(weights, cells, np.uint64(bits), delta)
    assert same_weights(python_net, numba_net)


# --- L'apprentissage ------------------------------------------------------------


@given(boards)
@compiled
def test_best_move_fast_matches_best_move(board: np.ndarray) -> None:
    net = random_network()
    weights, cells = network_arrays(net)
    bits = to_bitboard(board)
    action, afterstate, reward = best_move_fast(weights, cells, np.uint64(bits))
    expected = best_move(net, bits)
    if expected is None:
        assert action == -1
    else:
        assert (int(action), int(afterstate), int(reward)) == (
            expected.action,
            expected.afterstate,
            expected.reward,
        )


def test_best_move_fast_without_legal_move() -> None:
    weights, cells = network_arrays(NTupleNetwork())
    action, afterstate, reward = best_move_fast(weights, cells, np.uint64(GAME_OVER))
    assert (int(action), int(afterstate), int(reward)) == (-1, GAME_OVER, 0)


@pytest.mark.parametrize("first_seed", [0, 100, 200])
def test_play_and_learn_numba_plays_and_learns_like_python(first_seed: int) -> None:
    # Trois parties de suite sur le même réseau : les parties, puis les poids
    # appris, doivent être exactement les mêmes qu'avec ta version Python.
    python_net, numba_net = NTupleNetwork(), NTupleNetwork()
    for seed in range(first_seed, first_seed + 3):
        assert play_and_learn_numba(numba_net, 0.1, seed) == play_and_learn(python_net, 0.1, seed)
    assert same_weights(python_net, numba_net)


def test_play_and_learn_numba_with_tuples_of_different_lengths() -> None:
    python_net, numba_net = NTupleNetwork(MIXED_TUPLES), NTupleNetwork(MIXED_TUPLES)
    for seed in range(3):
        assert play_and_learn_numba(numba_net, 0.25, seed) == play_and_learn(python_net, 0.25, seed)
    assert same_weights(python_net, numba_net)


def test_play_and_learn_numba_returns_a_game_result() -> None:
    result = play_and_learn_numba(NTupleNetwork(), 0.1, 0)
    assert isinstance(result, GameResult)
    assert result.seed == 0
    assert type(result.score) is int
    assert type(result.moves) is int
    assert type(result.max_tile) is int
    assert result.moves > 0
    assert result.seconds > 0


def test_play_and_learn_numba_learns_in_the_network() -> None:
    net = NTupleNetwork()
    play_and_learn_numba(net, 0.1, 1)
    assert any(w.any() for w in net.weights)


def test_play_and_learn_numba_with_alpha_zero_keeps_the_weights() -> None:
    net, untouched = random_network(3), random_network(3)
    play_and_learn_numba(net, 0.0, 1)
    assert same_weights(net, untouched)


def test_play_and_learn_numba_is_much_faster() -> None:
    # Mêmes graines et réseaux neufs : les deux versions jouent les mêmes parties.
    play_and_learn_numba(NTupleNetwork(), 0.1, 0)  # compile avant de chronométrer
    seeds = range(3)
    start = time.perf_counter()
    python_net = NTupleNetwork()
    for seed in seeds:
        play_and_learn(python_net, 0.1, seed)
    python_time = time.perf_counter() - start
    start = time.perf_counter()
    numba_net = NTupleNetwork()
    for seed in seeds:
        play_and_learn_numba(numba_net, 0.1, seed)
    numba_time = time.perf_counter() - start
    assert numba_time * 10 < python_time
