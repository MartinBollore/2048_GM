"""Tests du réseau n-tuple (étape 3 de la phase 3).

Chaque test décrit un comportement attendu. Lance-les avec :

    pytest tests/agents/test_ntuple_network.py
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from gm2048.agents.ntuple import DEFAULT_TUPLES, NTupleNetwork, symmetries, tuple_index
from gm2048.engine import Board
from gm2048.engine.bitboard import BITBOARD_LIMIT, from_bitboard, to_bitboard, transpose

COUNTING = 0xFEDCBA9876543210
"""Le bitboard où la case i contient l'exposant i."""

boards = arrays(np.uint8, (4, 4), elements=st.integers(min_value=0, max_value=15))
bitboards = st.integers(min_value=0, max_value=BITBOARD_LIMIT - 1)

# Les 8 transformations du carré, écrites avec NumPy : 4 rotations, et les mêmes
# après un miroir gauche-droite.
NUMPY_SYMMETRIES: list[Callable[[Board], Board]] = [
    lambda b: b,
    lambda b: np.rot90(b, 1),
    lambda b: np.rot90(b, 2),
    lambda b: np.rot90(b, 3),
    lambda b: np.fliplr(b),
    lambda b: np.rot90(np.fliplr(b), 1),
    lambda b: np.rot90(np.fliplr(b), 2),
    lambda b: np.rot90(np.fliplr(b), 3),
]


def random_network(seed: int = 0) -> NTupleNetwork:
    """Un réseau aux poids tirés au hasard, pour que les valeurs ne soient pas toutes 0."""
    net = NTupleNetwork()
    rng = np.random.default_rng(seed)
    for w in net.weights:
        w[:] = rng.normal(size=w.size).astype(np.float32)
    return net


def board_with_distinct_features() -> int:
    """Un plateau dont les 40 poids lus sont tous différents (vérifié par les tests).

    Ses 16 exposants sont tous différents : deux lectures ne peuvent donc tomber sur
    le même indice que si elles lisent les mêmes cases dans le même ordre.
    """
    board = np.array([[3, 9, 0, 14], [7, 1, 12, 5], [10, 15, 4, 8], [2, 6, 13, 11]])
    return to_bitboard(board.astype(np.uint8))


# --- symmetries ---------------------------------------------------------------------


def test_there_are_8_symmetries() -> None:
    assert len(symmetries()) == 8


def test_each_symmetry_is_a_permutation_of_the_16_cells() -> None:
    for symmetry in symmetries():
        assert sorted(symmetry) == list(range(16))


def test_the_symmetries_are_all_different_and_include_the_identity() -> None:
    found = symmetries()
    assert len(set(found)) == 8
    assert tuple(range(16)) in found


def test_the_symmetries_are_the_8_transformations_of_the_square() -> None:
    # Lire les cases dans l'ordre d'une symétrie donne le plateau tourné ou retourné.
    board = np.arange(16).reshape(4, 4)
    found = {tuple(board.flatten()[list(s)]) for s in symmetries()}
    expected = {tuple(f(board).flatten()) for f in NUMPY_SYMMETRIES}
    assert found == expected


# --- tuple_index --------------------------------------------------------------------


def test_index_of_an_empty_board_is_zero() -> None:
    assert tuple_index(0, (0, 1, 2, 3)) == 0


@pytest.mark.parametrize(
    ("cells", "expected"),
    [
        ((0, 1, 2, 3), 0x3210),  # la première case dans les 4 bits de poids faible
        ((3, 2, 1, 0), 0x0123),  # l'ordre des cases compte
        ((5, 6, 9, 10), 0xA965),
        ((15,), 0xF),
        ((0, 1, 2, 3, 4, 5), 0x543210),
    ],
)
def test_index_on_the_counting_board(cells: tuple[int, ...], expected: int) -> None:
    assert tuple_index(COUNTING, cells) == expected


@given(bitboards)
def test_index_stays_inside_the_table(bits: int) -> None:
    for cells in DEFAULT_TUPLES:
        assert 0 <= tuple_index(bits, cells) < 16 ** len(cells)


@given(boards)
def test_index_reads_the_exponents_of_the_cells(board: Board) -> None:
    index = tuple_index(to_bitboard(board), (5, 6, 9, 10))
    flat = [int(e) for e in board.flatten()]
    assert index == flat[5] + 16 * flat[6] + 256 * flat[9] + 4096 * flat[10]


# --- Création du réseau -------------------------------------------------------------


def test_default_tuples() -> None:
    assert DEFAULT_TUPLES == ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 4, 5), (1, 2, 5, 6), (5, 6, 9, 10))
    assert NTupleNetwork().tuples == DEFAULT_TUPLES


def test_one_table_of_float32_zeros_per_tuple() -> None:
    net = NTupleNetwork()
    assert len(net.weights) == 5
    for w in net.weights:
        assert w.shape == (16**4,)
        assert w.dtype == np.float32
        assert not w.any()


def test_n_weights_counts_every_weight() -> None:
    assert NTupleNetwork().n_weights == 5 * 65_536
    assert NTupleNetwork([(0, 1, 2, 3, 4, 5), (0, 1)]).n_weights == 16**6 + 16**2


@pytest.mark.parametrize(
    "tuples",
    [
        [],  # aucun tuple
        [()],  # un tuple vide
        [(0, 1, 16)],  # une case qui n'existe pas
        [(-1, 0)],
        [(0, 1, 1)],  # la même case deux fois
        [(0, 1, 2, 3, 4, 5, 6)],  # 7 cases : la table aurait 268 millions de poids
    ],
)
def test_refuses_bad_tuples(tuples: list[tuple[int, ...]]) -> None:
    with pytest.raises(ValueError):
        NTupleNetwork(tuples)


@given(bitboards)
def test_a_new_network_values_every_board_at_zero(bits: int) -> None:
    assert NTupleNetwork().value(bits) == 0.0


# --- features -----------------------------------------------------------------------


@given(bitboards)
def test_40_weights_are_read_with_the_default_tuples(bits: int) -> None:
    features = NTupleNetwork().features(bits)
    assert len(features) == 5 * 8
    for table, index in features:
        assert 0 <= table < 5
        assert 0 <= index < 65_536


@given(bitboards)
def test_features_include_each_tuple_read_without_symmetry(bits: int) -> None:
    features = NTupleNetwork().features(bits)
    for table, cells in enumerate(DEFAULT_TUPLES):
        assert (table, tuple_index(bits, cells)) in features


@given(boards, st.sampled_from(NUMPY_SYMMETRIES))
def test_symmetric_boards_read_the_same_weights(
    board: Board, symmetry: Callable[[Board], Board]
) -> None:
    net = NTupleNetwork()
    original = net.features(to_bitboard(board))
    turned = net.features(to_bitboard(np.ascontiguousarray(symmetry(board))))
    assert sorted(original) == sorted(turned)


def test_the_test_board_reads_40_different_weights() -> None:
    features = NTupleNetwork().features(board_with_distinct_features())
    assert len(set(features)) == 40


# --- value --------------------------------------------------------------------------


@given(bitboards)
def test_value_is_the_sum_of_the_weights_read(bits: int) -> None:
    net = random_network()
    expected = sum(float(net.weights[t][i]) for t, i in net.features(bits))
    assert net.value(bits) == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_value_returns_a_python_float() -> None:
    assert type(random_network().value(COUNTING)) is float


@given(boards, st.sampled_from(NUMPY_SYMMETRIES))
def test_symmetric_boards_have_the_same_value(
    board: Board, symmetry: Callable[[Board], Board]
) -> None:
    net = random_network()
    turned = to_bitboard(np.ascontiguousarray(symmetry(board)))
    assert net.value(turned) == pytest.approx(net.value(to_bitboard(board)), rel=1e-5, abs=1e-5)


# --- update -------------------------------------------------------------------------


def test_update_raises_the_value_by_delta() -> None:
    net = random_network()
    bits = board_with_distinct_features()
    before = net.value(bits)
    net.update(bits, 2.5)
    assert net.value(bits) == pytest.approx(before + 2.5, rel=1e-5)


def test_update_shares_delta_between_the_40_weights_read() -> None:
    net = NTupleNetwork()
    bits = board_with_distinct_features()
    net.update(bits, 4.0)
    for t, i in net.features(bits):
        assert net.weights[t][i] == pytest.approx(4.0 / 40)


def test_update_leaves_the_other_weights_alone() -> None:
    net = NTupleNetwork()
    net.update(board_with_distinct_features(), 4.0)
    assert sum(int(np.count_nonzero(w)) for w in net.weights) == 40


def test_a_weight_read_several_times_gets_its_share_several_times() -> None:
    # Sur le plateau vide, chaque tuple lit 8 fois le même poids (l'indice 0) : ce
    # poids reçoit donc 8 fois la part de delta.
    net = NTupleNetwork()
    net.update(0, 4.0)
    for w in net.weights:
        assert w[0] == pytest.approx(8 * 4.0 / 40)


def test_update_also_raises_the_symmetric_boards() -> None:
    net = NTupleNetwork()
    board = from_bitboard(board_with_distinct_features())
    net.update(to_bitboard(board), 1.0)
    for symmetry in NUMPY_SYMMETRIES:
        turned = to_bitboard(np.ascontiguousarray(symmetry(board)))
        assert net.value(turned) == pytest.approx(1.0, rel=1e-5)


# --- save et load -------------------------------------------------------------------


def test_save_then_load_gives_the_same_network(tmp_path: Path) -> None:
    net = random_network(seed=3)
    path = tmp_path / "reseau.npz"
    net.save(path)
    loaded = NTupleNetwork.load(path)
    assert loaded.tuples == net.tuples
    assert len(loaded.weights) == len(net.weights)
    for mine, theirs in zip(net.weights, loaded.weights, strict=True):
        assert theirs.dtype == np.float32
        assert np.array_equal(mine, theirs)


def test_a_loaded_network_gives_the_same_values(tmp_path: Path) -> None:
    net = random_network(seed=4)
    path = tmp_path / "reseau.npz"
    net.save(path)
    loaded = NTupleNetwork.load(path)
    for bits in (0, COUNTING, transpose(COUNTING), board_with_distinct_features()):
        assert loaded.value(bits) == net.value(bits)


def test_save_and_load_keep_custom_tuples(tmp_path: Path) -> None:
    net = NTupleNetwork([(0, 1, 2), (15, 14), (3, 7, 11, 15)])
    net.update(COUNTING, 1.0)
    path = tmp_path / "reseau.npz"
    net.save(path)
    loaded = NTupleNetwork.load(path)
    assert loaded.tuples == ((0, 1, 2), (15, 14), (3, 7, 11, 15))
    assert loaded.value(COUNTING) == pytest.approx(1.0, rel=1e-5)
