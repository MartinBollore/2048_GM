"""Tests des coups sur bitboard (étape 2 de la phase 3).

Chaque test décrit un comportement attendu. Lance-les avec :

    pytest tests/engine/test_bitboard_moves.py

La plupart comparent le bitboard au moteur NumPy de la phase 1, plus lent mais déjà
testé : c'est la validation croisée.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from gm2048.engine import (
    ACTIONS,
    Action,
    Board,
    empty_cells,
    legal_actions,
    new_board,
    slide,
    spawn,
)
from gm2048.engine.bitboard import (
    ROW_COUNT,
    ROW_LEFT,
    ROW_MASK,
    ROW_REWARD,
    build_row_tables,
    empty_cells_bits,
    from_bitboard,
    legal_actions_bits,
    reverse_row,
    slide_bits,
    slide_row_left,
    spawn_bits,
    to_bitboard,
)

# Des exposants petits (0 à 5) donnent beaucoup de fusions et de plateaux pleins ;
# des exposants jusqu'à 14 couvrent toutes les tuiles qui peuvent encore fusionner.
small_boards = arrays(np.uint8, (4, 4), elements=st.integers(min_value=0, max_value=5))
wide_boards = arrays(np.uint8, (4, 4), elements=st.integers(min_value=0, max_value=14))
any_boards = st.one_of(small_boards, wide_boards)
rows = st.integers(min_value=0, max_value=ROW_MASK)
seeds = st.integers(min_value=0, max_value=2**32 - 1)


def row_of(*exponents: int) -> int:
    """Code une ligne à partir des exposants des colonnes 0, 1, 2 et 3."""
    return sum(e << (4 * c) for c, e in enumerate(exponents))


def board_with(row: int, col: int, exponent: int) -> Board:
    """Plateau vide sauf la case (``row``, ``col``)."""
    board = new_board()
    board[row, col] = exponent
    return board


# --- reverse_row --------------------------------------------------------------------


def test_reverse_row_example() -> None:
    assert reverse_row(0x4321) == 0x1234


def test_reverse_row_keeps_a_symmetric_row() -> None:
    assert reverse_row(0x1221) == 0x1221


@given(rows)
def test_reversing_twice_gives_the_row_back(row: int) -> None:
    assert reverse_row(reverse_row(row)) == row


@given(rows)
def test_reverse_row_stays_inside_16_bits(row: int) -> None:
    assert 0 <= reverse_row(row) <= ROW_MASK


# --- slide_row_left -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("before", "after", "points"),
    [
        ((0, 0, 0, 0), (0, 0, 0, 0), 0),  # ligne vide
        ((0, 0, 0, 1), (1, 0, 0, 0), 0),  # une tuile glisse jusqu'au bord
        ((1, 2, 3, 4), (1, 2, 3, 4), 0),  # rien ne bouge
        ((1, 1, 0, 0), (2, 0, 0, 0), 4),  # 2 + 2 = 4
        ((1, 0, 0, 1), (2, 0, 0, 0), 4),  # fusion par-dessus les trous
        ((1, 1, 1, 1), (2, 2, 0, 0), 8),  # deux fusions
        ((1, 1, 2, 0), (2, 2, 0, 0), 4),  # le 4 créé ne refusionne pas
        ((1, 1, 1, 0), (2, 1, 0, 0), 4),  # on fusionne depuis le bord
        ((3, 3, 3, 3), (4, 4, 0, 0), 32),
        ((10, 10, 0, 0), (11, 0, 0, 0), 2048),  # la tuile 2048
        ((14, 14, 0, 0), (15, 0, 0, 0), 32768),
    ],
)
def test_slide_row_left_examples(
    before: tuple[int, ...], after: tuple[int, ...], points: int
) -> None:
    assert slide_row_left(row_of(*before)) == (row_of(*after), points)


def test_two_15_do_not_merge() -> None:
    # La tuile 65 536 (exposant 16) ne tient pas sur 4 bits.
    assert slide_row_left(row_of(15, 15, 0, 0)) == (row_of(15, 15, 0, 0), 0)
    assert slide_row_left(row_of(0, 15, 0, 15)) == (row_of(15, 15, 0, 0), 0)


@given(rows)
def test_slide_row_left_stays_inside_16_bits(row: int) -> None:
    new_row, points = slide_row_left(row)
    assert 0 <= new_row <= ROW_MASK
    assert points >= 0


def test_slide_row_left_matches_the_numpy_engine_on_every_row() -> None:
    # Toutes les lignes dont les exposants vont de 0 à 14 (50 625 lignes) : on place
    # la ligne en haut d'un plateau vide et on la fait glisser avec le moteur NumPy.
    board = new_board()
    for exponents in itertools.product(range(15), repeat=4):
        board[0] = exponents
        expected = slide(board, Action.LEFT)
        new_row, points = slide_row_left(row_of(*exponents))
        assert new_row == row_of(*(int(e) for e in expected.afterstate[0])), exponents
        assert points == expected.reward, exponents


# --- Les tables -----------------------------------------------------------------------


def test_row_count() -> None:
    assert ROW_COUNT == 65_536


def test_tables_have_one_entry_per_row() -> None:
    assert len(ROW_LEFT) == ROW_COUNT
    assert len(ROW_REWARD) == ROW_COUNT


@given(rows)
def test_tables_match_slide_row_left(row: int) -> None:
    assert (ROW_LEFT[row], ROW_REWARD[row]) == slide_row_left(row)


def test_build_row_tables_builds_the_module_tables() -> None:
    assert build_row_tables() == (ROW_LEFT, ROW_REWARD)


# --- slide_bits -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("action", "row", "col"),
    [
        (Action.UP, 0, 1),
        (Action.RIGHT, 2, 3),
        (Action.DOWN, 3, 1),
        (Action.LEFT, 2, 0),
    ],
)
def test_each_action_moves_a_lone_tile_the_right_way(action: Action, row: int, col: int) -> None:
    # Un seul 2, en ligne 2 et colonne 1 : chaque coup le pousse vers un bord.
    after, points = slide_bits(to_bitboard(board_with(2, 1, 1)), action)
    assert after == to_bitboard(board_with(row, col, 1))
    assert points == 0


def test_slide_bits_accepts_a_plain_int() -> None:
    bits = to_bitboard(board_with(2, 1, 1))
    assert slide_bits(bits, 3) == slide_bits(bits, Action.LEFT)


@given(any_boards, st.sampled_from(ACTIONS))
def test_slide_bits_matches_the_numpy_engine(board: Board, action: Action) -> None:
    expected = slide(board, action)
    after, points = slide_bits(to_bitboard(board), action)
    assert np.array_equal(from_bitboard(after), expected.afterstate)
    assert points == expected.reward


@given(any_boards, st.sampled_from(ACTIONS))
def test_an_illegal_move_changes_nothing(board: Board, action: Action) -> None:
    bits = to_bitboard(board)
    after, points = slide_bits(bits, action)
    if not slide(board, action).moved:
        assert (after, points) == (bits, 0)


@given(any_boards, st.sampled_from(ACTIONS))
def test_slide_bits_returns_python_ints(board: Board, action: Action) -> None:
    after, points = slide_bits(to_bitboard(board), action)
    assert type(after) is int
    assert type(points) is int


# --- legal_actions_bits ---------------------------------------------------------------


def test_no_legal_action_on_a_blocked_board() -> None:
    blocked = np.array([[1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 1]], dtype=np.uint8)
    assert not legal_actions_bits(to_bitboard(blocked)).any()


@given(any_boards)
def test_legal_actions_bits_matches_the_numpy_engine(board: Board) -> None:
    mask = legal_actions_bits(to_bitboard(board))
    assert mask.dtype == np.bool_
    assert mask.shape == (4,)
    assert np.array_equal(mask, legal_actions(board))


# --- empty_cells_bits -----------------------------------------------------------------


def test_every_cell_of_the_empty_board_is_empty() -> None:
    assert empty_cells_bits(0) == list(range(16))


def test_a_full_board_has_no_empty_cell() -> None:
    assert empty_cells_bits(0x1111_1111_1111_1111) == []


@given(any_boards)
def test_empty_cells_bits_matches_the_numpy_engine(board: Board) -> None:
    assert empty_cells_bits(to_bitboard(board)) == empty_cells(board).tolist()


# --- spawn_bits -----------------------------------------------------------------------


@given(any_boards, seeds)
def test_spawn_bits_draws_like_the_numpy_engine(board: Board, seed: int) -> None:
    # Même graine, mêmes tirages : la même tuile apparaît au même endroit.
    if not (board == 0).any():
        return
    expected = spawn(board, np.random.default_rng(seed))
    bits = spawn_bits(to_bitboard(board), np.random.default_rng(seed))
    assert type(bits) is int
    assert np.array_equal(from_bitboard(bits), expected)


def test_spawn_bits_adds_one_tile_on_the_only_empty_cell() -> None:
    full_but_one = 0x1111_1111_1111_1110  # seule la case 0 est vide
    bits = spawn_bits(full_but_one, np.random.default_rng(0))
    assert bits & 0xF in (1, 2)
    assert bits >> 4 == full_but_one >> 4


def test_spawn_bits_refuses_a_full_board() -> None:
    with pytest.raises(ValueError):
        spawn_bits(0x1111_1111_1111_1111, np.random.default_rng(0))
