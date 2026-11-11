"""Tests du bitboard (étape 1 de la phase 3).

Chaque test décrit un comportement attendu. Lance-les avec :

    pytest tests/engine/test_bitboard.py
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from gm2048.engine import Board, new_board
from gm2048.engine.bitboard import (
    BITBOARD_LIMIT,
    ROW_MASK,
    from_bitboard,
    get_cell,
    get_row,
    set_row,
    to_bitboard,
    transpose,
)

# Tous les exposants possibles, de 0 (case vide) à 15.
boards = arrays(np.uint8, (4, 4), elements=st.integers(min_value=0, max_value=15))
bitboards = st.integers(min_value=0, max_value=BITBOARD_LIMIT - 1)
rows = st.integers(min_value=0, max_value=3)
row_values = st.integers(min_value=0, max_value=ROW_MASK)

COUNTING = 0xFEDCBA9876543210
"""Le bitboard des exposants 0 à 15 rangés dans l'ordre des cases."""


def counting_board() -> Board:
    """Le plateau [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]."""
    return np.arange(16, dtype=np.uint8).reshape(4, 4)


def board_with(row: int, col: int, exponent: int) -> Board:
    """Plateau vide sauf la case (``row``, ``col``)."""
    board = new_board()
    board[row, col] = exponent
    return board


# --- to_bitboard ----------------------------------------------------------------


def test_empty_board_is_zero() -> None:
    assert to_bitboard(new_board()) == 0


@pytest.mark.parametrize(
    ("row", "col", "expected"),
    [
        (0, 0, 0x1),  # en haut à gauche : les 4 bits de poids faible
        (0, 1, 0x10),  # la case suivante : 4 bits plus haut
        (0, 3, 0x1000),
        (1, 0, 0x1_0000),  # la deuxième ligne commence au bit 16
        (2, 1, 0x10_0000_0000),
        (3, 3, 0x1000_0000_0000_0000),  # en bas à droite : les 4 bits de poids fort
    ],
)
def test_each_cell_has_its_own_four_bits(row: int, col: int, expected: int) -> None:
    assert to_bitboard(board_with(row, col, 1)) == expected


def test_the_exponent_is_written_in_the_cell() -> None:
    assert to_bitboard(board_with(0, 0, 11)) == 0xB  # la tuile 2048
    assert to_bitboard(board_with(1, 2, 11)) == 0xB << 24


def test_counting_board() -> None:
    assert to_bitboard(counting_board()) == COUNTING


def test_a_full_board_of_15_uses_all_64_bits() -> None:
    board = np.full((4, 4), 15, dtype=np.uint8)
    assert to_bitboard(board) == BITBOARD_LIMIT - 1


def test_returns_a_python_int() -> None:
    # Un entier NumPy est limité à 64 bits et se comporte différemment de int :
    # le bitboard doit être un vrai int Python.
    assert type(to_bitboard(counting_board())) is int


def test_does_not_modify_the_board() -> None:
    board = counting_board()
    to_bitboard(board)
    assert np.array_equal(board, counting_board())


def test_refuses_an_exponent_that_does_not_fit_in_four_bits() -> None:
    with pytest.raises(ValueError):
        to_bitboard(board_with(2, 2, 16))


# --- from_bitboard ---------------------------------------------------------------


def test_zero_is_the_empty_board() -> None:
    assert np.array_equal(from_bitboard(0), new_board())


def test_from_bitboard_returns_a_4x4_uint8_board() -> None:
    board = from_bitboard(COUNTING)
    assert board.shape == (4, 4)
    assert board.dtype == np.uint8


def test_decodes_the_counting_board() -> None:
    assert np.array_equal(from_bitboard(COUNTING), counting_board())


def test_decodes_the_highest_cell() -> None:
    assert np.array_equal(from_bitboard(0xB << 60), board_with(3, 3, 11))


@pytest.mark.parametrize("bits", [-1, BITBOARD_LIMIT, BITBOARD_LIMIT + 5])
def test_refuses_an_integer_outside_64_bits(bits: int) -> None:
    with pytest.raises(ValueError):
        from_bitboard(bits)


@given(boards)
def test_board_round_trip(board: Board) -> None:
    assert np.array_equal(from_bitboard(to_bitboard(board)), board)


@given(bitboards)
def test_bitboard_round_trip(bits: int) -> None:
    assert to_bitboard(from_bitboard(bits)) == bits


# --- get_cell ---------------------------------------------------------------------


def test_get_cell_on_the_counting_board() -> None:
    for row in range(4):
        for col in range(4):
            assert get_cell(COUNTING, row, col) == 4 * row + col


@given(boards)
def test_get_cell_reads_the_same_value_as_numpy(board: Board) -> None:
    bits = to_bitboard(board)
    for row in range(4):
        for col in range(4):
            assert get_cell(bits, row, col) == board[row, col]


# --- get_row ----------------------------------------------------------------------


@pytest.mark.parametrize(("row", "expected"), [(0, 0x3210), (1, 0x7654), (2, 0xBA98), (3, 0xFEDC)])
def test_get_row_on_the_counting_board(row: int, expected: int) -> None:
    assert get_row(COUNTING, row) == expected


@given(boards, rows)
def test_get_row_puts_column_0_in_the_low_bits(board: Board, row: int) -> None:
    expected = sum(int(board[row, col]) << (4 * col) for col in range(4))
    assert get_row(to_bitboard(board), row) == expected


@given(bitboards, rows)
def test_a_row_fits_in_16_bits(bits: int, row: int) -> None:
    assert 0 <= get_row(bits, row) <= ROW_MASK


# --- set_row ----------------------------------------------------------------------


def test_set_row_on_an_empty_board() -> None:
    assert set_row(0, 2, 0x4321) == 0x4321 << 32


def test_set_row_erases_the_old_row() -> None:
    # Toutes les cases valent 15 : seule la ligne 1 doit passer à 0.
    assert set_row(BITBOARD_LIMIT - 1, 1, 0) == 0xFFFF_FFFF_0000_FFFF


@given(bitboards, rows, row_values)
def test_set_row_changes_only_that_row(bits: int, row: int, value: int) -> None:
    new_bits = set_row(bits, row, value)
    for other in range(4):
        expected = value if other == row else get_row(bits, other)
        assert get_row(new_bits, other) == expected


@given(bitboards, rows)
def test_writing_back_a_row_changes_nothing(bits: int, row: int) -> None:
    assert set_row(bits, row, get_row(bits, row)) == bits


@given(bitboards, rows, row_values)
def test_set_row_stays_inside_64_bits(bits: int, row: int, value: int) -> None:
    assert 0 <= set_row(bits, row, value) < BITBOARD_LIMIT


# --- transpose --------------------------------------------------------------------


def test_transpose_the_counting_board() -> None:
    assert transpose(COUNTING) == to_bitboard(counting_board().T)


def test_transpose_moves_a_cell_from_r_c_to_c_r() -> None:
    assert transpose(to_bitboard(board_with(0, 3, 5))) == to_bitboard(board_with(3, 0, 5))


@given(boards)
def test_transpose_matches_numpy(board: Board) -> None:
    assert transpose(to_bitboard(board)) == to_bitboard(board.T)


@given(bitboards)
def test_transposing_twice_gives_the_board_back(bits: int) -> None:
    assert transpose(transpose(bits)) == bits
