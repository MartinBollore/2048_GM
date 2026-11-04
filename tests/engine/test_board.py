"""Tests des règles du jeu (module board)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from gm2048.engine import (
    ACTIONS,
    SPAWN_FOUR_PROBABILITY,
    Action,
    Board,
    empty_cells,
    from_values,
    is_game_over,
    legal_actions,
    max_tile,
    new_board,
    slide,
    spawn,
    to_values,
)

# Des exposants petits (0 à 5) donnent beaucoup de fusions et de plateaux pleins.
boards = arrays(np.uint8, (4, 4), elements=st.integers(min_value=0, max_value=5))


def board_with_first_row(row: list[int]) -> Board:
    return from_values([row, [0] * 4, [0] * 4, [0] * 4])


# --- Glissement d'une ligne -----------------------------------------------------


@pytest.mark.parametrize(
    ("row", "expected", "reward"),
    [
        ([2, 2, 2, 2], [4, 4, 0, 0], 8),
        ([2, 2, 4, 0], [4, 4, 0, 0], 4),  # pas de double fusion en un coup
        ([4, 4, 8, 8], [8, 16, 0, 0], 24),
        ([2, 0, 0, 2], [4, 0, 0, 0], 4),
        ([2, 2, 2, 0], [4, 2, 0, 0], 4),  # la fusion part du bord d'arrivée
        ([0, 2, 2, 2], [4, 2, 0, 0], 4),
        ([4, 0, 4, 4], [8, 4, 0, 0], 8),
        ([0, 0, 0, 2], [2, 0, 0, 0], 0),
        ([2, 4, 8, 16], [2, 4, 8, 16], 0),
        ([1024, 1024, 0, 0], [2048, 0, 0, 0], 2048),
    ],
)
def test_slide_left_row(row: list[int], expected: list[int], reward: int) -> None:
    result = slide(board_with_first_row(row), Action.LEFT)
    assert to_values(result.afterstate)[0].tolist() == expected
    assert result.reward == reward
    assert result.moved == (row != expected)


def test_slide_each_direction() -> None:
    board = from_values(
        [
            [2, 0, 2, 4],
            [0, 4, 0, 4],
            [2, 4, 8, 0],
            [0, 0, 8, 2],
        ]
    )
    expected = {
        Action.LEFT: [[4, 4, 0, 0], [8, 0, 0, 0], [2, 4, 8, 0], [8, 2, 0, 0]],
        Action.RIGHT: [[0, 0, 4, 4], [0, 0, 0, 8], [0, 2, 4, 8], [0, 0, 8, 2]],
        Action.UP: [[4, 8, 2, 8], [0, 0, 16, 2], [0, 0, 0, 0], [0, 0, 0, 0]],
        Action.DOWN: [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 2, 8], [4, 8, 16, 2]],
    }
    for action, values in expected.items():
        assert to_values(slide(board, action).afterstate).tolist() == values, action.name


def test_slide_does_not_modify_input() -> None:
    board = board_with_first_row([2, 2, 4, 4])
    before = board.copy()
    slide(board, Action.LEFT)
    assert np.array_equal(board, before)


def test_slide_empty_board_does_not_move() -> None:
    for action in ACTIONS:
        result = slide(new_board(), action)
        assert not result.moved
        assert result.reward == 0


# --- Propriétés sur des plateaux aléatoires -----------------------------------


@given(boards)
def test_directions_are_symmetric(board: Board) -> None:
    left = slide(board, Action.LEFT)
    right = slide(board[:, ::-1], Action.LEFT)
    up = slide(board.T, Action.LEFT)
    down = slide(board[::-1, :].T, Action.LEFT)
    assert np.array_equal(slide(board, Action.RIGHT).afterstate, right.afterstate[:, ::-1])
    assert np.array_equal(slide(board, Action.UP).afterstate, up.afterstate.T)
    assert np.array_equal(slide(board, Action.DOWN).afterstate, down.afterstate.T[::-1, :])
    assert left.reward >= 0


@given(boards, st.sampled_from(ACTIONS))
def test_slide_conserves_tile_sum(board: Board, action: Action) -> None:
    result = slide(board, action)
    assert to_values(result.afterstate).sum() == to_values(board).sum()


@given(boards, st.sampled_from(ACTIONS))
def test_reward_matches_merges(board: Board, action: Action) -> None:
    result = slide(board, action)
    merges = np.count_nonzero(board) - np.count_nonzero(result.afterstate)
    assert merges >= 0
    assert (result.reward == 0) == (merges == 0)
    assert result.moved == (not np.array_equal(board, result.afterstate))


@given(boards)
def test_legal_actions_match_slide(board: Board) -> None:
    mask = legal_actions(board)
    assert mask.shape == (4,)
    for action in ACTIONS:
        assert mask[action] == slide(board, action).moved


# --- Fin de partie ----------------------------------------------------------------


def test_full_board_without_merge_is_game_over() -> None:
    board = from_values(
        [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ]
    )
    assert not legal_actions(board).any()
    assert is_game_over(board)


def test_full_board_with_one_merge_is_not_game_over() -> None:
    board = from_values(
        [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [8, 8, 16, 32],  # seule fusion possible : 8 + 8, horizontalement
        ]
    )
    assert legal_actions(board).tolist() == [False, True, False, True]
    assert not is_game_over(board)


# --- Apparition des tuiles ----------------------------------------------------------


def test_spawn_fills_exactly_one_empty_cell() -> None:
    rng = np.random.default_rng(0)
    board = board_with_first_row([2, 4, 0, 0])
    result = spawn(board, rng)
    changed = np.flatnonzero(result != board)
    assert changed.size == 1
    assert board.flat[changed[0]] == 0
    assert result.flat[changed[0]] in (1, 2)
    assert np.array_equal(board, board_with_first_row([2, 4, 0, 0]))  # entrée intacte


def test_spawn_on_full_board_raises() -> None:
    with pytest.raises(ValueError):
        spawn(np.ones((4, 4), dtype=np.uint8), np.random.default_rng(0))


def test_spawn_probabilities() -> None:
    rng = np.random.default_rng(123)
    exponents = [int(spawn(new_board(), rng).max()) for _ in range(20_000)]
    fours = exponents.count(2) / len(exponents)
    assert set(exponents) == {1, 2}
    assert abs(fours - SPAWN_FOUR_PROBABILITY) < 0.01


def test_spawn_uses_every_empty_cell() -> None:
    rng = np.random.default_rng(7)
    cells = {int(np.flatnonzero(spawn(new_board(), rng))[0]) for _ in range(2_000)}
    assert cells == set(range(16))


# --- Conversions ------------------------------------------------------------------


def test_values_round_trip() -> None:
    values = [[0, 2, 4, 8], [16, 32, 64, 128], [256, 512, 1024, 2048], [4096, 8192, 0, 2]]
    board = from_values(values)
    assert board.dtype == np.uint8
    assert board[2, 3] == 11
    assert to_values(board).tolist() == values


@pytest.mark.parametrize("bad", [3, 1, -2, 6, 1000])
def test_from_values_rejects_invalid_tiles(bad: int) -> None:
    with pytest.raises(ValueError):
        board_with_first_row([bad, 0, 0, 0])


def test_from_values_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        from_values([[2, 2], [2, 2]])


def test_max_tile_and_empty_cells() -> None:
    board = board_with_first_row([2, 0, 512, 0])
    assert max_tile(board) == 512
    assert max_tile(new_board()) == 0
    assert empty_cells(board).tolist() == [1, *range(3, 16)]
