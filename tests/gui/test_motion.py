"""Tests du suivi des tuiles pendant un coup."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from gm2048.engine import ACTIONS, Action, Board, Game, from_values, slide
from gm2048.gui.motion import (
    NewTile,
    TileMove,
    afterstate_of,
    appearance,
    line_cells,
    plan_transition,
    trace_slide,
)

boards = arrays(np.uint8, (4, 4), elements=st.integers(min_value=0, max_value=6))


def one_row(values: list[int]) -> Board:
    return from_values([values, [0] * 4, [0] * 4, [0] * 4])


def test_line_cells_start_from_the_edge_tiles_slide_towards() -> None:
    assert line_cells(Action.LEFT, 1) == [(1, 0), (1, 1), (1, 2), (1, 3)]
    assert line_cells(Action.RIGHT, 0) == [(0, 3), (0, 2), (0, 1), (0, 0)]
    assert line_cells(Action.UP, 2) == [(0, 2), (1, 2), (2, 2), (3, 2)]
    assert line_cells(Action.DOWN, 3) == [(3, 3), (2, 3), (1, 3), (0, 3)]


def test_four_equal_tiles_make_two_merges() -> None:
    moves, merged = trace_slide(one_row([2, 2, 2, 2]), Action.LEFT)
    assert moves == (
        TileMove(1, (0, 0), (0, 0), merges=True),
        TileMove(1, (0, 1), (0, 0), merges=True),
        TileMove(1, (0, 2), (0, 1), merges=True),
        TileMove(1, (0, 3), (0, 1), merges=True),
    )
    assert merged == (NewTile((0, 0), 2), NewTile((0, 1), 2))


def test_a_new_tile_does_not_merge_again_in_the_same_move() -> None:
    moves, merged = trace_slide(one_row([2, 2, 4, 0]), Action.LEFT)
    assert merged == (NewTile((0, 0), 2),)
    assert TileMove(2, (0, 2), (0, 1), merges=False) in moves


def test_tiles_slide_right_across_gaps() -> None:
    moves, merged = trace_slide(one_row([2, 0, 0, 2]), Action.RIGHT)
    assert moves == (
        TileMove(1, (0, 3), (0, 3), merges=True),
        TileMove(1, (0, 0), (0, 3), merges=True),
    )
    assert merged == (NewTile((0, 3), 2),)


def test_tiles_slide_down_a_column() -> None:
    board = from_values([[8, 0, 0, 0], [0] * 4, [4, 0, 0, 0], [0] * 4])
    moves, merged = trace_slide(board, Action.DOWN)
    assert moves == (
        TileMove(2, (2, 0), (3, 0), merges=False),
        TileMove(3, (0, 0), (2, 0), merges=False),
    )
    assert merged == ()


@given(board=boards, action=st.sampled_from(ACTIONS))
def test_tracing_agrees_with_the_engine(board: Board, action: Action) -> None:
    moves, merged = trace_slide(board, action)
    expected = slide(board, action)
    assert np.array_equal(afterstate_of(moves, merged), expected.afterstate)
    assert sum(1 << tile.exponent for tile in merged) == expected.reward


@given(board=boards, action=st.sampled_from(ACTIONS))
def test_every_tile_moves_once_along_its_line_towards_the_edge(
    board: Board, action: Action
) -> None:
    moves, _ = trace_slide(board, action)
    starts = sorted(move.start for move in moves)
    assert starts == sorted((int(r), int(c)) for r, c in np.argwhere(board != 0))
    for move in moves:
        line = next(i for i in range(4) if move.start in line_cells(action, i))
        cells = line_cells(action, line)
        assert move.end in cells
        assert cells.index(move.end) <= cells.index(move.start)
        assert board[move.start] == move.exponent


def test_plan_transition_finds_the_spawned_tile() -> None:
    game = Game.from_board(one_row([2, 2, 0, 0]), seed=0)
    before = game.board
    result = game.step(Action.LEFT)
    transition = plan_transition(before, Action.LEFT, result.board)
    assert transition.action is Action.LEFT
    assert transition.merged == (NewTile((0, 0), 2),)
    assert transition.reward == 4
    assert len(transition.spawned) == 1
    spawned = transition.spawned[0]
    assert spawned.exponent in (1, 2)
    assert result.afterstate[spawned.cell] == 0
    assert np.array_equal(transition.board, result.board)


def test_plan_transition_rejects_an_impossible_board() -> None:
    before = one_row([2, 2, 0, 0])
    with pytest.raises(ValueError, match="ne correspond pas"):
        plan_transition(before, Action.LEFT, one_row([2, 2, 0, 0]))


def test_appearance_makes_every_tile_appear() -> None:
    board = from_values([[2, 0, 0, 0], [0] * 4, [0, 0, 4, 0], [0] * 4])
    transition = appearance(board)
    assert transition.moves == ()
    assert transition.merged == ()
    assert set(transition.spawned) == {NewTile((0, 0), 1), NewTile((2, 2), 2)}
    assert transition.action is None
    assert transition.reward == 0
