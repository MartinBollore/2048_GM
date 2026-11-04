"""Tests d'une partie complète (module game)."""

from __future__ import annotations

import numpy as np
import pytest

from gm2048.engine import ACTIONS, Action, Game, from_values, slide

GAME_OVER_BOARD = from_values(
    [
        [2, 4, 2, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [4, 2, 4, 2],
    ]
)


def test_reset_places_two_small_tiles() -> None:
    game = Game(seed=0)
    board = game.board
    assert np.count_nonzero(board) == 2
    assert set(board[board > 0].tolist()) <= {1, 2}
    assert game.score == 0
    assert game.moves == 0
    assert not game.done


def play_scripted(seed: int, actions: list[Action]) -> list[list[list[int]]]:
    game = Game(seed=seed)
    boards = [game.board.tolist()]
    for action in actions:
        if game.done:
            break
        game.step(action)
        boards.append(game.board.tolist())
    return boards


def test_same_seed_gives_same_game() -> None:
    actions = [Action.LEFT, Action.UP, Action.RIGHT, Action.DOWN] * 25
    assert play_scripted(42, actions) == play_scripted(42, actions)
    assert play_scripted(42, actions) != play_scripted(43, actions)


def test_reset_with_seed_restarts_the_same_game() -> None:
    game = Game(seed=5)
    first = game.board
    game.step(Action.LEFT)
    assert np.array_equal(game.reset(seed=5), first)


def test_valid_move_spawns_one_tile_and_scores() -> None:
    game = Game.from_board(from_values([[2, 2, 0, 0], [0] * 4, [0] * 4, [0] * 4]), seed=1)
    result = game.step(Action.LEFT)
    assert result.moved
    assert result.reward == 4
    expected = slide(from_values([[2, 2, 0, 0], [0] * 4, [0] * 4, [0] * 4]), Action.LEFT)
    assert np.array_equal(result.afterstate, expected.afterstate)
    assert result.afterstate[0, 0] == 2  # exposant 2 = tuile 4
    new_cells = np.flatnonzero(result.board != result.afterstate)
    assert new_cells.size == 1
    assert result.afterstate.flat[new_cells[0]] == 0
    assert result.board.flat[new_cells[0]] in (1, 2)
    assert game.score == 4
    assert game.moves == 1


def test_invalid_move_changes_nothing() -> None:
    board = from_values([[2, 4, 0, 0], [0] * 4, [0] * 4, [0] * 4])
    game = Game.from_board(board, seed=1)
    result = game.step(Action.LEFT)
    assert not result.moved
    assert result.reward == 0
    assert np.array_equal(game.board, board)
    assert game.score == 0
    assert game.moves == 0


def test_game_over_is_detected_and_blocks_moves() -> None:
    game = Game.from_board(GAME_OVER_BOARD)
    assert game.done
    assert not game.legal_actions().any()
    with pytest.raises(RuntimeError):
        game.step(Action.LEFT)


def test_board_property_returns_a_copy() -> None:
    game = Game(seed=0)
    board = game.board
    board[:] = 7
    assert not np.array_equal(game.board, board)


def test_from_board_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        Game.from_board(np.zeros((3, 3), dtype=np.uint8))


@pytest.mark.parametrize("seed", range(5))
def test_random_game_runs_to_the_end(seed: int) -> None:
    rng = np.random.default_rng(seed)
    game = Game(seed=seed)
    total_reward = 0
    while not game.done:
        legal = np.flatnonzero(game.legal_actions())
        result = game.step(ACTIONS[int(rng.choice(legal))])
        assert result.moved
        total_reward += result.reward
        assert game.moves < 10_000
    assert game.score == total_reward
    assert game.max_tile >= 4
