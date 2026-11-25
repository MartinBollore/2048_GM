"""Tests des animations et de la session affichée."""

from __future__ import annotations

import pytest

from gm2048.engine import Action, Game, from_values
from gm2048.gui.motion import appearance, plan_transition
from gm2048.gui.session import (
    MERGE_SWELL,
    POP_SECONDS,
    POPUP_SECONDS,
    SLIDE_SECONDS,
    TOAST_SECONDS,
    Animation,
    Session,
    Sprite,
    ease_out_back,
    ease_out_cubic,
    swell,
)


def test_easing_curves_start_at_0_and_end_at_1() -> None:
    for curve in (ease_out_cubic, ease_out_back):
        assert curve(0.0) == pytest.approx(0.0)
        assert curve(1.0) == pytest.approx(1.0)
    assert ease_out_cubic(0.5) > 0.5  # rapide au départ
    assert max(ease_out_back(i / 100) for i in range(101)) > 1.0  # dépasse un peu


def test_swell_goes_back_to_normal_size() -> None:
    assert swell(0.0) == pytest.approx(1.0)
    assert swell(1.0) == pytest.approx(1.0)
    assert swell(0.5) == pytest.approx(1 + MERGE_SWELL)


def merge_transition() -> Animation:
    """Deux 2 fusionnent à gauche, puis un 2 apparaît en bas à droite."""
    before = from_values([[0, 0, 2, 2], [0] * 4, [0] * 4, [0] * 4])
    after = from_values([[4, 0, 0, 0], [0] * 4, [0] * 4, [0, 0, 0, 2]])
    return Animation(plan_transition(before, Action.LEFT, after))


def test_tiles_start_on_their_cells() -> None:
    animation = merge_transition()
    assert sorted(animation.sprites(), key=lambda s: s.col) == [
        Sprite(1, 0.0, 2.0),
        Sprite(1, 0.0, 3.0),
    ]


def test_tiles_are_between_cells_while_sliding() -> None:
    animation = merge_transition()
    animation.advance(SLIDE_SECONDS / 2)
    columns = sorted(sprite.col for sprite in animation.sprites())
    assert 0.0 < columns[0] < 2.0
    assert 0.0 < columns[1] < 3.0
    assert not animation.done


def test_merged_tile_swells_and_new_tile_grows_after_the_slide() -> None:
    animation = merge_transition()
    animation.advance(SLIDE_SECONDS + POP_SECONDS / 2)
    sprites = {(sprite.row, sprite.col): sprite for sprite in animation.sprites()}
    assert set(sprites) == {(0.0, 0.0), (3.0, 3.0)}
    assert sprites[0.0, 0.0].exponent == 2
    assert sprites[0.0, 0.0].scale > 1.0
    assert 0.0 < sprites[3.0, 3.0].alpha <= 1.0
    assert sprites[3.0, 3.0].scale > 0.0


def test_at_the_end_the_final_board_is_shown_still() -> None:
    animation = merge_transition()
    animation.advance(10.0)
    assert animation.done
    assert animation.elapsed == pytest.approx(animation.duration)
    assert {
        (s.exponent, s.row, s.col, round(s.scale, 9), s.alpha) for s in animation.sprites()
    } == {
        (2, 0.0, 0.0, 1.0, 1.0),
        (1, 3.0, 3.0, 1.0, 1.0),
    }


def test_zero_duration_shows_the_final_board_at_once() -> None:
    before = from_values([[0, 0, 2, 2], [0] * 4, [0] * 4, [0] * 4])
    after = from_values([[4, 0, 0, 0], [0] * 4, [0] * 4, [0, 0, 0, 2]])
    animation = Animation(plan_transition(before, Action.LEFT, after), duration_scale=0.0)
    assert animation.done
    assert {(s.exponent, s.row, s.col, s.scale) for s in animation.sprites()} == {
        (2, 0.0, 0.0, 1.0),
        (1, 3.0, 3.0, 1.0),
    }


def test_a_new_board_has_no_slide_phase() -> None:
    animation = Animation(appearance(from_values([[2, 0, 0, 0], [0] * 4, [0] * 4, [0] * 4])))
    assert animation.slide_seconds == 0.0
    assert animation.duration == pytest.approx(POP_SECONDS)
    assert animation.sprites()[0].scale == pytest.approx(0.0)


def two_twos() -> Game:
    return Game.from_board(from_values([[2, 2, 0, 0], [0] * 4, [0] * 4, [0] * 4]), seed=0)


def test_playing_a_move_starts_its_animation_and_a_score_popup() -> None:
    session = Session(two_twos())
    session.update(1.0)
    assert not session.animating
    result = session.play(Action.LEFT)
    assert result.moved
    assert session.animating
    assert session.last_action is Action.LEFT
    assert session.popup is not None
    assert session.popup.points == 4
    assert session.best_score == 4


def test_the_popup_flies_away_then_disappears() -> None:
    session = Session(two_twos())
    session.play(Action.LEFT)
    assert session.popup is not None
    session.update(POPUP_SECONDS / 2)
    assert session.popup.progress == pytest.approx(0.5)
    session.update(POPUP_SECONDS)
    assert session.popup is None


def test_no_popup_without_animation() -> None:
    session = Session(two_twos(), animation_scale=0.0)
    session.play(Action.LEFT)
    assert session.popup is None


def test_a_useless_move_changes_nothing() -> None:
    session = Session(two_twos())
    session.update(1.0)
    animation = session.animation
    result = session.play(Action.UP)
    assert not result.moved
    assert session.animation is animation
    assert session.last_action is None


def test_reaching_2048_shows_the_banner_once() -> None:
    board = from_values([[1024, 1024, 0, 0], [0] * 4, [0] * 4, [0] * 4])
    session = Session(Game.from_board(board, seed=0))
    session.play(Action.LEFT)
    assert session.toast_age == 0.0
    session.update(TOAST_SECONDS / 2)
    assert session.toast_age == pytest.approx(TOAST_SECONDS / 2)
    session.update(TOAST_SECONDS)
    assert session.toast_age is None
    session.play(Action.RIGHT)
    assert session.toast_age is None


def test_game_over_waits_for_the_last_animation() -> None:
    board = from_values([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
    session = Session(Game.from_board(board, score=100))
    assert session.game.done
    assert not session.over  # les tuiles sont encore en train d'apparaître
    session.update(POP_SECONDS + 0.1)
    assert session.over
    assert session.over_age == pytest.approx(0.1)  # le temps compte après l'animation
    session.update(0.5)
    assert session.over_age == pytest.approx(0.6)


def test_new_game_keeps_the_best_score() -> None:
    session = Session(two_twos())
    session.play(Action.LEFT)
    session.new_game(Game(seed=1))
    assert session.games_started == 2
    assert session.best_score == 4
    assert session.game.score == 0
    assert session.popup is None
    assert session.last_action is None


def test_playing_after_game_over_is_an_error() -> None:
    board = from_values([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
    session = Session(Game.from_board(board))
    with pytest.raises(RuntimeError):
        session.play(Action.LEFT)
