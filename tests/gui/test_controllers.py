"""Tests des contrôleurs : un humain au clavier, un agent à son rythme."""

from __future__ import annotations

import time
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Any

import pytest

from gm2048.agents.baselines import RandomAgent
from gm2048.engine import Action, ActionMask, Board, Game, from_values
from gm2048.evaluation import play_game
from gm2048.gui.controllers import (
    DELAYS,
    AgentController,
    Command,
    HumanController,
    animation_scale_for,
    faster,
    format_seconds,
    slower,
)
from gm2048.gui.session import POP_SECONDS, SLIDE_SECONDS


def test_faster_and_slower_walk_through_the_delays() -> None:
    assert faster(0.2) == 0.15
    assert slower(0.2) == 0.3
    assert faster(0.25) == 0.2  # un délai hors de la liste rejoint le plus proche
    assert slower(0.25) == 0.3
    assert faster(DELAYS[0]) == DELAYS[0]
    assert slower(DELAYS[-1]) == DELAYS[-1]
    assert slower(5.0) == 5.0


def test_animations_fit_in_the_delay() -> None:
    assert animation_scale_for(0.0) == 0.0
    assert animation_scale_for(10.0) == 1.0
    scale = animation_scale_for(0.1)
    assert (SLIDE_SECONDS + POP_SECONDS) * scale == pytest.approx(0.08)


def test_format_seconds_uses_a_decimal_comma() -> None:
    assert format_seconds(0.2) == "0,20 s"


# --- Humain ---------------------------------------------------------------------


def test_human_plays_with_direction_commands() -> None:
    controller = HumanController(seed=0)
    controller.session.new_game(
        Game.from_board(from_values([[2, 2, 0, 0], [0] * 4, [0] * 4, [0] * 4]), seed=0)
    )
    controller.handle(Command.LEFT)
    assert controller.session.game.score == 4
    controller.handle(Command.PAUSE)  # sans effet pour un humain
    assert not controller.paused


def test_human_new_game_uses_the_next_seed() -> None:
    controller = HumanController(seed=7)
    assert "graine 7" in controller.status()
    controller.handle(Command.NEW_GAME)
    assert controller.session.games_started == 2
    assert "Partie 2" in controller.status()
    assert "graine 8" in controller.status()
    assert controller.session.game.board.tolist() == Game(seed=8).board.tolist()


def test_human_without_seed_has_no_seed_in_status() -> None:
    assert "graine" not in HumanController().status()


def test_human_moves_are_ignored_after_game_over() -> None:
    controller = HumanController()
    board = from_values([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
    controller.session.new_game(Game.from_board(board))
    controller.handle(Command.LEFT)  # ne lève pas d'erreur
    assert controller.session.game.moves == 0


# --- Agent ----------------------------------------------------------------------


def make_controller(executor: Executor, delay: float = 0.2, seed: int = 0) -> AgentController:
    return AgentController(RandomAgent, seed=seed, delay=delay, executor=executor)


def test_agent_waits_for_the_delay_between_moves(inline_executor: Executor) -> None:
    controller = make_controller(inline_executor, delay=0.5)
    controller.update(0.3)
    assert controller.session.game.moves == 0
    controller.update(0.3)
    assert controller.session.game.moves == 1
    controller.update(0.3)
    assert controller.session.game.moves == 1


def test_agent_does_not_play_while_paused(inline_executor: Executor) -> None:
    controller = make_controller(inline_executor, delay=0.0)
    controller.handle(Command.PAUSE)
    assert controller.paused
    controller.update(1.0)
    assert controller.session.game.moves == 0
    controller.handle(Command.PAUSE)
    controller.update(1.0)
    assert controller.session.game.moves == 1


def test_step_plays_exactly_one_move_while_paused(inline_executor: Executor) -> None:
    controller = make_controller(inline_executor, delay=0.0)
    controller.handle(Command.STEP)  # en pleine partie : met d'abord en pause
    assert controller.paused
    controller.update(1.0)
    assert controller.session.game.moves == 0
    controller.handle(Command.STEP)
    controller.update(1.0)
    controller.update(1.0)
    assert controller.session.game.moves == 1
    controller.handle(Command.RIGHT)  # la flèche droite fait la même chose
    controller.update(1.0)
    assert controller.session.game.moves == 2


def test_speed_commands_change_the_delay_and_the_animations(inline_executor: Executor) -> None:
    controller = make_controller(inline_executor, delay=0.2)
    controller.handle(Command.SLOWER)
    assert controller.delay == 0.3
    assert controller.session.animation_scale == animation_scale_for(0.3)
    controller.handle(Command.FASTER)
    controller.handle(Command.FASTER)
    assert controller.delay == 0.15
    assert "délai 0,15 s" in controller.status()


def test_new_game_uses_the_next_seed_for_board_and_agent(inline_executor: Executor) -> None:
    seeds: list[int | None] = []

    def make_agent(seed: int | None) -> RandomAgent:
        seeds.append(seed)
        return RandomAgent(seed)

    controller = AgentController(make_agent, seed=5, executor=inline_executor)
    controller.handle(Command.NEW_GAME)
    assert seeds == [5, 6]
    assert controller.seed == 6
    assert "partie 2" in controller.status()
    assert controller.session.game.board.tolist() == Game(seed=6).board.tolist()


def test_watching_replays_the_same_game_as_the_evaluation(inline_executor: Executor) -> None:
    controller = make_controller(inline_executor, delay=0.0, seed=3)
    while not controller.session.game.done:
        controller.update(0.01)
    expected = play_game(RandomAgent(3), seed=3)
    assert controller.session.game.score == expected.score
    assert controller.session.game.moves == expected.moves


class StuckAgent:
    name = "bloqué"

    def act(self, board: Board, legal_mask: ActionMask) -> Action:
        return Action.UP


def test_an_illegal_move_from_the_agent_is_an_error(inline_executor: Executor) -> None:
    board = from_values([[2, 0, 0, 0], [0] * 4, [0] * 4, [0] * 4])
    controller = AgentController(lambda seed: StuckAgent(), delay=0.0, executor=inline_executor)
    controller.session.new_game(Game.from_board(board, seed=0))
    with pytest.raises(RuntimeError, match="bloqué"):
        controller.update(1.0)


class BrokenAgent:
    name = "cassé"

    def act(self, board: Board, legal_mask: ActionMask) -> Action:
        raise NotImplementedError("pas encore écrit")


def test_an_agent_error_reaches_the_window(inline_executor: Executor) -> None:
    controller = AgentController(lambda seed: BrokenAgent(), delay=0.0, executor=inline_executor)
    with pytest.raises(NotImplementedError, match="pas encore écrit"):
        controller.update(1.0)


def test_a_slow_agent_is_shown_as_thinking(stalled_executor: Executor) -> None:
    controller = AgentController(RandomAgent, delay=0.1, executor=stalled_executor)
    controller.update(0.2)
    assert not controller.thinking
    controller.update(0.5)
    assert controller.thinking
    assert "réflexion" in controller.status()
    assert controller.session.game.moves == 0


def test_the_agent_thinks_in_a_background_thread() -> None:
    controller = AgentController(RandomAgent, delay=0.0)
    try:
        deadline = time.monotonic() + 5
        while controller.session.game.moves < 3 and time.monotonic() < deadline:
            controller.update(0.02)
            time.sleep(0.001)
        assert controller.session.game.moves >= 3
    finally:
        controller.close()


def test_close_stops_the_executor(inline_executor: Any) -> None:
    AgentController(RandomAgent, executor=inline_executor).close()
    assert inline_executor.shut_down


def test_close_with_the_default_executor() -> None:
    controller = AgentController(RandomAgent)
    controller.close()
    assert isinstance(controller._executor, ThreadPoolExecutor)
    with pytest.raises(RuntimeError):
        controller._executor.submit(print)


def test_negative_delay_is_rejected(inline_executor: Executor) -> None:
    with pytest.raises(ValueError):
        AgentController(RandomAgent, delay=-1.0, executor=inline_executor)
