"""Tests du module d'évaluation (étapes 2 et 7 de la phase 2).

Chaque test décrit un comportement attendu. Lance-les avec :

    pytest tests/test_evaluation.py
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from gm2048.agents.baselines import RandomAgent
from gm2048.engine import Action, ActionMask, Board, Game
from gm2048.evaluation import (
    EvaluationReport,
    GameResult,
    evaluate,
    format_report,
    play_game,
)


class FirstLegalAgent:
    """Agent de test, sans hasard : joue toujours le premier coup légal."""

    name = "first-legal"

    def act(self, board: Board, legal_mask: ActionMask) -> Action:
        return Action(int(np.flatnonzero(legal_mask)[0]))


class CountingAgent(FirstLegalAgent):
    """Comme FirstLegalAgent, mais compte combien de fois on lui demande un coup."""

    def __init__(self) -> None:
        self.calls = 0

    def act(self, board: Board, legal_mask: ActionMask) -> Action:
        self.calls += 1
        assert legal_mask.any(), "on a demandé un coup alors qu'aucun n'était légal"
        return super().act(board, legal_mask)


class SlowAgent(CountingAgent):
    """Comme CountingAgent, mais réfléchit 1 ms avant chaque coup."""

    def act(self, board: Board, legal_mask: ActionMask) -> Action:
        time.sleep(0.001)
        return super().act(board, legal_mask)


def report_of(*games: tuple[int, int]) -> EvaluationReport:
    """Rapport fabriqué à la main à partir de couples (score, tuile max)."""
    results = tuple(
        GameResult(seed=i, score=score, max_tile=tile, moves=10)
        for i, (score, tile) in enumerate(games)
    )
    return EvaluationReport(agent_name="test", results=results)


# --- play_game ---------------------------------------------------------------


def test_play_game_matches_a_game_played_by_hand() -> None:
    agent = FirstLegalAgent()
    game = Game(seed=3)
    while not game.done:
        game.step(agent.act(game.board, game.legal_actions()))

    result = play_game(agent, seed=3)

    assert result == GameResult(seed=3, score=game.score, max_tile=game.max_tile, moves=game.moves)


def test_play_game_asks_the_agent_once_per_move() -> None:
    agent = CountingAgent()
    result = play_game(agent, seed=0)
    assert result.moves > 0
    assert agent.calls == result.moves


def test_play_game_is_reproducible() -> None:
    first = play_game(RandomAgent(seed=5), seed=9)
    second = play_game(RandomAgent(seed=5), seed=9)
    assert first == second


def test_different_seeds_give_different_games() -> None:
    results = {play_game(RandomAgent(seed=0), seed=seed) for seed in range(5)}
    assert len(results) > 1


def test_play_game_measures_the_duration_of_the_game() -> None:
    # L'agent dort 1 ms par coup : la partie dure donc au moins une milliseconde par
    # coup. La mesure doit inclure le temps de réflexion de l'agent.
    agent = SlowAgent()
    result = play_game(agent, seed=0)
    assert result.seconds >= 0.001 * agent.calls
    assert result.seconds < 60


def test_a_fast_game_still_has_a_positive_duration() -> None:
    assert play_game(FirstLegalAgent(), seed=1).seconds > 0


def test_the_duration_is_ignored_when_comparing_results() -> None:
    # Deux fois la même partie ne durent jamais exactement le même temps, mais ce sont
    # bien les mêmes parties : ``seconds`` ne compte pas dans ``==``.
    assert GameResult(0, 100, 128, 10, seconds=0.5) == GameResult(0, 100, 128, 10, seconds=2.0)
    assert GameResult(0, 100, 128, 10).seconds == 0.0


# --- evaluate ----------------------------------------------------------------


def test_evaluate_plays_the_requested_number_of_games() -> None:
    report = evaluate(FirstLegalAgent(), n_games=5, first_seed=10)
    assert report.n_games == 5
    assert [result.seed for result in report.results] == [10, 11, 12, 13, 14]


def test_evaluate_starts_at_seed_zero_by_default() -> None:
    report = evaluate(FirstLegalAgent(), n_games=3)
    assert [result.seed for result in report.results] == [0, 1, 2]


def test_evaluate_records_the_agent_name() -> None:
    assert evaluate(RandomAgent(seed=0), n_games=1).agent_name == "random"


def test_evaluate_uses_play_game_for_each_seed() -> None:
    agent = FirstLegalAgent()
    report = evaluate(agent, n_games=4, first_seed=20)
    assert report.results == tuple(play_game(agent, seed) for seed in range(20, 24))


def test_evaluate_calls_on_game_done_after_each_game() -> None:
    received: list[GameResult] = []
    report = evaluate(FirstLegalAgent(), n_games=3, first_seed=5, on_game_done=received.append)
    assert tuple(received) == report.results


@pytest.mark.parametrize("n_games", [0, -1])
def test_evaluate_refuses_less_than_one_game(n_games: int) -> None:
    with pytest.raises(ValueError):
        evaluate(FirstLegalAgent(), n_games=n_games)


# --- statistiques du rapport -------------------------------------------------


def test_mean_score() -> None:
    report = report_of((100, 256), (200, 2048), (600, 4096))
    assert report.mean_score == pytest.approx(300.0)


def test_win_rate_counts_games_reaching_the_tile() -> None:
    report = report_of((100, 256), (200, 2048), (600, 4096))
    assert report.win_rate() == pytest.approx(2 / 3)
    assert report.win_rate(256) == pytest.approx(1.0)
    assert report.win_rate(8192) == pytest.approx(0.0)


def test_median_score_with_an_odd_number_of_games() -> None:
    report = report_of((100, 256), (5000, 2048), (300, 512))
    assert report.median_score == pytest.approx(300.0)
    assert isinstance(report.median_score, float)


def test_median_score_with_an_even_number_of_games() -> None:
    report = report_of((100, 256), (200, 256), (600, 512), (900, 1024))
    assert report.median_score == pytest.approx(400.0)


def test_max_score_is_the_best_game() -> None:
    report = report_of((100, 256), (5000, 2048), (300, 512))
    assert report.max_score == 5000


def test_median_resists_one_exceptional_game() -> None:
    report = report_of((100, 256), (110, 256), (120, 256), (100_000, 8192))
    assert report.median_score == pytest.approx(115.0)
    assert report.mean_score > 20_000


def timed_report(*games: tuple[int, float]) -> EvaluationReport:
    """Rapport fabriqué à la main à partir de couples (coups, secondes)."""
    results = tuple(
        GameResult(seed=i, score=0, max_tile=2, moves=moves, seconds=seconds)
        for i, (moves, seconds) in enumerate(games)
    )
    return EvaluationReport(agent_name="test", results=results)


def test_seconds_per_move_is_total_time_over_total_moves() -> None:
    # 1 s pour 100 coups, puis 3 s pour 900 coups : 4 s pour 1 000 coups, soit 4 ms.
    # La moyenne des deux parties (10 ms et 3,3 ms par coup) donnerait autre chose.
    report = timed_report((100, 1.0), (900, 3.0))
    assert report.seconds_per_move == pytest.approx(0.004)


def test_seconds_per_move_without_any_move_is_zero() -> None:
    assert timed_report((0, 0.0), (0, 0.0)).seconds_per_move == 0.0


def test_tile_counts_are_sorted_by_tile() -> None:
    report = report_of((10, 512), (20, 128), (30, 512))
    assert report.tile_counts == {128: 1, 512: 2}
    assert list(report.tile_counts) == [128, 512]


def test_format_report_shows_the_main_figures() -> None:
    text = format_report(report_of((100, 256), (200, 2048), (600, 4096)))
    assert "test" in text
    assert "300" in text
    assert "66.7%" in text


def test_format_report_shows_the_median_and_the_time_per_move() -> None:
    text = format_report(timed_report((100, 1.0), (900, 3.0)))
    assert "Score médian : 0" in text
    assert "Meilleur score : 0" in text
    assert "Temps par coup : 4.00 ms" in text
