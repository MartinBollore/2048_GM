"""Tests du benchmark (étape 7 de la phase 2).

Chaque test décrit un comportement attendu. Lance-les avec :

    pytest tests/test_benchmark.py
"""

from __future__ import annotations

import pytest

from gm2048 import benchmark
from gm2048.agents import Agent
from gm2048.agents.baselines import GreedyAgent, RandomAgent
from gm2048.benchmark import BENCHMARK_TILES, markdown_table, run_benchmark
from gm2048.evaluation import EvaluationReport, GameResult, evaluate


def report_of(name: str, *games: tuple[int, int, int, float]) -> EvaluationReport:
    """Rapport fabriqué à la main à partir de (score, tuile max, coups, secondes)."""
    results = tuple(
        GameResult(seed=i, score=score, max_tile=tile, moves=moves, seconds=seconds)
        for i, (score, tile, moves, seconds) in enumerate(games)
    )
    return EvaluationReport(agent_name=name, results=results)


# --- run_benchmark ------------------------------------------------------------------


def test_returns_one_report_per_agent_in_order() -> None:
    reports = run_benchmark(["greedy", "random"], n_games=2)
    assert [report.agent_name for report in reports] == ["greedy", "random"]
    assert all(report.n_games == 2 for report in reports)


def test_gives_the_same_results_as_evaluate() -> None:
    # Chaque agent est créé avec la graine de la première partie, comme dans
    # ``gm2048 eval`` : le benchmark et eval donnent donc les mêmes chiffres.
    random_report, greedy_report = run_benchmark(["random", "greedy"], n_games=3, first_seed=5)
    assert random_report == evaluate(RandomAgent(seed=5), 3, first_seed=5)
    assert greedy_report == evaluate(GreedyAgent(seed=5), 3, first_seed=5)


def test_every_agent_plays_the_same_games() -> None:
    reports = run_benchmark(["random", "greedy"], n_games=3, first_seed=10)
    for report in reports:
        assert [result.seed for result in report.results] == [10, 11, 12]


def test_calls_on_game_done_after_every_game_of_every_agent() -> None:
    seen: list[GameResult] = []
    run_benchmark(["random", "greedy"], n_games=3, on_game_done=seen.append)
    assert len(seen) == 6


def test_the_depth_only_goes_to_agents_that_have_one(monkeypatch: pytest.MonkeyPatch) -> None:
    # On remplace make_agent par un espion qui note ce qu'on lui demande et renvoie un
    # agent aléatoire : le test reste rapide, même pour expectimax.
    calls: list[tuple[str, int | None, int | None]] = []

    def spy(name: str, seed: int | None = None, depth: int | None = None) -> Agent:
        calls.append((name, seed, depth))
        return RandomAgent(seed=seed)

    monkeypatch.setattr(benchmark, "make_agent", spy)
    run_benchmark(["random", "expectimax"], n_games=1, first_seed=3, depth=1)
    assert calls == [("random", 3, None), ("expectimax", 3, 1)]


def test_refuses_a_depth_when_no_agent_has_one() -> None:
    with pytest.raises(ValueError):
        run_benchmark(["random", "greedy"], n_games=1, depth=2)


def test_checks_every_name_before_playing_the_first_game() -> None:
    # Un benchmark peut durer des heures : une faute de frappe dans le dernier nom
    # doit être signalée tout de suite, pas après avoir joué les premiers agents.
    seen: list[GameResult] = []
    with pytest.raises(ValueError):
        run_benchmark(["random", "inconnu"], n_games=1, on_game_done=seen.append)
    assert seen == []


def test_refuses_an_empty_list_of_agents() -> None:
    with pytest.raises(ValueError):
        run_benchmark([], n_games=1)


@pytest.mark.parametrize("n_games", [0, -3])
def test_refuses_less_than_one_game(n_games: int) -> None:
    with pytest.raises(ValueError):
        run_benchmark(["random"], n_games=n_games)


# --- markdown_table -----------------------------------------------------------------


def test_benchmark_tiles() -> None:
    assert BENCHMARK_TILES == (512, 1024, 2048, 4096)


def test_markdown_table_exact_text() -> None:
    reports = [
        report_of("random", (1000, 128, 100, 0.01), (1400, 256, 140, 0.03)),
        report_of(
            "heuristic",
            (9000, 512, 500, 0.5),
            (30000, 2048, 1500, 1.5),
            (60000, 4096, 2500, 3.0),
            (20000, 1024, 1000, 1.0),
        ),
    ]
    expected = "\n".join(
        [
            "| Agent | Parties | Score moyen | Score médian | ≥ 512 | ≥ 1024 | ≥ 2048 | ≥ 4096 "
            "| ms par coup |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            "| random | 2 | 1200 | 1200 | 0.0% | 0.0% | 0.0% | 0.0% | 0.17 |",
            "| heuristic | 4 | 29750 | 25000 | 100.0% | 75.0% | 50.0% | 25.0% | 1.09 |",
        ]
    )
    assert markdown_table(reports) == expected


def test_markdown_table_with_other_tiles() -> None:
    report = report_of("greedy", (2000, 256, 200, 0.1), (3000, 512, 300, 0.1))
    lines = markdown_table([report], tiles=[256, 512]).splitlines()
    assert (
        lines[0] == "| Agent | Parties | Score moyen | Score médian | ≥ 256 | ≥ 512 | ms par coup |"
    )
    assert lines[1] == "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |"
    assert lines[2] == "| greedy | 2 | 2500 | 2500 | 100.0% | 50.0% | 0.40 |"


def test_markdown_table_has_one_line_per_report_plus_two() -> None:
    reports = [report_of(f"agent{i}", (100, 128, 10, 0.0)) for i in range(5)]
    assert len(markdown_table(reports).splitlines()) == 7


def test_markdown_table_refuses_an_empty_list() -> None:
    with pytest.raises(ValueError):
        markdown_table([])
