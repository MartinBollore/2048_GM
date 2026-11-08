"""Tests de la ligne de commande et de la boucle de jeu."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from gm2048 import cli
from gm2048.agents import Agent, registry
from gm2048.agents.baselines import ExpectimaxAgent, GreedyAgent, HeuristicAgent, RandomAgent
from gm2048.benchmark import markdown_table, run_benchmark
from gm2048.cli import QUIT, agent_factory, app, parse_key, play_loop, progress_info
from gm2048.engine import Action, Game, from_values
from gm2048.evaluation import EvaluationReport, GameResult, evaluate


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("\x1b[A", Action.UP),
        ("\x1b[B", Action.DOWN),
        ("\x1b[C", Action.RIGHT),
        ("\x1b[D", Action.LEFT),
        ("\xe0H", Action.UP),
        ("\xe0P", Action.DOWN),
        ("\xe0M", Action.RIGHT),
        ("\xe0K", Action.LEFT),
        ("z", Action.UP),
        ("Z", Action.UP),
        ("w", Action.UP),
        ("q", Action.LEFT),
        ("a", Action.LEFT),
        ("s", Action.DOWN),
        ("d", Action.RIGHT),
        (" d\n", Action.RIGHT),
        ("x", QUIT),
        ("\x1b", QUIT),
        ("k", None),
        ("", None),
    ],
)
def test_parse_key(key: str, expected: object) -> None:
    assert parse_key(key) == expected


class Recorder:
    def __init__(self, keys: list[str]) -> None:
        self._keys: Iterator[str] = iter(keys)
        self.output: list[str] = []
        self.reads = 0

    def read_key(self) -> str:
        self.reads += 1
        return next(self._keys)

    def echo(self, text: str) -> None:
        self.output.append(text)

    @property
    def text(self) -> str:
        return "\n".join(self.output)


def run(game: Game, keys: list[str]) -> Recorder:
    recorder = Recorder(keys)
    play_loop(game, recorder.read_key, recorder.echo, clear=lambda: None)
    return recorder


def test_play_loop_moves_then_quits() -> None:
    game = Game.from_board(from_values([[2, 2, 0, 0], [0] * 4, [0] * 4, [0] * 4]), seed=0)
    recorder = run(game, ["q", "x"])
    assert game.score == 4
    assert game.moves == 1
    assert "Partie abandonnée. Score : 4" in recorder.text


def test_play_loop_reports_unknown_key_and_useless_move() -> None:
    game = Game.from_board(from_values([[2, 4, 0, 0], [0] * 4, [0] * 4, [0] * 4]), seed=0)
    recorder = run(game, ["k", "q", "x"])
    assert "Touche non reconnue" in recorder.text
    assert "Ce coup ne déplace aucune tuile." in recorder.text
    assert game.moves == 0


def test_play_loop_stops_at_game_over() -> None:
    board = from_values([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
    recorder = run(Game.from_board(board, score=100), [])
    assert recorder.reads == 0
    assert "Partie terminée ! Score final : 100, meilleure tuile : 4." in recorder.text


def test_play_loop_quits_on_ctrl_c() -> None:
    def interrupt() -> str:
        raise KeyboardInterrupt

    output: list[str] = []
    play_loop(Game(seed=0), interrupt, output.append, clear=lambda: None)
    assert "Partie abandonnée" in output[-1]


def test_cli_play_in_line_mode() -> None:
    result = CliRunner().invoke(app, ["play", "--terminal", "--seed", "1"], input="z\nq\ns\nd\nx\n")
    assert result.exit_code == 0, result.output
    assert "Score : " in result.output
    assert "Partie abandonnée" in result.output


def test_cli_play_ends_cleanly_on_end_of_input() -> None:
    result = CliRunner().invoke(app, ["play", "--line", "--seed", "1"], input="")
    assert result.exit_code == 0, result.output
    assert "Partie abandonnée" in result.output


class FakeWindow:
    """Remplace la fenêtre pygame : enregistre les appels sans rien ouvrir."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def play(self, **options: object) -> None:
        self.calls.append(("play", options))

    def watch(self, make_agent: object, **options: object) -> None:
        self.calls.append(("watch", {"make_agent": make_agent, **options}))


@pytest.fixture
def fake_window(monkeypatch: pytest.MonkeyPatch) -> FakeWindow:
    pytest.importorskip("pygame")
    from gm2048.gui import window

    fake = FakeWindow()
    monkeypatch.setattr(window, "play", fake.play)
    monkeypatch.setattr(window, "watch", fake.watch)
    return fake


def test_cli_play_opens_the_window_by_default(fake_window: FakeWindow) -> None:
    result = CliRunner().invoke(app, ["play", "--seed", "4", "--theme", "sombre"])
    assert result.exit_code == 0, result.output
    assert fake_window.calls == [("play", {"seed": 4, "theme": "sombre"})]


def test_cli_play_falls_back_to_the_terminal_without_pygame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "_gui_available", lambda: False)
    result = CliRunner().invoke(app, ["play", "--seed", "1"], input="x\n")
    assert result.exit_code == 0, result.output
    assert "pygame-ce" in result.output
    assert "Partie abandonnée" in result.output


def test_cli_watch_opens_the_window_with_the_agent(fake_window: FakeWindow) -> None:
    result = CliRunner().invoke(
        app, ["watch", "--agent", "greedy", "--seed", "3", "--delay", "0.5"]
    )
    assert result.exit_code == 0, result.output
    assert fake_window.calls == [
        ("watch", {"make_agent": GreedyAgent, "seed": 3, "delay": 0.5, "theme": "clair"})
    ]


def test_cli_watch_uses_the_heuristic_agent_by_default(fake_window: FakeWindow) -> None:
    result = CliRunner().invoke(app, ["watch"])
    assert result.exit_code == 0, result.output
    assert fake_window.calls[0][1]["make_agent"] is HeuristicAgent


@pytest.mark.parametrize(
    "arguments",
    [
        ["watch", "--agent", "inconnu"],
        ["watch", "--theme", "fluo"],
        ["watch", "--delay", "-1"],
        ["play", "--theme", "fluo"],
    ],
)
def test_cli_rejects_bad_window_options(arguments: list[str], fake_window: FakeWindow) -> None:
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code != 0
    assert fake_window.calls == []


def test_cli_watch_passes_the_depth_to_the_agent(fake_window: FakeWindow) -> None:
    result = CliRunner().invoke(app, ["watch", "--agent", "expectimax", "--depth", "1"])
    assert result.exit_code == 0, result.output
    make_agent = fake_window.calls[0][1]["make_agent"]
    assert callable(make_agent)
    agent = make_agent(0)
    assert isinstance(agent, ExpectimaxAgent)
    assert agent.depth == 1


@pytest.mark.parametrize(
    "arguments",
    [
        ["watch", "--agent", "greedy", "--depth", "2"],
        ["watch", "--agent", "expectimax", "--depth", "0"],
    ],
)
def test_cli_watch_rejects_a_bad_depth(arguments: list[str], fake_window: FakeWindow) -> None:
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code != 0
    assert fake_window.calls == []


def test_cli_watch_explains_how_to_install_pygame(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_gui_available", lambda: False)
    result = CliRunner().invoke(app, ["watch"])
    assert result.exit_code == 1
    assert 'pip install -e ".[gui]"' in result.output


def test_cli_eval_prints_a_report() -> None:
    result = CliRunner().invoke(app, ["eval", "--agent", "random", "--games", "3", "--seed", "0"])
    assert result.exit_code == 0, result.output
    assert "random" in result.output
    assert "3 parties" in result.output


def test_cli_eval_rejects_an_unknown_agent() -> None:
    result = CliRunner().invoke(app, ["eval", "--agent", "inconnu"])
    assert result.exit_code != 0


def test_cli_eval_rejects_zero_games() -> None:
    result = CliRunner().invoke(app, ["eval", "--games", "0"])
    assert result.exit_code != 0


def test_cli_eval_keeps_the_progress_bar_out_of_the_report() -> None:
    result = CliRunner().invoke(app, ["eval", "--games", "3"])
    assert result.exit_code == 0, result.output
    assert "Évaluation de random" not in result.stdout
    assert "Score moyen" in result.stdout


def test_progress_info_is_empty_before_the_first_game() -> None:
    assert progress_info([]) is None


def test_progress_info_shows_mean_score_and_best_tile() -> None:
    played = [
        GameResult(seed=0, score=100, max_tile=128, moves=10),
        GameResult(seed=1, score=300, max_tile=256, moves=20),
    ]
    assert progress_info(played) == "score moyen 200, meilleure tuile 256"


def test_agent_factory_without_depth_keeps_the_default_agent() -> None:
    agent = agent_factory("expectimax")(0)
    assert isinstance(agent, ExpectimaxAgent)
    assert agent.depth == ExpectimaxAgent(0).depth


def test_agent_factory_passes_the_depth() -> None:
    agent = agent_factory("expectimax", depth=1)(0)
    assert isinstance(agent, ExpectimaxAgent)
    assert agent.depth == 1


def test_agent_factory_refuses_a_depth_for_agents_without_search() -> None:
    with pytest.raises(typer.BadParameter):
        agent_factory("random", depth=2)


def test_cli_eval_passes_the_depth_to_the_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    # On remplace evaluate par une version qui ne joue qu'une partie de l'agent aléatoire,
    # mais retient l'agent reçu : le test reste rapide.
    received: list[Agent] = []
    real_evaluate = cli.evaluate

    def spy(agent: Agent, *args: Any, **kwargs: Any) -> EvaluationReport:
        received.append(agent)
        return real_evaluate(RandomAgent(0), *args, **kwargs)

    monkeypatch.setattr(cli, "evaluate", spy)
    result = CliRunner().invoke(
        app, ["eval", "--agent", "expectimax", "--depth", "1", "--games", "1"]
    )
    assert result.exit_code == 0, result.output
    assert isinstance(received[0], ExpectimaxAgent)
    assert received[0].depth == 1
    assert "Profondeur de recherche : 1" in result.stdout


def test_cli_eval_rejects_a_depth_for_agents_without_search() -> None:
    result = CliRunner().invoke(app, ["eval", "--agent", "random", "--depth", "2"])
    assert result.exit_code != 0


def test_cli_eval_rejects_a_depth_of_zero() -> None:
    result = CliRunner().invoke(app, ["eval", "--agent", "expectimax", "--depth", "0"])
    assert result.exit_code != 0


# --- Étape 7 : le registre et gm2048 benchmark ---------------------------------------


def test_cli_uses_the_agent_registry() -> None:
    # Une seule liste d'agents dans tout le projet : la ligne de commande reprend
    # celle du registre au lieu d'en tenir une copie.
    assert cli.AGENTS is registry.AGENTS
    assert cli.DEPTH_AGENTS is registry.DEPTH_AGENTS


BENCH = ["benchmark", "--agent", "random", "--agent", "greedy", "--games", "2"]


def table_rows(output: str) -> dict[str, list[str]]:
    """Les lignes d'agents d'un tableau Markdown, indexées par nom d'agent."""
    rows = {}
    for line in output.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if line.startswith("| ") and cells[0] not in ("Agent", ":---"):
            rows[cells[0]] = cells
    return rows


def test_cli_benchmark_prints_a_markdown_table() -> None:
    result = CliRunner().invoke(app, BENCH)
    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert lines[0].startswith("| Agent | Parties | Score moyen | Score médian |")
    assert lines[1].startswith("| :--- |")
    assert list(table_rows(result.stdout)) == ["random", "greedy"]


def test_cli_benchmark_gives_the_same_scores_as_evaluate() -> None:
    result = CliRunner().invoke(app, [*BENCH, "--seed", "7"])
    assert result.exit_code == 0, result.output
    rows = table_rows(result.stdout)
    random_report = evaluate(RandomAgent(seed=7), 2, first_seed=7)
    greedy_report = evaluate(GreedyAgent(seed=7), 2, first_seed=7)
    assert rows["random"][1:4] == [
        "2",
        f"{random_report.mean_score:.0f}",
        f"{random_report.median_score:.0f}",
    ]
    assert rows["greedy"][2] == f"{greedy_report.mean_score:.0f}"


def test_cli_benchmark_keeps_the_progress_bar_out_of_the_table() -> None:
    # La sortie standard ne contient que le tableau : on peut la rediriger dans un
    # fichier (gm2048 benchmark > benchmark.md) et la coller dans le README.
    result = CliRunner().invoke(app, BENCH)
    assert result.exit_code == 0, result.output
    assert len(result.stdout.strip().splitlines()) == 4


def test_cli_benchmark_uses_all_agents_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Un espion remplace run_benchmark : il note la liste reçue, puis ne fait jouer
    # que l'agent aléatoire, pour que le test reste rapide.
    received: list[list[str]] = []

    def spy(agent_names: list[str], n_games: int, *args: Any, **kwargs: Any) -> Any:
        received.append(list(agent_names))
        return run_benchmark(["random"], n_games, *args, **kwargs)

    monkeypatch.setattr(cli, "run_benchmark", spy)
    result = CliRunner().invoke(app, ["benchmark", "--games", "1"])
    assert result.exit_code == 0, result.output
    assert received == [list(registry.AGENTS)]


def test_cli_benchmark_passes_the_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[int | None] = []

    def spy(
        agent_names: list[str],
        n_games: int,
        first_seed: int = 0,
        depth: int | None = None,
        on_game_done: Any = None,
    ) -> list[EvaluationReport]:
        received.append(depth)
        return [evaluate(RandomAgent(seed=0), 1)]

    monkeypatch.setattr(cli, "run_benchmark", spy)
    result = CliRunner().invoke(
        app, ["benchmark", "--agent", "expectimax", "--depth", "1", "--games", "1"]
    )
    assert result.exit_code == 0, result.output
    assert received == [1]
    assert "Profondeur de recherche : 1" in result.stdout


@pytest.mark.parametrize(
    "arguments",
    [
        ["benchmark", "--agent", "inconnu"],
        ["benchmark", "--agent", "random", "--agent", "inconnu"],
        ["benchmark", "--agent", "random", "--depth", "2"],
        ["benchmark", "--agent", "expectimax", "--depth", "0"],
        ["benchmark", "--games", "0"],
    ],
)
def test_cli_benchmark_rejects_bad_options(arguments: list[str]) -> None:
    # Code 2 : une erreur d'utilisation avec un message clair, pas une trace Python.
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 2, result.output


def test_markdown_table_is_what_the_benchmark_prints() -> None:
    # Le tableau affiché est bien celui de markdown_table (à la colonne du temps près).
    result = CliRunner().invoke(app, BENCH)
    assert result.exit_code == 0, result.output
    expected = markdown_table(run_benchmark(["random", "greedy"], 2))
    printed = [line.rsplit("|", 2)[0] for line in result.stdout.strip().splitlines()]
    assert printed == [line.rsplit("|", 2)[0] for line in expected.splitlines()]
