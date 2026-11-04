"""Tests de la ligne de commande et de la boucle de jeu."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from typer.testing import CliRunner

from gm2048.cli import QUIT, app, parse_key, play_loop
from gm2048.engine import Action, Game, from_values


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
    result = CliRunner().invoke(app, ["play", "--seed", "1"], input="z\nq\ns\nd\nx\n")
    assert result.exit_code == 0, result.output
    assert "Score : " in result.output
    assert "Partie abandonnée" in result.output


def test_cli_play_ends_cleanly_on_end_of_input() -> None:
    result = CliRunner().invoke(app, ["play", "--seed", "1"], input="")
    assert result.exit_code == 0, result.output
    assert "Partie abandonnée" in result.output
