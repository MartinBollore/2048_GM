"""Tests de l'affichage texte."""

from __future__ import annotations

from gm2048.engine import from_values, new_board, render


def test_render_shows_tiles_and_empty_cells() -> None:
    board = from_values([[2, 0, 0, 0], [0, 2048, 0, 0], [0] * 4, [0, 0, 0, 4]])
    text = render(board)
    lines = text.splitlines()
    assert len(lines) == 9  # 4 lignes de cases + 5 séparateurs
    assert "2048" in lines[3]
    assert lines[1].split("|")[1].strip() == "2"
    assert lines[1].split("|")[2].strip() == "."
    assert all(len(line) == len(lines[0]) for line in lines)


def test_render_widens_cells_for_big_tiles() -> None:
    board = from_values([[65536, 0, 0, 0], [0] * 4, [0] * 4, [0] * 4])
    lines = render(board).splitlines()
    assert "65536" in lines[1]
    assert all(len(line) == len(lines[0]) for line in lines)


def test_render_header() -> None:
    text = render(new_board(), score=120, moves=15)
    assert text.splitlines()[0] == "Score : 120   Coups : 15   Meilleure tuile : 0"
