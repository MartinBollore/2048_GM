"""Tests de la fenêtre de courbes (``gm2048 plot`` et ``gm2048 train --plot``).

La fenêtre ne fait que relire ``log.csv`` : on vérifie que ses courbes contiennent
bien ce que dit le journal, qu'elles suivent un journal qui grandit, et qu'elles
acceptent les journaux d'avant l'ajout du meilleur score et des tuiles.
"""

from __future__ import annotations

import csv
import dataclasses
import math
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from gm2048.training.loop import LOG_FILE, TrainingConfig, read_log, resume, train
from gm2048.training.plot import PLOT_TILES, TrainingPlot

SMALL = TrainingConfig(games=30, alpha=0.1, seed=3, checkpoint_every=10, log_every=10)


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    yield
    plt.close("all")


def ydata(line: object) -> list[float]:
    return [float(y) for y in np.asarray(line.get_ydata())]  # type: ignore[attr-defined]


def xdata(line: object) -> list[float]:
    return [float(x) for x in np.asarray(line.get_xdata())]  # type: ignore[attr-defined]


def test_curves_follow_the_log(tmp_path: Path) -> None:
    run = tmp_path / "run"
    train(SMALL, run)
    rows = read_log(run)
    plot = TrainingPlot([run])
    assert xdata(plot.mean_lines[run]) == [10, 20, 30]
    assert ydata(plot.mean_lines[run]) == [row["mean_score"] for row in rows]
    assert ydata(plot.max_lines[run]) == [row["max_score"] for row in rows]
    for tile in PLOT_TILES:
        assert ydata(plot.tile_lines[run, tile]) == [row[f"reach_{tile}"] for row in rows]


def test_refresh_follows_a_growing_log(tmp_path: Path) -> None:
    run = tmp_path / "run"
    train(dataclasses.replace(SMALL, games=10), run)
    plot = TrainingPlot([run])
    assert xdata(plot.mean_lines[run]) == [10]
    assert plot.refresh() is False  # rien n'a changé
    resume(run, games=30)
    assert plot.refresh() is True
    assert xdata(plot.mean_lines[run]) == [10, 20, 30]
    assert "30 / 30 parties" in plot.title()


def test_a_training_that_has_not_started_yet(tmp_path: Path) -> None:
    run = tmp_path / "pas-encore"
    plot = TrainingPlot([run])  # gm2048 train --plot ouvre la fenêtre avant la 1re partie
    assert xdata(plot.mean_lines[run]) == []
    train(SMALL, run)
    assert plot.refresh() is True
    assert xdata(plot.mean_lines[run]) == [10, 20, 30]


def test_an_old_log_without_max_score_and_tiles(tmp_path: Path) -> None:
    run = tmp_path / "run"
    train(SMALL, run)
    with (run / LOG_FILE).open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    with (run / LOG_FILE).open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([row[:5] for row in rows])  # colonnes de l'ancien format
    plot = TrainingPlot([run])
    assert all(math.isnan(y) for y in ydata(plot.max_lines[run]))
    assert plot.max_note.get_visible()
    # La courbe 2048 reprend win_rate, qui existait déjà.
    assert ydata(plot.tile_lines[run, 2048]) == [float(row[2]) for row in rows[1:]]
    assert all(math.isnan(y) for y in ydata(plot.tile_lines[run, 4096]))


def test_several_trainings_are_overlaid(tmp_path: Path) -> None:
    a, b = tmp_path / "alpha-0.1", tmp_path / "alpha-0.2"
    train(SMALL, a)
    train(dataclasses.replace(SMALL, alpha=0.2, games=20), b)
    plot = TrainingPlot([a, b])
    assert xdata(plot.mean_lines[a]) == [10, 20, 30]
    assert xdata(plot.mean_lines[b]) == [10, 20]
    assert plot.mean_lines[a].get_label() == "alpha-0.1"
    assert plot.mean_lines[a].get_color() != plot.mean_lines[b].get_color()
    title = plot.title()
    assert "alpha-0.1 : 30 / 30 parties" in title
    assert "alpha-0.2 : 20 / 20 parties" in title


def test_the_command_line_does_not_load_matplotlib() -> None:
    # matplotlib met du temps à se charger : seuls gm2048 plot et --plot l'importent.
    code = "import sys, gm2048.cli; assert 'matplotlib' not in sys.modules, 'matplotlib importé'"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
