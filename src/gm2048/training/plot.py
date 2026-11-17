"""Les courbes d'un entraînement, dans une fenêtre qui se met à jour toute seule.

``gm2048 plot runs/x`` ouvre cette fenêtre, et ``gm2048 train --plot`` l'ouvre à côté
de l'entraînement. Elle ne communique pas avec le processus qui entraîne : elle relit
``log.csv`` à intervalles réguliers, et la boucle d'entraînement y ajoute une ligne à
chaque bloc de parties. On peut donc l'ouvrir et la fermer à tout moment, sur un
entraînement en cours ou terminé, et superposer plusieurs entraînements pour les
comparer.

Trois graphiques, un point par bloc du journal :

- le score moyen du bloc ;
- le meilleur score du bloc ;
- la part des parties du bloc qui atteignent 512, 1024, 2048, … (une courbe par tuile).

C'est le seul module du projet qui importe matplotlib.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter
from matplotlib.typing import LineStyleType

from gm2048.training.loop import LOG_FILE, STATE_FILE, load_state, read_log

PLOT_TILES = (512, 1024, 2048, 4096, 8192, 16384)
"""Les tuiles qui ont une courbe dans le troisième graphique."""

TILE_COLORS = {
    512: "#9a8c98",
    1024: "#4c9a8f",
    2048: "#2a62c9",
    4096: "#c9822a",
    8192: "#c0392b",
    16384: "#6c3483",
}
"""Une couleur par tuile, la même pour tous les entraînements."""

RUN_STYLES: tuple[LineStyleType, ...] = ("-", "--", ":", "-.")
"""Dans le graphique des tuiles, les entraînements se distinguent par le trait."""


def _thousands(value: float, _position: object) -> str:
    """12345 → « 12 345 », à la française."""
    return f"{value:,.0f}".replace(",", " ")


def _values(rows: list[dict[str, float | None]], column: str) -> list[float]:
    """Une colonne du journal, avec NaN à la place des valeurs absentes (un trou)."""
    return [math.nan if row[column] is None else float(row[column] or 0.0) for row in rows]


class TrainingPlot:
    """La figure qui suit un ou plusieurs entraînements."""

    def __init__(self, runs: Sequence[Path]) -> None:
        """Prépare les trois graphiques pour ``runs``, puis lit une première fois les
        journaux. Un dossier qui ne contient pas encore de journal donne des courbes
        vides, qui se rempliront aux lectures suivantes."""
        self.runs = list(runs)
        self.figure, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 9), layout="constrained")
        self.mean_ax, self.max_ax, self.tiles_ax = axes
        self._stamps: dict[Path, float | None] = {}
        self._games: dict[Path, list[float]] = {run: [] for run in self.runs}
        self._best: dict[Path, list[float]] = {run: [] for run in self.runs}

        self.mean_ax.set_ylabel("Score moyen")
        self.max_ax.set_ylabel("Meilleur score")
        self.tiles_ax.set_ylabel("Parties qui atteignent")
        self.tiles_ax.set_xlabel("Parties jouées")
        for ax in (self.mean_ax, self.max_ax):
            ax.yaxis.set_major_formatter(FuncFormatter(_thousands))
        self.tiles_ax.xaxis.set_major_formatter(FuncFormatter(_thousands))
        self.tiles_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        self.tiles_ax.set_ylim(-0.02, 1.02)
        for ax in axes:
            ax.grid(True, alpha=0.3)

        self.mean_lines: dict[Path, Line2D] = {}
        self.max_lines: dict[Path, Line2D] = {}
        self.tile_lines: dict[tuple[Path, int], Line2D] = {}
        for k, run in enumerate(self.runs):
            color, style = f"C{k}", RUN_STYLES[k % len(RUN_STYLES)]
            (self.mean_lines[run],) = self.mean_ax.plot([], [], color=color, label=run.name)
            (self.max_lines[run],) = self.max_ax.plot([], [], color=color, label=run.name)
            for tile in PLOT_TILES:
                (self.tile_lines[run, tile],) = self.tiles_ax.plot(
                    [], [], color=TILE_COLORS[tile], linestyle=style
                )

        if len(self.runs) > 1:
            self.mean_ax.legend(loc="lower right")
        handles = [Line2D([], [], color=TILE_COLORS[t], label=f"{t}") for t in PLOT_TILES]
        if len(self.runs) > 1:
            handles += [
                Line2D([], [], color="0.4", linestyle=RUN_STYLES[k % len(RUN_STYLES)], label=r.name)
                for k, r in enumerate(self.runs)
            ]
        self.tiles_ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5))
        self.max_note = self.max_ax.text(
            0.5,
            0.5,
            "Pas encore de meilleur score : ce journal a été commencé par une version\n"
            "précédente. Les prochains blocs l'auront.",
            transform=self.max_ax.transAxes,
            ha="center",
            va="center",
            color="0.4",
            visible=False,
        )
        self.refresh()

    def refresh(self) -> bool:
        """Relit les journaux qui ont changé depuis la dernière lecture et met les
        courbes à jour. Renvoie ``True`` si quelque chose a changé."""
        changed = False
        for run in self.runs:
            log = run / LOG_FILE
            stamp = log.stat().st_mtime if log.is_file() else None
            if run in self._stamps and self._stamps[run] == stamp:
                continue
            try:
                rows = read_log(run)
            except (OSError, ValueError):
                continue  # ligne en cours d'écriture : on réessaiera au prochain tour
            self._stamps[run] = stamp
            games = self._games[run] = _values(rows, "games")
            self._best[run] = _values(rows, "max_score")
            self.mean_lines[run].set_data(games, _values(rows, "mean_score"))
            self.max_lines[run].set_data(games, self._best[run])
            for tile in PLOT_TILES:
                reach = _values(rows, f"reach_{tile}")
                if tile == 2048:
                    # Les anciens journaux n'ont que win_rate, qui est la même chose.
                    win = _values(rows, "win_rate")
                    reach = [w if math.isnan(r) else r for r, w in zip(reach, win, strict=True)]
                self.tile_lines[run, tile].set_data(games, reach)
            changed = True

        if changed:
            no_max = all(math.isnan(y) for best in self._best.values() for y in best)
            has_rows = any(self._games.values())
            self.max_note.set_visible(no_max and has_rows)
            for ax in (self.mean_ax, self.max_ax, self.tiles_ax):
                ax.relim()
                ax.autoscale_view(scaley=ax is not self.tiles_ax)
            self.figure.suptitle(self.title())
        return changed

    def title(self) -> str:
        """Une ligne par entraînement : où il en est."""
        lines = []
        for run, games in self._games.items():
            done = int(games[-1]) if games else 0
            total = ""
            if (run / STATE_FILE).is_file():
                try:
                    total = f" / {_thousands(load_state(run)[0].games, None)}"
                except (OSError, ValueError, KeyError, TypeError):
                    total = ""
            lines.append(f"{run.name} : {_thousands(done, None)}{total} parties")
        return "\n".join(lines)


def show_training(runs: Sequence[Path], interval: float = 2.0) -> None:
    """Ouvre la fenêtre et relit les journaux toutes les ``interval`` secondes, jusqu'à
    ce qu'on la ferme."""
    plot = TrainingPlot(runs)
    manager = plot.figure.canvas.manager
    if manager is not None:
        manager.set_window_title("gm2048 · entraînement")
    timer = plot.figure.canvas.new_timer(interval=int(interval * 1000))

    def tick() -> None:
        if plot.refresh():
            plot.figure.canvas.draw_idle()

    timer.add_callback(tick)
    timer.start()
    plt.show()
