"""Ligne de commande ``gm2048``."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import Annotated, Final, Literal

import typer

from gm2048.agents import Agent
from gm2048.agents.baselines import GreedyAgent, RandomAgent
from gm2048.engine import Action, Game, render
from gm2048.evaluation import evaluate, format_report

AGENTS: Final[dict[str, Callable[[int | None], Agent]]] = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
}
"""Les agents disponibles en ligne de commande : nom → fonction qui crée l'agent
à partir d'une graine. Ajouter un agent ici le rend utilisable par ``gm2048 eval``."""

app = typer.Typer(
    help="2048 GM : moteur de jeu 2048 et IA capable d'y jouer.",
    no_args_is_help=True,
    add_completion=False,
)

QUIT: Final = "quit"
Command = Action | Literal["quit"]

HELP_LINE = "Flèches, ZQSD ou WASD pour jouer ; x ou Échap pour quitter."

_KEYS: Final[dict[str, Action]] = {
    # Flèches sous Linux et macOS (séquences ANSI, deux variantes selon le terminal).
    "\x1b[A": Action.UP,
    "\x1b[B": Action.DOWN,
    "\x1b[C": Action.RIGHT,
    "\x1b[D": Action.LEFT,
    "\x1bOA": Action.UP,
    "\x1bOB": Action.DOWN,
    "\x1bOC": Action.RIGHT,
    "\x1bOD": Action.LEFT,
    # Flèches sous Windows (préfixe \xe0 ou \x00 suivi d'un code de touche).
    "\xe0H": Action.UP,
    "\xe0P": Action.DOWN,
    "\xe0M": Action.RIGHT,
    "\xe0K": Action.LEFT,
    "\x00H": Action.UP,
    "\x00P": Action.DOWN,
    "\x00M": Action.RIGHT,
    "\x00K": Action.LEFT,
    # Clavier AZERTY (ZQSD) et QWERTY (WASD).
    "z": Action.UP,
    "w": Action.UP,
    "q": Action.LEFT,
    "a": Action.LEFT,
    "s": Action.DOWN,
    "d": Action.RIGHT,
}
_QUIT_KEYS: Final = frozenset({"\x1b", "x"})


def parse_key(key: str) -> Command | None:
    """Traduit une touche (ou une ligne saisie) en coup, ``QUIT`` ou ``None``."""
    if not key.startswith(("\x1b", "\xe0", "\x00")):
        key = key.strip().lower()
    if key in _QUIT_KEYS:
        return QUIT
    return _KEYS.get(key)


def play_loop(
    game: Game,
    read_key: Callable[[], str],
    echo: Callable[[str], None],
    clear: Callable[[], None],
) -> None:
    """Boucle de jeu interactive, indépendante du terminal (donc testable)."""
    message = HELP_LINE
    while True:
        clear()
        echo(render(game.board, score=game.score, moves=game.moves))
        if game.done:
            echo(
                f"Partie terminée ! Score final : {game.score}, meilleure tuile : {game.max_tile}."
            )
            return
        echo(message)
        try:
            command = parse_key(read_key())
        except (KeyboardInterrupt, EOFError):
            command = QUIT
        if command == QUIT:
            echo(f"Partie abandonnée. Score : {game.score}, meilleure tuile : {game.max_tile}.")
            return
        if command is None:
            message = f"Touche non reconnue. {HELP_LINE}"
            continue
        result = game.step(command)
        message = HELP_LINE if result.moved else "Ce coup ne déplace aucune tuile."


def _read_char() -> str:
    return typer.getchar()


def _read_line() -> str:
    return input("> ")


def _clear_screen() -> None:
    if os.name == "nt":
        os.system("cls")
    else:
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()


def _no_clear() -> None:
    pass


@app.callback()
def main() -> None:
    """2048 GM : moteur de jeu 2048 et IA capable d'y jouer."""


@app.command()
def play(
    seed: Annotated[
        int | None, typer.Option(help="Graine du hasard, pour rejouer la même partie.")
    ] = None,
    line: Annotated[
        bool,
        typer.Option(
            "--line",
            help="Saisie ligne par ligne (touche puis Entrée). Automatique hors d'un terminal.",
        ),
    ] = False,
) -> None:
    """Jouer une partie au clavier dans le terminal."""
    line_mode = line or not sys.stdin.isatty()
    play_loop(
        Game(seed=seed),
        read_key=_read_line if line_mode else _read_char,
        echo=typer.echo,
        clear=_no_clear if line_mode else _clear_screen,
    )


@app.command("eval")
def eval_(
    agent: Annotated[str, typer.Option(help=f"Agent à évaluer : {', '.join(AGENTS)}.")] = "random",
    games: Annotated[int, typer.Option(min=1, help="Nombre de parties à jouer.")] = 100,
    seed: Annotated[int, typer.Option(help="Graine de la première partie (et de l'agent).")] = 0,
) -> None:
    """Faire jouer un agent sur une série de parties et afficher ses résultats."""
    if agent not in AGENTS:
        raise typer.BadParameter(
            f"agent inconnu {agent!r}. Choix possibles : {', '.join(AGENTS)}.",
            param_hint="--agent",
        )
    report = evaluate(AGENTS[agent](seed), n_games=games, first_seed=seed)
    typer.echo(format_report(report))
