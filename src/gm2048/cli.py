"""Ligne de commande ``gm2048``."""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Callable, Sequence
from typing import Annotated, Final, Literal

import typer

from gm2048.agents import Agent
from gm2048.agents.baselines import GreedyAgent, HeuristicAgent, RandomAgent
from gm2048.engine import Action, Game, render
from gm2048.evaluation import GameResult, evaluate, format_report
from gm2048.gui.theme import THEMES

AGENTS: Final[dict[str, Callable[[int | None], Agent]]] = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "heuristic": HeuristicAgent,
}
"""Les agents disponibles en ligne de commande : nom → fonction qui crée l'agent
à partir d'une graine. Ajouter un agent ici le rend utilisable par ``gm2048 eval`` et
``gm2048 watch``."""

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


GUI_MISSING = (
    'La fenêtre graphique a besoin de pygame-ce. Pour l\'installer :\n    pip install -e ".[gui]"'
)

THEME_NAMES: Final = tuple(THEMES)
"""Les thèmes de la fenêtre."""

ThemeOption = Annotated[
    str, typer.Option(help=f"Thème de la fenêtre : {', '.join(THEME_NAMES)} (touche T).")
]


def _check_theme(theme: str) -> None:
    if theme not in THEME_NAMES:
        raise typer.BadParameter(
            f"thème inconnu {theme!r}. Choix possibles : {', '.join(THEME_NAMES)}.",
            param_hint="--theme",
        )


def _check_agent(agent: str) -> None:
    if agent not in AGENTS:
        raise typer.BadParameter(
            f"agent inconnu {agent!r}. Choix possibles : {', '.join(AGENTS)}.",
            param_hint="--agent",
        )


def _gui_available() -> bool:
    """Vrai si pygame est installé (sans l'importer)."""
    return importlib.util.find_spec("pygame") is not None


@app.command()
def play(
    seed: Annotated[
        int | None, typer.Option(help="Graine du hasard, pour rejouer la même partie.")
    ] = None,
    terminal: Annotated[
        bool,
        typer.Option("--terminal", help="Jouer dans le terminal plutôt que dans une fenêtre."),
    ] = False,
    line: Annotated[
        bool,
        typer.Option(
            "--line",
            help="Dans le terminal, saisie ligne par ligne (touche puis Entrée). "
            "Automatique hors d'un vrai terminal.",
        ),
    ] = False,
    theme: ThemeOption = "clair",
) -> None:
    """Jouer une partie au clavier, dans une fenêtre animée (ou dans le terminal)."""
    _check_theme(theme)
    if not (terminal or line):
        if _gui_available():
            from gm2048.gui import window

            window.play(seed=seed, theme=theme)
            return
        typer.echo(f"{GUI_MISSING}\nEn attendant, la partie se joue dans le terminal.", err=True)
    line_mode = line or not sys.stdin.isatty()
    play_loop(
        Game(seed=seed),
        read_key=_read_line if line_mode else _read_char,
        echo=typer.echo,
        clear=_no_clear if line_mode else _clear_screen,
    )


@app.command()
def watch(
    agent: Annotated[str, typer.Option(help=f"Agent à regarder : {', '.join(AGENTS)}.")] = (
        "heuristic"
    ),
    seed: Annotated[
        int, typer.Option(help="Graine de la première partie (les suivantes : +1, +2, …).")
    ] = 0,
    delay: Annotated[
        float, typer.Option(min=0.0, help="Secondes entre deux coups (touches + et - en direct).")
    ] = 0.2,
    theme: ThemeOption = "clair",
) -> None:
    """Regarder un agent jouer, dans une fenêtre animée."""
    _check_agent(agent)
    _check_theme(theme)
    if not _gui_available():
        typer.echo(GUI_MISSING, err=True)
        raise typer.Exit(code=1)
    from gm2048.gui import window

    window.watch(AGENTS[agent], seed=seed, delay=delay, theme=theme)


@app.command("eval")
def eval_(
    agent: Annotated[str, typer.Option(help=f"Agent à évaluer : {', '.join(AGENTS)}.")] = "random",
    games: Annotated[int, typer.Option(min=1, help="Nombre de parties à jouer.")] = 100,
    seed: Annotated[int, typer.Option(help="Graine de la première partie (et de l'agent).")] = 0,
) -> None:
    """Faire jouer un agent sur une série de parties et afficher ses résultats."""
    _check_agent(agent)
    played: list[GameResult] = []

    def show_info(_: GameResult | None) -> str | None:
        return progress_info(played)

    # La barre s'écrit sur la sortie d'erreur : la sortie standard ne contient
    # que le rapport, qu'on peut donc rediriger dans un fichier sans la barre.
    with typer.progressbar(
        length=games,
        label=f"Évaluation de {agent}",
        show_pos=True,
        item_show_func=show_info,
        file=sys.stderr,
    ) as bar:

        def on_game_done(result: GameResult) -> None:
            played.append(result)
            bar.current_item = result  # déclenche le rafraîchissement de progress_info
            bar.update(1)

        report = evaluate(
            AGENTS[agent](seed), n_games=games, first_seed=seed, on_game_done=on_game_done
        )
    typer.echo(format_report(report))


def progress_info(played: Sequence[GameResult]) -> str | None:
    """Texte affiché à droite de la barre : score moyen et meilleure tuile jusqu'ici."""
    if not played:
        return None
    mean = sum(result.score for result in played) / len(played)
    best = max(result.max_tile for result in played)
    return f"score moyen {mean:.0f}, meilleure tuile {best}"
