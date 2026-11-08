"""Ligne de commande ``gm2048``."""

from __future__ import annotations

import importlib.util
import io
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, Final, Literal

import typer

from gm2048.agents import Agent
from gm2048.agents.registry import AGENTS, DEPTH_AGENTS, make_agent
from gm2048.benchmark import markdown_table, run_benchmark
from gm2048.engine import Action, Game, render
from gm2048.evaluation import GameResult, evaluate, format_report
from gm2048.gui.theme import THEMES


def agent_factory(name: str, depth: int | None = None) -> Callable[[int | None], Agent]:
    """Fonction qui crée l'agent ``name`` à partir d'une graine.

    Sans ``depth``, l'agent garde ses réglages par défaut. Avec ``depth``, l'agent
    doit faire partie de :data:`DEPTH_AGENTS`, sinon ``typer.BadParameter`` est levée.
    """
    if depth is None:
        return AGENTS[name]
    if name not in DEPTH_AGENTS:
        raise typer.BadParameter(
            f"l'agent {name!r} n'a pas de profondeur de recherche. "
            f"Agents concernés : {', '.join(DEPTH_AGENTS)}.",
            param_hint="--depth",
        )

    return lambda seed: make_agent(name, seed, depth)


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

DepthOption = Annotated[
    int | None,
    typer.Option(
        min=1,
        help=f"Profondeur de recherche ({', '.join(DEPTH_AGENTS)} seulement). "
        "Par défaut, celle de l'agent.",
    ),
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


def _use_utf8_stdout() -> None:
    """Fait écrire la sortie standard en UTF-8.

    Sous Windows, quand la sortie est redirigée vers un fichier
    (``gm2048 benchmark > benchmark.md``), Python l'encode par défaut en cp1252,
    qui ne connaît pas certains caractères comme « ≥ » : l'affichage final
    planterait, après des heures de calcul. On le règle donc dès le début.
    """
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")


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
    depth: DepthOption = None,
) -> None:
    """Regarder un agent jouer, dans une fenêtre animée."""
    _check_agent(agent)
    _check_theme(theme)
    make_agent = agent_factory(agent, depth)
    if not _gui_available():
        typer.echo(GUI_MISSING, err=True)
        raise typer.Exit(code=1)
    from gm2048.gui import window

    window.watch(make_agent, seed=seed, delay=delay, theme=theme)


@app.command("eval")
def eval_(
    agent: Annotated[str, typer.Option(help=f"Agent à évaluer : {', '.join(AGENTS)}.")] = "random",
    games: Annotated[int, typer.Option(min=1, help="Nombre de parties à jouer.")] = 100,
    seed: Annotated[int, typer.Option(help="Graine de la première partie (et de l'agent).")] = 0,
    depth: DepthOption = None,
) -> None:
    """Faire jouer un agent sur une série de parties et afficher ses résultats."""
    _check_agent(agent)
    make_agent = agent_factory(agent, depth)
    label = f"Évaluation de {agent}" + ("" if depth is None else f" (profondeur {depth})")
    played: list[GameResult] = []

    def show_info(_: GameResult | None) -> str | None:
        return progress_info(played)

    # La barre s'écrit sur la sortie d'erreur : la sortie standard ne contient
    # que le rapport, qu'on peut donc rediriger dans un fichier sans la barre.
    with typer.progressbar(
        length=games,
        label=label,
        show_pos=True,
        item_show_func=show_info,
        file=sys.stderr,
    ) as bar:

        def on_game_done(result: GameResult) -> None:
            played.append(result)
            bar.current_item = result  # déclenche le rafraîchissement de progress_info
            bar.update(1)

        report = evaluate(
            make_agent(seed), n_games=games, first_seed=seed, on_game_done=on_game_done
        )
    if depth is not None:
        typer.echo(f"Profondeur de recherche : {depth}")
    typer.echo(format_report(report))


def progress_info(played: Sequence[GameResult]) -> str | None:
    """Texte affiché à droite de la barre : score moyen, meilleur score et meilleure
    tuile jusqu'ici."""
    if not played:
        return None
    mean = sum(result.score for result in played) / len(played)
    best_score = max(result.score for result in played)
    best_tile = max(result.max_tile for result in played)
    return f"score moyen {mean:.0f}, meilleur score {best_score}, meilleure tuile {best_tile}"


@app.command()
def benchmark(
    agent: Annotated[
        list[str] | None,
        typer.Option(
            help="Agent à comparer ; répète l'option pour en mettre plusieurs. "
            f"Par défaut, tous : {', '.join(AGENTS)}."
        ),
    ] = None,
    games: Annotated[int, typer.Option(min=1, help="Nombre de parties par agent.")] = 100,
    seed: Annotated[int, typer.Option(help="Graine de la première partie (et des agents).")] = 0,
    depth: DepthOption = None,
    output: Annotated[
        Path | None,
        typer.Option(
            dir_okay=False,
            help="Fichier où écrire le tableau (en UTF-8), au lieu de l'afficher. "
            "Plus sûr que la redirection > sous PowerShell, qui abîme les accents.",
        ),
    ] = None,
) -> None:
    """Comparer plusieurs agents sur les mêmes parties et afficher un tableau Markdown.

    Étape 7, à toi de jouer. Le comportement attendu :

    - Sans ``--agent``, tous les agents du registre, dans l'ordre de ``AGENTS``.
    - Un nom inconnu, ou un ``--depth`` alors qu'aucun agent choisi n'a de profondeur,
      donne une erreur d'utilisation (``typer.BadParameter``), avant de jouer la
      moindre partie. ``--depth`` ne s'applique qu'aux agents de ``DEPTH_AGENTS``.
    - Les parties sont jouées par ``run_benchmark`` et le tableau produit par
      ``markdown_table`` : importe-les depuis ``gm2048.benchmark`` sous ces noms.
    - Une barre de progression sur la sortie d'erreur, comme dans ``eval_``, avec une
      étape par partie jouée (tous agents confondus).
    - Sur la sortie standard, uniquement le tableau, précédé de la ligne
      ``Profondeur de recherche : N`` si ``--depth`` est donné.

    Attention au temps : expectimax à profondeur 2 met plus d'une minute et demie par
    partie. Pour 1 000 parties, ``--depth 1`` est plus raisonnable.
    """
    _use_utf8_stdout()
    if agent is None:
        agent = list(AGENTS)

    for name in agent:
        _check_agent(name)

    if depth is not None and not any(name in DEPTH_AGENTS for name in agent):
        raise typer.BadParameter(
            "aucun des agents choisis n'a de profondeur de recherche. "
            f"Agents concernés : {', '.join(DEPTH_AGENTS)}.",
            param_hint="--depth",
        )
    if output is not None and not output.parent.is_dir():
        # Vérifié avant de jouer : on ne découvre pas une faute de frappe après des heures.
        raise typer.BadParameter(
            f"le dossier {str(output.parent)!r} n'existe pas.", param_hint="--output"
        )

    label = "Benchmark" + ("" if depth is None else f" (profondeur {depth})")
    played: list[GameResult] = []

    def show_info(_: GameResult | None) -> str | None:
        return progress_info(played)

    # Comme pour eval : la barre sur la sortie d'erreur, le tableau seul sur la sortie
    # standard (gm2048 benchmark > benchmark.md ne récupère que le tableau).
    with typer.progressbar(
        length=games * len(agent),
        label=label,
        show_pos=True,
        item_show_func=show_info,
        file=sys.stderr,
    ) as bar:

        def on_game_done(result: GameResult) -> None:
            played.append(result)
            bar.current_item = result
            bar.update(1)

        evaluation_reports = run_benchmark(
            agent_names=agent,
            n_games=games,
            first_seed=seed,
            depth=depth,
            on_game_done=on_game_done,
        )

    text = markdown_table(evaluation_reports)
    if depth is not None:
        # La ligne vide est indispensable : collé à une ligne de texte, un tableau
        # Markdown n'est pas reconnu et s'affiche comme un seul paragraphe.
        text = f"Profondeur de recherche : {depth}\n\n{text}"
    if output is None:
        typer.echo(text)
    else:
        output.write_text(text + "\n", encoding="utf-8")
        typer.echo(f"Tableau écrit dans {output}", err=True)
