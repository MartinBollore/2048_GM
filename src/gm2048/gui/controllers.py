"""Qui joue, et quand : un humain au clavier, ou un agent à son rythme.

La fenêtre traduit les touches en :class:`Command` et appelle, à chaque image,
``handle`` pour chaque commande puis ``update(dt)``. Le contrôleur décide quoi
en faire : jouer un coup, mettre en pause, changer de vitesse…

Un agent peut mettre du temps à choisir son coup (un expectimax profond, par
exemple). Pour que la fenêtre reste fluide pendant qu'il réfléchit,
:class:`AgentController` lui demande son coup dans un fil d'exécution à part,
dès que le plateau est connu, pendant que l'animation précédente se joue.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from enum import Enum, auto
from typing import Final, Protocol

from gm2048.agents import Agent
from gm2048.engine import Action, Game
from gm2048.gui.session import POP_SECONDS, SLIDE_SECONDS, Session


class Command(Enum):
    """Ce que l'utilisateur demande au clavier."""

    UP = auto()
    RIGHT = auto()
    DOWN = auto()
    LEFT = auto()
    PAUSE = auto()
    STEP = auto()
    FASTER = auto()
    SLOWER = auto()
    NEW_GAME = auto()
    THEME = auto()
    QUIT = auto()


DIRECTIONS: Final[dict[Command, Action]] = {
    Command.UP: Action.UP,
    Command.RIGHT: Action.RIGHT,
    Command.DOWN: Action.DOWN,
    Command.LEFT: Action.LEFT,
}

DELAYS: Final = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)
"""Les délais entre deux coups proposés par les touches + et -, en secondes."""

_THINKING_GRACE = 0.3
"""Attente au-delà du délai avant d'afficher « réflexion » (évite le clignotement)."""


def faster(delay: float) -> float:
    """Le délai immédiatement plus court que ``delay`` dans :data:`DELAYS`."""
    shorter = [d for d in DELAYS if d < delay - 1e-9]
    return shorter[-1] if shorter else DELAYS[0]


def slower(delay: float) -> float:
    """Le délai immédiatement plus long que ``delay`` dans :data:`DELAYS`."""
    longer = [d for d in DELAYS if d > delay + 1e-9]
    return longer[0] if longer else max(delay, DELAYS[-1])


def animation_scale_for(delay: float) -> float:
    """Durée des animations adaptée au délai : elles tiennent dans 80 % du délai."""
    return min(1.0, 0.8 * delay / (SLIDE_SECONDS + POP_SECONDS))


def format_seconds(seconds: float) -> str:
    """``0.2`` → ``"0,20 s"`` (virgule décimale à la française)."""
    return f"{seconds:.2f} s".replace(".", ",")


class Controller(Protocol):
    """Ce que la fenêtre attend d'un contrôleur."""

    session: Session
    help_lines: tuple[str, ...]
    """Aide affichée en bas de la fenêtre."""

    @property
    def paused(self) -> bool: ...

    def handle(self, command: Command) -> None: ...

    def update(self, dt: float) -> None: ...

    def status(self) -> str:
        """Ligne d'information affichée au-dessus du plateau."""
        ...

    def close(self) -> None: ...


def _seed_for(first_seed: int | None, game_index: int) -> int | None:
    """Graine de la partie numéro ``game_index`` (0 pour la première)."""
    return None if first_seed is None else first_seed + game_index


class HumanController:
    """Un humain joue au clavier."""

    help_lines: tuple[str, ...] = (
        "Flèches, ZQSD ou WASD : jouer",
        "R : nouvelle partie   T : thème   Échap : quitter",
    )

    def __init__(self, seed: int | None = None) -> None:
        self.first_seed = seed
        self.session = Session(Game(seed=seed))

    @property
    def paused(self) -> bool:
        return False

    def handle(self, command: Command) -> None:
        if command in DIRECTIONS and not self.session.game.done:
            self.session.play(DIRECTIONS[command])
        elif command is Command.NEW_GAME:
            seed = _seed_for(self.first_seed, self.session.games_started)
            self.session.new_game(Game(seed=seed))

    def update(self, dt: float) -> None:
        self.session.update(dt)

    def status(self) -> str:
        parts = [f"Partie {self.session.games_started}", f"{self.session.game.moves} coups"]
        seed = _seed_for(self.first_seed, self.session.games_started - 1)
        if seed is not None:
            parts.append(f"graine {seed}")
        return " · ".join(parts)

    def close(self) -> None:
        pass


class AgentController:
    """Un agent joue, un coup tous les ``delay`` secondes au plus.

    La partie numéro ``k`` (en comptant à partir de 0) utilise la graine
    ``seed + k``, pour le plateau comme pour l'agent : la première partie est
    exactement celle que joue ``gm2048 eval --seed <seed>`` en premier.

    ``executor`` exécute les appels à ``agent.act`` ; par défaut, un fil
    d'exécution dédié. Les tests en passent un qui exécute tout de suite.
    """

    help_lines: tuple[str, ...] = (
        "Espace : pause   → : coup suivant   + / - : vitesse",
        "Entrée : nouvelle partie   T : thème   Échap : quitter",
    )

    def __init__(
        self,
        make_agent: Callable[[int | None], Agent],
        seed: int = 0,
        delay: float = 0.2,
        executor: Executor | None = None,
    ) -> None:
        if delay < 0:
            raise ValueError(f"Le délai doit être positif ou nul, reçu {delay}.")
        self._make_agent = make_agent
        self.first_seed = seed
        self.delay = delay
        self._paused = False
        self._step_requested = False
        self._since_move = 0.0
        self._pending: Future[Action] | None = None
        if executor is None:
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gm2048-agent")
        self._executor = executor
        self.agent = make_agent(seed)
        self.session = Session(Game(seed=seed), animation_scale_for(delay))

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def seed(self) -> int:
        """Graine de la partie en cours."""
        return self.first_seed + self.session.games_started - 1

    @property
    def thinking(self) -> bool:
        """Vrai si l'agent fait attendre la partie depuis un moment."""
        return (
            self._pending is not None
            and not self._pending.done()
            and self._since_move > self.delay + _THINKING_GRACE
        )

    def handle(self, command: Command) -> None:
        if command is Command.PAUSE:
            self._paused = not self._paused
            self._step_requested = False
        elif command in (Command.STEP, Command.RIGHT):
            # En pleine partie, la première pression met en pause ; ensuite, un coup par pression.
            if self._paused:
                self._step_requested = True
            else:
                self._paused = True
        elif command is Command.FASTER:
            self._set_delay(faster(self.delay))
        elif command is Command.SLOWER:
            self._set_delay(slower(self.delay))
        elif command is Command.NEW_GAME:
            self._new_game()

    def update(self, dt: float) -> None:
        self.session.update(dt)
        self._since_move += dt
        game = self.session.game
        if game.done:
            return
        if self._pending is None:
            self._pending = self._executor.submit(self.agent.act, game.board, game.legal_actions())
        if self.session.animating:
            return
        if self._paused:
            if not self._step_requested:
                return
        elif self._since_move < self.delay:
            return
        if not self._pending.done():
            return
        action = self._pending.result()  # relance ici une éventuelle erreur de l'agent
        self._pending = None
        self._step_requested = False
        result = self.session.play(action)
        if not result.moved:
            raise RuntimeError(
                f"L'agent {self.agent.name!r} a choisi le coup {Action(action).name}, "
                "qui ne déplace aucune tuile."
            )
        self._since_move = 0.0

    def status(self) -> str:
        parts = [
            self.agent.name,
            f"partie {self.session.games_started}",
            f"graine {self.seed}",
            f"{self.session.game.moves} coups",
            f"délai {format_seconds(self.delay)}",
        ]
        if self.thinking:
            parts.append("réflexion…")
        return " · ".join(parts)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _set_delay(self, delay: float) -> None:
        self.delay = delay
        self.session.animation_scale = animation_scale_for(delay)

    def _new_game(self) -> None:
        seed = self.first_seed + self.session.games_started
        # Un coup encore en réflexion concerne l'ancienne partie : on l'oublie.
        self._pending = None
        self._step_requested = False
        self._since_move = 0.0
        self.agent = self._make_agent(seed)
        self.session.new_game(Game(seed=seed))
