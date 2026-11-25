"""La partie affichée : animations dans le temps, points gagnés, fin de partie.

Rien ici ne dépend de pygame. Le temps avance par petits pas (``update(dt)``,
``dt`` en secondes) et :meth:`Animation.sprites` dit, à chaque instant, quelles
tuiles dessiner, où et à quelle taille. La fenêtre n'a plus qu'à les peindre.

Un coup s'anime en deux temps :

1. **glissement** : chaque tuile file de sa case de départ à sa case d'arrivée ;
2. **apparition** : les tuiles nées d'une fusion gonflent puis reprennent leur
   taille, la nouvelle tuile surgit en grandissant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gm2048.engine import Action, Game, StepResult
from gm2048.evaluation import WIN_TILE
from gm2048.gui.motion import Transition, appearance, plan_transition

SLIDE_SECONDS = 0.10
"""Durée du glissement des tuiles, à vitesse normale."""
POP_SECONDS = 0.14
"""Durée de l'apparition des nouvelles tuiles, à vitesse normale."""
POPUP_SECONDS = 0.7
"""Durée de vie du « +points » qui s'envole au-dessus du score."""
TOAST_SECONDS = 2.6
"""Durée d'affichage du bandeau « tuile 2048 atteinte »."""

MERGE_SWELL = 0.18
"""De combien grossit (en proportion) une tuile née d'une fusion, au plus fort."""


# --- Courbes d'animation --------------------------------------------------------
#
# Chaque courbe transforme une progression ``u`` entre 0 et 1 (le temps) en une
# valeur à dessiner. Une progression linéaire paraît mécanique : on démarre vite
# et on freine en arrivant, comme un objet qui glisse puis s'arrête.


def ease_out_cubic(u: float) -> float:
    """Départ rapide, arrivée en douceur : 0 → 0, 1 → 1."""
    return 1 - (1 - u) ** 3


def ease_out_back(u: float, overshoot: float = 1.4) -> float:
    """Comme :func:`ease_out_cubic`, mais dépasse un peu 1 avant d'y revenir."""
    v = u - 1
    return 1 + (overshoot + 1) * v**3 + overshoot * v**2


def swell(u: float) -> float:
    """Taille d'une tuile née d'une fusion : 1, puis un peu plus, puis 1."""
    return 1 + MERGE_SWELL * math.sin(math.pi * u)


def _lerp(start: float, end: float, u: float) -> float:
    return start + (end - start) * u


@dataclass(frozen=True, slots=True)
class Sprite:
    """Une tuile à dessiner, à une position éventuellement entre deux cases."""

    exponent: int
    row: float
    col: float
    scale: float = 1.0
    """Taille relative : 1.0 = une case entière."""
    alpha: float = 1.0
    """Opacité, de 0.0 (invisible) à 1.0."""


class Animation:
    """Le déroulé dans le temps d'une :class:`~gm2048.gui.motion.Transition`.

    ``duration_scale`` raccourcit l'animation : 1.0 = vitesse normale, 0.0 =
    aucune animation (on affiche directement le plateau d'arrivée).
    """

    def __init__(self, transition: Transition, duration_scale: float = 1.0) -> None:
        self.transition = transition
        scale = max(0.0, duration_scale)
        self.slide_seconds = SLIDE_SECONDS * scale if transition.moves else 0.0
        self.pop_seconds = POP_SECONDS * scale
        self.elapsed = 0.0

    @property
    def duration(self) -> float:
        return self.slide_seconds + self.pop_seconds

    @property
    def done(self) -> bool:
        return self.elapsed >= self.duration

    def advance(self, dt: float) -> None:
        self.elapsed = min(self.elapsed + dt, self.duration)

    def finish(self) -> None:
        self.elapsed = self.duration

    def sprites(self) -> list[Sprite]:
        """Les tuiles à dessiner maintenant, dans l'ordre où les peindre."""
        if self.elapsed < self.slide_seconds:
            return self._sliding(ease_out_cubic(self.elapsed / self.slide_seconds))
        if self.pop_seconds > 0:
            return self._popping(min(1.0, (self.elapsed - self.slide_seconds) / self.pop_seconds))
        return self._popping(1.0)

    def _sliding(self, u: float) -> list[Sprite]:
        # Les tuiles immobiles d'abord, pour que celles qui glissent passent dessus.
        moves = sorted(self.transition.moves, key=lambda move: move.start != move.end)
        return [
            Sprite(
                move.exponent,
                _lerp(move.start[0], move.end[0], u),
                _lerp(move.start[1], move.end[1], u),
            )
            for move in moves
        ]

    def _popping(self, u: float) -> list[Sprite]:
        transition = self.transition
        fresh = {tile.cell for tile in transition.merged} | {
            tile.cell for tile in transition.spawned
        }
        sprites = [
            Sprite(int(transition.board[row, col]), float(row), float(col))
            for row, col in np.argwhere(transition.board != 0)
            if (int(row), int(col)) not in fresh
        ]
        sprites.extend(
            Sprite(tile.exponent, tile.cell[0], tile.cell[1], scale=swell(u))
            for tile in transition.merged
        )
        sprites.extend(
            Sprite(
                tile.exponent,
                tile.cell[0],
                tile.cell[1],
                scale=max(0.0, ease_out_back(u)),
                alpha=min(1.0, 2.5 * u),
            )
            for tile in transition.spawned
        )
        return sprites


@dataclass(slots=True)
class ScorePopup:
    """Des points gagnés qui s'envolent au-dessus du score."""

    points: int
    age: float = 0.0

    @property
    def progress(self) -> float:
        """De 0.0 (vient d'apparaître) à 1.0 (disparaît)."""
        return min(1.0, self.age / POPUP_SECONDS)


class Session:
    """La partie affichée dans la fenêtre, et ce qui l'entoure.

    Elle garde la partie en cours, son animation, les points qui s'envolent, le
    bandeau de victoire et le meilleur score obtenu depuis l'ouverture de la
    fenêtre. ``animation_scale`` règle la durée des animations (voir
    :class:`Animation`) ; à 0, sans animation, les points ne s'envolent pas non
    plus (ils changeraient à chaque image).
    """

    def __init__(self, game: Game, animation_scale: float = 1.0) -> None:
        self.animation_scale = animation_scale
        self.best_score = 0
        self.games_started = 0
        self.popup: ScorePopup | None = None
        """Les derniers points gagnés, qui s'envolent (un seul à la fois)."""
        self.toast_age: float | None = None
        """Temps écoulé depuis l'arrivée à 2048, ou ``None`` si pas de bandeau."""
        self.over_age = 0.0
        """Temps écoulé depuis la fin de la partie (pour l'apparition en fondu)."""
        self.last_action: Action | None = None
        self.game = game
        self.animation = Animation(appearance(game.board), animation_scale)
        self._won = False
        self.new_game(game)

    def new_game(self, game: Game) -> None:
        """Affiche une nouvelle partie (le meilleur score est conservé)."""
        self.game = game
        self.games_started += 1
        self.animation = Animation(appearance(game.board), self.animation_scale)
        self.popup = None
        self.toast_age = None
        self.over_age = 0.0
        self.last_action = None
        self._won = game.max_tile >= WIN_TILE
        self.best_score = max(self.best_score, game.score)

    def play(self, action: Action) -> StepResult:
        """Joue un coup et lance son animation (l'animation précédente est abandonnée).

        Lève ``RuntimeError`` si la partie est déjà terminée.
        """
        before = self.game.board
        result = self.game.step(action)
        if not result.moved:
            return result
        self.animation = Animation(
            plan_transition(before, action, result.board), self.animation_scale
        )
        self.last_action = Action(action)
        if result.reward and self.animation_scale > 0:
            self.popup = ScorePopup(result.reward)
        self.best_score = max(self.best_score, self.game.score)
        if not self._won and self.game.max_tile >= WIN_TILE:
            self._won = True
            self.toast_age = 0.0
        return result

    @property
    def animating(self) -> bool:
        return not self.animation.done

    @property
    def over(self) -> bool:
        """Vrai quand la partie est finie et que la dernière animation est terminée."""
        return not self.animating and self.game.done

    def update(self, dt: float) -> None:
        """Fait avancer le temps de ``dt`` secondes."""
        animation_left = self.animation.duration - self.animation.elapsed
        self.animation.advance(dt)
        if self.popup is not None:
            self.popup.age += dt
            if self.popup.age >= POPUP_SECONDS:
                self.popup = None
        if self.toast_age is not None:
            self.toast_age += dt
            if self.toast_age >= TOAST_SECONDS:
                self.toast_age = None
        if self.over:
            # Seul compte le temps écoulé après la fin de la dernière animation.
            self.over_age += max(0.0, dt - animation_left)

    def sprites(self) -> list[Sprite]:
        return self.animation.sprites()
