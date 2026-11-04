"""Une partie de 2048 : plateau, score, hasard et fin de partie."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from gm2048.engine.board import (
    Action,
    ActionMask,
    Board,
    is_game_over,
    legal_actions,
    max_tile,
    new_board,
    slide,
    spawn,
)


@dataclass(frozen=True, slots=True)
class StepResult:
    """Résultat d'un coup joué dans une :class:`Game`."""

    afterstate: Board
    """Plateau après glissement, avant l'apparition de la nouvelle tuile."""
    board: Board
    """Plateau après l'apparition de la nouvelle tuile."""
    reward: int
    """Points gagnés par ce coup (somme des tuiles créées par fusion)."""
    moved: bool
    """Faux si le coup ne déplaçait aucune tuile : rien n'a changé."""
    done: bool
    """Vrai si la partie est terminée après ce coup."""


class Game:
    """Une partie de 2048.

    Tout le hasard passe par un ``numpy.random.Generator`` interne : avec la
    même graine et les mêmes coups, on obtient exactement la même partie.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._board = new_board()
        self.score = 0
        self.moves = 0
        self.reset()

    @classmethod
    def from_board(cls, board: npt.ArrayLike, seed: int | None = None, score: int = 0) -> Game:
        """Crée une partie à partir d'un plateau d'exposants donné."""
        game = cls(seed)
        start = np.array(board, dtype=np.uint8)
        if start.shape != game._board.shape:
            raise ValueError(f"Le plateau doit être de forme 4x4, reçu {start.shape}.")
        game._board = start
        game.score = score
        return game

    def reset(self, seed: int | None = None) -> Board:
        """Recommence une partie avec deux tuiles au hasard.

        Sans graine, le générateur continue sa séquence : des parties
        successives sont différentes mais toutes reproductibles.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        board = spawn(new_board(), self._rng)
        self._board = spawn(board, self._rng)
        self.score = 0
        self.moves = 0
        return self.board

    def step(self, action: Action | int) -> StepResult:
        """Joue un coup.

        Un coup qui ne déplace aucune tuile ne change rien : pas de nouvelle
        tuile, pas de point, et le compteur de coups n'avance pas.
        """
        if self.done:
            raise RuntimeError("La partie est terminée : appelez reset() pour recommencer.")
        result = slide(self._board, Action(action))
        if not result.moved:
            return StepResult(result.afterstate, self.board, 0, moved=False, done=False)
        self._board = spawn(result.afterstate, self._rng)
        self.score += result.reward
        self.moves += 1
        return StepResult(result.afterstate, self.board, result.reward, moved=True, done=self.done)

    @property
    def board(self) -> Board:
        """Copie du plateau courant (le modifier ne change pas la partie)."""
        return self._board.copy()

    @property
    def done(self) -> bool:
        """Vrai si plus aucun coup n'est possible."""
        return is_game_over(self._board)

    @property
    def max_tile(self) -> int:
        """Valeur de la plus grande tuile du plateau."""
        return max_tile(self._board)

    def legal_actions(self) -> ActionMask:
        """Masque des coups légaux, indexé par :class:`Action`."""
        return legal_actions(self._board)
