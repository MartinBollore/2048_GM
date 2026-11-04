"""Règles de 2048 sur un plateau NumPy.

Le plateau est un tableau 4x4 d'exposants (``uint8``) : 0 = case vide,
1 = tuile 2, 2 = tuile 4, ..., 11 = tuile 2048.

Toutes les fonctions de ce module sont pures : elles ne modifient jamais le
plateau reçu et renvoient un nouveau tableau. Le glissement (déterministe) est
séparé de l'apparition d'une tuile (aléatoire), ce qui permet aux agents de
raisonner sur l'« afterstate », le plateau juste après le glissement.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import numpy.typing as npt

BOARD_SIZE = 4
"""Côté du plateau."""

SPAWN_FOUR_PROBABILITY = 0.1
"""Probabilité qu'une nouvelle tuile soit un 4 plutôt qu'un 2."""

Board = npt.NDArray[np.uint8]
"""Plateau 4x4 d'exposants."""

ActionMask = npt.NDArray[np.bool_]
"""Masque de 4 booléens, indexé par :class:`Action`."""


class Action(IntEnum):
    """Les quatre coups possibles, dans l'ordre haut, droite, bas, gauche."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


ACTIONS: tuple[Action, ...] = tuple(Action)


@dataclass(frozen=True, slots=True)
class SlideResult:
    """Résultat d'un glissement, avant l'apparition d'une nouvelle tuile."""

    afterstate: Board
    """Plateau après glissement et fusions."""
    reward: int
    """Somme des valeurs des tuiles créées par fusion."""
    moved: bool
    """Vrai si le coup a modifié le plateau (coup légal)."""


# --- Construction et conversion ---------------------------------------------


def new_board() -> Board:
    """Renvoie un plateau vide."""
    return np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)


def from_values(values: npt.ArrayLike) -> Board:
    """Construit un plateau à partir des valeurs affichées (0, 2, 4, 8, ...).

    Lève ``ValueError`` si la grille n'est pas 4x4 ou contient une valeur qui
    n'est ni 0 ni une puissance de 2 supérieure ou égale à 2.
    """
    grid = np.asarray(values, dtype=np.int64)
    if grid.shape != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError(f"Le plateau doit être de forme 4x4, reçu {grid.shape}.")
    board = new_board()
    for (row, col), value in np.ndenumerate(grid):
        tile = int(value)
        if tile == 0:
            continue
        if tile < 2 or tile & (tile - 1) != 0:
            raise ValueError(f"Valeur de tuile invalide : {tile}.")
        board[row, col] = tile.bit_length() - 1
    return board


def to_values(board: Board) -> npt.NDArray[np.int64]:
    """Convertit un plateau d'exposants en valeurs affichées (0, 2, 4, 8, ...)."""
    exponents = board.astype(np.int64)
    return np.where(exponents > 0, np.left_shift(1, exponents), 0)


def max_tile(board: Board) -> int:
    """Valeur de la plus grande tuile du plateau (0 si le plateau est vide)."""
    exponent = int(board.max())
    return 1 << exponent if exponent > 0 else 0


def empty_cells(board: Board) -> npt.NDArray[np.intp]:
    """Indices (à plat, de 0 à 15) des cases vides."""
    return np.flatnonzero(board == 0)


# --- Glissement ---------------------------------------------------------------
#
# Chaque direction est ramenée à un glissement vers la gauche : on réoriente le
# plateau pour que le coup devienne « gauche », on fait glisser chaque ligne,
# puis on remet le plateau dans son orientation d'origine.


def _orient(board: Board, action: Action) -> Board:
    """Réoriente le plateau pour que ``action`` devienne un glissement à gauche."""
    if action is Action.LEFT:
        return board
    if action is Action.RIGHT:
        return board[:, ::-1]
    if action is Action.UP:
        return board.T
    return board[::-1, :].T  # DOWN : chaque colonne lue de bas en haut


def _unorient(board: Board, action: Action) -> Board:
    """Inverse de :func:`_orient`."""
    if action is Action.LEFT:
        return board
    if action is Action.RIGHT:
        return board[:, ::-1]
    if action is Action.UP:
        return board.T
    return board.T[::-1, :]  # DOWN


def _slide_row_left(row: Board) -> tuple[list[int], int]:
    """Fait glisser une ligne vers la gauche.

    Chaque tuile fusionne au plus une fois par coup, en partant du bord vers
    lequel on glisse : ``[2, 2, 2, 2]`` donne ``[4, 4, 0, 0]`` et
    ``[2, 2, 4, 0]`` donne ``[4, 4, 0, 0]`` (et non ``[8, 0, 0, 0]``).
    """
    tiles = [int(t) for t in row if t != 0]
    result: list[int] = []
    reward = 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            exponent = tiles[i] + 1
            result.append(exponent)
            reward += 1 << exponent
            i += 2
        else:
            result.append(tiles[i])
            i += 1
    result.extend([0] * (len(row) - len(result)))
    return result, reward


def slide(board: Board, action: Action) -> SlideResult:
    """Applique un coup sans faire apparaître de nouvelle tuile."""
    action = Action(action)
    oriented = _orient(board, action)
    slid = np.empty_like(oriented)
    reward = 0
    for r in range(BOARD_SIZE):
        row, row_reward = _slide_row_left(oriented[r])
        slid[r] = row
        reward += row_reward
    afterstate = np.ascontiguousarray(_unorient(slid, action))
    moved = not np.array_equal(afterstate, board)
    return SlideResult(afterstate=afterstate, reward=reward, moved=moved)


def _can_move_left(board: Board) -> bool:
    """Vrai si un glissement à gauche modifierait le plateau."""
    left, right = board[:, :-1], board[:, 1:]
    hole_before_tile = (left == 0) & (right != 0)
    mergeable = (left == right) & (left != 0)
    return bool(np.any(hole_before_tile | mergeable))


def legal_actions(board: Board) -> ActionMask:
    """Masque des coups qui modifient le plateau, indexé par :class:`Action`."""
    return np.array([_can_move_left(_orient(board, a)) for a in ACTIONS], dtype=np.bool_)


def is_game_over(board: Board) -> bool:
    """Vrai si plus aucun coup ne modifie le plateau."""
    return not legal_actions(board).any()


# --- Apparition d'une tuile ---------------------------------------------------


def spawn(board: Board, rng: np.random.Generator) -> Board:
    """Ajoute un 2 (90 %) ou un 4 (10 %) sur une case vide tirée au hasard.

    Lève ``ValueError`` si le plateau est plein.
    """
    empty = empty_cells(board)
    if empty.size == 0:
        raise ValueError("Aucune case vide : impossible d'ajouter une tuile.")
    cell = int(rng.choice(empty))
    exponent = 2 if rng.random() < SPAWN_FOUR_PROBABILITY else 1
    result = board.copy()
    result.flat[cell] = exponent
    return result
