"""Suivre chaque tuile pendant un coup, pour pouvoir l'animer.

Le moteur (:mod:`gm2048.engine`) ne connaît que des plateaux : avant le coup,
après le glissement, après l'apparition d'une tuile. Pour dessiner une tuile
qui glisse, il faut savoir *d'où* elle part et *où* elle arrive. Ce module le
retrouve en rejouant le glissement tuile par tuile, avec exactement les mêmes
règles que :func:`gm2048.engine.slide` (les tests le vérifient sur des
centaines de plateaux tirés au hasard).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gm2048.engine import BOARD_SIZE, Action, Board, new_board

Cell = tuple[int, int]
"""Une case du plateau : (ligne, colonne), de (0, 0) en haut à gauche à (3, 3)."""


@dataclass(frozen=True, slots=True)
class TileMove:
    """Le trajet d'une tuile pendant un glissement."""

    exponent: int
    """Exposant de la tuile qui se déplace (1 pour la tuile 2)."""
    start: Cell
    """Case de départ."""
    end: Cell
    """Case d'arrivée (égale à ``start`` si la tuile ne bouge pas)."""
    merges: bool
    """Vrai si la tuile fusionne avec une autre en arrivant sur ``end``."""


@dataclass(frozen=True, slots=True)
class NewTile:
    """Une tuile qui apparaît : née d'une fusion, ou ajoutée au hasard."""

    cell: Cell
    exponent: int


@dataclass(frozen=True, slots=True)
class Transition:
    """Tout ce qu'il faut pour animer le passage d'un plateau au suivant."""

    moves: tuple[TileMove, ...]
    """Trajet de chaque tuile du plateau de départ."""
    merged: tuple[NewTile, ...]
    """Tuiles créées par les fusions, à la place des paires qui fusionnent."""
    spawned: tuple[NewTile, ...]
    """Tuiles ajoutées au hasard après le glissement (une seule après un coup)."""
    board: Board
    """Plateau d'arrivée, une fois l'animation terminée."""
    action: Action | None = None
    """Coup joué, ou ``None`` pour l'apparition d'un plateau neuf."""

    @property
    def reward(self) -> int:
        """Points gagnés : la somme des tuiles créées par fusion."""
        return sum(1 << tile.exponent for tile in self.merged)


def line_cells(action: Action, line: int) -> list[Cell]:
    """Les cases d'une ligne (ou colonne) dans l'ordre où les tuiles s'y empilent.

    La première case est celle du bord vers lequel on glisse : pour un coup à
    droite, la ligne 0 donne ``[(0, 3), (0, 2), (0, 1), (0, 0)]``.
    """
    last = BOARD_SIZE - 1
    steps = range(BOARD_SIZE)
    if action is Action.LEFT:
        return [(line, k) for k in steps]
    if action is Action.RIGHT:
        return [(line, last - k) for k in steps]
    if action is Action.UP:
        return [(k, line) for k in steps]
    return [(last - k, line) for k in steps]


def trace_slide(board: Board, action: Action) -> tuple[tuple[TileMove, ...], tuple[NewTile, ...]]:
    """Trajets des tuiles et fusions d'un glissement, sans apparition de tuile.

    Comme dans le moteur, chaque tuile fusionne au plus une fois par coup, en
    partant du bord vers lequel on glisse.
    """
    action = Action(action)
    moves: list[TileMove] = []
    merged: list[NewTile] = []
    for line in range(BOARD_SIZE):
        cells = line_cells(action, line)
        tiles = [(cell, int(board[cell])) for cell in cells if board[cell] != 0]
        target = 0
        i = 0
        while i < len(tiles):
            cell, exponent = tiles[i]
            end = cells[target]
            if i + 1 < len(tiles) and tiles[i + 1][1] == exponent:
                moves.append(TileMove(exponent, cell, end, merges=True))
                moves.append(TileMove(exponent, tiles[i + 1][0], end, merges=True))
                merged.append(NewTile(end, exponent + 1))
                i += 2
            else:
                moves.append(TileMove(exponent, cell, end, merges=False))
                i += 1
            target += 1
    return tuple(moves), tuple(merged)


def afterstate_of(moves: tuple[TileMove, ...], merged: tuple[NewTile, ...]) -> Board:
    """Reconstruit le plateau juste après le glissement à partir des trajets."""
    board = new_board()
    for move in moves:
        if not move.merges:
            board[move.end] = move.exponent
    for tile in merged:
        board[tile.cell] = tile.exponent
    return board


def plan_transition(before: Board, action: Action, after: Board) -> Transition:
    """Décrit un coup : ``before`` est le plateau avant, ``after`` celui après l'apparition.

    Lève ``ValueError`` si ``after`` ne peut pas venir de ``before`` avec ce coup.
    """
    action = Action(action)
    moves, merged = trace_slide(before, action)
    afterstate = afterstate_of(moves, merged)
    spawned: list[NewTile] = []
    for row, col in np.argwhere(after != afterstate):
        cell = (int(row), int(col))
        if afterstate[cell] != 0:
            raise ValueError(f"Le plateau d'arrivée ne correspond pas au coup {action.name}.")
        spawned.append(NewTile(cell, int(after[cell])))
    return Transition(moves, merged, tuple(spawned), after.copy(), action)


def appearance(board: Board) -> Transition:
    """Décrit l'apparition d'un plateau neuf : toutes ses tuiles surgissent."""
    spawned = tuple(
        NewTile((int(row), int(col)), int(board[row, col])) for row, col in np.argwhere(board != 0)
    )
    return Transition(moves=(), merged=(), spawned=spawned, board=board.copy())
