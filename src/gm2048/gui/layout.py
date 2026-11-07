"""Géométrie de la fenêtre : où placer le titre, les scores, le plateau, l'aide.

Tout est dessiné sur une maquette de 500 x 720 points, agrandie ou réduite
pour remplir la fenêtre sans la déformer. Redimensionner la fenêtre recalcule
simplement la mise en page.
"""

from __future__ import annotations

from dataclasses import dataclass

from gm2048.engine import BOARD_SIZE

Rect = tuple[int, int, int, int]
"""Un rectangle : (x, y, largeur, hauteur) en pixels."""

DESIGN_WIDTH = 500
DESIGN_HEIGHT = 720

_MARGIN = 20
_HEADER_TOP = 44
"""Haut des cartes de score : la place au-dessus sert aux points qui s'envolent."""
_BOARD_TOP = 152
_BOARD_SIZE = DESIGN_WIDTH - 2 * _MARGIN
_GAP_RATIO = 0.03
"""Espace entre deux cases, en proportion du côté du plateau."""


@dataclass(frozen=True, slots=True)
class Layout:
    """Positions des éléments, en pixels, pour une taille de fenêtre donnée."""

    scale: float
    """Facteur entre la maquette et la fenêtre."""
    title: Rect
    score_card: Rect
    best_card: Rect
    status: Rect
    board: Rect
    footer: Rect
    gap: float
    """Espace entre deux cases."""
    cell: float
    """Côté d'une case."""

    def px(self, design_units: float) -> int:
        """Convertit une longueur de la maquette en pixels (au moins 1)."""
        return max(1, round(design_units * self.scale))

    def cell_origin(self, row: float, col: float) -> tuple[float, float]:
        """Coin haut gauche de la case ``(row, col)`` ; accepte des positions entre deux cases."""
        x, y, _, _ = self.board
        step = self.cell + self.gap
        return x + self.gap + col * step, y + self.gap + row * step

    def tile_rect(self, row: float, col: float, scale: float = 1.0) -> Rect:
        """Rectangle d'une tuile, agrandie ou réduite autour du centre de sa case."""
        x, y = self.cell_origin(row, col)
        size = self.cell * scale
        offset = (self.cell - size) / 2
        return round(x + offset), round(y + offset), round(size), round(size)


def compute_layout(width: int, height: int) -> Layout:
    """Mise en page pour une fenêtre de ``width`` x ``height`` pixels."""
    scale = max(0.05, min(width / DESIGN_WIDTH, height / DESIGN_HEIGHT))
    left = (width - DESIGN_WIDTH * scale) / 2
    top = (height - DESIGN_HEIGHT * scale) / 2

    def rect(x: float, y: float, w: float, h: float) -> Rect:
        return (
            round(left + x * scale),
            round(top + y * scale),
            round(w * scale),
            round(h * scale),
        )

    card_width, card_height, card_gap = 104, 60, 10
    right = DESIGN_WIDTH - _MARGIN
    board = rect(_MARGIN, _BOARD_TOP, _BOARD_SIZE, _BOARD_SIZE)
    side = board[2]
    gap = side * _GAP_RATIO
    return Layout(
        scale=scale,
        title=rect(_MARGIN, _HEADER_TOP, 240, card_height),
        score_card=rect(right - 2 * card_width - card_gap, _HEADER_TOP, card_width, card_height),
        best_card=rect(right - card_width, _HEADER_TOP, card_width, card_height),
        status=rect(_MARGIN, _BOARD_TOP - 36, _BOARD_SIZE, 30),
        board=(board[0], board[1], side, side),
        footer=rect(_MARGIN, _BOARD_TOP + _BOARD_SIZE + 16, _BOARD_SIZE, 64),
        gap=gap,
        cell=(side - (BOARD_SIZE + 1) * gap) / BOARD_SIZE,
    )
