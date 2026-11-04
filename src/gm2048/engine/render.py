"""Affichage texte d'un plateau."""

from __future__ import annotations

from gm2048.engine.board import BOARD_SIZE, Board, max_tile, to_values

_MIN_CELL_WIDTH = 4


def render(board: Board, score: int | None = None, moves: int | None = None) -> str:
    """Dessine le plateau en texte, avec le score et le nombre de coups si fournis.

    Exemple (sans score) ::

        +------+------+------+------+
        |    2 |    . |    . |    . |
        ...
    """
    values = to_values(board)
    width = max(_MIN_CELL_WIDTH, len(str(int(values.max()))))
    separator = "+" + "+".join(["-" * (width + 2)] * BOARD_SIZE) + "+"

    lines: list[str] = []
    header = []
    if score is not None:
        header.append(f"Score : {score}")
    if moves is not None:
        header.append(f"Coups : {moves}")
    if header:
        header.append(f"Meilleure tuile : {max_tile(board)}")
        lines.append("   ".join(header))

    lines.append(separator)
    for row in values:
        cells = [f" {(str(int(v)) if v else '.'):>{width}} " for v in row]
        lines.append("|" + "|".join(cells) + "|")
        lines.append(separator)
    return "\n".join(lines)
