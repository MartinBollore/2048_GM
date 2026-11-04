"""Moteur de jeu 2048 : règles pures, sans aucune dépendance à l'IA."""

from gm2048.engine.board import (
    ACTIONS,
    BOARD_SIZE,
    SPAWN_FOUR_PROBABILITY,
    Action,
    ActionMask,
    Board,
    SlideResult,
    empty_cells,
    from_values,
    is_game_over,
    legal_actions,
    max_tile,
    new_board,
    slide,
    spawn,
    to_values,
)
from gm2048.engine.game import Game, StepResult
from gm2048.engine.render import render

__all__ = [
    "ACTIONS",
    "BOARD_SIZE",
    "SPAWN_FOUR_PROBABILITY",
    "Action",
    "ActionMask",
    "Board",
    "Game",
    "SlideResult",
    "StepResult",
    "empty_cells",
    "from_values",
    "is_game_over",
    "legal_actions",
    "max_tile",
    "new_board",
    "render",
    "slide",
    "spawn",
    "to_values",
]
