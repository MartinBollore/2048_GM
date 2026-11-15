"""L'apprentissage du réseau n-tuple : jouer des parties et corriger les poids."""

from gm2048.training.td import Move, best_move, play_and_learn, td_update

__all__ = ["Move", "best_move", "play_and_learn", "td_update"]
