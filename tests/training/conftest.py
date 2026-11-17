"""Outils communs aux tests de l'entraînement.

Les tests des courbes ne doivent jamais ouvrir de vraie fenêtre (la CI n'a pas
d'écran) : le moteur « Agg » de matplotlib dessine en mémoire.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
