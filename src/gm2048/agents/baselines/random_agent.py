"""Agent aléatoire : il joue un coup légal tiré au hasard.

C'est le « plancher » de référence : toute IA digne de ce nom doit faire mieux
que lui. Il sert aussi à tester le reste du code (évaluation, environnement)
avec un agent très simple.

À toi de jouer : remplace les ``raise NotImplementedError`` par ton code, puis
lance ``pytest tests/agents/test_random_agent.py``. Les tests décrivent
exactement ce qui est attendu : lis-les avant de commencer.
"""

from __future__ import annotations

import numpy as np

from gm2048.engine import Action, ActionMask, Board


class RandomAgent:
    """Choisit au hasard, avec la même probabilité, parmi les coups légaux."""

    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        """Crée l'agent.

        L'agent garde son propre ``numpy.random.Generator``, créé à partir de
        ``seed`` : deux agents créés avec la même graine doivent jouer
        exactement les mêmes coups.
        """

        self.rng = np.random.default_rng(seed)

    def act(self, board: Board, legal_mask: ActionMask) -> Action:
        """Renvoie un coup légal tiré au hasard.

        ``legal_mask`` est un tableau NumPy de 4 booléens, indexé par
        :class:`Action` : ``legal_mask[Action.LEFT]`` vaut ``True`` si glisser
        vers la gauche modifie le plateau. Cet agent n'a pas besoin de regarder
        ``board`` : le paramètre est là pour respecter le contrat ``Agent``.

        Lève ``ValueError`` si aucun coup n'est légal.
        """

        idx_legal_moves = [i for i, x in enumerate(legal_mask) if x]
        if len(idx_legal_moves) == 0:
            raise ValueError("Aucun coup légal disponible")

        move = self.rng.choice(idx_legal_moves)

        return Action(int(move))
