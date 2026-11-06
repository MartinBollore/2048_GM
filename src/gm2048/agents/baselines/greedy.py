"""Agent glouton : il joue le coup légal qui rapporte le plus de points tout de suite.

C'est la première IA qui « réfléchit » : pour chaque coup légal, elle simule le
glissement dans sa tête, sans toucher au vrai plateau, et regarde combien de
points ce coup rapporterait. Elle ne voit qu'un coup à l'avance, et ignore la
tuile qui apparaîtra ensuite.

À toi de jouer : remplace les ``raise NotImplementedError`` par ton code, puis
lance ``pytest tests/agents/test_greedy.py``. Les tests décrivent exactement ce
qui est attendu : lis-les avant de commencer.
"""

from __future__ import annotations

import numpy as np

from gm2048.engine import Action, ActionMask, Board, slide


class GreedyAgent:
    """Joue le coup légal au meilleur gain immédiat, et tire au sort en cas d'égalité."""

    name = "greedy"

    def __init__(self, seed: int | None = None) -> None:
        """Crée l'agent.

        Le glouton n'a besoin du hasard que pour départager les coups qui
        rapportent autant de points. Comme :class:`RandomAgent`, il garde son
        propre ``numpy.random.Generator``, créé à partir de ``seed`` : deux
        agents créés avec la même graine doivent jouer exactement les mêmes
        coups.
        """

        self.rng = np.random.default_rng(seed)

    def act(self, board: Board, legal_mask: ActionMask) -> Action:
        """Renvoie le coup légal qui rapporte le plus de points immédiatement.

        Pour simuler un coup, utilise ``slide(board, action)`` du moteur
        (``from gm2048.engine import slide``). Il renvoie un ``SlideResult``
        dont l'attribut ``reward`` est la somme des points gagnés par toutes les
        fusions du coup. ``slide`` ne modifie pas ``board`` : il calcule un
        nouveau plateau, l'afterstate.

        Seuls les coups autorisés par ``legal_mask`` comptent. Si plusieurs
        coups font le meilleur gain (par exemple quand aucun coup ne fusionne
        et que tous valent 0 point), l'agent tire l'un d'eux au hasard, avec la
        même probabilité pour chacun.

        Lève ``ValueError`` si aucun coup n'est légal.
        """

        idx_legal_moves = [i for i, x in enumerate(legal_mask) if x]
        if len(idx_legal_moves) == 0:
            raise ValueError("Aucun coup légal disponible")

        list_rewards = []
        for idx in idx_legal_moves:
            slide_result = slide(board=board, action=Action(int(idx)))
            list_rewards.append(slide_result.reward)

        max_reward = max(list_rewards)
        idx_max = [i for i, r in enumerate(list_rewards) if r == max_reward]

        move = self.rng.choice([idx_legal_moves[i] for i in idx_max])

        return Action(int(move))
