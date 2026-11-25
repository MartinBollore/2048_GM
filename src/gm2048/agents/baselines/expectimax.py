"""Agent expectimax : il regarde plusieurs coups à l'avance, hasard compris.

L'heuristique notait le plateau juste après son coup. Expectimax va plus loin :
il imagine aussi la tuile qui va apparaître, puis son propre coup suivant, et
ainsi de suite jusqu'à une profondeur donnée. Il alterne deux sortes de nœuds :

- les nœuds **max** (ton tour) : on essaie chaque coup légal et on garde le
  meilleur ;
- les nœuds **hasard** : on fait la moyenne sur toutes les tuiles qui peuvent
  apparaître, pondérée par leur probabilité (un 2 dans 90 % des cas, un 4 dans
  10 %, sur une case vide tirée au hasard).

Les formules (``s`` est un plateau, ``s'`` l'afterstate du coup ``a``, ``r`` les
points du coup, ``n`` le nombre de cases vides de ``s'``) :

    V_max(s, d)     = max sur les coups légaux a de [ r(s, a) + V_hasard(s', d) ]
    V_hasard(s', d) = 1/n * somme sur les cases vides c de
                      [ 0.9 * V_max(s' + un 2 en c, d - 1) + 0.1 * V_max(s' + un 4 en c, d - 1) ]
    V_max(s, 0)     = note heuristique du plateau (evaluate_board)

Une partie perdue (plus aucun coup légal) vaut :data:`GAME_OVER_VALUE`, à
n'importe quelle profondeur, y compris 0 : c'est le pire qui puisse arriver.

``depth`` compte les nœuds hasard traversés. Avec ``depth=1``, l'agent essaie
chaque coup, imagine chaque tuile possible, puis note le plateau obtenu. Avec
``depth=2``, il joue encore un coup et imagine encore une tuile avant de noter.

Le nombre de plateaux à examiner explose avec la profondeur. Le cache (un
``dict``) retient la valeur des plateaux déjà calculés, car plusieurs chemins
mènent souvent au même plateau.

À toi de jouer : remplace les ``raise NotImplementedError`` par ton code, puis
lance ``pytest tests/agents/test_expectimax.py``. Ordre conseillé :
``__init__``, ``_chance_value`` et ``_max_value`` ensemble (ils s'appellent
l'un l'autre), puis ``act``.
"""

from __future__ import annotations

import numpy as np

from gm2048.agents.baselines.heuristic import DEFAULT_WEIGHTS, HeuristicWeights, evaluate_board
from gm2048.engine import (
    SPAWN_FOUR_PROBABILITY,
    Action,
    ActionMask,
    Board,
    empty_cells,
    is_game_over,
    legal_actions,
    slide,
)

GAME_OVER_VALUE = -1_000_000.0
"""Valeur d'une partie perdue : bien pire que n'importe quelle note de plateau."""


class ExpectimaxAgent:
    """Joue le coup qui maximise la valeur espérée, calculée ``depth`` tuiles à l'avance."""

    name = "expectimax"

    def __init__(
        self,
        seed: int | None = None,
        depth: int = 2,
        weights: HeuristicWeights | None = None,
    ) -> None:
        """Crée l'agent.

        - ``seed`` départage les égalités, comme pour les agents précédents.
        - ``depth`` (au moins 1, sinon ``ValueError``) est gardé dans ``self.depth``.
        - ``weights`` est gardé dans ``self.weights`` ; s'il vaut ``None``,
          l'agent prend ``DEFAULT_WEIGHTS`` du module ``heuristic``.
        - ``self.cache`` est un ``dict`` vide au départ. Ses clés sont des
          couples ``(board.tobytes(), depth)``, ses valeurs des ``float``.
        """

        self.rng = np.random.default_rng(seed)

        if depth < 1:
            raise ValueError("La profondeur d'Expectimax doit être d'au moins 1")
        self.depth = depth

        if weights is None:
            weights = DEFAULT_WEIGHTS
        self.weights = weights

        self.cache: dict[tuple[bytes, int], float] = {}

    def act(self, board: Board, legal_mask: ActionMask) -> Action:
        """Renvoie le coup légal à la meilleure valeur : ``r + V_hasard(s', self.depth)``.

        Vide le cache au début de chaque appel : il ne sert qu'à l'intérieur d'une
        même recherche. Seuls les coups autorisés par ``legal_mask`` comptent. En
        cas d'égalité, tire au hasard parmi les meilleurs coups. Lève
        ``ValueError`` si aucun coup n'est légal.
        """

        self.cache = {}

        idx_legal_moves = [i for i, x in enumerate(legal_mask) if x]
        if len(idx_legal_moves) == 0:
            raise ValueError("Aucun coup légal disponible")

        list_rewards = []
        for idx in idx_legal_moves:
            slide_result = slide(board=board, action=Action(int(idx)))

            points = slide_result.reward
            afterstate = slide_result.afterstate
            v_random = self._chance_value(afterstate=afterstate, depth=self.depth)

            list_rewards.append(points + v_random)

        max_reward = max(list_rewards)
        idx_max = [i for i, r in enumerate(list_rewards) if r == max_reward]

        move = self.rng.choice([idx_legal_moves[i] for i in idx_max])

        return Action(int(move))

    def _max_value(self, board: Board, depth: int) -> float:
        """Valeur ``V_max`` d'un plateau où c'est à toi de jouer.

        Dans l'ordre :

        1. si ``(board.tobytes(), depth)`` est dans le cache, renvoie la valeur rangée ;
        2. si plus aucun coup n'est légal, la valeur est ``GAME_OVER_VALUE`` ;
        3. sinon, si ``depth`` vaut 0, la valeur est ``evaluate_board(board, self.weights)`` ;
        4. sinon, la valeur est le maximum, sur les coups légaux, de
           ``r + self._chance_value(afterstate, depth)``.

        Range la valeur dans le cache avant de la renvoyer.
        """

        key = (board.tobytes(), depth)
        if key in self.cache:
            v = self.cache[key]
        elif len(empty_cells(board)) == 0 and is_game_over(board):
            v = GAME_OVER_VALUE
        elif depth == 0:
            v = evaluate_board(board=board, weights=self.weights)
        else:
            idx_legal_moves = [i for i, x in enumerate(legal_actions(board)) if x]
            if len(idx_legal_moves) == 0:
                raise ValueError("Aucun coup légal disponible")

            v = float("-inf")
            for idx in idx_legal_moves:
                slide_result = slide(board=board, action=Action(int(idx)))

                points = slide_result.reward
                afterstate = slide_result.afterstate
                v_random = self._chance_value(afterstate=afterstate, depth=depth)

                reward = points + v_random
                if reward > v:
                    v = reward

        self.cache[key] = v
        return v

    def _chance_value(self, afterstate: Board, depth: int) -> float:
        """Valeur ``V_hasard`` d'un afterstate : la moyenne sur toutes les tuiles possibles.

        Pour chaque case vide, on pose un 2 (exposant 1), puis un 4 (exposant 2),
        et on calcule ``_max_value(..., depth - 1)`` du plateau obtenu. Les deux
        valeurs sont pondérées par 0,9 et 0,1 (le moteur fournit
        ``SPAWN_FOUR_PROBABILITY``), puis on fait la moyenne sur les cases
        vides. L'afterstate d'un coup légal a toujours au moins une case vide.

        Attention : ne modifie pas ``afterstate`` lui-même, travaille sur une copie.
        """

        empty = empty_cells(afterstate)
        n = len(empty)

        v = 0.0
        for cell in empty:
            board_copy1 = afterstate.copy()

            board_copy1.flat[cell] = 1
            v_max1 = self._max_value(board=board_copy1, depth=depth - 1)

            board_copy2 = afterstate.copy()

            board_copy2.flat[cell] = 2
            v_max2 = self._max_value(board=board_copy2, depth=depth - 1)

            v = v + ((1 - SPAWN_FOUR_PROBABILITY) * v_max1 + SPAWN_FOUR_PROBABILITY * v_max2) / n

        return v
