"""Agent heuristique : il note chaque plateau comme le ferait un bon joueur.

Le glouton ne regardait que les points gagnés tout de suite. L'heuristique
regarde aussi à quoi ressemble le plateau après le coup (l'afterstate), avec
quatre critères qu'utilisent les bons joueurs :

- **cases vides** : plus il reste de place, moins on risque d'être bloqué ;
- **monotonie** : des lignes et des colonnes rangées dans l'ordre permettent
  d'enchaîner les fusions en cascade ;
- **régularité** : deux tuiles voisines proches finiront par fusionner ;
- **coin** : la plus grosse tuile est à l'abri dans un coin.

Chaque critère est une fonction qui renvoie un nombre : plus il est grand,
meilleur est le plateau. Les poids (:class:`HeuristicWeights`) disent combien
compte chaque critère. Pour choisir son coup, l'agent calcule, pour chaque coup
légal : points gagnés + note de l'afterstate, et joue le meilleur.

Tous les critères travaillent sur les **exposants** du plateau (1 pour la tuile
2, 2 pour la tuile 4, …), pas sur les valeurs : un écart d'un cran entre 1024 et
2048 compte autant qu'entre 2 et 4.

Attention : le plateau est un tableau de ``uint8``, des entiers *non signés*.
Avant d'écrire tes soustractions, essaie ``np.uint8(1) - np.uint8(3)`` dans une
console Python.

À toi de jouer : remplace les ``raise NotImplementedError`` par ton code, puis
lance ``pytest tests/agents/test_heuristic.py``. Ordre conseillé : les quatre
critères, puis ``evaluate_board``, puis l'agent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gm2048.engine import Action, ActionMask, Board, empty_cells, slide

# --- Les quatre critères --------------------------------------------------------


def empty_score(board: Board) -> float:
    """Nombre de cases vides du plateau.

    Exemples : 16.0 pour un plateau vide, 0.0 pour un plateau plein.
    """

    id_empty_cells = empty_cells(board)
    return float(len(id_empty_cells))


def monotonicity_score(board: Board) -> float:
    """Pénalité pour les lignes et colonnes qui ne sont pas rangées dans l'ordre.

    Pour chaque ligne et chaque colonne (8 en tout), on lit les 4 exposants dans
    l'ordre et on regarde les écarts entre cases voisines, cases vides comprises
    (une case vide vaut 0) :

    - ``montées`` = somme des écarts positifs (quand l'exposant augmente) ;
    - ``descentes`` = somme des écarts négatifs, en valeur absolue.

    Une ligne rangée dans un sens a 0 montée ou 0 descente. La pénalité de la
    ligne est donc ``min(montées, descentes)`` : le sens n'a pas d'importance,
    seul compte le désordre.

    Le score est **moins** la somme des 8 pénalités : 0.0 pour un plateau
    parfaitement rangé, négatif sinon.

    Exemple : la ligne de tuiles ``2 8 4 _`` a pour exposants ``1 3 2 0``. Les
    écarts sont +2, -1, -2 : 2 de montée, 3 de descente, pénalité 2.
    """

    board_size = board.shape[0]
    penality = 0

    values = board.astype(np.int64)
    for i in range(0, 2 * board_size):
        line_col = values[i, :] if i < board_size else values[:, i - board_size]

        diff = []
        for j in range(0, board_size - 1):
            diff.append(line_col[j + 1] - line_col[j])

        up = sum(elt for elt in diff if elt > 0)
        down = abs(sum(elt for elt in diff if elt < 0))
        penality = penality + min(up, down)

    return -penality


def smoothness_score(board: Board) -> float:
    """Pénalité pour les écarts entre tuiles voisines.

    On additionne, pour chaque paire de cases voisines (horizontalement ou
    verticalement) **toutes les deux non vides**, l'écart d'exposant en valeur
    absolue. Une case vide n'est voisine de rien.

    Le score est **moins** cette somme : 0.0 si toutes les voisines sont
    identiques (ou séparées par des cases vides), négatif sinon.

    Exemple : ``2 8 _ 4`` sur une ligne donne exposants ``1 3 0 2``. Seule la
    paire (2, 8) compte : écart |1 - 3| = 2, score -2.0.
    """

    board_size = board.shape[0]
    penality = 0

    values = board.astype(np.int64)
    for i in range(0, 2 * board_size):
        line_col = values[i, :] if i < board_size else values[:, i - board_size]

        for j in range(0, board_size - 1):
            elt1 = line_col[j]
            elt2 = line_col[j + 1]
            if elt1 != 0 and elt2 != 0:
                penality = penality + abs(elt2 - elt1)

    return -penality


def corner_score(board: Board) -> float:
    """Bonus si la plus grosse tuile est dans un coin.

    Renvoie l'exposant de la plus grosse tuile si au moins une case de coin la
    contient, 0.0 sinon (et 0.0 pour un plateau vide). Exemple : une tuile 64
    dans un coin rapporte 6.0.
    """

    max_t = board.max()
    bonus = max_t != 0 and (
        board[0, 0] == max_t
        or board[0, -1] == max_t
        or board[-1, 0] == max_t
        or board[-1, -1] == max_t
    )

    return float(max_t) if bonus else 0.0


# --- Les poids et la note du plateau ---------------------------------------------


@dataclass(frozen=True, slots=True)
class HeuristicWeights:
    """Importance de chaque critère dans la note du plateau."""

    empty: float
    monotonicity: float
    smoothness: float
    corner: float


DEFAULT_WEIGHTS = HeuristicWeights(empty=1.0, monotonicity=1.0, smoothness=1.0, corner=1.0)
"""Poids de départ, tous égaux. À toi de les régler par l'expérience."""


def evaluate_board(board: Board, weights: HeuristicWeights) -> float:
    """Note du plateau : somme des quatre critères, chacun multiplié par son poids."""

    evaluation = (
        empty_score(board) * weights.empty
        + monotonicity_score(board) * weights.monotonicity
        + smoothness_score(board) * weights.smoothness
        + corner_score(board) * weights.corner
    )

    return evaluation


# --- L'agent ---------------------------------------------------------------------


class HeuristicAgent:
    """Joue le coup qui maximise : points gagnés + note de l'afterstate."""

    name = "heuristic"

    def __init__(self, seed: int | None = None, weights: HeuristicWeights | None = None) -> None:
        """Crée l'agent.

        ``seed`` sert, comme pour le glouton, à départager les égalités : la
        graine vient en premier pour que ``gm2048 eval`` puisse créer l'agent
        avec ``HeuristicAgent(seed)``, comme les autres.

        ``weights`` est gardé dans l'attribut ``self.weights`` ; s'il vaut
        ``None``, l'agent prend :data:`DEFAULT_WEIGHTS`.
        """

        self.rng = np.random.default_rng(seed)
        if weights is not None:
            self.weights = weights
        else:
            self.weights = DEFAULT_WEIGHTS

    def act(self, board: Board, legal_mask: ActionMask) -> Action:
        """Renvoie le coup légal à la meilleure valeur : points gagnés + note de l'afterstate.

        Pour chaque coup légal, ``slide(board, action)`` donne les points gagnés
        (``reward``) et le plateau juste après le coup (``afterstate``), avant
        l'apparition de la nouvelle tuile. La valeur du coup est
        ``reward + evaluate_board(afterstate, self.weights)``.

        En cas d'égalité, tire au hasard parmi les meilleurs coups. Lève
        ``ValueError`` si aucun coup n'est légal.
        """

        idx_legal_moves = [i for i, x in enumerate(legal_mask) if x]
        if len(idx_legal_moves) == 0:
            raise ValueError("Aucun coup légal disponible")

        list_rewards = []
        for idx in idx_legal_moves:
            slide_result = slide(board=board, action=Action(int(idx)))
            additional_reward = evaluate_board(board=slide_result.afterstate, weights=self.weights)
            list_rewards.append(slide_result.reward + additional_reward)

        max_reward = max(list_rewards)
        idx_max = [i for i, r in enumerate(list_rewards) if r == max_reward]

        move = self.rng.choice([idx_legal_moves[i] for i in idx_max])

        return Action(int(move))
