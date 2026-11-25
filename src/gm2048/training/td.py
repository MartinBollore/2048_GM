"""Apprendre par différence temporelle, TD(0) (phase 3, étape 4).

L'agent joue comme le glouton de la phase 2, mais avec une note apprise. Pour chaque
coup légal, il calcule les points gagnés ``r`` et l'afterstate ``s'`` (le plateau
juste après le glissement, avant l'apparition de la tuile), puis joue le coup qui
maximise ``r + V(s')``, où ``V`` est la note donnée par le réseau.

Au coup suivant, il découvre ce que valait vraiment ``s'`` : les points du coup
d'après, plus la note de l'afterstate d'après. L'écart entre les deux est l'erreur
TD, ``delta``, et le réseau corrige un peu la note de ``s'`` dans sa direction :

    delta = r_suivant + V(s'_suivant) - V(s')
    V(s') <- V(s') + alpha * delta

Quand la partie se termine, il n'y a plus de coup d'après : la cible vaut 0,
puisqu'une partie finie ne rapportera plus aucun point.

Tout se joue sur des bitboards, avec ``slide_bits`` et ``spawn_bits`` : le moteur
NumPy serait trop lent pour des milliers de parties.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from gm2048.agents.ntuple.network import NTupleNetwork
from gm2048.engine import max_tile
from gm2048.engine.bitboard import Bitboard, from_bitboard, slide_bits, spawn_bits
from gm2048.engine.board import ACTIONS, Action
from gm2048.evaluation import GameResult


@dataclass(frozen=True, slots=True)
class Move:
    """Un coup envisagé : l'action, l'afterstate obtenue et les points gagnés."""

    action: Action
    """Le coup joué."""
    afterstate: Bitboard
    """Le plateau juste après le glissement, avant l'apparition de la tuile."""
    reward: int
    """Les points gagnés par les fusions du coup."""


def best_move(net: NTupleNetwork, bits: Bitboard) -> Move | None:
    """Le coup légal qui maximise ``reward + net.value(afterstate)``.

    Un coup est légal s'il change le plateau : son afterstate est différente de
    ``bits``. Inutile d'appeler ``legal_actions_bits``, qui referait les quatre
    glissements : compare directement l'afterstate au plateau.

    Les coups sont examinés dans l'ordre de ``Action`` (haut, droite, bas, gauche).
    En cas d'égalité, le premier examiné l'emporte : pas de hasard ici, l'apparition
    des tuiles suffit à varier les parties.

    Renvoie ``None`` si aucun coup n'est légal (la partie est finie). Ne modifie pas
    le réseau.
    """

    max_score = -math.inf
    best_action = ACTIONS[0]
    best_afterstate = bits
    best_reward = 0

    nb_illegal_move = 0
    for action in ACTIONS:
        afterstate, reward = slide_bits(bits=bits, action=action)

        if bits != afterstate:
            v_afterstate = net.value(bits=afterstate)

            if reward + v_afterstate > max_score:
                max_score = reward + v_afterstate

                best_action = action
                best_afterstate = afterstate
                best_reward = reward
        else:
            nb_illegal_move += 1

    if nb_illegal_move < len(ACTIONS):
        return Move(action=best_action, afterstate=best_afterstate, reward=best_reward)
    else:
        return None


def td_update(
    net: NTupleNetwork, afterstate: Bitboard, next_move: Move | None, alpha: float
) -> float:
    """Corrige la note de ``afterstate`` et renvoie l'erreur TD ``delta``.

    ``afterstate`` est l'afterstate du coup précédent, et ``next_move`` le coup
    choisi juste après, depuis le plateau suivant (``None`` si la partie est finie).

    - la cible vaut ``next_move.reward + net.value(next_move.afterstate)``, ou 0 si
      ``next_move`` est ``None`` ;
    - ``delta = cible - net.value(afterstate)`` ;
    - la note de ``afterstate`` est corrigée avec ``net.update(afterstate, alpha * delta)``.

    Calcule bien la cible et ``delta`` avant la correction. Renvoie ``delta``, avant
    multiplication par ``alpha``.
    """

    cible = next_move.reward + net.value(bits=next_move.afterstate) if next_move is not None else 0

    delta = cible - net.value(bits=afterstate)
    net.update(bits=afterstate, delta=alpha * delta)

    return delta


def play_and_learn(net: NTupleNetwork, alpha: float, seed: int) -> GameResult:
    """Joue une partie complète avec ``best_move`` en apprenant après chaque coup.

    La partie commence comme ``Game(seed)`` : un générateur
    ``np.random.default_rng(seed)``, puis deux tuiles posées avec ``spawn_bits`` sur
    le plateau vide (le bitboard 0). Avec ``spawn_bits``, qui fait les mêmes tirages
    que le moteur NumPy, la partie est donc la même que dans ``Game``, coup pour coup.

    À chaque tour :

    1. choisis ``move = best_move(net, bits)`` ;
    2. s'il y a une afterstate précédente, corrige-la avec
       ``td_update(net, précédente, move, alpha)``. En fin de partie, ``move`` vaut
       ``None`` : c'est la mise à jour avec une cible de 0 ;
    3. si ``move`` est ``None``, la partie est finie ;
    4. sinon, ajoute ses points au score, compte le coup, retiens son afterstate,
       puis fais apparaître une tuile dessus pour obtenir le plateau suivant.

    Chaque afterstate jouée est ainsi corrigée exactement une fois. Avec
    ``alpha = 0``, les poids ne changent pas.

    Renvoie un :class:`GameResult` : la graine, le score, la valeur de la plus grande
    tuile (par exemple 2048, pas l'exposant 11), le nombre de coups, et la durée en
    secondes mesurée avec ``time.perf_counter()``.
    """

    rng = np.random.default_rng(seed)
    score = 0
    nb_moves = 0

    start = time.perf_counter()
    bits = 0
    bits = spawn_bits(bits=bits, rng=rng)
    bits = spawn_bits(bits=bits, rng=rng)

    move = best_move(net=net, bits=bits)
    while move is not None:
        bits = spawn_bits(bits=move.afterstate, rng=rng)
        score += move.reward
        nb_moves += 1

        next_move = best_move(net=net, bits=bits)
        td_update(net=net, afterstate=move.afterstate, next_move=next_move, alpha=alpha)

        move = next_move
    time_elapse = time.perf_counter() - start

    board = from_bitboard(bits=bits)
    result = GameResult(
        seed=seed, score=score, max_tile=max_tile(board=board), moves=nb_moves, seconds=time_elapse
    )

    return result
