"""L'apprentissage TD compilé avec Numba (phase 3, étape Numba).

Ce module refait, en code compilé, ce que ``td.py`` fait en Python : jouer une
partie avec le réseau et corriger les poids après chaque coup. Le but est d'aller
beaucoup plus vite, sans changer un seul résultat : avec la même graine, la partie
jouée et les poids obtenus doivent être exactement ceux de ``play_and_learn``.
C'est ce que vérifient les tests, comme le bitboard avait été vérifié contre NumPy.

Ce qu'il faut savoir sur Numba :

- ``@njit`` compile la fonction en code machine au premier appel (une ou deux
  secondes), puis la réutilise. ``cache=True`` garde le code compilé sur le disque
  pour les lancements suivants.
- Dans une fonction compilée, il n'y a plus d'objets Python : pas de dataclass, pas
  de ``Action``, pas de ``NTupleNetwork``. Seulement des nombres, des tableaux NumPy
  et des tuples. Une fonction compilée peut appeler une autre fonction compilée.
- **Un bitboard est un** ``np.uint64`` (entier non signé de 64 bits). Piège : si tu
  combines un ``np.uint64`` et un entier ordinaire avec ``|``, ``&`` ou ``+``, Numba
  donne un entier *signé* de 64 bits. Un plateau dont le bit de poids fort est à 1
  (un exposant de 8 ou plus dans la case 15) devient alors négatif, et ne vaut plus
  le bitboard Python. Écris donc les masques et les valeurs en ``np.uint64`` :
  utilise les constantes ci-dessous, ``np.uint64(0)`` pour partir de zéro, et
  ``np.uint64(4 * i)`` pour un décalage calculé.
- Le générateur ``np.random.Generator`` fonctionne dans Numba, avec
  ``rng.integers(0, n)`` et ``rng.random()``. ``rng.choice(cases)`` de
  ``spawn_bits`` fait exactement le même tirage que ``cases[rng.integers(0, n)]`` :
  les deux versions font donc apparaître les mêmes tuiles.
- Le réseau entre dans les fonctions compilées sous forme de deux tuples de
  tableaux : ``weights`` (les tables de poids, les mêmes tableaux que
  ``net.weights``) et ``cells`` (pour chaque tuple, un tableau de forme
  ``(8, len(tuple))`` : les cases lues pour chaque symétrie). Voir
  :func:`network_arrays`.

Les actions sont des entiers, dans l'ordre de ``Action`` : 0 haut, 1 droite, 2 bas,
3 gauche.
"""

from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
from numba import njit

from gm2048.agents.ntuple.network import NTupleNetwork, symmetries
from gm2048.engine.bitboard import (
    CELL_BITS,
    ROW_ACTION_LEFT,
    ROW_ACTION_RIGHT,
    ROW_BITS,
    ROW_REWARD_LEFT,
    ROW_REWARD_RIGHT,
    TRANSPOSE_DIAG_MASK,
    TRANSPOSE_MINUS_3,
    TRANSPOSE_MINUS_6,
    TRANSPOSE_MINUS_9,
    TRANSPOSE_PLUS_3,
    TRANSPOSE_PLUS_6,
    TRANSPOSE_PLUS_9,
    from_bitboard,
)
from gm2048.engine.board import BOARD_SIZE, SPAWN_FOUR_PROBABILITY, max_tile
from gm2048.evaluation import GameResult

# --- Les tables et les constantes, en types NumPy ----------------------------------
#
# Numba ne sait pas lire les listes Python de bitboard.py efficacement : on les
# recopie une fois dans des tableaux NumPy. Une fonction compilée qui lit une de ces
# variables globales la fige comme une constante.

ROW_LEFT = np.array(ROW_ACTION_LEFT, dtype=np.uint64)
"""``ROW_LEFT[row]`` : la ligne ``row`` après un glissement à gauche."""
ROW_RIGHT = np.array(ROW_ACTION_RIGHT, dtype=np.uint64)
"""``ROW_RIGHT[row]`` : la ligne ``row`` après un glissement à droite."""
REWARD_LEFT = np.array(ROW_REWARD_LEFT, dtype=np.int64)
"""Les points gagnés par un glissement à gauche."""
REWARD_RIGHT = np.array(ROW_REWARD_RIGHT, dtype=np.int64)
"""Les points gagnés par un glissement à droite."""

CELL_MASK = np.uint64(0xF)
ROW_MASK = np.uint64(0xFFFF)
DIAG = np.uint64(TRANSPOSE_DIAG_MASK)
PLUS_3, MINUS_3 = np.uint64(TRANSPOSE_PLUS_3), np.uint64(TRANSPOSE_MINUS_3)
PLUS_6, MINUS_6 = np.uint64(TRANSPOSE_PLUS_6), np.uint64(TRANSPOSE_MINUS_6)
PLUS_9, MINUS_9 = np.uint64(TRANSPOSE_PLUS_9), np.uint64(TRANSPOSE_MINUS_9)
SHIFT_12, SHIFT_24, SHIFT_36 = (
    np.uint64(3 * CELL_BITS),
    np.uint64(6 * CELL_BITS),
    np.uint64(9 * CELL_BITS),
)


# --- Le moteur ----------------------------------------------------------------------


@njit(cache=True)
def transpose_fast(bits):
    """``transpose`` de bitboard.py, sur un ``np.uint64``. Renvoie un ``np.uint64``."""

    bits = (
        bits & DIAG
        | ((bits & PLUS_3) << SHIFT_12)
        | ((bits & MINUS_3) >> SHIFT_12)
        | ((bits & PLUS_6) << SHIFT_24)
        | ((bits & MINUS_6) >> SHIFT_24)
        | ((bits & PLUS_9) << SHIFT_36)
        | ((bits & MINUS_9) >> SHIFT_36)
    )

    return np.uint64(bits)


@njit(cache=True)
def slide_fast(bits, action):
    """``slide_bits`` de bitboard.py : renvoie ``(afterstate, points)``.

    ``bits`` est un ``np.uint64``, ``action`` un entier de 0 à 3. L'afterstate est un
    ``np.uint64``, les points un entier. Un coup illégal renvoie ``(bits, 0)``.
    """

    if action == 0 or action == 2:
        bits = transpose_fast(bits=bits)

    reward = 0
    bits_slided = np.uint64(0)
    for i in range(BOARD_SIZE):
        k = np.uint64(i * ROW_BITS)
        row = (bits >> k) & np.uint64(ROW_MASK)

        if action == 3 or action == 0:
            row_slided = ROW_LEFT[row]
            reward += REWARD_LEFT[row]
        else:
            row_slided = ROW_RIGHT[row]
            reward += REWARD_RIGHT[row]

        bits_slided = bits_slided | (row_slided << k)

    if action == 0 or action == 2:
        bits_slided = transpose_fast(bits=bits_slided)

    return bits_slided, reward


@njit(cache=True)
def spawn_fast(bits, rng):
    """``spawn_bits`` de bitboard.py : un 2 (90 %) ou un 4 (10 %) sur une case vide.

    Mêmes tirages, dans le même ordre : ``rng.integers(0, n)`` pour choisir parmi les
    ``n`` cases vides (rangées dans l'ordre croissant), puis ``rng.random()`` comparé
    à 0.1 pour décider d'un 4. Lève ``ValueError`` si le plateau est plein.
    """

    empty = np.empty(16, dtype=np.int64)
    c = 0
    for i in range(BOARD_SIZE**2):
        k = np.uint64(i * CELL_BITS)
        exposant = (bits >> k) & np.uint64(CELL_MASK)

        if exposant == 0:
            empty[c] = i
            c += 1

    if c == 0:
        raise ValueError("Aucune case vide : impossible d'ajouter une tuile.")

    cell = empty[rng.integers(0, c)]
    exponent = np.uint64(2) if rng.random() < SPAWN_FOUR_PROBABILITY else np.uint64(1)

    k = np.uint64(cell * CELL_BITS)
    bits = bits & ~(np.uint64(CELL_MASK) << k) | (exponent << k)

    return bits


# --- Le réseau ----------------------------------------------------------------------


def network_arrays(
    net: NTupleNetwork,
) -> tuple[tuple[npt.NDArray[np.float32], ...], tuple[npt.NDArray[np.int64], ...]]:
    """Le réseau sous la forme que comprennent les fonctions compilées : ``(weights, cells)``.

    - ``weights`` est ``tuple(net.weights)`` : les mêmes tableaux, pas des copies.
      Quand une fonction compilée modifie ``weights[t]``, c'est donc bien le réseau
      qui apprend ;
    - ``cells[t]`` est un tableau ``np.int64`` de forme ``(8, len(net.tuples[t]))`` :
      la ligne ``s`` contient les cases lues pour la symétrie ``s``, c'est-à-dire
      ``symmetries()[s][c]`` pour chaque case ``c`` du tuple, dans l'ordre.

    Cette fonction n'est pas compilée : c'est du Python ordinaire.
    """

    weights = tuple(net.weights)

    sym = symmetries()
    _cells = []
    for t in net.tuples:
        cell = []
        for s in sym:
            cell.append(tuple(s[i] for i in t))

        _cells.append(cell)
    cells = tuple(np.array(table, dtype=np.int64) for table in _cells)

    return weights, cells


@njit(cache=True)
def value_fast(weights, cells, bits):
    """``net.value`` : la somme des poids lus, sous forme de nombre à virgule.

    Additionne dans le même ordre que ta version Python (tuple par tuple, symétrie
    par symétrie), en partant de ``0.0`` : le résultat doit être exactement le même.
    """

    v = 0.0
    for table in range(len(cells)):
        cell = cells[table]
        for c in cell:
            index = np.uint64(0)
            for i in range(len(c)):
                exposant = (bits >> np.uint64(c[i] * CELL_BITS)) & CELL_MASK
                index = index | (exposant << np.uint64(i * CELL_BITS))

            v += float(weights[table][index])

    return float(v)


@njit(cache=True)
def update_fast(weights, cells, bits, delta):
    """``net.update`` : chacun des ``m`` poids lus reçoit ``delta / m``.

    Pour des poids identiques à ta version Python, calcule la part une seule fois et
    convertis-la en ``np.float32`` avant de l'ajouter, comme le fait NumPy quand tu
    ajoutes un nombre Python à une case d'un tableau float32.
    """

    m = 0
    for table in range(len(cells)):
        m += len(cells[table])

    d = np.float32(delta / m)
    for table in range(len(cells)):
        cell = cells[table]
        for c in cell:
            index = np.uint64(0)
            for i in range(len(c)):
                exposant = (bits >> np.uint64(c[i] * CELL_BITS)) & CELL_MASK
                index = index | (exposant << np.uint64(i * CELL_BITS))

            weights[table][index] += d


# --- L'apprentissage ----------------------------------------------------------------


@njit(cache=True)
def best_move_fast(weights, cells, bits):
    """``best_move`` : renvoie ``(action, afterstate, points)`` du meilleur coup.

    Mêmes règles qu'en Python : coups examinés de 0 à 3, coup légal si l'afterstate
    diffère de ``bits``, le premier l'emporte en cas d'égalité. Comme ``None``
    n'existe pas ici, l'absence de coup légal renvoie ``(-1, bits, 0)``.
    """

    max_score = 0
    best_action = -1
    best_afterstate = bits
    best_reward = 0.0

    for action in range(4):
        afterstate, reward = slide_fast(bits=bits, action=action)

        if bits != afterstate:
            v_afterstate = value_fast(weights=weights, cells=cells, bits=afterstate)

            if reward + v_afterstate > max_score or best_action == -1:
                max_score = reward + v_afterstate

                best_action = action
                best_afterstate = afterstate
                best_reward = reward

    return best_action, best_afterstate, best_reward


@njit(cache=True)
def play_and_learn_fast(weights, cells, alpha, rng):
    """``play_and_learn`` compilé : renvoie ``(score, nombre de coups, plateau final)``.

    Même déroulé que ta version Python, avec les mêmes calculs dans le même ordre :
    deux tuiles sur le plateau vide, puis à chaque tour le coup choisi, la tuile, et
    la correction TD de l'afterstate précédente. Il n'y a pas de ``td_update`` séparé
    ici : écris ses trois lignes directement dans la boucle (la cible, ``delta``, puis
    ``update_fast(weights, cells, afterstate, alpha * delta)``), avec une cible de 0
    en fin de partie.
    """

    score = 0
    nb_moves = 0

    bits = np.uint64(0)
    bits = spawn_fast(bits=bits, rng=rng)
    bits = spawn_fast(bits=bits, rng=rng)

    action, afterstate, reward = best_move_fast(weights=weights, cells=cells, bits=bits)
    while action != -1:
        bits = spawn_fast(bits=afterstate, rng=rng)
        score += reward
        nb_moves += 1

        next_action, next_afterstate, next_reward = best_move_fast(
            weights=weights, cells=cells, bits=bits
        )
        if next_action != -1:
            cible = next_reward + value_fast(weights=weights, cells=cells, bits=next_afterstate)
        else:
            cible = 0

        delta = cible - value_fast(weights=weights, cells=cells, bits=afterstate)
        update_fast(weights=weights, cells=cells, bits=afterstate, delta=alpha * delta)

        action = next_action
        afterstate = next_afterstate
        reward = next_reward

    return score, nb_moves, bits


def play_and_learn_numba(net: NTupleNetwork, alpha: float, seed: int) -> GameResult:
    """Même chose que ``td.play_and_learn``, avec le même résultat, mais compilé.

    Crée le générateur avec ``np.random.default_rng(seed)``, prépare le réseau avec
    :func:`network_arrays`, appelle :func:`play_and_learn_fast`, puis construit le
    :class:`GameResult` (convertis les nombres renvoyés avec ``int()``). La durée
    est mesurée autour de tout cela avec ``time.perf_counter()``.
    """

    rng = np.random.default_rng(seed)
    weights, cells = network_arrays(net=net)

    start = time.perf_counter()
    score, nb_moves, bits = play_and_learn_fast(weights=weights, cells=cells, alpha=alpha, rng=rng)
    time_elapse = time.perf_counter() - start

    board = from_bitboard(bits=int(bits))
    result = GameResult(
        seed=seed,
        score=int(score),
        max_tile=max_tile(board=board),
        moves=int(nb_moves),
        seconds=time_elapse,
    )

    return result
