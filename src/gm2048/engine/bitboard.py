"""Le plateau de 2048 dans un seul entier de 64 bits (phase 3, étape 1).

Une case contient un exposant de 0 à 15, qui tient sur 4 bits. Les 16 cases du
plateau tiennent donc dans 64 bits, c'est-à-dire dans un seul entier Python.

La case numéro ``i = 4 * ligne + colonne`` occupe les bits ``4 * i`` à
``4 * i + 3`` : la case en haut à gauche (0, 0) est dans les 4 bits de poids
faible, celle en bas à droite (3, 3) dans les 4 bits de poids fort.

Conséquence : la ligne ``r`` occupe les bits ``16 * r`` à ``16 * r + 15``. Une
ligne entière se lit d'un coup, comme un nombre de 0 à 65 535 dont les 4 bits de
poids faible sont la colonne 0. C'est ce qui permettra, à l'étape 2, de calculer
chaque coup avec des tables précalculées.

Exemple, avec les exposants 0 à 15 rangés dans l'ordre des cases :

    plateau                 bitboard
    [[ 0,  1,  2,  3],
     [ 4,  5,  6,  7],      0xFEDCBA9876543210
     [ 8,  9, 10, 11],
     [12, 13, 14, 15]]

Un chiffre hexadécimal vaut 4 bits, donc une case. ``hex()`` écrit les chiffres
de poids fort en premier : il lit le plateau à l'envers, de la case 15 à la case 0.

Attention aux types : une case d'un plateau NumPy est un ``np.uint8``, qui ne
garde que 8 bits. ``board[3, 3] << 60`` vaut donc 0, sans aucune erreur.
Convertis chaque case avec ``int()`` avant de la décaler.
"""

from __future__ import annotations

import numpy as np

from gm2048.engine import Action
from gm2048.engine.board import (
    ACTIONS,
    BOARD_SIZE,
    SPAWN_FOUR_PROBABILITY,
    ActionMask,
    Board,
    new_board,
)

Bitboard = int
"""Plateau codé dans un entier de 64 bits : 16 cases de 4 bits."""

CELL_BITS = 4
"""Nombre de bits par case : un exposant de 0 à 15."""

CELL_MASK = 0xF
"""Masque d'une case : ses 4 bits à 1."""

ROW_BITS = 16
"""Nombre de bits par ligne : 4 cases de 4 bits."""

ROW_MASK = 0xFFFF
"""Masque d'une ligne : ses 16 bits à 1."""

MAX_EXPONENT = 15
"""Plus grand exposant qui tient dans une case (la tuile 32 768)."""

BITBOARD_LIMIT = 1 << 64
"""Un bitboard valide est un entier de 0 à ``BITBOARD_LIMIT - 1``."""


# --- Conversion entre NumPy et bitboard ----------------------------------------


def to_bitboard(board: Board) -> Bitboard:
    """Code un plateau NumPy 4x4 d'exposants dans un entier de 64 bits.

    Renvoie un ``int`` Python (pas un entier NumPy) et ne modifie pas ``board``.

    Lève ``ValueError`` si une case dépasse ``MAX_EXPONENT`` : ses bits
    déborderaient sur la case suivante.

    >>> from gm2048.engine import new_board
    >>> to_bitboard(new_board())
    0
    >>> board = new_board()
    >>> board[0, 1] = 1  # un 2 dans la deuxième case de la première ligne
    >>> hex(to_bitboard(board))
    '0x10'
    """

    if np.max(board) > MAX_EXPONENT:
        raise ValueError("Le plateau ne peut pas comporter d'exposant supérieur à 15")

    bitboard = 0
    for i in range(0, BOARD_SIZE):
        bit_row = 0
        for j in range(0, BOARD_SIZE):
            k = j * CELL_BITS
            bit_row = bit_row | (int(board[i, j]) << k)

        bitboard = set_row(bits=bitboard, row=i, value=bit_row)

    return bitboard


def from_bitboard(bits: Bitboard) -> Board:
    """Décode un bitboard en plateau NumPy 4x4 d'exposants (``np.uint8``).

    C'est l'inverse de :func:`to_bitboard` : ``from_bitboard(to_bitboard(board))``
    redonne ``board``.

    Lève ``ValueError`` si ``bits`` est négatif ou ne tient pas sur 64 bits
    (``bits >= BITBOARD_LIMIT``).
    """

    if bits >= BITBOARD_LIMIT:
        raise ValueError("La valeur d'un bitboard ne peut dépasser ou égaler sa valeur limite")
    elif bits < 0:
        raise ValueError("La valeur d'un bitboard ne peut être négative")

    board = new_board()
    for i in range(0, BOARD_SIZE):
        for j in range(0, BOARD_SIZE):
            exposant = get_cell(bits=bits, row=i, col=j)

            board[i, j] = exposant

    return board


# --- Lecture et écriture --------------------------------------------------------
#
# Ces fonctions serviront à chaque coup, des millions de fois pendant un
# entraînement : elles ne vérifient pas leurs arguments, pour rester rapides.
# C'est à l'appelant de donner une ligne et une colonne de 0 à 3, et une ligne
# de 0 à ``ROW_MASK``.


def get_cell(bits: Bitboard, row: int, col: int) -> int:
    """Exposant de la case (``row``, ``col``), de 0 à 15.

    >>> get_cell(0xFEDCBA9876543210, 1, 2)
    6
    """

    bit_row = get_row(bits=bits, row=row)

    k = col * CELL_BITS
    exposant = (bit_row >> k) & CELL_MASK

    return exposant


def get_row(bits: Bitboard, row: int) -> int:
    """Les 16 bits de la ligne ``row``, comme un nombre de 0 à ``ROW_MASK``.

    Les 4 bits de poids faible du résultat sont la colonne 0.

    >>> hex(get_row(0xFEDCBA9876543210, 1))
    '0x7654'
    """

    k = row * ROW_BITS
    bit_row = (bits >> k) & ROW_MASK

    return bit_row


def set_row(bits: Bitboard, row: int, value: int) -> Bitboard:
    """Renvoie un nouveau bitboard où la ligne ``row`` vaut ``value``.

    Les trois autres lignes ne changent pas. ``value`` est une ligne codée comme
    le résultat de :func:`get_row`, de 0 à ``ROW_MASK``.

    >>> hex(set_row(0, 2, 0x4321))
    '0x432100000000'
    """

    k = row * ROW_BITS
    bits = bits & ~(ROW_MASK << k)
    bits = bits | (value << k)

    return bits


def transpose(bits: Bitboard) -> Bitboard:
    """Échange lignes et colonnes : la case (r, c) passe en (c, r).

    C'est l'équivalent de ``board.T`` pour un bitboard. À l'étape 2, il ramènera
    les coups haut et bas à des coups gauche et droite, qui travaillent ligne par
    ligne. Transposer deux fois redonne le plateau de départ.
    """

    for i in range(0, BOARD_SIZE - 1):
        for j in range(i + 1, BOARD_SIZE):
            n_ij = i * BOARD_SIZE + j
            k_ij = n_ij * CELL_BITS
            exp_ij = (bits >> k_ij) & CELL_MASK

            n_ji = j * BOARD_SIZE + i
            k_ji = n_ji * CELL_BITS
            exp_ji = (bits >> k_ji) & CELL_MASK

            bits = bits & ~(CELL_MASK << k_ij) | (exp_ji << k_ij)
            bits = bits & ~(CELL_MASK << k_ji) | (exp_ij << k_ji)

    return bits


# --- Étape 2 : les coups en tables précalculées ---------------------------------
#
# Une ligne de 4 cases tient sur 16 bits : il n'existe que 65 536 lignes possibles.
# Au chargement du module, on calcule une fois pour toutes, pour chacune, la ligne
# obtenue après un glissement à gauche et les points gagnés. Pendant la partie, un
# coup ne calcule plus aucun glissement : il lit dans ces deux tables.


ROW_COUNT = 1 << ROW_BITS
"""Nombre de lignes possibles : 65 536."""


def reverse_row(row: int) -> int:
    """Inverse l'ordre des 4 cases d'une ligne : la colonne 0 passe en colonne 3, etc.

    Sert à glisser vers la droite : inverser, glisser à gauche, puis inverser de nouveau.

    >>> hex(reverse_row(0x4321))
    '0x1234'
    """

    row_inv = 0
    for i in range(0, BOARD_SIZE):
        k_i = i * CELL_BITS
        exposant = (row >> k_i) & CELL_MASK

        j = BOARD_SIZE - i - 1
        k_j = j * CELL_BITS
        row_inv = row_inv | (exposant << k_j)

    return row_inv


def slide_row_left(row: int) -> tuple[int, int]:
    """Glisse une ligne de 16 bits vers la gauche, c'est-à-dire vers la colonne 0.

    Renvoie ``(nouvelle_ligne, points)``. Les règles sont celles du moteur de la
    phase 1 : les tuiles se tassent vers la colonne 0, deux tuiles égales voisines
    fusionnent, chaque tuile fusionne au plus une fois par coup en partant de la
    colonne 0, et les points sont la somme des valeurs des tuiles créées
    (``1 << exposant``).

    Une seule différence : deux 15 voisins ne fusionnent pas, car l'exposant 16 (la
    tuile 65 536) ne tient pas sur 4 bits. Ce cas n'arrive jamais en pratique.

    Cette fonction n'a pas besoin d'être rapide : elle ne sert qu'à remplir les tables.

    >>> new_row, points = slide_row_left(0x1111)  # [2, 2, 2, 2] devient [4, 4, 0, 0]
    >>> hex(new_row), points
    ('0x22', 8)
    """

    tiles = []
    for i in range(BOARD_SIZE):
        k = i * CELL_BITS
        t = (row >> k) & CELL_MASK
        if t != 0:
            tiles.append(int(t))

    row_board_slided: list[int] = []
    reward = 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1] and tiles[i] != MAX_EXPONENT:
            exponent = tiles[i] + 1
            row_board_slided.append(exponent)
            reward += 1 << exponent
            i += 2
        else:
            row_board_slided.append(tiles[i])
            i += 1
    row_board_slided.extend([0] * (BOARD_SIZE - len(row_board_slided)))

    row_slided = 0
    for i in range(BOARD_SIZE):
        k = i * CELL_BITS
        row_slided = row_slided | (row_board_slided[i] << k)

    return row_slided, reward


def build_row_tables() -> tuple[list[int], list[int]]:
    """Calcule les deux tables, pour les ``ROW_COUNT`` lignes possibles.

    Renvoie ``(row_left, row_reward)`` : ``row_left[row]`` est la ligne ``row``
    après un glissement à gauche, et ``row_reward[row]`` les points gagnés, comme
    le donne :func:`slide_row_left`.
    """

    row_left = []
    row_reward = []

    bit_row_limit = 1 << ROW_BITS
    for row in range(bit_row_limit):
        row_slided, reward = slide_row_left(row=row)

        row_left.append(row_slided)
        row_reward.append(reward)

    return row_left, row_reward


# tables, calculées au chargement du module par un seul appel à build_row_tables.
ROW_LEFT: list[int] = []
"""``ROW_LEFT[row]`` : la ligne ``row`` après un glissement à gauche."""

ROW_REWARD: list[int] = []
"""``ROW_REWARD[row]`` : les points gagnés par ce glissement."""

ROW_LEFT, ROW_REWARD = build_row_tables()


def slide_bits(bits: Bitboard, action: int) -> tuple[Bitboard, int]:
    """Applique un coup sans faire apparaître de tuile : renvoie ``(afterstate, points)``.

    ``action`` suit l'ordre de ``gm2048.engine.Action`` : 0 haut, 1 droite, 2 bas,
    3 gauche. Le résultat est celui de ``engine.slide`` sur le plateau NumPy
    équivalent. Un coup illégal renvoie ``(bits, 0)`` : le plateau ne change pas.

    Aucun glissement n'est calculé ici, seulement des lectures dans ``ROW_LEFT`` et
    ``ROW_REWARD`` : :func:`reverse_row` ramène la droite à la gauche, et
    :func:`transpose` ramène le haut et le bas à la gauche et à la droite.
    """

    bits = orient(bits=bits, action=action)

    reward = 0
    bits_slided = 0
    for i in range(BOARD_SIZE):
        k = i * ROW_BITS
        row = (bits >> k) & ROW_MASK

        row_slided = ROW_LEFT[row]
        reward += ROW_REWARD[row]

        bits_slided = bits_slided | (row_slided << k)

    bits_slided = unorient(bits=bits_slided, action=action)

    return bits_slided, reward


def orient(bits: Bitboard, action: int) -> Bitboard:
    """Réoriente le plateau pour que ``action`` devienne un glissement à gauche."""
    if action == Action.UP or action == Action.DOWN:
        bits = transpose(bits)

    if action == Action.RIGHT or action == Action.DOWN:
        for i in range(BOARD_SIZE):
            k = i * ROW_BITS
            row = reverse_row(row=(bits >> k) & ROW_MASK)
            bits = bits & ~(ROW_MASK << k) | (row << k)

    return bits


def unorient(bits: Bitboard, action: int) -> Bitboard:
    if action == Action.RIGHT or action == Action.DOWN:
        for i in range(BOARD_SIZE):
            k = i * ROW_BITS
            row = reverse_row(row=(bits >> k) & ROW_MASK)
            bits = bits & ~(ROW_MASK << k) | (row << k)

    if action == Action.UP or action == Action.DOWN:
        bits = transpose(bits)

    return bits


def legal_actions_bits(bits: Bitboard) -> ActionMask:
    """Masque des coups qui modifient le plateau, comme ``engine.legal_actions``.

    Un tableau NumPy de 4 booléens, indexé par action.
    """

    action_mask = []
    for action in ACTIONS:
        afterstate, _ = slide_bits(bits=bits, action=action)

        action_mask.append(bits != afterstate)

    return np.array(action_mask, dtype=np.bool_)


def empty_cells_bits(bits: Bitboard) -> list[int]:
    """Numéros des cases vides, de 0 à 15, dans l'ordre croissant.

    Même résultat que ``engine.empty_cells``, sous forme de liste.
    """

    empty_cells = []
    for i in range(BOARD_SIZE**2):
        k = i * CELL_BITS
        exposant = (bits >> k) & CELL_MASK

        if exposant == 0:
            empty_cells.append(i)

    return empty_cells


def spawn_bits(bits: Bitboard, rng: np.random.Generator) -> Bitboard:
    """Ajoute un 2 (90 %) ou un 4 (10 %) sur une case vide tirée au hasard.

    Fait exactement les mêmes tirages que ``engine.spawn``, dans le même ordre :
    ``rng.choice`` sur la liste des cases vides, puis ``rng.random()`` comparé à
    ``SPAWN_FOUR_PROBABILITY`` pour décider d'un 4. Avec la même graine, les deux
    moteurs font donc apparaître la même tuile au même endroit.

    Lève ``ValueError`` si le plateau est plein.
    """

    empty = empty_cells_bits(bits=bits)
    if len(empty) == 0:
        raise ValueError("Aucune case vide : impossible d'ajouter une tuile.")

    cell = int(rng.choice(empty))
    exponent = 2 if rng.random() < SPAWN_FOUR_PROBABILITY else 1

    k = cell * CELL_BITS
    bits = bits & ~(CELL_MASK << k) | (exponent << k)

    return bits
