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

# BOARD_SIZE, à importer depuis gm2048.engine.board à côté de Board.
from gm2048.engine.board import BOARD_SIZE, Board, new_board

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
