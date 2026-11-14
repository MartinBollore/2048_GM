"""Le réseau n-tuple (phase 3, étape 3).

Le réseau note un plateau : plus la note est haute, plus le plateau est prometteur.
Il ne peut pas retenir une note par plateau (il en existe des milliards de
milliards). À la place, il regarde de petits groupes de cases, les « tuples », et
garde une note pour chaque motif possible de chaque tuple.

Un tuple de 4 cases a 16 ** 4 = 65 536 motifs possibles, puisque chaque case contient
un exposant de 0 à 15. Sa table a donc 65 536 poids. Le motif lu sur le plateau sert
d'indice dans la table, et la note du plateau est la somme des poids lus.

Chaque tuple est lu sur les 8 symétries du plateau (4 rotations et leurs miroirs) :
un bon motif dans un coin est aussi bon dans les trois autres. Avec 5 tuples, le
réseau lit donc 5 x 8 = 40 poids par plateau.

À l'étape 4, l'apprentissage corrigera ces poids après chaque coup, avec ``update``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from gm2048.engine.bitboard import CELL_BITS, CELL_MASK, MAX_EXPONENT, Bitboard
from gm2048.engine.board import BOARD_SIZE

Tuple = tuple[int, ...]
"""Un tuple : des numéros de cases, de 0 à 15 (case ``i = 4 * ligne + colonne``)."""

Weights = npt.NDArray[np.float32]
"""Une table de poids : un tableau NumPy à une dimension, en float32."""

DEFAULT_TUPLES: tuple[Tuple, ...] = (
    (0, 1, 2, 3),  # la ligne du haut, le long du bord
    (4, 5, 6, 7),  # la deuxième ligne
    (0, 1, 4, 5),  # le carré 2x2 du coin
    (1, 2, 5, 6),  # le carré 2x2 du milieu du bord
    (5, 6, 9, 10),  # le carré 2x2 du centre
)
"""Les tuples par défaut : 2 lignes et 3 carrés, qui couvrent tout le plateau par symétrie."""

MAX_TUPLE_LENGTH = 6
"""Au-delà de 6 cases, une table dépasse 268 millions de poids (plus d'1 Go)."""


# --- Les symétries et l'indice ------------------------------------------------------


def symmetries() -> tuple[tuple[int, ...], ...]:
    """Les 8 symétries du carré, chacune sous forme de permutation des 16 cases.

    Une symétrie ``s`` est un tuple de 16 numéros de cases. Lire le plateau dans
    l'ordre ``s[0], s[1], ..., s[15]`` donne le plateau tourné ou retourné. La
    première symétrie est l'identité, ``(0, 1, 2, ..., 15)`` ; les autres sont les
    rotations d'un quart, d'un demi et de trois quarts de tour, puis les mêmes
    après un miroir.

    Elles ne dépendent pas du plateau : on les calcule une fois, à la création du
    réseau.
    """

    a = np.arange(BOARD_SIZE**2).reshape(BOARD_SIZE, BOARD_SIZE)
    sym = [np.rot90(a, k) for k in range(4)] + [np.rot90(np.fliplr(a), k) for k in range(4)]

    return tuple(tuple(m.ravel().tolist()) for m in sym)


def tuple_index(bits: Bitboard, cells: Sequence[int]) -> int:
    """L'indice du motif que forment les cases ``cells`` dans la table de leur tuple.

    Les exposants des cases sont mis bout à bout, 4 bits chacun : l'exposant de
    ``cells[0]`` dans les 4 bits de poids faible, puis celui de ``cells[1]``, etc.
    Pour un tuple de 4 cases, l'indice va donc de 0 à 65 535, exactement comme une
    ligne du bitboard.

    >>> hex(tuple_index(0xFEDCBA9876543210, (5, 6, 9, 10)))
    '0xa965'
    """

    index = 0
    for i in range(len(cells)):
        exposant = (bits >> cells[i] * CELL_BITS) & CELL_MASK
        index = index | (exposant << i * CELL_BITS)

    return index


# --- Le réseau ----------------------------------------------------------------------


class NTupleNetwork:
    """Un réseau n-tuple : une table de poids par tuple, lue sur les 8 symétries.

    Attributs :

    - ``tuples`` : les tuples du réseau, un tuple d'entiers par table ;
    - ``weights`` : la liste des tables, une par tuple, de ``16 ** len(tuple)`` poids
      en float32, tous à 0 dans un réseau neuf.
    """

    def __init__(self, tuples: Sequence[Tuple] = DEFAULT_TUPLES) -> None:
        """Crée un réseau neuf, avec une table à zéro par tuple.

        Lève ``ValueError`` s'il n'y a aucun tuple, ou si un tuple est vide, a plus
        de ``MAX_TUPLE_LENGTH`` cases, contient une case hors de 0 à 15, ou deux
        fois la même case.
        """

        if len(tuples) == 0:
            raise ValueError("Aucun tuple fourni")

        for t in tuples:
            if len(t) == 0:
                raise ValueError("Un tuple est vide")
            elif len(t) > MAX_TUPLE_LENGTH:
                raise ValueError(
                    "Un tuple a un nombre d'éléments supérieur au nombre maximal autorisé"
                )
            elif min(t) < 0 or max(t) >= BOARD_SIZE ** 2:
                raise ValueError(
                    "Certains indices dans les tuples sont soient inférieur à 0 "
                    "ou supérieur à MAX_EXPONENT"
                )

            ts = sorted(t)
            elt = ts[0]
            i = 1
            while i < len(ts) and elt != ts[i]:
                elt = ts[i]
                i += 1

            if i < len(ts):
                raise ValueError("Un tuple possède deux indices égaux")

        sym = symmetries()
        _cells = []
        for t in tuples:
            cell = []
            for s in sym:
                cell.append(tuple(s[i] for i in t))

            _cells.append(cell)

        weights: list[Weights] = []
        for t in tuples:
            w = np.zeros(shape=16 ** len(t), dtype=np.float32)
            weights.append(w)

        self.tuples: tuple[Tuple, ...] = tuple(tuples)
        self.weights: list[Weights] = weights
        self._cells = _cells

    @property
    def n_weights(self) -> int:
        """Le nombre total de poids, toutes tables confondues."""

        n = sum([w.size for w in self.weights])
        return n

    def features(self, bits: Bitboard) -> list[tuple[int, int]]:
        """Les poids que lit le réseau pour ce plateau, sous forme ``(table, indice)``.

        Pour chaque tuple et chaque symétrie, on lit les cases du tuple vues à
        travers la symétrie, ``s[c] for c in tuple``, et on calcule leur indice avec
        :func:`tuple_index`. Avec les tuples par défaut, cela fait 5 x 8 = 40 paires.

        Deux plateaux symétriques l'un de l'autre lisent les mêmes paires, dans un
        autre ordre : c'est ce qui leur donne la même valeur.
        """

        feat = []
        for table in range(len(self._cells)):
            cell = self._cells[table]
            for c in cell:
                index = tuple_index(bits=bits, cells=c)
                feat.append((table, index))

        return feat

    def value(self, bits: Bitboard) -> float:
        """La note du plateau : la somme des poids lus par :meth:`features`.

        Renvoie un ``float`` Python.
        """

        feat = self.features(bits=bits)

        v = 0.0
        for table, index in feat:
            v += float(self.weights[table][index])

        return float(v)

    def update(self, bits: Bitboard, delta: float) -> None:
        """Corrige la note du plateau de ``delta``, en la répartissant sur les poids lus.

        Chacun des ``m`` poids lus par :meth:`features` reçoit ``delta / m`` (m = 40
        avec les tuples par défaut). Si les ``m`` poids sont tous différents, la note
        du plateau monte donc exactement de ``delta``. Un poids lu plusieurs fois
        reçoit sa part plusieurs fois.
        """

        feat = self.features(bits=bits)

        m = len(feat)
        for table, index in feat:
            self.weights[table][index] += delta / m

    # --- Sauvegarde -----------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Enregistre les tuples et les poids dans un fichier ``.npz``.

        ``path`` doit finir par ``.npz`` : sinon, NumPy ajoute l'extension lui-même.
        """

        d: dict[str, Any] = {}
        for i in range(len(self.tuples)):
            d[f"tuple_{i}"] = self.tuples[i]

        for i in range(len(self.weights)):
            d[f"weight_{i}"] = self.weights[i]

        path = path.with_suffix(".npz")
        np.savez(path, **d)

    @classmethod
    def load(cls, path: Path) -> NTupleNetwork:
        """Recrée un réseau à partir d'un fichier écrit par :meth:`save`."""

        tuples = []
        weights = []

        path = path.with_suffix(".npz")
        with np.load(path) as data:
            i = 0
            while f"tuple_{i}" in data:
                tuples.append(tuple(data[f"tuple_{i}"].tolist()))
                i += 1

            i = 0
            while f"weight_{i}" in data:
                weights.append(data[f"weight_{i}"])
                i += 1

        tuple_network = NTupleNetwork(tuples=tuples)
        tuple_network.weights = weights

        return tuple_network
