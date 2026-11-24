"""L'agent n-tuple (phase 3, étape 6) : il joue avec les poids appris, sans plus les
modifier.

Pendant l'entraînement, le réseau apprend en jouant. Ici, il ne fait plus que jouer :
à chaque coup, il choisit l'action qui maximise ``points + valeur de l'afterstate``,
exactement comme ``best_move`` de l'étape 4, mais sans aucune mise à jour. C'est cet
agent que ``gm2048 benchmark`` mesurera, avec l'objectif de la phase : au moins une
partie sur deux qui atteint 2048.

À toi de jouer : complète ``__init__`` et ``act``, ajoute l'agent au registre
(``gm2048/agents/registry.py``) sous le nom ``"ntuple"``, puis lance
``pytest tests/agents/test_ntuple_agent.py tests/agents/test_registry.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gm2048.agents.ntuple.network import NTupleNetwork
from gm2048.engine import Action, ActionMask, Board
from gm2048.engine.bitboard import to_bitboard

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[4] / "models" / "ntuple.npz"
"""Les poids utilisés par défaut : ``models/ntuple.npz``, à la racine du projet.

``parents[4]`` remonte de ``agent.py`` à la racine : ``ntuple``, ``agents``,
``gm2048``, ``src``, puis le projet. Comme ton installation est en mode éditable
(``pip install -e``), ce chemin pointe bien vers ton dossier, quel que soit le dossier
d'où tu lances une commande.
"""


class NTupleAgent:
    """Joue le coup qui maximise ``points + valeur de l'afterstate`` selon le réseau."""

    name = "ntuple"

    def __init__(self, seed: int | None = None, weights: Path | str | None = None) -> None:
        """Charge le réseau depuis ``weights``, ou depuis ``DEFAULT_WEIGHTS`` si rien
        n'est donné.

        - ``seed`` est ignoré : l'agent ne tire rien au sort, deux agents avec les mêmes
          poids jouent les mêmes coups. Il est là pour que le registre puisse créer cet
          agent comme les autres, avec ``AGENTS["ntuple"](seed)``.
        - Lis ``DEFAULT_WEIGHTS`` ici, au moment de l'appel, et pas comme valeur par
          défaut de l'argument : un test le remplace par des poids de test.
        - Si le fichier n'existe pas, lève ``FileNotFoundError`` avec un message qui
          donne le chemin cherché et explique comment obtenir des poids (le test
          cherche « gm2048 train » dans le message).
        - Garde le réseau chargé dans ``self.net``.

        Pour la vitesse, ``act`` doit utiliser la version compilée de l'étape Numba :
        prépare ici, une fois pour toutes, les tableaux de ``network_arrays(self.net)``.
        Importe ``gm2048.training.fast`` à l'intérieur de cette méthode, pas en haut du
        fichier : le registre importe tous les agents, et ``gm2048 play`` n'a pas à
        payer le chargement de Numba (environ une seconde).
        """

        import gm2048.training.fast as fast

        if isinstance(weights, str):
            weights = Path(weights)

        if weights is None:
            self.net = NTupleNetwork.load(DEFAULT_WEIGHTS)
        elif not weights.exists():
            raise FileNotFoundError(f"Le fichier {weights} n'existe pas (gm2048 train)")
        else:
            self.net = NTupleNetwork.load(weights)

        self.fast = fast
        self.net_arrays = fast.network_arrays(net=self.net)

    def act(self, board: Board, legal_mask: ActionMask) -> Action:
        """Renvoie le meilleur coup légal selon le réseau.

        Le même coup que ``best_move`` de ``td.py``, mais calculé par
        ``best_move_fast`` : convertis le plateau avec ``to_bitboard`` (de
        ``gm2048.engine.bitboard``), en ``np.uint64``. ``best_move_fast`` renvoie
        ``(action, afterstate, points)``, avec une action de -1 s'il n'y a aucun coup
        légal : lève alors ``ValueError``.

        Renvoie un ``Action``, pas un simple entier. ``legal_mask`` ne sert pas ici :
        ``best_move_fast`` ne considère déjà que les coups qui changent le plateau.
        """

        weights = self.net_arrays[0]
        cells = self.net_arrays[1]

        bitboard = np.uint64(to_bitboard(board=board))
        move, _, _ = self.fast.best_move_fast(weights=weights, cells=cells, bits=bitboard)

        if move == -1:
            raise ValueError("Aucun coup legal")
        else:
            return Action(int(move))
