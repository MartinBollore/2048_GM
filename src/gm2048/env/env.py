"""Environnement Gymnasium : le jeu 2048 au format standard de l'apprentissage par renforcement.

Gymnasium définit une interface que presque toutes les bibliothèques d'IA par
renforcement savent utiliser. Un environnement y fait trois choses :

- ``reset()`` commence une partie et renvoie la première observation ;
- ``step(action)`` joue un coup et renvoie ce qui s'est passé ;
- ``render()`` affiche la partie (ici, en texte).

L'agent ne voit que des **observations** (ici, le plateau d'exposants) et ne
choisit que des **actions** (ici, un entier de 0 à 3). Les deux sont décrits par
des **espaces** (``observation_space`` et ``action_space``), pour que n'importe
quel algorithme sache à quoi s'attendre sans connaître le jeu.

``step`` renvoie cinq valeurs : ``(observation, reward, terminated, truncated, info)``.

- ``reward`` : les points gagnés par ce coup, en ``float``.
- ``terminated`` : la partie est finie d'elle-même (plus aucun coup possible).
- ``truncated`` : la partie est coupée de l'extérieur, ici parce qu'on a atteint
  ``max_steps`` coups. La différence compte pour l'apprentissage : une partie
  coupée aurait pu continuer, une partie terminée non.
- ``info`` : un dictionnaire d'informations utiles qui ne font pas partie de
  l'observation. Ici : ``"action_mask"`` (les coups légaux), ``"score"`` et
  ``"max_tile"``.

Le moteur fait déjà tout le travail : cet environnement est un adaptateur
autour de :class:`~gm2048.engine.Game`. Une partie jouée ici doit donner
exactement les mêmes plateaux et le même score qu'avec ``Game`` et la même graine.
Tu auras besoin d'importer ``Action``, ``Game``, ``legal_actions`` et ``render`` depuis
``gm2048.engine``, et ``spaces`` depuis ``gymnasium``.

À toi de jouer : remplace les ``raise NotImplementedError`` par ton code, puis
lance ``pytest tests/env/test_env.py``. Ordre conseillé : ``__init__``,
``_info``, ``reset``, ``step``, puis ``render``.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from gm2048.engine import BOARD_SIZE, Action, Game, legal_actions, render

Observation = npt.NDArray[np.uint8]
"""Une observation : le plateau d'exposants, un tableau 4x4 de ``uint8``."""


class Env2048(gym.Env[Observation, np.int64]):
    """Le jeu 2048 vu comme un environnement Gymnasium."""

    metadata: dict[str, Any] = {"render_modes": ["ansi"], "render_fps": 4}  # noqa: RUF012
    """``"ansi"`` : ``render()`` renvoie le plateau sous forme de texte.

    ``render_fps`` est réclamé par le vérificateur de Gymnasium, même sans animation. Le
    ``noqa`` dit à ruff que ce dictionnaire de classe est voulu : Gymnasium le déclare ainsi.
    """

    def __init__(self, render_mode: str | None = None, max_steps: int | None = None) -> None:
        """Crée l'environnement, sans commencer de partie.

        - ``render_mode`` vaut ``None`` ou ``"ansi"`` (sinon ``ValueError``). Garde-le
          dans ``self.render_mode``.
        - ``max_steps`` (``None`` ou au moins 1, sinon ``ValueError``) est le nombre
          de coups après lequel la partie est coupée. Garde-le dans ``self.max_steps``.
        - ``self.observation_space`` est un ``gymnasium.spaces.Box`` de valeurs 0 à
          15, de forme (4, 4), de type ``np.uint8`` : un exposant de 15 correspond à
          la tuile 32 768, bien au-delà de ce qu'on atteindra.
        - ``self.action_space`` est un ``gymnasium.spaces.Discrete(4)`` : les quatre
          valeurs de :class:`~gm2048.engine.Action`.
        - Aucune partie n'existe avant le premier ``reset()``.
        """

        if render_mode is not None and render_mode != "ansi":
            raise ValueError("Render_mode est soit 'ansi', soit None")
        self.render_mode = render_mode

        if max_steps is not None and max_steps < 1:
            raise ValueError("Le nombre max de steps doit être d'au moins 1")
        self.max_steps = max_steps

        # noinspection PyTypeChecker
        self.observation_space = gym.spaces.Box(
            low=0, high=15, shape=(BOARD_SIZE, BOARD_SIZE), dtype=np.uint8
        )

        self.action_space = gym.spaces.Discrete(4)

        self._game: Game | None = None
        self._step: int | None = None

    def _info(self) -> dict[str, Any]:
        """Le dictionnaire ``info`` de la partie en cours.

        Trois clés : ``"action_mask"`` (le masque des coups légaux, tableau NumPy de
        4 booléens), ``"score"`` (``int``) et ``"max_tile"`` (``int``, la valeur de la
        plus grosse tuile, par exemple 128).
        """

        info: dict[str, Any] = {}
        if self._game is not None:
            info["action_mask"] = legal_actions(self._game.board)
            info["score"] = self._game.score
            info["max_tile"] = self._game.max_tile

        return info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        """Commence une nouvelle partie et renvoie ``(observation, info)``.

        Commence toujours par ``super().reset(seed=seed)`` : Gymnasium prépare alors
        ``self.np_random``, le générateur aléatoire de l'environnement.

        - Avec une graine, la partie est ``Game(seed=seed)`` : exactement la même
          partie que dans le reste du projet.
        - Sans graine, la partie reçoit une graine tirée avec ``self.np_random``
          (par exemple ``int(self.np_random.integers(2**31))``). Ainsi, après un
          ``reset(seed=1)``, les ``reset()`` suivants donnent des parties différentes
          mais toujours les mêmes d'une exécution à l'autre.

        ``options`` n'est pas utilisé, mais fait partie de l'interface Gymnasium.
        Remet aussi le compteur de coups à zéro.
        """

        super().reset(seed=seed)

        game_seed = seed if seed is not None else int(self.np_random.integers(2**31))

        self._game = Game(seed=game_seed)
        self._step = 0

        return self._game.board, self._info()

    def step(self, action: np.int64 | int) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Joue ``action`` et renvoie ``(observation, reward, terminated, truncated, info)``.

        - ``action`` peut être un ``int`` ou un entier NumPy (``action_space.sample()``
          renvoie un ``np.int64``). Une valeur hors de 0 à 3 lève ``ValueError``.
        - Avant le premier ``reset()``, lève ``RuntimeError``.
        - Un coup illégal ne change rien : récompense 0.0, même observation,
          ``terminated`` faux. Il compte quand même comme un coup pour ``max_steps``,
          sinon un agent qui insiste sur un coup illégal ne serait jamais coupé.
        - ``truncated`` est vrai quand le nombre de coups atteint ``max_steps`` et
          que la partie n'est pas terminée d'elle-même. Sans ``max_steps``, il est
          toujours faux.
        - ``reward`` est un ``float``, ``terminated`` et ``truncated`` des ``bool`` Python.
        """

        if action < 0 or action > 3:
            raise ValueError("Une action doit être un entier de 0 à 3")

        if self._game is None or self._step is None:
            raise RuntimeError("Impossible de faire step car aucune Game n'est lancée")

        if self._game.done:
            raise RuntimeError("Impossible de continuer la Game car elle est terminée")

        result = self._game.step(Action(int(action)))
        observation = self._game.board
        reward = float(result.reward)

        self._step = self._step + 1

        terminated = self._game.done
        truncated = (
            (not terminated) and (self.max_steps is not None) and (self._step >= self.max_steps)
        )
        info = self._info()

        return observation, reward, terminated, truncated, info

    def render(self) -> str | None:
        """En mode ``"ansi"``, renvoie le texte de ``gm2048.engine.render`` pour la partie
        en cours, avec le score et le nombre de coups (``moves`` de ``Game``). Sinon,
        ou avant le premier ``reset()``, renvoie ``None``.
        """

        info = self._info()
        if self.render_mode == "ansi" and self._game is not None:
            r = render(board=self._game.board, score=info["score"], moves=self._game.moves)
        else:
            r = None

        return r
