"""Environnement Gymnasium du jeu 2048, pour les agents qui apprennent en jouant.

Importer ce module enregistre l'environnement auprès de Gymnasium : on peut alors
le créer avec ``gymnasium.make("gm2048/2048-v0")``.
"""

import gymnasium as gym

from gm2048.env.env import Env2048

ENV_ID = "gm2048/2048-v0"

if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point="gm2048.env.env:Env2048")

__all__ = ["ENV_ID", "Env2048"]
