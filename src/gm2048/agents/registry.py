"""Le registre des agents : associer un nom à une classe.

Jusqu'ici, la liste des agents vivait dans ``cli.py`` (``AGENTS`` et
``DEPTH_AGENTS``). Mais d'autres parties du projet ont besoin de créer un agent à
partir de son nom : le benchmark de cette étape, et en phase 3 les scripts
d'entraînement. Elles ne doivent pas dépendre de la ligne de commande (Typer) pour
ça. Le registre devient donc un petit module à part, que tout le monde importe, y
compris ``cli.py``.

Un registre, c'est un dictionnaire : nom → classe. En Python, une classe est un
objet comme un autre : on peut la ranger dans un dictionnaire et l'appeler plus
tard pour créer un agent, par exemple ``AGENTS["greedy"](seed)``. Ajouter un agent
au projet revient alors à ajouter une ligne ici, et il devient disponible partout :
``gm2048 eval``, ``gm2048 watch`` et ``gm2048 benchmark``.

À toi de jouer : remplis ``AGENTS`` et ``DEPTH_AGENTS``, remplace le
``raise NotImplementedError`` de ``make_agent``, puis lance
``pytest tests/agents/test_registry.py``. Importe les classes d'agents depuis
``gm2048.agents.baselines``, jamais depuis ``gm2048.cli`` : un test vérifie que ce
module n'importe ni Typer ni la ligne de commande.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

from gm2048.agents.base import Agent
from gm2048.agents.baselines import ExpectimaxAgent, GreedyAgent, HeuristicAgent, RandomAgent

AGENTS: Final[dict[str, Callable[..., Agent]]] = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "heuristic": HeuristicAgent,
    "expectimax": ExpectimaxAgent,
}
"""Les agents du projet : nom → classe de l'agent.

Chaque nom doit être identique à l'attribut ``name`` de l'agent : c'est ce nom qui
apparaît dans les rapports et dans le tableau du benchmark.

``Callable[..., Agent]`` se lit « quelque chose qu'on appelle, avec n'importe quels
arguments, et qui renvoie un ``Agent`` » : c'est exactement ce qu'est une classe
d'agent.
"""

DEPTH_AGENTS: Final[tuple[str, ...]] = ("expectimax",)
"""Les noms des agents qui acceptent une profondeur de recherche (``depth``)."""


def make_agent(name: str, seed: int | None = None, depth: int | None = None) -> Agent:
    """Crée l'agent ``name`` avec la graine ``seed``.

    - Un nom absent de ``AGENTS`` lève ``ValueError``, avec un message qui cite le
      nom fautif et rappelle les noms possibles.
    - Sans ``depth``, l'agent garde sa profondeur par défaut.
    - Avec ``depth``, l'agent doit faire partie de ``DEPTH_AGENTS`` (sinon
      ``ValueError``), et il reçoit cette profondeur.

    C'est la version « bibliothèque » de ``agent_factory`` dans ``cli.py`` : elle lève
    ``ValueError`` et non ``typer.BadParameter``, parce qu'elle ne sait rien de la
    ligne de commande. C'est à la ligne de commande de traduire l'erreur pour
    l'utilisateur.
    """

    if name not in AGENTS:
        raise ValueError(
            f"L'agent {name!r} n'existe pas. Les agents disponibles sont : {', '.join(AGENTS)}."
        )

    if depth is None:
        agent = AGENTS[name](seed)
    elif name in DEPTH_AGENTS:
        agent = AGENTS[name](seed, depth)
    else:
        raise ValueError(
            f"L'agent {name!r} n'a pas de profondeur de calcul. "
            f"Les agents avec une profondeur de calcul sont : {', '.join(DEPTH_AGENTS)}."
        )

    return agent
