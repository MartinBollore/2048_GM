"""Le contrat commun à tous les agents.

Un ``Protocol`` décrit ce qu'un objet doit savoir faire, sans imposer
d'héritage : toute classe qui possède un attribut ``name`` et une méthode
``act`` avec la bonne signature *est* un ``Agent`` aux yeux de mypy. On parle de
typage structurel (du « duck typing » vérifié par l'outil de typage).

``@runtime_checkable`` permet en plus de tester ``isinstance(objet, Agent)``
pendant l'exécution. Attention : ce test vérifie seulement que les attributs
existent, pas leurs types. C'est mypy qui vérifie les types.

Grâce à ce contrat, le module d'évaluation (étape 2) pourra faire jouer
n'importe quel agent sans savoir comment il réfléchit.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from gm2048.engine import Action, ActionMask, Board


@runtime_checkable
class Agent(Protocol):
    """Un joueur de 2048 : il regarde le plateau et choisit un coup."""

    name: str
    """Nom court de l'agent, utilisé dans les rapports (par exemple ``"random"``)."""

    def act(self, board: Board, legal_mask: ActionMask) -> Action:
        """Choisit un coup parmi ceux marqués ``True`` dans ``legal_mask``."""
        ...
