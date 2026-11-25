"""Outils communs aux tests de la fenêtre.

Les tests ne doivent jamais ouvrir de vraie fenêtre (la CI n'a pas d'écran) :
le pilote vidéo « dummy » de SDL fait semblant d'en avoir un.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import Executor, Future
from typing import Any

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


class InlineExecutor(Executor):
    """Exécute chaque tâche tout de suite, dans le fil courant : tests déterministes."""

    def __init__(self) -> None:
        self.calls = 0
        self.shut_down = False

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
        self.calls += 1
        future: Future[Any] = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as error:
            future.set_exception(error)
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self.shut_down = True


class StalledExecutor(Executor):
    """Ne termine jamais aucune tâche : un agent qui réfléchit indéfiniment."""

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
        return Future()


@pytest.fixture
def inline_executor() -> InlineExecutor:
    return InlineExecutor()


@pytest.fixture
def stalled_executor() -> StalledExecutor:
    return StalledExecutor()
