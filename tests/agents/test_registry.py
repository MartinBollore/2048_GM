"""Tests du registre des agents (étape 7 de la phase 2).

Chaque test décrit un comportement attendu. Lance-les avec :

    pytest tests/agents/test_registry.py
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from gm2048.agents.baselines import ExpectimaxAgent, GreedyAgent, HeuristicAgent, RandomAgent
from gm2048.agents.registry import AGENTS, DEPTH_AGENTS, make_agent
from gm2048.engine import Game

EXPECTED_CLASSES = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "heuristic": HeuristicAgent,
    "expectimax": ExpectimaxAgent,
}


# --- Le contenu du registre ---------------------------------------------------------


def test_the_registry_lists_the_four_agents() -> None:
    assert set(AGENTS) == set(EXPECTED_CLASSES)


@pytest.mark.parametrize(("name", "cls"), EXPECTED_CLASSES.items())
def test_each_name_points_to_its_class(name: str, cls: type) -> None:
    assert AGENTS[name] is cls


@pytest.mark.parametrize("name", EXPECTED_CLASSES)
def test_each_agent_is_registered_under_its_own_name(name: str) -> None:
    # Le nom du registre et l'attribut ``name`` de l'agent doivent être les mêmes :
    # c'est ce nom qui apparaît dans les rapports et dans le tableau du benchmark.
    assert make_agent(name).name == name


def test_only_expectimax_has_a_search_depth() -> None:
    assert tuple(DEPTH_AGENTS) == ("expectimax",)


def test_every_depth_agent_is_in_the_registry() -> None:
    assert all(name in AGENTS for name in DEPTH_AGENTS)


# --- make_agent ---------------------------------------------------------------------


@pytest.mark.parametrize(("name", "cls"), EXPECTED_CLASSES.items())
def test_make_agent_builds_the_right_agent(name: str, cls: type) -> None:
    assert isinstance(make_agent(name, seed=0), cls)


def test_make_agent_passes_the_seed() -> None:
    # Même graine, mêmes coups : make_agent transmet bien la graine à l'agent.
    game = Game(seed=4)
    from_registry, by_hand = make_agent("random", seed=11), RandomAgent(seed=11)
    for _ in range(30):
        if game.done:
            break
        mask = game.legal_actions()
        action = from_registry.act(game.board, mask)
        assert action == by_hand.act(game.board, mask)
        game.step(action)


def test_make_agent_without_depth_keeps_the_default_depth() -> None:
    agent = make_agent("expectimax", seed=0)
    assert isinstance(agent, ExpectimaxAgent)
    assert agent.depth == ExpectimaxAgent(seed=0).depth


def test_make_agent_passes_the_depth() -> None:
    agent = make_agent("expectimax", seed=0, depth=1)
    assert isinstance(agent, ExpectimaxAgent)
    assert agent.depth == 1


@pytest.mark.parametrize("name", ["random", "greedy", "heuristic"])
def test_make_agent_refuses_a_depth_for_agents_without_search(name: str) -> None:
    with pytest.raises(ValueError):
        make_agent(name, depth=2)


def test_make_agent_refuses_an_unknown_name_and_lists_the_choices() -> None:
    with pytest.raises(ValueError) as excinfo:
        make_agent("inconnu")
    message = str(excinfo.value)
    assert "inconnu" in message
    assert all(name in message for name in EXPECTED_CLASSES)


# --- Indépendance vis-à-vis de la ligne de commande ---------------------------------


@pytest.mark.parametrize("module", ["gm2048.agents.registry", "gm2048.benchmark"])
def test_does_not_depend_on_the_command_line(module: str) -> None:
    # Le registre et le benchmark servent aussi hors de la ligne de commande (les
    # agents qui apprennent, en phase 3, en auront besoin) : ils ne doivent pas
    # importer Typer ni le module cli. On le vérifie dans un Python neuf.
    code = (
        f"import sys, {module}; "
        "assert 'typer' not in sys.modules, 'typer importé'; "
        "assert 'gm2048.cli' not in sys.modules, 'gm2048.cli importé'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
