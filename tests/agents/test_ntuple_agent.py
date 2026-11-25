"""Tests de l'agent n-tuple (phase 3, étape 6).

Chaque test décrit un comportement attendu. Lance-les avec :

    pytest tests/agents/test_ntuple_agent.py

Les tests n'utilisent pas ``models/ntuple.npz`` : ils entraînent un petit réseau de
200 parties (moins d'une seconde) et le sauvegardent dans un dossier temporaire.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from gm2048.agents import Agent
from gm2048.agents.ntuple import NTupleNetwork
from gm2048.agents.ntuple import agent as agent_module
from gm2048.agents.ntuple.agent import NTupleAgent
from gm2048.engine import Action, Game, from_values
from gm2048.engine.bitboard import to_bitboard
from gm2048.evaluation import evaluate
from gm2048.training.fast import play_and_learn_numba
from gm2048.training.td import best_move

# Plein, sans aucune fusion possible : la partie est finie.
NO_MOVE = from_values(
    [
        [2, 4, 2, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [4, 2, 4, 2],
    ]
)


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Les poids d'un réseau entraîné sur 200 parties."""
    net = NTupleNetwork()
    for seed in range(200):
        play_and_learn_numba(net, 0.1, seed)
    path = tmp_path_factory.mktemp("weights") / "trained.npz"
    net.save(path)
    return path


@pytest.fixture(scope="module")
def fresh(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Les poids d'un réseau neuf : tous à zéro, donc tous les coups se valent."""
    path = tmp_path_factory.mktemp("weights") / "fresh.npz"
    NTupleNetwork().save(path)
    return path


def same_weights(a: NTupleNetwork, b: NTupleNetwork) -> bool:
    return all(np.array_equal(x, y) for x, y in zip(a.weights, b.weights, strict=True))


# --- Le contrat ---------------------------------------------------------------------


def test_name(trained: Path) -> None:
    assert NTupleAgent.name == "ntuple"
    assert NTupleAgent(weights=trained).name == "ntuple"


def test_is_an_agent(trained: Path) -> None:
    assert isinstance(NTupleAgent(weights=trained), Agent)


def test_takes_the_seed_first_like_the_other_agents(trained: Path) -> None:
    # Le registre crée les agents avec AGENTS[name](seed).
    assert NTupleAgent(0, weights=trained).name == "ntuple"


# --- Les poids ----------------------------------------------------------------------


def test_loads_the_given_weights(trained: Path) -> None:
    agent = NTupleAgent(weights=trained)
    assert same_weights(agent.net, NTupleNetwork.load(trained))


def test_accepts_a_path_as_text(trained: Path) -> None:
    agent = NTupleAgent(weights=str(trained))
    assert same_weights(agent.net, NTupleNetwork.load(trained))


def test_default_weights_are_models_ntuple_npz_at_the_project_root() -> None:
    expected = Path(__file__).resolve().parents[2] / "models" / "ntuple.npz"
    assert expected == agent_module.DEFAULT_WEIGHTS


def test_uses_the_default_weights_without_argument(
    trained: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # On fait pointer DEFAULT_WEIGHTS vers les poids de test : l'agent doit le lire
    # au moment où il est créé, pas avant.
    monkeypatch.setattr(agent_module, "DEFAULT_WEIGHTS", trained)
    assert same_weights(NTupleAgent().net, NTupleNetwork.load(trained))


def test_missing_weights_explain_how_to_get_them(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        NTupleAgent(weights=tmp_path / "absent.npz")
    message = str(excinfo.value)
    assert "absent.npz" in message
    assert "gm2048 train" in message


# --- Le jeu -------------------------------------------------------------------------


@pytest.mark.parametrize("weights", ["trained", "fresh"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_plays_like_best_move(weights: str, seed: int, request: pytest.FixtureRequest) -> None:
    # À chaque coup d'une partie complète, l'agent choisit le coup de best_move
    # (étape 4) avec les mêmes poids. Avec le réseau neuf, tout vaut 0 : seuls les
    # points départagent, puis l'ordre des actions.
    path: Path = request.getfixturevalue(weights)
    agent = NTupleAgent(weights=path)
    net = NTupleNetwork.load(path)
    game = Game(seed=seed)
    while not game.done:
        mask = game.legal_actions()
        action = agent.act(game.board, mask)
        expected = best_move(net, to_bitboard(game.board))
        assert expected is not None
        assert action == expected.action
        assert isinstance(action, Action)
        assert mask[action]
        game.step(action)


def test_does_not_learn(trained: Path) -> None:
    agent = NTupleAgent(weights=trained)
    evaluate(agent, n_games=3)
    assert same_weights(agent.net, NTupleNetwork.load(trained))


def test_the_seed_changes_nothing(trained: Path) -> None:
    # L'agent ne tire rien au sort : la graine ne sert qu'au registre.
    a = evaluate(NTupleAgent(1, weights=trained), n_games=3)
    b = evaluate(NTupleAgent(2, weights=trained), n_games=3)
    assert a.results == b.results


def test_refuses_to_play_without_legal_move(trained: Path) -> None:
    agent = NTupleAgent(weights=trained)
    with pytest.raises(ValueError):
        agent.act(NO_MOVE, np.zeros(4, dtype=np.bool_))


# --- La vitesse ---------------------------------------------------------------------


def test_is_much_faster_than_best_move_in_python(trained: Path) -> None:
    # Le benchmark joue 1 000 parties de plus de 1 000 coups : l'agent doit choisir
    # ses coups avec la version compilée (best_move_fast), pas avec best_move.
    agent = NTupleAgent(weights=trained)
    net = NTupleNetwork.load(trained)
    game = Game(seed=0)
    boards = []
    while not game.done:
        boards.append(game.board.copy())
        game.step(agent.act(game.board, game.legal_actions()))
    boards = boards[:300]
    mask = np.ones(4, dtype=np.bool_)

    start = time.perf_counter()
    for board in boards:
        best_move(net, to_bitboard(board))
    python = time.perf_counter() - start

    start = time.perf_counter()
    for board in boards:
        agent.act(board, mask)
    compiled = time.perf_counter() - start

    assert compiled * 5 < python, f"{compiled:.4f} s contre {python:.4f} s en Python"


def test_importing_the_agent_does_not_load_numba() -> None:
    # Numba met environ une seconde à se charger. Le registre importe tous les
    # agents : gm2048 play ne doit pas payer ce temps. Vérifié dans un Python neuf.
    code = (
        "import sys, gm2048.agents.ntuple.agent, gm2048.agents.registry; "
        "assert 'numba' not in sys.modules, 'numba importé'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
