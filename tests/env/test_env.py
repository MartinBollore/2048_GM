"""Tests de l'environnement Gymnasium (étape 6 de la phase 2).

Chaque test décrit un comportement attendu. Lance-les avec :

    pytest tests/env/test_env.py
"""

from __future__ import annotations

import warnings

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from gm2048.agents.baselines import RandomAgent
from gm2048.engine import Action, Game, legal_actions, render
from gm2048.env import ENV_ID, Env2048


def first_seed_with_an_illegal_move() -> tuple[int, Action]:
    """Une graine dont le plateau de départ a au moins un coup illégal, et ce coup."""
    for seed in range(100):
        mask = Game(seed=seed).legal_actions()
        if not mask.all():
            return seed, Action(int(np.flatnonzero(~mask)[0]))
    raise AssertionError("aucune graine trouvée")


def play_until_the_end(env: Env2048, seed: int) -> int:
    """Joue le premier coup légal jusqu'à la fin de la partie et renvoie le nombre de coups."""
    _, info = env.reset(seed=seed)
    terminated = False
    steps = 0
    while not terminated:
        action = int(np.flatnonzero(info["action_mask"])[0])
        _, _, terminated, _, info = env.step(action)
        steps += 1
    return steps


# --- L'interface Gymnasium ----------------------------------------------------------


def test_passes_the_gymnasium_checker_without_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(Env2048(), skip_render_check=True)


def test_can_be_created_with_gymnasium_make() -> None:
    env = gym.make(ENV_ID)
    assert isinstance(env.unwrapped, Env2048)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(gym.make(ENV_ID, render_mode="ansi").unwrapped)


def test_observation_space() -> None:
    assert Env2048().observation_space == spaces.Box(0, 15, shape=(4, 4), dtype=np.uint8)


def test_action_space() -> None:
    assert Env2048().action_space == spaces.Discrete(4)


@pytest.mark.parametrize("render_mode", ["human", "rgb_array", "texte"])
def test_refuses_an_unknown_render_mode(render_mode: str) -> None:
    with pytest.raises(ValueError):
        Env2048(render_mode=render_mode)


@pytest.mark.parametrize("max_steps", [0, -5])
def test_refuses_max_steps_below_one(max_steps: int) -> None:
    with pytest.raises(ValueError):
        Env2048(max_steps=max_steps)


def test_keeps_its_settings() -> None:
    env = Env2048(render_mode="ansi", max_steps=50)
    assert env.render_mode == "ansi"
    assert env.max_steps == 50


# --- reset --------------------------------------------------------------------------


def test_reset_returns_an_observation_and_an_info() -> None:
    env = Env2048()
    obs, info = env.reset(seed=0)
    assert obs.shape == (4, 4)
    assert obs.dtype == np.uint8
    assert env.observation_space.contains(obs)
    assert set(info) == {"action_mask", "score", "max_tile"}
    assert info["score"] == 0


@pytest.mark.parametrize("seed", [0, 1, 42, 2026])
def test_reset_with_a_seed_starts_the_same_game_as_game(seed: int) -> None:
    obs, info = Env2048().reset(seed=seed)
    game = Game(seed=seed)
    np.testing.assert_array_equal(obs, game.board)
    assert info["max_tile"] == game.max_tile


def test_reset_gives_the_legal_moves_in_info() -> None:
    obs, info = Env2048().reset(seed=3)
    assert info["action_mask"].dtype == np.bool_
    np.testing.assert_array_equal(info["action_mask"], legal_actions(obs))


def test_reset_without_seed_is_reproducible_after_a_seeded_reset() -> None:
    first, second = Env2048(), Env2048()
    first.reset(seed=7)
    second.reset(seed=7)
    for _ in range(3):
        obs_first, _ = first.reset()
        obs_second, _ = second.reset()
        np.testing.assert_array_equal(obs_first, obs_second)


def test_reset_without_seed_starts_new_games() -> None:
    env = Env2048()
    env.reset(seed=7)
    boards = {env.reset()[0].tobytes() for _ in range(20)}
    assert len(boards) > 1


def test_reset_restarts_a_finished_game() -> None:
    env = Env2048()
    play_until_the_end(env, seed=0)
    obs, info = env.reset(seed=0)
    assert info["score"] == 0
    np.testing.assert_array_equal(obs, Game(seed=0).board)


# --- step ---------------------------------------------------------------------------


def test_step_before_reset_raises() -> None:
    with pytest.raises(RuntimeError):
        Env2048().step(0)


def test_step_returns_five_values_of_the_right_types() -> None:
    env = Env2048()
    _, info = env.reset(seed=0)
    action = int(np.flatnonzero(info["action_mask"])[0])
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert set(info) == {"action_mask", "score", "max_tile"}


def test_step_accepts_numpy_integers() -> None:
    env = Env2048()
    env.reset(seed=0)
    env.step(np.int64(Action.LEFT))


@pytest.mark.parametrize("action", [4, -1, 10])
def test_step_refuses_an_unknown_action(action: int) -> None:
    env = Env2048()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(action)


def test_an_illegal_move_changes_nothing() -> None:
    seed, illegal = first_seed_with_an_illegal_move()
    env = Env2048()
    before, _ = env.reset(seed=seed)
    obs, reward, terminated, truncated, info = env.step(illegal)
    np.testing.assert_array_equal(obs, before)
    assert reward == 0.0
    assert not terminated
    assert not truncated
    assert info["score"] == 0


@pytest.mark.parametrize("seed", range(5))
def test_plays_exactly_the_same_game_as_game(seed: int) -> None:
    # Le critère de fin de l'étape : mêmes coups, même graine, même partie.
    env, game = Env2048(), Game(seed=seed)
    env_agent, game_agent = RandomAgent(seed=seed), RandomAgent(seed=seed)
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    terminated = False
    while not terminated:
        action = env_agent.act(obs, info["action_mask"])
        obs, reward, terminated, truncated, info = env.step(action)
        result = game.step(game_agent.act(game.board, game.legal_actions()))
        total_reward += reward
        np.testing.assert_array_equal(obs, game.board)
        assert reward == result.reward
        assert terminated == game.done
        assert not truncated
        np.testing.assert_array_equal(info["action_mask"], legal_actions(obs))
    assert info["score"] == game.score
    assert total_reward == game.score
    assert info["max_tile"] == game.max_tile


def test_stepping_after_the_end_raises() -> None:
    env = Env2048()
    play_until_the_end(env, seed=1)
    with pytest.raises(RuntimeError):
        env.step(0)


# --- truncated ----------------------------------------------------------------------


def test_truncates_after_max_steps() -> None:
    env = Env2048(max_steps=5)
    _, info = env.reset(seed=0)
    for step in range(1, 6):
        action = int(np.flatnonzero(info["action_mask"])[0])
        _, _, terminated, truncated, info = env.step(action)
        assert not terminated
        assert truncated == (step == 5)


def test_illegal_moves_count_towards_max_steps() -> None:
    seed, illegal = first_seed_with_an_illegal_move()
    env = Env2048(max_steps=3)
    env.reset(seed=seed)
    results = [env.step(illegal) for _ in range(3)]
    assert [truncated for _, _, _, truncated, _ in results] == [False, False, True]


def test_reset_restarts_the_step_counter() -> None:
    env = Env2048(max_steps=2)
    _, info = env.reset(seed=0)
    env.step(int(np.flatnonzero(info["action_mask"])[0]))
    _, info = env.reset(seed=0)
    _, _, _, truncated, _ = env.step(int(np.flatnonzero(info["action_mask"])[0]))
    assert not truncated


def test_a_game_that_ends_on_the_last_allowed_move_is_terminated_not_truncated() -> None:
    length = play_until_the_end(Env2048(), seed=1)
    env = Env2048(max_steps=length)
    _, info = env.reset(seed=1)
    for _ in range(length):
        action = int(np.flatnonzero(info["action_mask"])[0])
        _, _, terminated, truncated, info = env.step(action)
    assert terminated
    assert not truncated


def test_never_truncates_without_max_steps() -> None:
    env = Env2048()
    _, info = env.reset(seed=2)
    for _ in range(300):
        action = int(np.flatnonzero(info["action_mask"])[0])
        _, _, terminated, truncated, info = env.step(action)
        assert not truncated
        if terminated:
            break


# --- render -------------------------------------------------------------------------


def test_render_in_ansi_mode_shows_the_board() -> None:
    env = Env2048(render_mode="ansi")
    _, info = env.reset(seed=0)
    env.step(int(np.flatnonzero(info["action_mask"])[0]))
    game = Game(seed=0)
    game.step(int(np.flatnonzero(game.legal_actions())[0]))
    assert env.render() == render(game.board, score=game.score, moves=game.moves)


def test_render_without_mode_returns_none() -> None:
    env = Env2048()
    env.reset(seed=0)
    assert env.render() is None


def test_render_before_reset_returns_none() -> None:
    assert Env2048(render_mode="ansi").render() is None
