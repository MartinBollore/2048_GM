"""Tests de la boucle d'entraînement (étape 5 de la phase 3).

Les entraînements des tests sont minuscules (quelques dizaines de parties), avec
des sauvegardes et des lignes de journal très fréquentes. Lance-les avec :

    pytest tests/training/test_loop.py
"""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from gm2048.agents.ntuple import DEFAULT_TUPLES, NTupleNetwork
from gm2048.evaluation import GameResult
from gm2048.training import loop
from gm2048.training.fast import play_and_learn_numba
from gm2048.training.loop import (
    LOG_COLUMNS,
    LOG_FILE,
    STATE_FILE,
    WEIGHTS_FILE,
    TrainingConfig,
    load_state,
    resume,
    train,
)

SMALL = TrainingConfig(games=60, alpha=0.1, seed=3, checkpoint_every=20, log_every=10)
MIXED_TUPLES = ((0, 1, 2, 3), (0, 1, 4), (5, 6, 9, 10))


class Interrupt(Exception):
    """Simule une coupure (comme un Ctrl+C) au milieu d'un entraînement."""


def interrupt_after(n: int):  # type: ignore[no-untyped-def]
    """Un on_game_done qui coupe l'entraînement lors de son n-ième appel."""
    calls = 0

    def on_game_done(result: GameResult) -> None:
        nonlocal calls
        calls += 1
        if calls == n:
            raise Interrupt

    return on_game_done


def manual_training(config: TrainingConfig) -> tuple[NTupleNetwork, list[GameResult]]:
    """Ce que doit faire train, sans fichiers : les parties seed, seed + 1, ..."""
    net = NTupleNetwork(config.tuples)
    results = [
        play_and_learn_numba(net, config.alpha, config.seed + i) for i in range(config.games)
    ]
    return net, results


def same_weights(a: NTupleNetwork, b: NTupleNetwork) -> bool:
    return all(np.array_equal(x, y) for x, y in zip(a.weights, b.weights, strict=True))


def read_log(out_dir: Path) -> list[list[str]]:
    with (out_dir / LOG_FILE).open(newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def log_without_seconds(out_dir: Path) -> list[list[str]]:
    """Le journal sans la colonne des durées, qui change d'une exécution à l'autre."""
    seconds = LOG_COLUMNS.index("seconds")
    return [row[:seconds] + row[seconds + 1 :] for row in read_log(out_dir)]


# --- TrainingConfig -------------------------------------------------------------


def test_training_config_defaults() -> None:
    config = TrainingConfig()
    assert config.games == 100_000
    assert config.alpha == 0.1
    assert config.seed == 0
    assert config.tuples == DEFAULT_TUPLES
    assert config.checkpoint_every == 5_000
    assert config.log_every == 1_000


def test_training_config_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        SMALL.games = 10  # type: ignore[misc]


@pytest.mark.parametrize(
    "changes",
    [
        {"games": 0},
        {"alpha": 0.0},
        {"alpha": -0.1},
        {"checkpoint_every": 0},
        {"log_every": 0},
        {"checkpoint_every": 25, "log_every": 10},  # 25 n'est pas un multiple de 10
    ],
)
def test_training_config_rejects_bad_settings(changes: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(SMALL, **changes)


# --- train ----------------------------------------------------------------------


def test_train_plays_the_games_seed_plus_i(tmp_path: Path) -> None:
    expected, _ = manual_training(SMALL)
    net = train(SMALL, tmp_path / "run")
    assert same_weights(net, expected)


def test_train_uses_the_settings_of_the_config(tmp_path: Path) -> None:
    config = dataclasses.replace(SMALL, alpha=0.25, seed=11, tuples=MIXED_TUPLES, games=20)
    expected, _ = manual_training(config)
    net = train(config, tmp_path / "run")
    assert net.tuples == MIXED_TUPLES
    assert same_weights(net, expected)


def test_train_creates_the_folder_and_its_parents(tmp_path: Path) -> None:
    out = tmp_path / "runs" / "a" / "b"
    train(dataclasses.replace(SMALL, games=10), out)
    assert (out / WEIGHTS_FILE).is_file()
    assert (out / STATE_FILE).is_file()
    assert (out / LOG_FILE).is_file()


def test_train_saves_the_final_weights(tmp_path: Path) -> None:
    net = train(SMALL, tmp_path / "run")
    assert same_weights(NTupleNetwork.load(tmp_path / "run" / WEIGHTS_FILE), net)


def test_train_writes_the_state(tmp_path: Path) -> None:
    train(SMALL, tmp_path / "run")
    state = json.loads((tmp_path / "run" / STATE_FILE).read_text(encoding="utf-8"))
    assert state["games_done"] == 60
    assert state["config"]["games"] == 60
    assert state["config"]["alpha"] == 0.1
    assert load_state(tmp_path / "run") == (SMALL, 60)


def test_load_state_gives_back_tuples(tmp_path: Path) -> None:
    config = dataclasses.replace(SMALL, tuples=MIXED_TUPLES, games=10)
    train(config, tmp_path / "run")
    loaded, _ = load_state(tmp_path / "run")
    assert loaded.tuples == MIXED_TUPLES
    assert all(isinstance(t, tuple) for t in loaded.tuples)


def test_train_writes_one_log_line_per_block(tmp_path: Path) -> None:
    _, results = manual_training(SMALL)
    train(SMALL, tmp_path / "run")
    rows = read_log(tmp_path / "run")
    assert tuple(rows[0]) == LOG_COLUMNS
    assert [int(row[0]) for row in rows[1:]] == [10, 20, 30, 40, 50, 60]
    for k, row in enumerate(rows[1:]):
        block = results[10 * k : 10 * (k + 1)]
        _games, mean_score, win_rate, mean_moves, seconds = row
        assert float(mean_score) == pytest.approx(sum(r.score for r in block) / 10)
        assert float(win_rate) == pytest.approx(sum(r.max_tile >= 2048 for r in block) / 10)
        assert float(mean_moves) == pytest.approx(sum(r.moves for r in block) / 10)
        assert float(seconds) > 0


def test_train_log_statistics_on_invented_games(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Le journal sur des parties inventées, pour vérifier chaque colonne au chiffre près.

    En vrai, 60 parties d'apprentissage n'atteignent jamais 2048 : ``win_rate`` vaut 0
    et ne vérifie rien. Ici ``play_and_learn_numba`` est remplacé, dans ``loop.py``, par
    une fausse partie : ``train`` doit donc l'appeler par ce nom-là.
    """
    tiles = [1024, 2048, 4096, 512, 2048]

    def fake_play_and_learn(net: NTupleNetwork, alpha: float, seed: int) -> GameResult:
        k = seed % 5
        return GameResult(
            seed=seed, score=1000 * (k + 1), max_tile=tiles[k], moves=10 * k, seconds=0.5
        )

    monkeypatch.setattr(loop, "play_and_learn_numba", fake_play_and_learn)
    config = TrainingConfig(games=8, seed=0, checkpoint_every=5, log_every=5)
    train(config, tmp_path / "run")
    rows = read_log(tmp_path / "run")
    assert [int(row[0]) for row in rows[1:]] == [5, 8]
    first, last = ([float(x) for x in row[1:]] for row in rows[1:])
    # Parties 0 à 4 : 2048, 4096 et 2048 comptent comme gagnées, 1024 et 512 non.
    assert first == pytest.approx([3000.0, 3 / 5, 20.0, 2.5])
    # Parties 5 à 7 (seed % 5 = 0, 1, 2) : 1024, 2048, 4096.
    assert last == pytest.approx([2000.0, 2 / 3, 10.0, 1.5])


def test_train_logs_the_last_incomplete_block(tmp_path: Path) -> None:
    config = dataclasses.replace(SMALL, games=35)
    _, results = manual_training(config)
    train(config, tmp_path / "run")
    rows = read_log(tmp_path / "run")
    assert [int(row[0]) for row in rows[1:]] == [10, 20, 30, 35]
    last = results[30:]
    assert float(rows[-1][1]) == pytest.approx(sum(r.score for r in last) / 5)


def test_train_saves_at_the_end_even_off_a_checkpoint(tmp_path: Path) -> None:
    config = dataclasses.replace(SMALL, games=35)  # 35 n'est pas un multiple de 20
    expected, _ = manual_training(config)
    train(config, tmp_path / "run")
    assert load_state(tmp_path / "run") == (config, 35)
    assert same_weights(NTupleNetwork.load(tmp_path / "run" / WEIGHTS_FILE), expected)


def test_train_calls_on_game_done_for_each_game_in_order(tmp_path: Path) -> None:
    seen: list[GameResult] = []
    train(SMALL, tmp_path / "run", on_game_done=seen.append)
    assert [r.seed for r in seen] == list(range(3, 63))


def test_train_refuses_to_overwrite_a_training(tmp_path: Path) -> None:
    out = tmp_path / "run"
    train(dataclasses.replace(SMALL, games=10), out)
    before = (out / WEIGHTS_FILE).read_bytes(), read_log(out)
    with pytest.raises(FileExistsError):
        train(SMALL, out)
    assert ((out / WEIGHTS_FILE).read_bytes(), read_log(out)) == before


def test_train_saves_before_the_first_game(tmp_path: Path) -> None:
    out = tmp_path / "run"
    with pytest.raises(Interrupt):
        train(SMALL, out, on_game_done=interrupt_after(1))
    _, games_done = load_state(out)
    assert games_done == 0
    assert tuple(read_log(out)[0]) == LOG_COLUMNS


@pytest.mark.parametrize(("stop_at", "saved"), [(19, 0), (20, 20), (35, 20), (45, 40)])
def test_train_saves_every_checkpoint_every_games(tmp_path: Path, stop_at: int, saved: int) -> None:
    out = tmp_path / "run"
    with pytest.raises(Interrupt):
        train(SMALL, out, on_game_done=interrupt_after(stop_at))
    _, games_done = load_state(out)
    assert games_done == saved
    # Les poids sauvegardés sont bien ceux du réseau après `saved` parties.
    expected, _ = manual_training(dataclasses.replace(SMALL, games=max(saved, 1)))
    if saved == 0:
        expected = NTupleNetwork(SMALL.tuples)
    assert same_weights(NTupleNetwork.load(out / WEIGHTS_FILE), expected)


# --- resume ---------------------------------------------------------------------


@pytest.mark.parametrize("stop_at", [1, 5, 20, 35, 59])
def test_interrupted_then_resumed_equals_straight(tmp_path: Path, stop_at: int) -> None:
    straight = train(SMALL, tmp_path / "straight")
    out = tmp_path / "cut"
    with pytest.raises(Interrupt):
        train(SMALL, out, on_game_done=interrupt_after(stop_at))
    resumed = resume(out)
    assert same_weights(resumed, straight)
    assert same_weights(NTupleNetwork.load(out / WEIGHTS_FILE), straight)
    assert log_without_seconds(out) == log_without_seconds(tmp_path / "straight")
    assert load_state(out) == (SMALL, 60)


def test_resume_can_be_interrupted_again(tmp_path: Path) -> None:
    straight = train(SMALL, tmp_path / "straight")
    out = tmp_path / "cut"
    with pytest.raises(Interrupt):
        train(SMALL, out, on_game_done=interrupt_after(15))
    with pytest.raises(Interrupt):
        resume(out, on_game_done=interrupt_after(30))
    resumed = resume(out)
    assert same_weights(resumed, straight)
    assert log_without_seconds(out) == log_without_seconds(tmp_path / "straight")


def test_resume_keeps_the_settings_of_the_training(tmp_path: Path) -> None:
    config = dataclasses.replace(SMALL, alpha=0.25, seed=11, tuples=MIXED_TUPLES)
    straight = train(config, tmp_path / "straight")
    out = tmp_path / "cut"
    with pytest.raises(Interrupt):
        train(config, out, on_game_done=interrupt_after(30))
    assert same_weights(resume(out), straight)


def test_resume_calls_on_game_done_for_the_remaining_games(tmp_path: Path) -> None:
    out = tmp_path / "cut"
    with pytest.raises(Interrupt):
        train(SMALL, out, on_game_done=interrupt_after(30))
    seen: list[GameResult] = []
    resume(out, on_game_done=seen.append)
    assert [r.seed for r in seen] == list(range(3 + 20, 63))


def test_resume_extends_a_finished_training(tmp_path: Path) -> None:
    straight = train(SMALL, tmp_path / "straight")
    out = tmp_path / "short"
    train(dataclasses.replace(SMALL, games=40), out)
    extended = resume(out, games=60)
    assert same_weights(extended, straight)
    assert load_state(out) == (SMALL, 60)
    assert log_without_seconds(out) == log_without_seconds(tmp_path / "straight")


def test_an_interrupted_extension_resumes_up_to_the_new_total(tmp_path: Path) -> None:
    """Prolonger, couper, puis reprendre sans --games : on va jusqu'au nouveau total."""
    straight = train(SMALL, tmp_path / "straight")
    out = tmp_path / "short"
    train(dataclasses.replace(SMALL, games=40), out)
    with pytest.raises(Interrupt):
        resume(out, games=60, on_game_done=interrupt_after(5))
    assert load_state(out) == (SMALL, 40)
    assert same_weights(resume(out), straight)
    assert log_without_seconds(out) == log_without_seconds(tmp_path / "straight")


def test_resume_of_a_finished_training_plays_nothing(tmp_path: Path) -> None:
    out = tmp_path / "run"
    net = train(SMALL, out)
    seen: list[GameResult] = []
    again = resume(out, on_game_done=seen.append)
    assert seen == []
    assert same_weights(again, net)
    assert len(read_log(out)) == 1 + 6


def test_resume_without_a_training(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resume(tmp_path / "nothing")


def test_resume_cannot_go_back(tmp_path: Path) -> None:
    out = tmp_path / "run"
    train(SMALL, out)
    with pytest.raises(ValueError):
        resume(out, games=40)
