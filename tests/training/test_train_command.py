"""Tests de la commande ``gm2048 train`` (étape 5 de la phase 3)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from gm2048.agents.ntuple import NTupleNetwork
from gm2048.cli import app
from gm2048.evaluation import GameResult
from gm2048.training.loop import WEIGHTS_FILE, TrainingConfig, load_state, train

runner = CliRunner()
OPTIONS = ["--games", "30", "--seed", "5", "--checkpoint-every", "10", "--log-every", "5"]
CONFIG = TrainingConfig(games=30, seed=5, checkpoint_every=10, log_every=5)


def same_weights(a: NTupleNetwork, b: NTupleNetwork) -> bool:
    return all(np.array_equal(x, y) for x, y in zip(a.weights, b.weights, strict=True))


def test_cli_train_runs_a_training(tmp_path: Path) -> None:
    expected = train(CONFIG, tmp_path / "api")
    result = runner.invoke(app, ["train", "--out", str(tmp_path / "cli"), *OPTIONS])
    assert result.exit_code == 0, result.output
    assert load_state(tmp_path / "cli") == (CONFIG, 30)
    assert same_weights(NTupleNetwork.load(tmp_path / "cli" / WEIGHTS_FILE), expected)


def test_cli_train_passes_alpha(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["train", "--out", str(tmp_path / "cli"), *OPTIONS, "--alpha", "0.25"]
    )
    assert result.exit_code == 0, result.output
    config, _ = load_state(tmp_path / "cli")
    assert config.alpha == 0.25


def test_cli_train_needs_out() -> None:
    result = runner.invoke(app, ["train", *OPTIONS])
    assert result.exit_code != 0


def test_cli_train_refuses_an_existing_training(tmp_path: Path) -> None:
    out = tmp_path / "run"
    train(CONFIG, out)
    before = (out / WEIGHTS_FILE).read_bytes()
    result = runner.invoke(app, ["train", "--out", str(out), *OPTIONS])
    assert result.exit_code == 2  # erreur d'utilisation
    assert (out / WEIGHTS_FILE).read_bytes() == before


def test_cli_train_rejects_bad_settings_before_playing(tmp_path: Path) -> None:
    out = tmp_path / "run"
    result = runner.invoke(
        app, ["train", "--out", str(out), "--checkpoint-every", "7", "--log-every", "5"]
    )
    assert result.exit_code == 2
    assert not (out / "state.json").exists()


def test_cli_train_resumes(tmp_path: Path) -> None:
    expected = train(CONFIG, tmp_path / "straight")
    out = tmp_path / "cut"

    calls = 0

    def stop(_: GameResult) -> None:
        nonlocal calls
        calls += 1
        if calls == 17:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        train(CONFIG, out, on_game_done=stop)
    result = runner.invoke(app, ["train", "--out", str(out), "--resume"])
    assert result.exit_code == 0, result.output
    assert load_state(out) == (CONFIG, 30)
    assert same_weights(NTupleNetwork.load(out / WEIGHTS_FILE), expected)


def test_cli_train_resume_extends_with_games(tmp_path: Path) -> None:
    out = tmp_path / "run"
    train(CONFIG, out)
    result = runner.invoke(app, ["train", "--out", str(out), "--resume", "--games", "40"])
    assert result.exit_code == 0, result.output
    config, done = load_state(out)
    assert (config.games, done) == (40, 40)


def test_cli_train_resume_needs_a_training(tmp_path: Path) -> None:
    result = runner.invoke(app, ["train", "--out", str(tmp_path / "nothing"), "--resume"])
    assert result.exit_code == 2


def test_cli_train_resume_cannot_go_back(tmp_path: Path) -> None:
    out = tmp_path / "run"
    train(CONFIG, out)
    result = runner.invoke(app, ["train", "--out", str(out), "--resume", "--games", "20"])
    assert result.exit_code == 2
    assert load_state(out) == (CONFIG, 30)


def test_cli_train_says_where_the_files_are(tmp_path: Path) -> None:
    out = tmp_path / "run"
    result = runner.invoke(app, ["train", "--out", str(out), *OPTIONS])
    assert result.exit_code == 0, result.output
    assert WEIGHTS_FILE in result.stdout


# --- La fenêtre de courbes ----------------------------------------------------------


def test_cli_train_plot_opens_the_window_in_another_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launched: list[list[str]] = []

    def fake_popen(command: list[str], **kwargs: object) -> None:
        launched.append(command)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    out = tmp_path / "run"
    result = CliRunner().invoke(app, ["train", "--out", str(out), *OPTIONS, "--plot"])
    assert result.exit_code == 0, result.output
    assert launched == [[sys.executable, "-m", "gm2048", "plot", str(out)]]
    assert load_state(out) == (CONFIG, 30)


def test_cli_train_without_plot_opens_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launched: list[list[str]] = []
    monkeypatch.setattr(subprocess, "Popen", lambda command, **kwargs: launched.append(command))
    result = CliRunner().invoke(app, ["train", "--out", str(tmp_path / "run"), *OPTIONS])
    assert result.exit_code == 0, result.output
    assert launched == []


def test_cli_plot_shows_the_given_trainings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gm2048.training import plot

    shown: list[tuple[list[Path], float]] = []

    def fake_show(runs: list[Path], interval: float) -> None:
        shown.append((runs, interval))

    monkeypatch.setattr(plot, "show_training", fake_show)
    a, b = tmp_path / "a", tmp_path / "b"
    result = CliRunner().invoke(app, ["plot", str(a), str(b), "--interval", "5"])
    assert result.exit_code == 0, result.output
    assert shown == [([a, b], 5.0)]
