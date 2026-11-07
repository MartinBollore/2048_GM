"""Tests de la fenêtre pygame, avec le pilote vidéo « dummy » (aucune fenêtre ne s'ouvre)."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import Executor

import pytest

pygame = pytest.importorskip("pygame")

from gm2048.agents.baselines import RandomAgent  # noqa: E402
from gm2048.engine import Action, Game, from_values  # noqa: E402
from gm2048.gui.controllers import AgentController, Command, HumanController  # noqa: E402
from gm2048.gui.layout import compute_layout  # noqa: E402
from gm2048.gui.theme import DARK, LIGHT  # noqa: E402
from gm2048.gui.window import (  # noqa: E402
    Renderer,
    WindowApp,
    arrow_points,
    command_for_key,
    format_points,
    run,
)

WIDTH, HEIGHT = 500, 720


@pytest.fixture(autouse=True)
def display() -> Iterator[None]:
    pygame.display.init()
    pygame.font.init()
    yield
    pygame.quit()


def key(code: int, text: str = "") -> pygame.event.Event:
    return pygame.event.Event(pygame.KEYDOWN, key=code, unicode=text)


def human_with(rows: list[list[int]]) -> HumanController:
    controller = HumanController(seed=0)
    controller.session.new_game(Game.from_board(from_values(rows), seed=0))
    controller.session.update(1.0)
    return controller


@pytest.mark.parametrize(
    ("code", "text", "expected"),
    [
        (pygame.K_UP, "", Command.UP),
        (pygame.K_z, "z", Command.UP),
        (pygame.K_w, "w", Command.UP),
        (pygame.K_q, "q", Command.LEFT),
        (pygame.K_a, "a", Command.LEFT),
        (pygame.K_s, "s", Command.DOWN),
        (pygame.K_d, "d", Command.RIGHT),
        (pygame.K_RIGHT, "", Command.RIGHT),
        (pygame.K_SPACE, " ", Command.PAUSE),
        (pygame.K_n, "n", Command.STEP),
        (pygame.K_EQUALS, "+", Command.FASTER),
        (pygame.K_KP_PLUS, "+", Command.FASTER),
        (pygame.K_6, "-", Command.SLOWER),  # le « - » du clavier AZERTY
        (pygame.K_MINUS, "-", Command.SLOWER),
        (pygame.K_RETURN, "\r", Command.NEW_GAME),
        (pygame.K_r, "r", Command.NEW_GAME),
        (pygame.K_t, "t", Command.THEME),
        (pygame.K_ESCAPE, "\x1b", Command.QUIT),
        (pygame.K_x, "x", Command.QUIT),
        (pygame.K_k, "k", None),
    ],
)
def test_command_for_key(code: int, text: str, expected: Command | None) -> None:
    assert command_for_key(code, text) == expected


def test_arrow_points_towards_the_move() -> None:
    def tip(action: Action) -> tuple[float, float]:
        x, y = arrow_points(action, 0, 0, 10)[0]
        return round(x, 6), round(y, 6)

    assert tip(Action.UP) == (0, -10)
    assert tip(Action.RIGHT) == (10, 0)
    assert tip(Action.DOWN) == (0, 10)
    assert tip(Action.LEFT) == (-10, 0)


def test_format_points_groups_thousands() -> None:
    assert format_points(0) == "0"
    assert format_points(123456) == "123 456"


def pixel(surface: pygame.Surface, x: float, y: float) -> tuple[int, int, int]:
    color = surface.get_at((round(x), round(y)))
    return color.r, color.g, color.b


def close_to(color: tuple[int, int, int], expected: tuple[int, int, int]) -> bool:
    """Les formes lissées (réduites de 3x) peuvent dévier d'un ou deux pour cent."""
    return all(abs(a - b) <= 6 for a, b in zip(color, expected, strict=True))


@pytest.mark.parametrize("theme", [LIGHT, DARK])
def test_tiles_and_empty_cells_are_painted_with_the_theme_colors(theme: object) -> None:
    assert theme is LIGHT or theme is DARK
    surface = pygame.Surface((WIDTH, HEIGHT))
    controller = human_with([[2, 0, 0, 0], [0, 2048, 0, 0], [0] * 4, [0] * 4])
    layout = compute_layout(WIDTH, HEIGHT)
    Renderer().draw(surface, controller, theme, layout)

    assert pixel(surface, 2, 2) == theme.background
    # Un point du dessus de la tuile, loin du nombre et des coins arrondis.
    for row, col, exponent in [(0, 0, 1), (1, 1, 11)]:
        x, y = layout.cell_origin(row, col)
        face = pixel(surface, x + layout.cell * 0.15, y + layout.cell * 0.2)
        assert close_to(face, theme.tile(exponent).face)
    x, y = layout.cell_origin(3, 3)
    assert close_to(pixel(surface, x + layout.cell / 2, y + layout.cell / 2), theme.empty_cell)


def test_every_screen_state_can_be_drawn() -> None:
    renderer = Renderer()
    surface = pygame.Surface((WIDTH, HEIGHT))
    layout = compute_layout(WIDTH, HEIGHT)
    controller = human_with([[1024, 1024, 0, 0], [0] * 4, [0] * 4, [512, 0, 0, 0]])
    controller.handle(Command.LEFT)  # 2048 : bandeau et points qui s'envolent
    for _ in range(5):
        controller.update(0.03)
        renderer.draw(surface, controller, LIGHT, layout)
    assert controller.session.toast_age is not None
    over = human_with([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
    over.update(1.0)
    assert over.session.over
    renderer.draw(surface, over, DARK, layout)
    huge = human_with([[131072, 65536, 32768, 16384], [0] * 4, [0] * 4, [0] * 4])
    renderer.draw(surface, huge, LIGHT, layout)
    tiny = pygame.Surface((60, 80))
    renderer.draw(tiny, over, DARK, compute_layout(60, 80))


def test_big_numbers_fit_in_their_tile() -> None:
    renderer = Renderer()
    font_size = renderer.fitted_font("heavy", 50, "131072", 86).size("131072")
    assert font_size[0] <= 86


def test_frame_handles_keys_and_quit() -> None:
    surface = pygame.Surface((WIDTH, HEIGHT))
    controller = human_with([[2, 2, 0, 0], [0] * 4, [0] * 4, [0] * 4])
    app = WindowApp(controller, LIGHT)
    assert app.frame(surface, 0.016, [key(pygame.K_LEFT)])
    assert controller.session.game.score == 4
    assert app.frame(surface, 0.016, [key(pygame.K_t, "t")])
    assert app.theme is DARK
    assert not app.frame(surface, 0.016, [key(pygame.K_ESCAPE, "\x1b")])
    assert not app.frame(surface, 0.016, [pygame.event.Event(pygame.QUIT)])


def test_frame_pauses_the_agent(inline_executor: Executor) -> None:
    surface = pygame.Surface((WIDTH, HEIGHT))
    controller = AgentController(RandomAgent, delay=0.0, executor=inline_executor)
    app = WindowApp(controller, LIGHT)
    app.frame(surface, 0.5, [key(pygame.K_SPACE, " ")])
    assert controller.paused
    assert controller.session.game.moves == 0
    app.frame(surface, 0.5, [key(pygame.K_SPACE, " ")])
    app.frame(surface, 0.5, [])
    assert controller.session.game.moves > 0


def test_run_plays_a_whole_game_and_closes_the_controller(inline_executor: Executor) -> None:
    controller = AgentController(RandomAgent, delay=0.0, executor=inline_executor)
    run(controller, title="test", fps=0, max_frames=40)
    assert controller.session.game.moves > 0
    assert inline_executor.shut_down  # type: ignore[attr-defined]


def test_run_stops_on_window_close(inline_executor: Executor) -> None:
    controller = AgentController(RandomAgent, delay=0.0, executor=inline_executor)

    def close_window(dt: float) -> None:
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        original_update(dt)

    original_update = controller.update
    controller.update = close_window  # type: ignore[method-assign]
    run(controller, title="test", fps=0, max_frames=1000)
    assert controller.session.game.moves <= 1


def test_run_rejects_an_unknown_theme(inline_executor: Executor) -> None:
    controller = AgentController(RandomAgent, executor=inline_executor)
    with pytest.raises(ValueError, match="Thème inconnu"):
        run(controller, theme="fluo")
