"""Tests des couleurs et de la mise en page."""

from __future__ import annotations

import pytest

from gm2048.gui.layout import DESIGN_HEIGHT, DESIGN_WIDTH, Layout, Rect, compute_layout
from gm2048.gui.theme import DARK, LIGHT, THEMES, TILES, next_theme, shade


def test_shade_darkens_and_lightens() -> None:
    assert shade((100, 200, 50), 0.5) == (50, 100, 25)
    assert shade((100, 200, 50), 1.0) == (100, 200, 50)
    assert shade((0, 0, 0), 2.0) == (255, 255, 255)


def test_every_tile_up_to_16384_has_its_own_color() -> None:
    faces = [style.face for style in TILES]
    assert len(set(faces)) == len(faces)


def test_big_tiles_reuse_the_last_style() -> None:
    assert LIGHT.tile(1) is TILES[0]
    assert LIGHT.tile(11) is TILES[10]
    assert LIGHT.tile(11).glow
    assert LIGHT.tile(30) is TILES[-1]


def test_tile_edge_is_darker_than_its_face() -> None:
    for style in TILES:
        assert sum(style.edge) < sum(style.face)


def test_themes_are_listed_by_name_and_cycle() -> None:
    assert list(THEMES) == ["clair", "sombre"]
    assert next_theme(LIGHT) is DARK
    assert next_theme(DARK) is LIGHT


SIZES = [(DESIGN_WIDTH, DESIGN_HEIGHT), (1000, 1440), (800, 600), (300, 900), (623, 897)]


def inside(rect: Rect, width: int, height: int) -> bool:
    x, y, w, h = rect
    return x >= 0 and y >= 0 and x + w <= width + 1 and y + h <= height + 1


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_layout_fits_the_window(width: int, height: int) -> None:
    layout = compute_layout(width, height)
    for rect in (
        layout.title,
        layout.score_card,
        layout.best_card,
        layout.status,
        layout.board,
        layout.footer,
    ):
        assert inside(rect, width, height)


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_layout_elements_are_stacked_without_overlap(width: int, height: int) -> None:
    layout = compute_layout(width, height)
    _, card_y, _, card_h = layout.score_card
    _, status_y, _, status_h = layout.status
    _, board_y, _, board_h = layout.board
    assert card_y + card_h <= status_y
    assert status_y + status_h <= board_y
    assert board_y + board_h <= layout.footer[1]
    assert layout.score_card[0] + layout.score_card[2] <= layout.best_card[0]


def cells_fit_in_board(layout: Layout) -> bool:
    bx, by, side, _ = layout.board
    for row in range(4):
        for col in range(4):
            x, y = layout.cell_origin(row, col)
            if x < bx or y < by or x + layout.cell > bx + side + 0.01:
                return False
            if y + layout.cell > by + side + 0.01:
                return False
    return True


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_board_is_square_and_holds_the_cells(width: int, height: int) -> None:
    layout = compute_layout(width, height)
    assert layout.board[2] == layout.board[3]
    assert cells_fit_in_board(layout)
    # Les 4 cases et les 5 espaces remplissent exactement le plateau.
    assert 4 * layout.cell + 5 * layout.gap == pytest.approx(layout.board[2])


def test_scaled_tiles_stay_centered_on_their_cell() -> None:
    layout = compute_layout(DESIGN_WIDTH, DESIGN_HEIGHT)
    x, y, w, h = layout.tile_rect(1, 2)
    bx, by, bw, bh = layout.tile_rect(1, 2, scale=1.2)
    assert bw > w
    assert abs((bx + bw / 2) - (x + w / 2)) <= 1
    assert abs((by + bh / 2) - (y + h / 2)) <= 1


def test_positions_between_cells_are_interpolated() -> None:
    layout = compute_layout(DESIGN_WIDTH, DESIGN_HEIGHT)
    left = layout.cell_origin(0, 1)[0]
    right = layout.cell_origin(0, 2)[0]
    assert layout.cell_origin(0, 1.5)[0] == pytest.approx((left + right) / 2)


def test_px_scales_design_lengths() -> None:
    assert compute_layout(DESIGN_WIDTH * 2, DESIGN_HEIGHT * 2).px(10) == 20
    assert compute_layout(10, 10).px(1) == 1  # jamais moins d'un pixel
