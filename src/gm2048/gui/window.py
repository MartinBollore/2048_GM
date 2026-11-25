"""La fenêtre pygame : dessine la session et lit le clavier.

C'est le seul module du paquet qui dépend de pygame. Il ne prend aucune
décision de jeu : il traduit les touches en :class:`~gm2048.gui.controllers.Command`,
fait avancer le contrôleur, puis peint ce que la session décrit.

Pour des bords bien lisses, les formes arrondies (tuiles, plateau, cartes) sont
dessinées trois fois plus grandes puis réduites ; le résultat est gardé en
cache, si bien que chaque image ne fait que recopier des tuiles déjà prêtes.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Final

import pygame

from gm2048.agents import Agent
from gm2048.engine import Action
from gm2048.gui.controllers import AgentController, Command, Controller, HumanController
from gm2048.gui.layout import DESIGN_HEIGHT, DESIGN_WIDTH, Layout, Rect, compute_layout
from gm2048.gui.session import TOAST_SECONDS, Session, Sprite, ease_out_cubic
from gm2048.gui.theme import GOLD, RGB, THEMES, Theme, next_theme

RGBA = tuple[int, int, int, int]

FPS: Final = 60
MAX_FRAME_SECONDS: Final = 0.1
"""Au-delà, un pas de temps est raccourci (fenêtre déplacée, machine occupée…)."""

_SUPERSAMPLE: Final = 3
_OVERLAY_FADE_SECONDS: Final = 0.35
_TOAST_TEXT: Final = "Tuile 2048 atteinte !"
_TOAST_INK: Final = (74, 46, 0)

# --- Clavier --------------------------------------------------------------------

_KEY_COMMANDS: Final[dict[int, Command]] = {
    pygame.K_UP: Command.UP,
    pygame.K_z: Command.UP,
    pygame.K_w: Command.UP,
    pygame.K_DOWN: Command.DOWN,
    pygame.K_s: Command.DOWN,
    pygame.K_LEFT: Command.LEFT,
    pygame.K_q: Command.LEFT,
    pygame.K_a: Command.LEFT,
    pygame.K_RIGHT: Command.RIGHT,
    pygame.K_d: Command.RIGHT,
    pygame.K_SPACE: Command.PAUSE,
    pygame.K_p: Command.PAUSE,
    pygame.K_n: Command.STEP,
    pygame.K_PLUS: Command.FASTER,
    pygame.K_KP_PLUS: Command.FASTER,
    pygame.K_EQUALS: Command.FASTER,
    pygame.K_MINUS: Command.SLOWER,
    pygame.K_KP_MINUS: Command.SLOWER,
    pygame.K_RETURN: Command.NEW_GAME,
    pygame.K_KP_ENTER: Command.NEW_GAME,
    pygame.K_r: Command.NEW_GAME,
    pygame.K_t: Command.THEME,
    pygame.K_ESCAPE: Command.QUIT,
    pygame.K_x: Command.QUIT,
}
# Le caractère tapé passe avant le code de touche : « + » et « - » ne sont pas sur
# les mêmes touches en AZERTY et en QWERTY.
_TEXT_COMMANDS: Final[dict[str, Command]] = {
    "+": Command.FASTER,
    "=": Command.FASTER,
    "-": Command.SLOWER,
}


def command_for_key(key: int, text: str = "") -> Command | None:
    """La commande associée à une touche (``key``) et au caractère tapé (``text``)."""
    return _TEXT_COMMANDS.get(text) or _KEY_COMMANDS.get(key)


# --- Petits outils de dessin ----------------------------------------------------


def rounded_surface(size: tuple[int, int], color: RGB | RGBA, radius: int) -> pygame.Surface:
    """Un rectangle arrondi aux bords lissés, sur fond transparent."""
    width, height = max(1, size[0]), max(1, size[1])
    big = pygame.Surface((width * _SUPERSAMPLE, height * _SUPERSAMPLE), pygame.SRCALPHA)
    pygame.draw.rect(big, color, big.get_rect(), border_radius=radius * _SUPERSAMPLE)
    return pygame.transform.smoothscale(big, (width, height))


def arrow_points(action: Action, cx: float, cy: float, r: float) -> list[tuple[float, float]]:
    """Les sommets d'une flèche triangulaire de rayon ``r`` pointant vers ``action``."""
    angle = math.radians(90 * int(action))  # haut, droite, bas, gauche : un quart de tour chacun
    base = [(0.0, -r), (0.85 * r, 0.55 * r), (-0.85 * r, 0.55 * r)]
    cos, sin = math.cos(angle), math.sin(angle)
    return [(cx + x * cos - y * sin, cy + x * sin + y * cos) for x, y in base]


def format_points(value: int) -> str:
    """``12345`` → ``"12 345"``."""
    return f"{value:,}".replace(",", " ")


# Polices essayées dans l'ordre : celles de Windows d'abord, puis des polices
# libres courantes sous Linux. Faute de mieux, la police fournie avec pygame.
_FONT_CHOICES: Final[dict[str, tuple[tuple[str, bool], ...]]] = {
    "heavy": (
        ("segoeuiblack", False),
        ("arialblack", False),
        ("segoeui", True),
        ("dejavusans", True),
    ),
    "bold": (("segoeui", True), ("arial", True), ("dejavusans", True)),
    "regular": (("segoeui", False), ("arial", False), ("dejavusans", False)),
}


class Renderer:
    """Peint une session dans une surface pygame, avec des caches de formes et de polices."""

    def __init__(self) -> None:
        self._font_paths: dict[str, str | None] = {}
        self._fonts: dict[tuple[str, int], pygame.font.Font] = {}
        self._shapes: dict[tuple[object, ...], pygame.Surface] = {}

    # Polices -------------------------------------------------------------------

    def font(self, role: str, size: int) -> pygame.font.Font:
        key = (role, max(1, size))
        if key not in self._fonts:
            if role not in self._font_paths:
                self._font_paths[role] = next(
                    (
                        path
                        for name, bold in _FONT_CHOICES[role]
                        if (path := pygame.font.match_font(name, bold=bold)) is not None
                    ),
                    None,
                )
            self._fonts[key] = pygame.font.Font(self._font_paths[role], key[1])
        return self._fonts[key]

    def fitted_font(self, role: str, size: int, text: str, max_width: int) -> pygame.font.Font:
        """La police ``role`` la plus grande (jusqu'à ``size``) où ``text`` tient en largeur."""
        font = self.font(role, size)
        while size > 6 and font.size(text)[0] > max_width:
            size = max(6, int(size * 0.9))
            font = self.font(role, size)
        return font

    def _blit_text(
        self,
        surface: pygame.Surface,
        text: str,
        font: pygame.font.Font,
        color: RGB,
        alpha: float = 1.0,
        **position: float | tuple[float, float],
    ) -> pygame.Rect:
        image = font.render(text, True, color)
        if alpha < 1:
            image.set_alpha(round(255 * max(0.0, alpha)))
        rect = image.get_rect(**position)
        surface.blit(image, rect)
        return rect

    # Formes en cache -----------------------------------------------------------

    def _rounded(self, size: tuple[int, int], color: RGB | RGBA, radius: int) -> pygame.Surface:
        """Comme :func:`rounded_surface`, gardé en cache (ne pas dessiner dessus : le copier)."""
        key = ("rounded", size, color, radius)
        if key not in self._shapes:
            self._shapes[key] = rounded_surface(size, color, radius)
        return self._shapes[key]

    def board_surface(self, theme: Theme, layout: Layout) -> pygame.Surface:
        """Le plateau vide : fond arrondi et cases creuses."""
        side = layout.board[2]
        key = ("board", theme.name, side)
        if key not in self._shapes:
            s = _SUPERSAMPLE
            big = pygame.Surface((side * s, side * s), pygame.SRCALPHA)
            pygame.draw.rect(big, theme.board, big.get_rect(), border_radius=round(layout.gap * s))
            cell_radius = round(layout.cell * 0.09 * s)
            for row in range(4):
                for col in range(4):
                    x, y = layout.cell_origin(row, col)
                    rect = (
                        round((x - layout.board[0]) * s),
                        round((y - layout.board[1]) * s),
                        round(layout.cell * s),
                        round(layout.cell * s),
                    )
                    pygame.draw.rect(big, theme.empty_cell, rect, border_radius=cell_radius)
            self._shapes[key] = pygame.transform.smoothscale(big, (side, side))
        return self._shapes[key]

    def tile_surface(self, exponent: int, cell: int, theme: Theme) -> pygame.Surface:
        """Une tuile prête à coller : rebord, dessus, nombre, et halo pour les grosses."""
        key = ("tile", exponent, cell, theme.name)
        if key in self._shapes:
            return self._shapes[key]
        style = theme.tile(exponent)
        s = _SUPERSAMPLE
        margin = max(2, round(cell * 0.07)) if style.glow else 0
        size = cell + 2 * margin
        lip = max(2, round(cell * 0.055))
        radius = round(cell * 0.09)
        big = pygame.Surface((size * s, size * s), pygame.SRCALPHA)
        if style.glow:
            for i, alpha in enumerate((35, 60, 95)):
                inset = round(i * margin * s / 3)
                rect = (inset, inset, size * s - 2 * inset, size * s - 2 * inset)
                pygame.draw.rect(
                    big, (*GOLD, alpha), rect, border_radius=(radius + margin) * s - inset
                )
        face = (margin * s, margin * s, cell * s, cell * s)
        pygame.draw.rect(big, style.edge, face, border_radius=radius * s)
        pygame.draw.rect(big, style.face, (*face[:3], (cell - lip) * s), border_radius=radius * s)
        image = pygame.transform.smoothscale(big, (size, size))

        text = str(1 << exponent)
        size_ratio = {1: 0.5, 2: 0.5, 3: 0.42, 4: 0.34, 5: 0.28}.get(len(text), 0.23)
        font = self.fitted_font("heavy", round(cell * size_ratio), text, round(cell * 0.86))
        self._blit_text(image, text, font, style.text, center=(size / 2, margin + (cell - lip) / 2))
        self._shapes[key] = image
        return image

    def arrow_badge(self, action: Action, size: int, theme: Theme) -> pygame.Surface:
        """Un petit carré avec une flèche : le dernier coup joué."""
        key = ("arrow", action, size, theme.name)
        if key not in self._shapes:
            s = _SUPERSAMPLE
            big = pygame.Surface((size * s, size * s), pygame.SRCALPHA)
            pygame.draw.rect(big, theme.card, big.get_rect(), border_radius=round(size * 0.25 * s))
            center = size * s / 2
            pygame.draw.polygon(
                big, theme.card_value, arrow_points(action, center, center, size * s * 0.28)
            )
            self._shapes[key] = pygame.transform.smoothscale(big, (size, size))
        return self._shapes[key]

    # L'image complète ----------------------------------------------------------

    def draw(
        self, surface: pygame.Surface, controller: Controller, theme: Theme, layout: Layout
    ) -> None:
        session = controller.session
        surface.fill(theme.background)
        self._draw_title(surface, theme, layout)
        self._draw_card(surface, "SCORE", session.game.score, layout.score_card, theme, layout)
        self._draw_card(surface, "MEILLEUR", session.best_score, layout.best_card, theme, layout)
        self._draw_popup(surface, session, theme, layout)
        self._draw_status(surface, controller, theme, layout)
        surface.blit(self.board_surface(theme, layout), layout.board[:2])
        for sprite in session.sprites():
            self._draw_sprite(surface, sprite, theme, layout)
        if session.over:
            self._draw_game_over(surface, session, theme, layout)
        if session.toast_age is not None:
            self._draw_toast(surface, session.toast_age, layout)
        self._draw_footer(surface, controller.help_lines, theme, layout)

    def _draw_title(self, surface: pygame.Surface, theme: Theme, layout: Layout) -> None:
        x, y, _, h = layout.title
        number = self._blit_text(
            surface,
            "2048",
            self.font("heavy", layout.px(52)),
            theme.accent,
            midleft=(x, y + h / 2),
        )
        self._blit_text(
            surface,
            "GM",
            self.font("heavy", layout.px(26)),
            theme.text,
            bottomleft=(number.right + layout.px(8), number.bottom - layout.px(7)),
        )

    def _draw_card(
        self,
        surface: pygame.Surface,
        label: str,
        value: int,
        rect: Rect,
        theme: Theme,
        layout: Layout,
    ) -> None:
        x, y, w, h = rect
        surface.blit(self._rounded((w, h), theme.card, layout.px(8)), (x, y))
        self._blit_text(
            surface,
            label,
            self.font("bold", layout.px(12)),
            theme.card_label,
            center=(x + w / 2, y + h * 0.27),
        )
        text = format_points(value)
        font = self.fitted_font("heavy", layout.px(26), text, w - layout.px(12))
        self._blit_text(surface, text, font, theme.card_value, center=(x + w / 2, y + h * 0.64))

    def _draw_popup(
        self, surface: pygame.Surface, session: Session, theme: Theme, layout: Layout
    ) -> None:
        popup = session.popup
        if popup is None:
            return
        x, y, w, _ = layout.score_card
        rise = ease_out_cubic(popup.progress) * layout.px(22)
        self._blit_text(
            surface,
            f"+{format_points(popup.points)}",
            self.font("heavy", layout.px(19)),
            theme.accent,
            alpha=1 - popup.progress,
            midbottom=(x + w / 2, y - layout.px(2) - rise),
        )

    def _draw_status(
        self, surface: pygame.Surface, controller: Controller, theme: Theme, layout: Layout
    ) -> None:
        x, y, w, h = layout.status
        badge = layout.px(28)
        right = x + w
        last_action = controller.session.last_action
        if last_action is not None:
            surface.blit(
                self.arrow_badge(last_action, badge, theme), (right - badge, y + (h - badge) / 2)
            )
        right -= badge + layout.px(10)
        if controller.paused:
            label = self._blit_text(
                surface,
                "EN PAUSE",
                self.font("bold", layout.px(13)),
                theme.accent,
                midright=(right, y + h / 2),
            )
            right = label.left - layout.px(10)
        text = controller.status()
        font = self.fitted_font("regular", layout.px(15), text, round(right - x))
        self._blit_text(surface, text, font, theme.muted, midleft=(x, y + h / 2))

    def _draw_sprite(
        self, surface: pygame.Surface, sprite: Sprite, theme: Theme, layout: Layout
    ) -> None:
        if sprite.scale <= 0 or sprite.alpha <= 0:
            return
        image = self.tile_surface(sprite.exponent, round(layout.cell), theme)
        if sprite.scale != 1:
            side = max(1, round(image.get_width() * sprite.scale))
            image = pygame.transform.smoothscale(image, (side, side))
        if sprite.alpha < 1:
            image = image.copy()
            image.set_alpha(round(255 * sprite.alpha))
        x, y = layout.cell_origin(sprite.row, sprite.col)
        center = (round(x + layout.cell / 2), round(y + layout.cell / 2))
        surface.blit(image, image.get_rect(center=center))

    def _draw_game_over(
        self, surface: pygame.Surface, session: Session, theme: Theme, layout: Layout
    ) -> None:
        x, y, side, _ = layout.board
        veil = self._rounded((side, side), (*theme.veil, 200), round(layout.gap)).copy()
        middle = side / 2
        self._blit_text(
            veil,
            "Partie terminée",
            self.font("heavy", layout.px(40)),
            theme.text,
            center=(middle, middle - layout.px(42)),
        )
        game = session.game
        self._blit_text(
            veil,
            f"Score {format_points(game.score)} · tuile max {format_points(game.max_tile)}",
            self.font("bold", layout.px(20)),
            theme.text,
            center=(middle, middle + layout.px(8)),
        )
        self._blit_text(
            veil,
            "Entrée ou R : nouvelle partie",
            self.font("regular", layout.px(16)),
            theme.muted,
            center=(middle, middle + layout.px(48)),
        )
        veil.set_alpha(round(255 * min(1.0, session.over_age / _OVERLAY_FADE_SECONDS)))
        surface.blit(veil, (x, y))

    def _draw_toast(self, surface: pygame.Surface, age: float, layout: Layout) -> None:
        alpha = max(0.0, min(1.0, age / 0.25, (TOAST_SECONDS - age) / 0.5))
        font = self.font("bold", layout.px(20))
        text_width, text_height = font.size(_TOAST_TEXT)
        width, height = text_width + layout.px(40), text_height + layout.px(18)
        toast = self._rounded((width, height), GOLD, height // 2).copy()
        self._blit_text(toast, _TOAST_TEXT, font, _TOAST_INK, center=(width / 2, height / 2))
        toast.set_alpha(round(255 * alpha))
        x, y, side, _ = layout.board
        surface.blit(toast, toast.get_rect(center=(x + side / 2, y + layout.px(4))))

    def _draw_footer(
        self, surface: pygame.Surface, lines: tuple[str, ...], theme: Theme, layout: Layout
    ) -> None:
        x, y, w, _ = layout.footer
        for i, line in enumerate(lines):
            font = self.fitted_font("regular", layout.px(14), line, w)
            self._blit_text(
                surface,
                line,
                font,
                theme.muted,
                center=(x + w / 2, y + layout.px(12) + i * layout.px(24)),
            )


# --- La fenêtre -------------------------------------------------------------------


class WindowApp:
    """Une image à la fois : clavier, temps, dessin. Testable sans vraie fenêtre."""

    def __init__(self, controller: Controller, theme: Theme) -> None:
        self.controller = controller
        self.theme = theme
        self.renderer = Renderer()

    def frame(
        self, surface: pygame.Surface, dt: float, events: Iterable[pygame.event.Event]
    ) -> bool:
        """Traite les événements, avance de ``dt`` secondes et dessine. Faux pour quitter."""
        for event in events:
            if event.type == pygame.QUIT:
                return False
            if event.type != pygame.KEYDOWN:
                continue
            command = command_for_key(event.key, getattr(event, "unicode", ""))
            if command is Command.QUIT:
                return False
            if command is Command.THEME:
                self.theme = next_theme(self.theme)
            elif command is not None:
                self.controller.handle(command)
        self.controller.update(dt)
        layout = compute_layout(*surface.get_size())
        self.renderer.draw(surface, self.controller, self.theme, layout)
        return True


def initial_window_size() -> tuple[int, int]:
    """Une fenêtre haute de 82 % de l'écran principal, aux proportions de la maquette."""
    try:
        desktops = pygame.display.get_desktop_sizes()
    except pygame.error:
        desktops = []
    screen_height = desktops[0][1] if desktops else 900
    height = max(520, round(screen_height * 0.82))
    return round(height * DESIGN_WIDTH / DESIGN_HEIGHT), height


def make_icon(renderer: Renderer) -> pygame.Surface:
    """L'icône de la fenêtre : une tuile dorée marquée « GM »."""
    icon = rounded_surface((64, 64), GOLD, 14)
    font = renderer.font("heavy", 30)
    image = font.render("GM", True, _TOAST_INK)
    icon.blit(image, image.get_rect(center=(32, 33)))
    return icon


def run(
    controller: Controller,
    *,
    title: str = "2048 GM",
    theme: str = "clair",
    fps: int = FPS,
    max_frames: int | None = None,
) -> None:
    """Ouvre la fenêtre et la fait vivre jusqu'à ce qu'on la ferme.

    ``max_frames`` arrête la boucle après ce nombre d'images (utile aux tests).
    """
    if theme not in THEMES:
        raise ValueError(f"Thème inconnu {theme!r}. Choix possibles : {', '.join(THEMES)}.")
    pygame.display.init()
    pygame.font.init()
    try:
        app = WindowApp(controller, THEMES[theme])
        pygame.display.set_caption(title)
        pygame.display.set_icon(make_icon(app.renderer))
        pygame.display.set_mode(initial_window_size(), pygame.RESIZABLE)
        clock = pygame.time.Clock()
        frames = 0
        while max_frames is None or frames < max_frames:
            dt = min(clock.tick(fps) / 1000, MAX_FRAME_SECONDS)
            screen = pygame.display.get_surface()
            if screen is None or not app.frame(screen, dt, pygame.event.get()):
                break
            pygame.display.flip()
            frames += 1
    finally:
        controller.close()
        pygame.quit()


def watch(
    make_agent: Callable[[int | None], Agent],
    *,
    seed: int = 0,
    delay: float = 0.2,
    theme: str = "clair",
) -> None:
    """Regarder un agent jouer dans une fenêtre animée."""
    controller = AgentController(make_agent, seed=seed, delay=delay)
    run(controller, title=f"2048 GM - {controller.agent.name}", theme=theme)


def play(*, seed: int | None = None, theme: str = "clair") -> None:
    """Jouer au clavier dans une fenêtre animée."""
    run(HumanController(seed), title="2048 GM", theme=theme)
