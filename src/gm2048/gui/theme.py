"""Les couleurs de la fenêtre : un thème clair et un thème sombre.

Les tuiles suivent un dégradé « coucher de soleil » : sable pour les petites
valeurs, puis abricot, orange, corail, rouge, framboise, prune et violet. La
tuile 2048 est dorée et brille ; au-delà, les tuiles passent à l'indigo puis à
la nuit. Les deux thèmes partagent les mêmes tuiles : seul le décor change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

RGB = tuple[int, int, int]


def shade(color: RGB, factor: float) -> RGB:
    """Assombrit (``factor`` < 1) ou éclaircit (``factor`` > 1) une couleur."""
    if factor <= 1:
        return (round(color[0] * factor), round(color[1] * factor), round(color[2] * factor))
    t = factor - 1
    return (
        round(color[0] + (255 - color[0]) * t),
        round(color[1] + (255 - color[1]) * t),
        round(color[2] + (255 - color[2]) * t),
    )


@dataclass(frozen=True, slots=True)
class TileStyle:
    """Apparence d'une tuile."""

    face: RGB
    """Couleur du dessus de la tuile."""
    text: RGB
    """Couleur du nombre."""
    glow: bool = False
    """Vrai pour un halo lumineux autour de la tuile."""

    @property
    def edge(self) -> RGB:
        """Couleur du rebord sous la tuile, qui lui donne du relief."""
        return shade(self.face, 0.78)


_DARK_TEXT: Final = (94, 72, 56)
_LIGHT_TEXT: Final = (255, 250, 243)
_GOLD: Final = (247, 190, 40)

TILES: Final[tuple[TileStyle, ...]] = (
    TileStyle((244, 233, 218), _DARK_TEXT),  # 2
    TileStyle((238, 214, 177), _DARK_TEXT),  # 4
    TileStyle((243, 164, 88), _LIGHT_TEXT),  # 8
    TileStyle((238, 133, 66), _LIGHT_TEXT),  # 16
    TileStyle((232, 104, 58), _LIGHT_TEXT),  # 32
    TileStyle((218, 78, 50), _LIGHT_TEXT),  # 64
    TileStyle((201, 58, 70), _LIGHT_TEXT),  # 128
    TileStyle((178, 46, 92), _LIGHT_TEXT),  # 256
    TileStyle((146, 42, 112), _LIGHT_TEXT),  # 512
    TileStyle((110, 46, 132), _LIGHT_TEXT),  # 1024
    TileStyle(_GOLD, (74, 46, 0), glow=True),  # 2048
    TileStyle((58, 52, 110), _LIGHT_TEXT, glow=True),  # 4096
    TileStyle((34, 40, 84), _LIGHT_TEXT, glow=True),  # 8192
    TileStyle((24, 24, 36), _GOLD, glow=True),  # 16384 et au-delà
)
"""Style des tuiles, indexé par exposant - 1 (la dernière sert aussi au-delà)."""

GOLD: Final = _GOLD
"""Couleur du bandeau de victoire et du halo des grosses tuiles."""


@dataclass(frozen=True, slots=True)
class Theme:
    """Les couleurs du décor."""

    name: str
    background: RGB
    board: RGB
    """Fond du plateau."""
    empty_cell: RGB
    text: RGB
    """Texte principal (titre, fin de partie)."""
    muted: RGB
    """Texte secondaire (ligne d'information, aide)."""
    accent: RGB
    """Couleur du titre et des points gagnés."""
    card: RGB
    """Fond des cartes de score."""
    card_label: RGB
    card_value: RGB
    veil: RGB
    """Voile posé sur le plateau en fin de partie."""

    def tile(self, exponent: int) -> TileStyle:
        """Style de la tuile d'exposant ``exponent`` (au moins 1)."""
        return TILES[min(max(exponent, 1), len(TILES)) - 1]


LIGHT: Final = Theme(
    name="clair",
    background=(248, 242, 233),
    board=(128, 101, 84),
    empty_cell=(150, 123, 104),
    text=(74, 58, 47),
    muted=(140, 120, 104),
    accent=(232, 104, 58),
    card=(128, 101, 84),
    card_label=(232, 215, 198),
    card_value=(255, 250, 243),
    veil=(248, 242, 233),
)

DARK: Final = Theme(
    name="sombre",
    background=(27, 24, 30),
    board=(45, 40, 48),
    empty_cell=(62, 55, 64),
    text=(242, 231, 219),
    muted=(160, 148, 141),
    accent=(243, 164, 88),
    card=(45, 40, 48),
    card_label=(160, 148, 141),
    card_value=(242, 231, 219),
    veil=(27, 24, 30),
)

THEMES: Final[dict[str, Theme]] = {LIGHT.name: LIGHT, DARK.name: DARK}
"""Les thèmes par nom, tels qu'on les choisit avec ``--theme``."""


def next_theme(theme: Theme) -> Theme:
    """Le thème suivant, pour passer de l'un à l'autre avec la touche T."""
    names = list(THEMES)
    return THEMES[names[(names.index(theme.name) + 1) % len(names)]]
