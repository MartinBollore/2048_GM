"""Évaluation des agents : les faire jouer et mesurer leurs résultats.

C'est ici que le contrat :class:`~gm2048.agents.Agent` sert enfin : les
fonctions de ce module acceptent *n'importe quel* agent, sans savoir comment il
choisit ses coups.

Les parties sont jouées à graines fixes : la partie numéro ``i`` d'une
évaluation commençant à ``first_seed`` utilise la graine ``first_seed + i``.
Deux agents évalués avec les mêmes graines affrontent donc exactement les mêmes
tirages de tuiles, ce qui rend la comparaison équitable.

À toi de jouer : remplace les ``raise NotImplementedError`` par ton code, puis
lance ``pytest tests/test_evaluation.py``. Ordre conseillé : ``play_game``,
``evaluate``, puis les trois statistiques de ``EvaluationReport``.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

from gm2048.agents import Agent
from gm2048.engine import Game

WIN_TILE = 2048
"""La tuile à atteindre pour gagner une partie."""


@dataclass(frozen=True, slots=True)
class GameResult:
    """Le bilan d'une partie terminée."""

    seed: int
    """Graine de la partie (permet de la rejouer à l'identique)."""
    score: int
    """Score final."""
    max_tile: int
    """Valeur de la plus grande tuile en fin de partie (par exemple 512)."""
    moves: int
    """Nombre de coups joués."""


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Les résultats d'un agent sur une série de parties."""

    agent_name: str
    """Nom de l'agent évalué (son attribut ``name``)."""
    results: tuple[GameResult, ...]
    """Une entrée par partie, dans l'ordre où elles ont été jouées."""

    @property
    def n_games(self) -> int:
        """Nombre de parties jouées."""
        return len(self.results)

    @property
    def mean_score(self) -> float:
        """Score moyen sur toutes les parties.

        Indice : ``np.mean`` renvoie un réel NumPy ; convertis-le avec
        ``float(...)`` pour respecter le type annoncé.
        """

        m = sum(r.score for r in self.results) / self.n_games
        return float(m)

    def win_rate(self, tile: int = WIN_TILE) -> float:
        """Proportion des parties (entre 0.0 et 1.0) où ``tile`` a été atteinte.

        Une partie compte si sa tuile max est supérieure ou égale à ``tile`` :
        une partie finie sur 4096 a forcément atteint 2048 en chemin.
        """

        nb_games_tile_won = sum(1 for r in self.results if r.max_tile >= tile)

        rate = nb_games_tile_won / self.n_games
        return rate

    @property
    def tile_counts(self) -> dict[int, int]:
        """Nombre de parties terminées sur chaque tuile max.

        Les clés sont les tuiles, rangées dans l'ordre croissant ; seules les
        tuiles effectivement obtenues apparaissent. Exemple :
        ``{128: 3, 256: 5, 512: 2}``.

        Indice : ``collections.Counter`` compte des éléments, et ``sorted``
        appliqué à ses ``items()`` les range par clé.
        """

        tile_counter = collections.Counter([x.max_tile for x in self.results])
        tile_counter_sorted = sorted(tile_counter.items())

        return dict(tile_counter_sorted)


def play_game(agent: Agent, seed: int) -> GameResult:
    """Fait jouer une partie complète à ``agent`` avec la graine ``seed``."""

    game = Game(seed=seed)
    while not game.done:
        action = agent.act(game.board, game.legal_actions())
        game.step(action)

    score = game.score
    max_tile = game.max_tile
    moves = game.moves
    game_result = GameResult(seed=seed, score=score, max_tile=max_tile, moves=moves)

    return game_result


def evaluate(agent: Agent, n_games: int, first_seed: int = 0) -> EvaluationReport:
    """Fait jouer ``n_games`` parties à ``agent``, avec les graines
    ``first_seed``, ``first_seed + 1``, …, ``first_seed + n_games - 1``.

    Le même objet agent joue toutes les parties, les unes après les autres.

    Lève ``ValueError`` si ``n_games`` est inférieur à 1.
    """

    if n_games < 1:
        raise ValueError("Le nombre de partie ne peut pas être nul")

    results = []
    for i in range(1, n_games + 1):
        seed = first_seed + i - 1
        results.append(play_game(agent=agent, seed=seed))

    evaluation_report = EvaluationReport(agent_name=agent.name, results=tuple(results))

    return evaluation_report


def format_report(report: EvaluationReport) -> str:
    """Met en forme un rapport d'évaluation pour l'afficher dans le terminal."""
    lines = [
        f"Agent : {report.agent_name} ({report.n_games} parties)",
        f"Score moyen : {report.mean_score:.0f}",
        f"Parties atteignant {WIN_TILE} : {report.win_rate():.1%}",
        "",
        "Tuile max   Parties   Part",
    ]
    for tile, count in report.tile_counts.items():
        lines.append(f"{tile:>9}   {count:>7}   {count / report.n_games:>6.1%}")
    return "\n".join(lines)
