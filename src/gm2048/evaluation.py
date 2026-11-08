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
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field

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
    seconds: float = field(default=0.0, compare=False)
    """Durée de la partie en secondes, temps de réflexion de l'agent compris.

    Ce champ est ignoré quand on compare deux résultats avec ``==`` : la même partie,
    rejouée, ne dure jamais exactement le même temps. Il vaut 0.0 par défaut, pour les
    résultats fabriqués à la main dans les tests.
    """


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

    @property
    def median_score(self) -> float:
        """Score médian : la moitié des parties fait moins, l'autre moitié fait plus.

        Avec un nombre pair de parties, c'est la moyenne des deux scores du milieu.
        La médiane résiste mieux que la moyenne aux quelques parties exceptionnelles :
        compare les deux dans le tableau du benchmark.

        Indice : le module ``statistics`` de Python a ce qu'il faut. Renvoie un ``float``.
        """

        m = statistics.median(r.score for r in self.results)
        return float(m)

    @property
    def seconds_per_move(self) -> float:
        """Temps moyen par coup, en secondes, sur l'ensemble des parties.

        C'est la durée totale de toutes les parties divisée par le nombre total de
        coups, et non la moyenne des temps par coup de chaque partie : une longue
        partie pèse plus qu'une courte, comme dans la réalité. Vaut 0.0 si aucun
        coup n'a été joué.
        """

        t_sum = sum(r.seconds for r in self.results)
        move_sum = sum(r.moves for r in self.results)
        tm = t_sum / move_sum if move_sum > 0 else 0

        return float(tm)

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
    """Fait jouer une partie complète à ``agent`` avec la graine ``seed``.

    Étape 7 : mesure aussi la durée de la partie avec ``time.perf_counter()`` et
    range-la dans le champ ``seconds`` du résultat. ``perf_counter`` renvoie un
    instant en secondes ; seule la différence entre deux appels a un sens.
    """

    game = Game(seed=seed)
    start = time.perf_counter()
    while not game.done:
        action = agent.act(game.board, game.legal_actions())
        game.step(action)

    score = game.score
    max_tile = game.max_tile
    moves = game.moves
    seconds = time.perf_counter() - start
    game_result = GameResult(
        seed=seed, score=score, max_tile=max_tile, moves=moves, seconds=seconds
    )

    return game_result


def evaluate(
    agent: Agent,
    n_games: int,
    first_seed: int = 0,
    on_game_done: Callable[[GameResult], None] | None = None,
) -> EvaluationReport:
    """Fait jouer ``n_games`` parties à ``agent``, avec les graines
    ``first_seed``, ``first_seed + 1``, …, ``first_seed + n_games - 1``.

    Le même objet agent joue toutes les parties, les unes après les autres.

    Si ``on_game_done`` est fourni, il est appelé avec le :class:`GameResult` de
    chaque partie dès qu'elle se termine. C'est ce qui permet à la ligne de
    commande d'afficher une barre de progression, sans que ce module ait à
    savoir quoi que ce soit de l'affichage.

    Lève ``ValueError`` si ``n_games`` est inférieur à 1.
    """

    if n_games < 1:
        raise ValueError("Le nombre de partie ne peut pas être nul")

    results = []
    for i in range(1, n_games + 1):
        seed = first_seed + i - 1
        result = play_game(agent=agent, seed=seed)
        results.append(result)
        if on_game_done is not None:
            on_game_done(result)

    evaluation_report = EvaluationReport(agent_name=agent.name, results=tuple(results))

    return evaluation_report


def format_report(report: EvaluationReport) -> str:
    """Met en forme un rapport d'évaluation pour l'afficher dans le terminal."""
    lines = [
        f"Agent : {report.agent_name} ({report.n_games} parties)",
        f"Score moyen : {report.mean_score:.0f}",
        f"Score médian : {report.median_score:.0f}",
        f"Parties atteignant {WIN_TILE} : {report.win_rate():.1%}",
        f"Temps par coup : {report.seconds_per_move * 1000:.2f} ms",
        "",
        "Tuile max   Parties   Part",
    ]
    for tile, count in report.tile_counts.items():
        lines.append(f"{tile:>9}   {count:>7}   {count / report.n_games:>6.1%}")
    return "\n".join(lines)
