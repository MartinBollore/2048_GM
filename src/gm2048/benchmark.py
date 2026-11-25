"""Benchmark : comparer plusieurs agents sur les mêmes parties.

``gm2048 eval`` mesure un agent. Le benchmark en mesure plusieurs, avec les mêmes
graines (donc le même plateau de départ et la même suite de tirages aléatoires),
et résume le tout dans un tableau Markdown, le format des fichiers README de
GitHub. Mesuré sur 1 000 parties, ce tableau devient la référence que les agents
qui apprennent (phases 3 et 4) devront battre.

Le module fait très peu de choses lui-même : il s'appuie sur le registre
(:func:`~gm2048.agents.registry.make_agent`) pour créer les agents, et sur
:func:`~gm2048.evaluation.evaluate` pour les faire jouer. C'est voulu : chaque
morceau du projet fait une seule chose, et les morceaux s'assemblent.

À toi de jouer : remplace les ``raise NotImplementedError`` par ton code, puis
lance ``pytest tests/test_benchmark.py``. Ordre conseillé : ``run_benchmark``,
puis ``markdown_table``. Il te faudra importer ``AGENTS``, ``DEPTH_AGENTS`` et
``make_agent`` depuis ``gm2048.agents.registry``, et ``evaluate`` depuis
``gm2048.evaluation``. Garde le nom ``make_agent`` : un test le remplace par un
espion pour vérifier ce que tu lui demandes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Final

from gm2048.agents.registry import AGENTS, DEPTH_AGENTS, make_agent
from gm2048.evaluation import EvaluationReport, GameResult, evaluate

BENCHMARK_TILES: Final[tuple[int, ...]] = (512, 1024, 2048, 4096)
"""Les tuiles dont le tableau donne le taux d'atteinte, une colonne chacune."""


def run_benchmark(
    agent_names: Sequence[str],
    n_games: int,
    first_seed: int = 0,
    depth: int | None = None,
    on_game_done: Callable[[GameResult], None] | None = None,
) -> list[EvaluationReport]:
    """Fait jouer chaque agent de ``agent_names`` sur les mêmes ``n_games`` parties.

    Renvoie un rapport par agent, dans l'ordre de ``agent_names``.

    - Chaque agent est créé avec ``make_agent(nom, seed=first_seed, ...)``, puis évalué
      avec ``evaluate`` à partir de ``first_seed`` : les chiffres sont donc les mêmes
      qu'avec ``gm2048 eval --agent nom --seed first_seed``.
    - ``depth`` n'est transmis qu'aux agents de ``DEPTH_AGENTS`` ; les autres sont
      créés sans profondeur.
    - ``on_game_done`` est transmis à ``evaluate`` : il est appelé après chaque
      partie, tous agents confondus.
    - ``ValueError`` si la liste est vide, si ``n_games`` est inférieur à 1, si un
      nom est inconnu, ou si ``depth`` est donné alors qu'aucun agent de la liste n'a
      de profondeur. Toutes ces vérifications ont lieu avant la première partie : un
      benchmark peut durer des heures, une faute de frappe doit se voir tout de suite.
    """

    if len(agent_names) == 0:
        raise ValueError("Aucun agent sélectionné")

    if n_games < 1:
        raise ValueError("Le nombre de partie doit être d'au moins 1")

    for name in agent_names:
        if name not in AGENTS:
            raise ValueError("L'agent " + name + " n'existe pas.")

    nb_agent_with_depth = 0
    for name in agent_names:
        if name in DEPTH_AGENTS:
            nb_agent_with_depth = nb_agent_with_depth + 1

    if nb_agent_with_depth == 0 and depth is not None:
        raise ValueError(
            "Une profondeur de calcul est choisie alors qu'aucun agent ne peut l'utiliser"
        )

    evaluation_reports = []
    for name in agent_names:
        if name in DEPTH_AGENTS:
            agent = make_agent(name=name, seed=first_seed, depth=depth)
        else:
            agent = make_agent(name=name, seed=first_seed)

        evaluation_reports.append(
            evaluate(agent=agent, n_games=n_games, first_seed=first_seed, on_game_done=on_game_done)
        )

    return evaluation_reports


def markdown_table(
    reports: Sequence[EvaluationReport], tiles: Sequence[int] = BENCHMARK_TILES
) -> str:
    """Met les rapports en forme dans un tableau Markdown, une ligne par rapport.

    Exemple avec ``tiles=(512, 2048)`` et deux rapports :

        | Agent | Parties | Score moyen | Score médian | Score max | ≥ 512 | ≥ 2048 | ms par coup |
        | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
        | random | 2 | 1200 | 1200 | 1400 | 0.0% | 0.0% | 0.17 |
        | heuristic | 4 | 29750 | 25000 | 60000 | 100.0% | 50.0% | 1.09 |

    Les règles :

    - Les colonnes, dans l'ordre : Agent, Parties, Score moyen, Score médian, Score max, une
      colonne ``≥ T`` par tuile ``T`` de ``tiles``, puis ms par coup.
    - La deuxième ligne règle l'alignement : ``:---`` (à gauche) pour la colonne
      Agent, ``---:`` (à droite) pour toutes les colonnes de chiffres.
    - Chaque ligne commence par ``| `` et finit par `` |``, et les cellules sont
      séparées par `` | ``.
    - Les scores sont arrondis à l'unité (``f"{x:.0f}"``). Les taux s'écrivent avec
      ``f"{x:.1%}"`` ; la colonne ``≥ T`` compte les parties dont la tuile max est
      d'au moins ``T``, comme ``win_rate``. Le temps par coup est en millisecondes,
      avec deux décimales.
    - Les lignes sont séparées par un saut de ligne, sans saut de ligne à la fin.
    - ``ValueError`` si ``reports`` est vide.

    Indice : construis chaque ligne comme une liste de cellules (des ``str``), puis
    assemble-la avec ``" | ".join(...)``.
    """
    if not reports:
        raise ValueError("il faut au moins un rapport pour faire un tableau")

    def row(cells: Sequence[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    header = [
        "Agent",
        "Parties",
        "Score moyen",
        "Score médian",
        "Score max",
        *(f"≥ {tile}" for tile in tiles),
        "ms par coup",
    ]
    alignment = [":---"] + ["---:"] * (len(header) - 1)
    lines = [row(header), row(alignment)]
    for report in reports:
        lines.append(
            row(
                [
                    report.agent_name,
                    str(report.n_games),
                    f"{report.mean_score:.0f}",
                    f"{report.median_score:.0f}",
                    str(report.max_score),
                    *(f"{report.win_rate(tile):.1%}" for tile in tiles),
                    f"{report.seconds_per_move * 1000:.2f}",
                ]
            )
        )
    return "\n".join(lines)
