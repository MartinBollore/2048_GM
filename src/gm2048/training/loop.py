"""La boucle d'entraînement (phase 3, étape 5).

Un entraînement joue des dizaines de milliers de parties avec
``play_and_learn_numba``. Il doit montrer où il en est, sauvegarder régulièrement, et
pouvoir reprendre après une coupure exactement là où il s'était arrêté : un
entraînement coupé puis repris doit donner les mêmes poids qu'un entraînement fait
d'une traite.

C'est simple grâce aux graines : la partie numéro ``i`` (en comptant à partir de 0)
utilise la graine ``config.seed + i``. Pour reprendre, il suffit donc de recharger
les poids et de savoir combien de parties ont déjà été jouées.

Un entraînement vit dans son propre dossier, qui contient trois fichiers :

- ``weights.npz`` : les poids du réseau, écrits avec ``net.save`` ;
- ``state.json`` : où en est l'entraînement, sous la forme
  ``{"games_done": 20000, "config": {...}}``, où ``config`` contient les champs de
  :class:`TrainingConfig` (``dataclasses.asdict`` le fait pour toi). En JSON, les
  tuples deviennent des listes : reconvertis-les en tuples à la lecture ;
- ``log.csv`` : le journal, une ligne toutes les ``log_every`` parties, avec les
  colonnes de :data:`LOG_COLUMNS`.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from gm2048.agents.ntuple.network import DEFAULT_TUPLES, NTupleNetwork, Tuple
from gm2048.evaluation import WIN_TILE, GameResult
from gm2048.training.fast import play_and_learn_numba

WEIGHTS_FILE = "weights.npz"
"""Le fichier des poids, dans le dossier de l'entraînement."""
STATE_FILE = "state.json"
"""Le fichier qui dit où en est l'entraînement."""
LOG_FILE = "log.csv"
"""Le journal de l'entraînement."""

REACH_TILES = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
"""Les tuiles dont le journal suit la part de parties qui les atteignent."""

LOG_COLUMNS = (
    "games",
    "mean_score",
    "win_rate",
    "mean_moves",
    "seconds",
    "max_score",
    *(f"reach_{tile}" for tile in REACH_TILES),
)
"""Les colonnes du journal, dans l'ordre. Chaque ligne résume un bloc de parties :

- ``games`` : le nombre total de parties jouées à la fin du bloc (1000, 2000, ...) ;
- ``mean_score`` : le score moyen des parties du bloc ;
- ``win_rate`` : la part des parties du bloc qui ont atteint 2048 (entre 0 et 1,
  avec ``WIN_TILE`` de ``gm2048.evaluation``) ;
- ``mean_moves`` : le nombre moyen de coups par partie du bloc ;
- ``seconds`` : la durée totale des parties du bloc (la somme de leurs ``seconds``) ;
- ``max_score`` : le meilleur score du bloc ;
- ``reach_128`` à ``reach_32768`` : la part des parties du bloc dont la plus grande
  tuile atteint au moins cette valeur (entre 0 et 1).

Les journaux écrits avant l'ajout de ``max_score`` et des ``reach_*`` n'ont que les
cinq premières colonnes : :func:`read_log` les lit quand même, et :func:`resume`
ajoute les colonnes manquantes, vides pour les anciennes lignes.
"""


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Les réglages d'un entraînement."""

    games: int = 100_000
    """Le nombre total de parties à jouer."""
    alpha: float = 0.1
    """Le taux d'apprentissage."""
    seed: int = 0
    """La graine de la première partie : la partie ``i`` utilise ``seed + i``."""
    tuples: tuple[Tuple, ...] = DEFAULT_TUPLES
    """Les tuples du réseau."""
    checkpoint_every: int = 5_000
    """Une sauvegarde toutes les ``checkpoint_every`` parties."""
    log_every: int = 1_000
    """Une ligne de journal toutes les ``log_every`` parties."""

    def __post_init__(self) -> None:
        """Lève ``ValueError`` si un réglage n'a pas de sens.

        ``games``, ``checkpoint_every`` et ``log_every`` doivent valoir au moins 1,
        ``alpha`` doit être strictement positif, et ``checkpoint_every`` doit être un
        multiple de ``log_every`` (pour qu'une sauvegarde tombe toujours sur une ligne
        du journal).
        """
        if self.games <= 0:
            raise ValueError("games doit être un entier strictement positif")
        elif not self.alpha > 0.0:  # écrit ainsi, refuse aussi alpha = nan
            raise ValueError("alpha doit être strictement supérieur à 0")
        elif self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every doit être un entier strictement positif")
        elif self.log_every <= 0:
            raise ValueError("log_every doit être un entier strictement positif")
        elif self.checkpoint_every % self.log_every != 0:
            raise ValueError("checkpoint_every doit être un multiple de log_every")


def _save(net: NTupleNetwork, config: TrainingConfig, games_done: int, out_dir: Path) -> None:
    """Sauvegarde les poids, puis l'état (dans cet ordre)."""
    net.save(out_dir / WEIGHTS_FILE)
    state = {"games_done": games_done, "config": asdict(config)}
    (out_dir / STATE_FILE).write_text(json.dumps(state, indent=2), encoding="utf-8")


def load_state(out_dir: Path) -> tuple[TrainingConfig, int]:
    """Lit ``state.json`` : renvoie ``(config, nombre de parties déjà jouées)``."""
    state = json.loads((out_dir / STATE_FILE).read_text(encoding="utf-8"))
    raw = state["config"]
    # JSON a transformé les tuples en listes : on les reconvertit.
    raw["tuples"] = tuple(tuple(t) for t in raw["tuples"])
    return TrainingConfig(**raw), int(state["games_done"])


def _log_row(games_done: int, block: Sequence[GameResult]) -> list[float]:
    """La ligne du journal qui résume ``block``, terminé après ``games_done`` parties."""
    n = len(block)
    return [
        games_done,
        sum(r.score for r in block) / n,
        sum(r.max_tile >= WIN_TILE for r in block) / n,
        sum(r.moves for r in block) / n,
        sum(r.seconds for r in block),
        max(r.score for r in block),
        *(sum(r.max_tile >= tile for r in block) / n for tile in REACH_TILES),
    ]


def read_log(out_dir: Path) -> list[dict[str, float | None]]:
    """Lit ``log.csv`` : une ligne par bloc, sous la forme ``{colonne: valeur}``.

    Toutes les colonnes de :data:`LOG_COLUMNS` sont présentes. Une colonne absente du
    fichier (journal d'une version précédente) ou vide vaut ``None``. Un fichier qui
    n'existe pas encore donne une liste vide.
    """
    path = out_dir / LOG_FILE
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as log:
        rows = list(csv.DictReader(log))
    return [{column: _number(row.get(column)) for column in LOG_COLUMNS} for row in rows]


def _number(text: str | None) -> float | None:
    return float(text) if text else None


def _run(
    net: NTupleNetwork,
    config: TrainingConfig,
    games_done: int,
    out_dir: Path,
    on_game_done: Callable[[GameResult], None] | None,
) -> NTupleNetwork:
    """La boucle commune à ``train`` et ``resume`` : joue les parties ``games_done`` à
    ``config.games - 1``, en complétant le journal et en sauvegardant au fil de l'eau."""
    block: list[GameResult] = []  # les parties depuis la dernière ligne du journal
    with (out_dir / LOG_FILE).open("a", newline="", encoding="utf-8") as log:
        writer = csv.writer(log)
        for i in range(games_done, config.games):
            result = play_and_learn_numba(net, config.alpha, config.seed + i)
            block.append(result)
            done = i + 1
            last = done == config.games
            if done % config.log_every == 0 or last:
                writer.writerow(_log_row(done, block))
                log.flush()  # la ligne survit même à un processus tué
                block = []
            if done % config.checkpoint_every == 0 or last:
                _save(net, config, done, out_dir)
            if on_game_done is not None:
                on_game_done(result)
    return net


def train(
    config: TrainingConfig,
    out_dir: Path,
    on_game_done: Callable[[GameResult], None] | None = None,
) -> NTupleNetwork:
    """Lance un nouvel entraînement dans ``out_dir`` et renvoie le réseau entraîné.

    - Crée ``out_dir`` s'il n'existe pas (avec ses dossiers parents). Lève
      ``FileExistsError`` s'il contient déjà un ``state.json`` : on n'écrase jamais un
      entraînement par accident.
    - Avant la première partie, crée un réseau neuf avec ``config.tuples``, écrit la
      ligne d'en-tête de ``log.csv`` et fait une première sauvegarde (0 partie jouée).
      Ainsi, même un entraînement coupé tout de suite peut être repris.
    - Joue les parties ``0`` à ``config.games - 1`` avec
      ``play_and_learn_numba(net, config.alpha, config.seed + i)``, importé en haut
      de ce fichier (un test le remplace par de fausses parties sous ce nom-là).

    Après chaque partie, dans cet ordre :

    1. si le nombre de parties jouées est un multiple de ``log_every``, ou si c'est la
       dernière partie, écrit une ligne de journal pour le bloc qui se termine ;
    2. si c'est un multiple de ``checkpoint_every``, ou la dernière partie, sauvegarde
       ``weights.npz`` puis ``state.json`` ;
    3. appelle ``on_game_done(result)`` s'il est donné (pour la barre de progression).

    Quand ``on_game_done`` est appelé pour une partie, tout ce qui la concerne est
    donc déjà écrit sur le disque.
    """
    if (out_dir / STATE_FILE).exists():
        raise FileExistsError(f"{out_dir} contient déjà un entraînement")
    out_dir.mkdir(parents=True, exist_ok=True)
    net = NTupleNetwork(config.tuples)
    with (out_dir / LOG_FILE).open("w", newline="", encoding="utf-8") as log:
        csv.writer(log).writerow(LOG_COLUMNS)
    _save(net, config, 0, out_dir)
    return _run(net, config, 0, out_dir, on_game_done)


def resume(
    out_dir: Path,
    games: int | None = None,
    on_game_done: Callable[[GameResult], None] | None = None,
) -> NTupleNetwork:
    """Reprend l'entraînement enregistré dans ``out_dir`` et renvoie le réseau entraîné.

    - Lève ``FileNotFoundError`` s'il n'y a pas de ``state.json``.
    - Recharge la configuration, le nombre de parties jouées et les poids.
    - Avec ``games``, prolonge l'entraînement jusqu'à ce nouveau total. Lève
      ``ValueError`` si ``games`` est plus petit que le nombre de parties déjà jouées.
      Sauvegarde tout de suite la configuration mise à jour, avant la première
      partie : une prolongation coupée se reprend ensuite sans redonner ``games``.
    - Supprime du journal les lignes écrites après la dernière sauvegarde (leur
      ``games`` dépasse le nombre de parties sauvegardées) : ces parties vont être
      rejouées, et leurs lignes réécrites.
    - Continue ensuite exactement comme ``train`` : mêmes graines, même journal
      (complété, pas réécrit), mêmes sauvegardes.

    Un entraînement déjà terminé, repris sans ``games``, ne joue aucune partie.
    """
    if not (out_dir / STATE_FILE).exists():
        raise FileNotFoundError(f"aucun entraînement à reprendre dans {out_dir}")
    config, games_done = load_state(out_dir)
    if games is not None:
        if games < games_done:
            raise ValueError(f"l'entraînement a déjà joué {games_done} parties, pas {games}")
        config = replace(config, games=games)
    net = NTupleNetwork.load(out_dir / WEIGHTS_FILE)

    # On efface les lignes du journal écrites après la dernière sauvegarde.
    log_path = out_dir / LOG_FILE
    with log_path.open(newline="", encoding="utf-8") as log:
        header, *rows = csv.reader(log)
    kept = [row for row in rows if int(row[0]) <= games_done]
    if tuple(header) != LOG_COLUMNS:
        # Journal d'une version précédente : on ajoute les colonnes qui lui manquent,
        # vides pour les lignes déjà écrites.
        position = {column: i for i, column in enumerate(header)}
        kept = [[row[position[c]] if c in position else "" for c in LOG_COLUMNS] for row in kept]
    with log_path.open("w", newline="", encoding="utf-8") as log:
        csv.writer(log).writerows([LOG_COLUMNS, *kept])

    _save(net, config, games_done, out_dir)  # enregistre tout de suite un nouveau total
    return _run(net, config, games_done, out_dir, on_game_done)
