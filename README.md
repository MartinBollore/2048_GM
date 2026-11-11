# 2048_GM

Ce projet a pour objectif de construire un modèle Python capable de jouer à 2048 mieux qu'un bon joueur.

L'objectif chiffré : atteindre la tuile 2048 dans au moins 90 % des parties, mesuré sur 1 000 parties à graines fixes.

## Installation

Python 3.12 ou plus récent est nécessaire.

```bash
git clone https://github.com/MartinBollore/2048_GM.git
cd 2048_GM
python -m venv .venv
```

Activer l'environnement virtuel :

- Windows : `.venv\Scripts\activate`
- Linux / macOS : `source .venv/bin/activate`

Puis installer le projet avec ses outils de développement :

```bash
pip install -e ".[dev]"
```

Cela installe aussi [pygame-ce](https://pyga.me/), qui sert à la fenêtre graphique. Pour installer
seulement le jeu et sa fenêtre, sans les outils de développement : `pip install -e ".[gui]"`.

## Jouer

```bash
gm2048 play                   # nouvelle partie, dans une fenêtre animée
gm2048 play --seed 42         # partie reproductible
gm2048 play --theme sombre    # thème sombre (la touche T change de thème en cours de partie)
```

Commandes : flèches, ZQSD (AZERTY) ou WASD (QWERTY) ; `R` ou Entrée pour une nouvelle partie ;
`T` pour changer de thème ; `x` ou Échap pour quitter. La fenêtre se redimensionne à volonté.

### Dans le terminal

```bash
gm2048 play --terminal
```

Le jeu lit alors les touches une par une dans un vrai terminal (terminal Windows, onglet *Terminal* de
PyCharm). Dans la console d'exécution de PyCharm, qui n'est pas un vrai terminal, il passe
automatiquement en saisie ligne par ligne : taper la touche puis Entrée. L'option `--line` force ce mode.
Sans pygame-ce, `gm2048 play` se rabat sur le terminal.

## Regarder un agent jouer

```bash
gm2048 watch                                  # l'agent heuristique, un coup toutes les 0,2 s
gm2048 watch --agent greedy --delay 0.5       # un autre agent, plus lentement
gm2048 watch --seed 42                        # la première partie de `gm2048 eval --seed 42`
gm2048 watch --agent expectimax --depth 1     # profondeur de recherche choisie
```

Les tuiles glissent, fusionnent et apparaissent avec des animations, dont la durée s'adapte au délai
entre deux coups. Au-dessus du plateau : l'agent, le numéro de la partie, sa graine, le nombre de coups,
le délai et une flèche qui montre le dernier coup joué.

| Touche | Effet |
| --- | --- |
| Espace | pause / reprise |
| → (ou N) | en pause : jouer un seul coup ; en pleine partie : mettre en pause |
| + / - | accélérer / ralentir (de 0 à 2 s entre deux coups) |
| Entrée ou R | partie suivante (graine suivante) |
| T | thème clair / sombre |
| Échap ou X | quitter |

L'agent choisit son coup dans un fil d'exécution à part : un agent lent (un expectimax profond, par
exemple) ne fige pas la fenêtre, et « réflexion… » s'affiche s'il fait attendre la partie.

## Évaluer un agent

```bash
gm2048 eval --agent random --games 1000   # 1 000 parties, graines 0 à 999
gm2048 eval --agent expectimax --depth 1 --games 20   # profondeur de recherche choisie
```

Affiche le score moyen, le score médian, le meilleur score, la part des parties atteignant 2048 et la répartition des tuiles max.
Pendant l'évaluation, une barre de progression indique l'avancement, le temps restant estimé,
le score moyen, le meilleur score et la meilleure tuile obtenus jusque-là. Elle s'écrit sur la sortie d'erreur :
`gm2048 eval > rapport.txt` n'enregistre que le rapport.

## Benchmark

| Agent | Parties | Score moyen | Score médian | Score max | ≥ 512 | ≥ 1024 | ≥ 2048 | ≥ 4096 | ms par coup |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 1000 | 1104 | 1056 | 4700 | 0.1% | 0.0% | 0.0% | 0.0% | 0.66 |
| greedy | 1000 | 1961 | 1636 | 5832 | 2.1% | 0.0% | 0.0% | 0.0% | 0.82 |
| heuristic | 1000 | 11545 | 11146 | 37300 | 89.4% | 56.7% | 9.2% | 0.0% | 1.11 |
| expectimax (Profondeur 1) | 1000 | 18235 | 16368 | 59236 | 97.4% | 81.4% | 31.8% | 0.7% | 4.66 |
| expectimax (Profondeur 2) | 500 | 19423 | 16260 | 60124 | 99.4% | 88.6% | 33.8% | 0.6% | 101.31 |



## Tests et qualité

```bash
pytest            # tests
ruff check .      # lint
ruff format .     # formatage
mypy              # vérification des types
```

La CI GitHub Actions relance ces vérifications sous Linux et Windows (Python 3.12 et 3.13) à chaque push
et sur chaque pull request.

## Structure

```text
src/gm2048/
├── engine/        # règles du jeu, sans aucune dépendance à l'IA
│   ├── board.py   # plateau d'exposants, glissement, fusions, coups légaux, apparition
│   ├── game.py    # Game : une partie avec score, hasard reproductible et fin de partie
│   └── render.py  # affichage texte
├── agents/        # le contrat Agent et les agents
│   └── baselines/ # agents de référence, sans apprentissage (aléatoire, …)
├── evaluation.py  # faire jouer un agent sur N parties et mesurer ses résultats
├── gui/           # fenêtre graphique (pygame-ce), pour jouer ou regarder un agent
│   ├── motion.py      # trajet de chaque tuile pendant un coup
│   ├── session.py     # animations dans le temps, score, fin de partie
│   ├── controllers.py # qui joue et quand : humain au clavier ou agent à son rythme
│   ├── theme.py       # couleurs des thèmes clair et sombre
│   ├── layout.py      # géométrie de la fenêtre
│   └── window.py      # dessin et clavier (le seul module qui importe pygame)
└── cli.py         # commande gm2048
tests/             # même découpage que src/
```

Le plateau est un tableau NumPy 4x4 d'exposants : 0 = case vide, 1 = tuile 2, 2 = tuile 4, …,
11 = tuile 2048. Le glissement (`slide`, déterministe) est séparé de l'apparition d'une tuile
(`spawn`, aléatoire), ce qui servira aux agents qui raisonnent sur le plateau juste après leur coup.

```python
from gm2048.engine import Action, Game

game = Game(seed=0)
result = game.step(Action.LEFT)
print(result.reward, game.score, game.legal_actions())
```

## Feuille de route

1. **Socle et moteur de jeu** : moteur NumPy, tests des règles, `gm2048 play` (cette étape).
2. **Environnement, baselines et évaluation** : environnement Gymnasium, agents aléatoire, glouton,
   heuristique et expectimax, benchmark sur 1 000 parties.
3. **Moteur rapide et réseau n-tuple** : moteur bitboard, boucle d'entraînement commune, agent n-tuple
   appris par TD.
4. **Deep RL** : DQN et PPO avec PyTorch, comparaison avec le n-tuple.
5. **Agent final** : expectimax guidé par le meilleur réseau appris.
