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

## Jouer

```bash
gm2048 play              # nouvelle partie
gm2048 play --seed 42    # partie reproductible
```

Commandes : flèches, ZQSD (AZERTY) ou WASD (QWERTY) ; `x` ou Échap pour quitter.

Le jeu lit les touches une par une dans un vrai terminal (terminal Windows, onglet *Terminal* de PyCharm).
Dans la console d'exécution de PyCharm, qui n'est pas un vrai terminal, il passe automatiquement en
saisie ligne par ligne : taper la touche puis Entrée. L'option `--line` force ce mode.

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
