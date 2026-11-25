"""Fenêtre graphique : jouer ou regarder un agent jouer, avec animations.

Le paquet est découpé pour que presque tout soit testable sans ouvrir de
fenêtre :

- :mod:`~gm2048.gui.motion` retrouve le trajet de chaque tuile pendant un coup ;
- :mod:`~gm2048.gui.session` déroule les animations dans le temps et gère la
  partie affichée ;
- :mod:`~gm2048.gui.controllers` décide quand jouer : au clavier pour un humain,
  au rythme choisi pour un agent ;
- :mod:`~gm2048.gui.theme` et :mod:`~gm2048.gui.layout` donnent les couleurs et
  la géométrie ;
- :mod:`~gm2048.gui.window` est le seul module qui dépend de pygame : il dessine
  et lit le clavier.

pygame est une dépendance optionnelle (``pip install -e ".[gui]"``) : rien
n'importe ce paquet tant qu'on n'ouvre pas de fenêtre.
"""

import os

# pygame affiche sa version à l'import : on le fait taire.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
# Sous Windows avec un affichage agrandi (125 %, 150 %…), sans cette option la
# fenêtre est étirée par le système et devient floue. Ignorée ailleurs.
os.environ.setdefault("SDL_WINDOWS_DPI_AWARENESS", "permonitorv2")
