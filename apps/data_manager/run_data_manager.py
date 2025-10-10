"""ThreadX Data Manager - point d'entrée."""

from __future__ import annotations

import sys
from pathlib import Path


  def _ensure_src_on_path() -> None:
    """Ajoute le projet et ./src au PYTHONPATH si nécessaire."""

    project_root = Path(__file__).resolve().parents[2]
    if not project_root.exists():
        raise FileNotFoundError(f"Racine du projet introuvable: {project_root}")

    for candidate in (project_root, project_root / "src"):
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


_ensure_src_on_path()

from tkinter import TclError

from apps.data_manager.main_window import main


def _run() -> None:
    try:
        main()
    except KeyboardInterrupt:
        # Arrêt propre pour éviter les traces stack inutiles en console
        print("Arrêt manuel du Data Manager", file=sys.stderr)
    except TclError:
        print(
            "Impossible de démarrer l'interface Tkinter. Vérifiez la configuration d'affichage.",
            file=sys.stderr,
        )
        return


if __name__ == "__main__":
    _run()
