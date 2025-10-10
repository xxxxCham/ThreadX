"""ThreadX Data Manager - point d'entrée."""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_imports() -> None:
    """Ajoute le projet et ./src au PYTHONPATH si nécessaire."""

    project_root = Path(__file__).resolve().parents[2]
    candidates = (project_root, project_root / "src")
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


_bootstrap_imports()

from tkinter import TclError

from apps.data_manager.main_window import main


def run() -> None:
    """Lance l'interface Data Manager avec gestion propre des interruptions."""

    try:
        main()
    except KeyboardInterrupt:
        print("Arrêt manuel du Data Manager", file=sys.stderr)
    except TclError as exc:
        raise SystemExit(
            "Impossible de démarrer l'interface Tkinter : afficher requis"
        ) from exc


if __name__ == "__main__":
    run()
