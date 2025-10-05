"""
ThreadX Point d'entrée unifié pour l'interface Streamlit
========================================================

Centralisation du point d'entrée pour l'application Streamlit.

Cette interface unifiée remplace les précédents scripts ad-hoc
et utilise l'infrastructure centralisée de ThreadX.

Usage:
------
    python -m threadx.ui.streamlit

Author: ThreadX Team
Version: Phase A
"""

import sys
import os
from pathlib import Path

# Pour assurer que le module threadx est dans le PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    """Point d'entrée pour l'application Streamlit."""
    try:
        # Import dynamique pour éviter les dépendances circulaires
        import streamlit.web.bootstrap as bootstrap
        from streamlit.web.bootstrap import run_streamlit_script

        # Chemin vers l'application Streamlit
        streamlit_app_path = (
            Path(__file__).parent.parent.parent.parent / "apps" / "streamlit" / "app.py"
        )

        if not streamlit_app_path.exists():
            raise FileNotFoundError(
                f"Application Streamlit non trouvée: {streamlit_app_path}"
            )

        # Paramètres Streamlit par défaut
        args = [
            "--",
            str(streamlit_app_path),
            "--server.port=8504",
            "--browser.gatherUsageStats=false",
        ]

        # Lancement de Streamlit via bootstrap
        sys.argv = ["streamlit", "run"] + args
        bootstrap.run()

    except ImportError as e:
        print(f"❌ Erreur: Streamlit n'est pas installé. {e}")
        print("\n💡 Installation: pip install streamlit")
        return 1
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1
    except Exception as e:
        print(f"❌ Erreur lors du lancement de l'application Streamlit: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
