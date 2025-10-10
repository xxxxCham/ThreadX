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
import subprocess
from pathlib import Path

# Pour assurer que le module threadx est dans le PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    """Point d'entrée pour l'application Streamlit."""
    try:
        # Chemin vers l'application Streamlit
        streamlit_app_path = (
            Path(__file__).parent.parent.parent.parent / "apps" / "streamlit" / "app.py"
        )

        if not streamlit_app_path.exists():
            raise FileNotFoundError(
                f"Application Streamlit non trouvée: {streamlit_app_path}"
            )

        # Lancement de Streamlit via subprocess
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(streamlit_app_path),
            "--server.port=8504",
            "--browser.gatherUsageStats=false",
        ]

        print(f"🚀 Lancement de Streamlit: {streamlit_app_path}")
        result = subprocess.run(cmd, check=True)
        return result.returncode

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
