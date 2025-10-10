Scripts de configuration pour exécution dans un environnement automatisé (Codex / CI / Codespaces).

Fichiers:

* `codex_setup.sh` : version Bash/Linux.
* `codex_setup.ps1` : version PowerShell/Windows.

Fonctionnalités:

1. Création (ou réutilisation) d'un environnement virtuel `.venv`.
2. Normalisation des dépendances PyTorch (retrait des suffixes CUDA type `+cu121`).
3. Installation des dépendances avec fallback CPU-only si nécessaire.
4. Installation du paquet local en mode editable `pip install -e .`.
5. Vérifications basiques d'import des modules clés.

Utilisation rapide:

Sous Linux/macOS: