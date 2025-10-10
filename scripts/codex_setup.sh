#!/usr/bin/env bash
set -euo pipefail

# Ce script prépare l'environnement dans un conteneur (Codex / Codespaces / CI)
# 1. Crée un venv
# 2. Installe dépendances CPU (torch sans build CUDA spécifique)
# 3. Installe le package en mode editable
# 4. Affiche un résumé

PYTHON=${PYTHON:-python}
VENV_PATH=".venv"

echo "[1/6] Python: $($PYTHON --version)"

if [ ! -d "$VENV_PATH" ]; then
  echo "[2/6] Création de l'environnement virtuel";
  $PYTHON -m venv "$VENV_PATH";
else
  echo "[2/6] Environnement virtuel déjà présent";
fi

# Activer venv
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

# Remplacer les dépendances torch CUDA par CPU only si besoin
echo "[3/6] Normalisation requirements.txt (remplacement +cu121 si présent)"
TMP_REQ=$(mktemp)
sed -E 's/\+cu[0-9]+//g' requirements.txt > "$TMP_REQ"
# Optionnel : pinner torch à une version compatible

# Installer deps
echo "[4/6] Installation des dépendances"
pip install --upgrade pip
pip install -r "$TMP_REQ" || {
  echo "Tentative fallback installation sélective";
  grep -E '^torch==|^torchvision==|^torchaudio==' "$TMP_REQ" > /tmp/torch_pkgs || true
  sed -i '/^torch==/d;/^torchvision==/d;/^torchaudio==/d' "$TMP_REQ"
  pip install -r "$TMP_REQ"
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
}

# Installer le package local
echo "[5/6] Installation en mode editable"
pip install -e .

# Résumé
echo "[6/6] Résumé"
python - <<'EOF'
import platform, sys, pkgutil
print('Python:', sys.version)
print('Platform:', platform.platform())
for name in ('torch','pandas','numpy','scikit_learn','streamlit'):
  try:
    __import__(name.replace('scikit_learn','sklearn'))
    print(f'OK: {name}')
  except Exception as e:
    print(f'WARN: {name} non importable: {e}')
EOF

echo "Setup terminé avec succès."