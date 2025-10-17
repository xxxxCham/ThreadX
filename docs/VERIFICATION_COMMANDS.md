# ThreadX v0.5.0 — Commandes de Vérification Post-Release

Ce document fournit toutes les commandes pour vérifier l'intégrité de la release **ThreadX v0.5.0**.

---

## 1. Vérification des Artefacts

### Lister les fichiers générés

```powershell
# Liste des artefacts dans dist/
Get-ChildItem dist | Select-Object Name, Length, LastWriteTime

# Vérifier l'archive de release
Get-Item ThreadX_release.zip | Select-Object Name, Length, LastWriteTime
```

**Attendu** :
```
Name                                Length LastWriteTime
----                                ------ -------------
threadx-0.5.0-py3-none-any.whl      315321 14/10/2025 21:03
threadx-0.5.0.tar.gz                302958 14/10/2025 21:03
SHA256SUMS.txt                        ...  14/10/2025 21:03
```

---

## 2. Vérification des Checksums

### Afficher les checksums générés

```powershell
Get-Content dist\SHA256SUMS.txt
```

**Attendu** :
```
SHA256                                                           File
------                                                           ----
681df123118def4ff2a886a2071d6fcf3e385ed40023b037eb69b170106c15ab threadx-0.5.0-py3-none-any.whl
323d90647b544926b7498a603441300d073db5bb09a127793b5477784b9bde13 threadx-0.5.0.tar.gz
```

### Re-calculer les checksums (vérification d'intégrité)

```powershell
# Calculer checksum du wheel
(Get-FileHash dist\threadx-0.5.0-py3-none-any.whl -Algorithm SHA256).Hash.ToLower()

# Calculer checksum du tarball
(Get-FileHash dist\threadx-0.5.0.tar.gz -Algorithm SHA256).Hash.ToLower()
```

**Attendu** :
```
681df123118def4ff2a886a2071d6fcf3e385ed40023b037eb69b170106c15ab
323d90647b544926b7498a603441300d073db5bb09a127793b5477784b9bde13
```

> ⚠️ **Important** : Les checksums doivent correspondre **exactement** à ceux dans `SHA256SUMS.txt`

---

## 3. Installation et Test du Package

### Créer un environnement de test propre

```powershell
# Créer un venv de test
python -m venv test_env
.\test_env\Scripts\Activate.ps1

# Installer le wheel
pip install dist\threadx-0.5.0-py3-none-any.whl

# Vérifier la version installée
python -c "import threadx; print(threadx.__version__)"
```

**Attendu** :
```
0.5.0
```

### Tester l'import des modules principaux

```powershell
python -c "from threadx.bridge import ThreadXBridge; print('Bridge OK')"
python -c "from threadx.ui.layout import create_layout; print('UI OK')"
python -c "from threadx.cli.main import app; print('CLI OK')"
```

**Attendu** :
```
Bridge OK
UI OK
CLI OK
```

### Lancer l'interface Dash (test smoke)

```powershell
# Lancer le serveur Dash
python -c "from dash import Dash; from threadx.ui.layout import create_layout; app = Dash(__name__); app.layout = create_layout(); print('http://127.0.0.1:8050'); app.run_server(debug=False)"
```

**Attendu** : Interface accessible sur `http://127.0.0.1:8050` sans erreur de démarrage

---

## 4. Vérification de la Documentation

### Compter les lignes de documentation

```powershell
# Total de lignes dans tous les docs
(Get-ChildItem docs\*.md -Recurse | Get-Content | Measure-Object -Line).Lines

# Détail par fichier
Get-ChildItem docs\*.md | ForEach-Object {
    "$($_.Name): $((Get-Content $_.FullName | Measure-Object -Line).Lines) lignes"
}
```

**Attendu** :
```
index.md: 41 lignes
getting_started.md: 152 lignes
user_guide.md: 294 lignes
dev_guide.md: 495 lignes
bridge_api.md: 640 lignes
release_notes_template.md: 119 lignes
Total: ~1741 lignes (docs/)
```

### Vérifier le contenu du CHANGELOG

```powershell
Get-Content CHANGELOG.md | Select-String "## \[0.5.0\]" -Context 5
```

**Attendu** :
```markdown
## [0.5.0] - 2025-10-14

### Added
- **Bridge Layer Asynchrone** : Module `threadx.bridge` avec 4 sous-modules
...
```

---

## 5. Tests et Linting

### Exécuter la suite de tests

```powershell
# Tests rapides
python -m pytest -q

# Tests verbeux avec détails
python -m pytest -v --tb=short

# Tests avec couverture de code
python -m pytest --cov=src/threadx --cov-report=term-missing
```

### Linter le code source

```powershell
# Vérification sans correction
python -m ruff check src

# Correction automatique (safe fixes)
python -m ruff check src --fix

# Correction aggressive (unsafe fixes)
python -m ruff check src --fix --unsafe-fixes
```

**Note** : ~1557 suggestions ruff détectées (principalement imports et types deprecated)

---

## 6. Vérification de pyproject.toml

### Valider la configuration

```powershell
# Afficher les métadonnées du projet
python -m build --help

# Tester la config sans build
python -c "import tomllib; f=open('pyproject.toml','rb'); config=tomllib.load(f); print(config['project']['version']); f.close()"
```

**Attendu** :
```
0.5.0
```

### Re-build pour vérifier la reproductibilité

```powershell
# Nettoyer dist/
Remove-Item dist\* -Force

# Re-build
python -m build

# Vérifier les checksums (doivent être identiques)
Get-FileHash dist\*.whl,dist\*.tar.gz -Algorithm SHA256
```

**Attendu** : Les mêmes checksums SHA-256 que dans `SHA256SUMS.txt`

---

## 7. Extraction et Inspection de l'Archive

### Contenu du ZIP de release

```powershell
# Lister le contenu de ThreadX_release.zip
Expand-Archive ThreadX_release.zip -DestinationPath temp_extract -Force
Get-ChildItem temp_extract -Recurse | Select-Object FullName
```

**Attendu** :
```
temp_extract/dist/threadx-0.5.0-py3-none-any.whl
temp_extract/dist/threadx-0.5.0.tar.gz
temp_extract/dist/SHA256SUMS.txt
temp_extract/CHANGELOG.md
temp_extract/RELEASE_NOTES.md
temp_extract/docs/index.md
temp_extract/docs/getting_started.md
temp_extract/docs/user_guide.md
temp_extract/docs/dev_guide.md
temp_extract/docs/bridge_api.md
temp_extract/docs/release_notes_template.md
```

### Nettoyer après extraction

```powershell
Remove-Item temp_extract -Recurse -Force
```

---

## 8. Vérification de Compatibilité Multiplateforme

### Vérifier le wheel (platform-agnostic)

```powershell
# Inspecter les métadonnées du wheel
python -m zipfile -l dist\threadx-0.5.0-py3-none-any.whl | Select-String "METADATA"
python -m zipfile -e dist\threadx-0.5.0-py3-none-any.whl temp_wheel

# Afficher les métadonnées
Get-Content temp_wheel\threadx-0.5.0.dist-info\METADATA | Select-String "Requires-Python"
```

**Attendu** :
```
Requires-Python: >=3.12
```

### Nettoyer

```powershell
Remove-Item temp_wheel -Recurse -Force
```

---

## 9. Simulation de Déploiement

### Publier sur Test PyPI (dry-run)

```powershell
# Installer twine
pip install twine

# Vérifier les artefacts avant upload
python -m twine check dist/*

# Upload sur Test PyPI (dry-run)
# python -m twine upload --repository testpypi dist/*
```

**Note** : Nécessite un compte Test PyPI et un token API

### Créer un tag Git

```powershell
# Tag local
git tag v0.5.0 -m "Release v0.5.0 - Architecture 3 Couches"

# Pousser le tag (dry-run)
# git push origin v0.5.0
```

---

## 10. Checklist Finale de Release

Avant de publier sur GitHub Releases, vérifier :

- [ ] Tous les checksums correspondent (`SHA256SUMS.txt` vs fichiers réels)
- [ ] Le wheel s'installe sans erreur dans un venv propre
- [ ] L'import des modules principaux fonctionne (`bridge`, `ui`, `cli`)
- [ ] L'interface Dash démarre sans erreur
- [ ] La documentation est complète (9 fichiers MD)
- [ ] `CHANGELOG.md` contient l'entrée v0.5.0
- [ ] `RELEASE_NOTES.md` contient les checksums corrects
- [ ] L'archive `ThreadX_release.zip` contient tous les fichiers attendus
- [ ] `pyproject.toml` version = `0.5.0`
- [ ] Tests passent (au moins les tests critiques)
- [ ] Linting ne montre aucune erreur bloquante

---

## Commandes Rapides (Résumé)

```powershell
# Vérifier les artefacts
Get-ChildItem dist

# Vérifier les checksums
Get-Content dist\SHA256SUMS.txt

# Installer et tester
python -m venv test_env
.\test_env\Scripts\Activate.ps1
pip install dist\threadx-0.5.0-py3-none-any.whl
python -c "import threadx; print(threadx.__version__)"

# Tests
python -m pytest -q

# Linting
python -m ruff check src --exit-zero

# Archive
Expand-Archive ThreadX_release.zip -DestinationPath temp_extract -Force
Get-ChildItem temp_extract -Recurse
```

---

## Support

Pour toute question ou problème :
- Documentation : `docs/getting_started.md`
- Issues : `https://github.com/xxxxCham/ThreadX/issues`
- CHANGELOG : `CHANGELOG.md`
