# Release 0.5.0 - 2025-10-14

## Features

- **Architecture 3 Couches Complète** : Implémentation stricte Engine/Bridge/UI avec séparation des responsabilités
- **Bridge Asynchrone** : ThreadXBridge avec orchestration async via ThreadPoolExecutor
- **Interface Web Dash** : 4 onglets complets (Data, Indicators, Backtest, Optimization) avec Bootstrap DARKLY
- **CLI Asynchrone** : Interface ligne de commande complète avec Typer et Rich
- **Data Management** : Téléchargement Binance (Single/Top 100/Groups), validation UDFI, batch indicateurs
- **Registry avec Checksums** : Système d'idempotence avec SHA-256 pour éviter re-downloads
- **Tests Complets** : Suite pytest avec mocks pour isolation UI/Bridge
- **Documentation Exhaustive** : 5 guides (index, getting_started, user_guide, dev_guide, bridge_api)

## Fixes

- Import circulaires résolus via lazy imports dans Controllers
- Memory leaks corrigés dans gestion threads async
- Validation UDFI stricte avec timezone UTC obligatoire
- Checksums registry pour idempotence garantie

## Changes

- Refonte complète architecture : monolithique → 3 couches
- IDs Dash avec préfixes (`data-`, `indicators-`, `bt-`, `opt-`)
- Configuration centralisée dans models.py (Pydantic)
- Variables d'environnement optionnelles (THREADX_DASH_PORT, etc.)

## Dependencies

- Added: dash 2.14+
- Added: dash-bootstrap-components 1.5+
- Added: plotly 5.17+
- Added: typer 0.9+
- Added: rich 13.7+
- Added: pydantic 2.5+
- Added: numpy 1.26+
- Added: pandas 2.1+
- Added: pyarrow 14.0+

## Breaking Changes

⚠️ **ATTENTION** : Cette version introduit une architecture complètement nouvelle.

- **Import paths** : Tous les imports Engine doivent passer par Bridge
  - Avant : `from threadx.backtest.engine import BacktestEngine`
  - Après : `bridge.run_backtest_async(...)`

- **API calls** : Toutes les opérations longues sont async
  - Avant : `result = backtest_engine.run(...)`
  - Après : `task_id = bridge.run_backtest_async(...); event = bridge.get_event(task_id)`

- **UI callbacks** : Pattern polling obligatoire
  - Nécessite `dcc.Interval` pour polling événements
  - Nécessite `dcc.Store` pour task_id persistence

### Migration Guide

1. **Remplacer imports Engine par Bridge** :
   ```python
   # Avant
   from threadx.backtest.engine import BacktestEngine
   engine = BacktestEngine()
   result = engine.run(...)

   # Après
   from threadx.bridge.async_coordinator import ThreadXBridge
   bridge = ThreadXBridge.get_instance()
   task_id = bridge.run_backtest_async(...)
   event = bridge.get_event(task_id)
   ```

2. **Migrer callbacks UI** :
   ```python
   # Avant : Callback synchrone
   @callback(Output("result", "data"), Input("btn", "n_clicks"))
   def run_backtest(n_clicks):
       result = engine.run(...)
       return result

   # Après : Callback async + polling
   @callback(Output("task-id", "data"), Input("btn", "n_clicks"))
   def start_backtest(n_clicks):
       task_id = bridge.run_backtest_async(...)
       return task_id

   @callback(Output("result", "data"), Input("interval", "n_intervals"), State("task-id", "data"))
   def poll_backtest(n, task_id):
       event = bridge.get_event(task_id)
       if event["status"] == "completed":
           return event["result"]
   ```

3. **Adapter CLI** :
   ```python
   # Avant : CLI synchrone
   @app.command()
   def backtest(symbol: str):
       result = engine.run(symbol)
       print(result)

   # Après : CLI async
   @app.command()
   def backtest(symbol: str):
       task_id = bridge.run_backtest_async(symbol=symbol)
       while True:
           event = bridge.get_event(task_id)
           if event["status"] == "completed":
               print(event["result"])
               break
   ```

## Performance

- **Async Execution** : Opérations longues non-bloquantes via ThreadPoolExecutor
- **Indicator Cache** : Cache automatique avec hits/misses tracking
- **Lazy Imports** : Imports Engine retardés pour startup rapide
- **Parallel Downloads** : Téléchargements Binance parallélisés (4 workers par défaut)

## Documentation

- [Getting Started](docs/getting_started.md) : Installation et premiers pas
- [User Guide](docs/user_guide.md) : Guide utilisateur complet avec IDs Dash
- [Developer Guide](docs/dev_guide.md) : Architecture, conventions, patterns
- [Bridge API](docs/bridge_api.md) : Référence API complète avec exemples JSON

## Artefacts de Release

### Fichiers Distribués

- `ThreadX-0.5.0-py3-none-any.whl` : Wheel Python
- `ThreadX-0.5.0.tar.gz` : Source distribution
- `SHA256SUMS.txt` : Checksums SHA-256
- `ThreadX_release.zip` : Archive complète avec docs

### Checksums SHA-256

```
681df123118def4ff2a886a2071d6fcf3e385ed40023b037eb69b170106c15ab  ThreadX-0.5.0-py3-none-any.whl
323d90647b544926b7498a603441300d073db5bb09a127793b5477784b9bde13  ThreadX-0.5.0.tar.gz
```

## Installation

### Depuis GitHub Releases

```bash
# Télécharger le wheel depuis GitHub Releases
pip install ThreadX-0.5.0-py3-none-any.whl
```

### Depuis les sources

```bash
git clone https://github.com/xxxxCham/ThreadX.git
cd ThreadX
git checkout v0.5.0
pip install -e .
```

## Vérification

```bash
# Vérifier la version
python -c "import threadx; print(threadx.__version__)"

# Exécuter les tests
pytest -v

# Vérifier le linting
ruff check src

# Lancer l'interface Dash
python apps/dash_app.py

# Lancer le CLI
python -m threadx.cli --help
```

## Contributors

- @xxxxCham - Développement principal et architecture

## Liens

- **Repository** : <https://github.com/xxxxCham/ThreadX>
- **Issues** : <https://github.com/xxxxCham/ThreadX/issues>
- **Changelog** : [CHANGELOG.md](CHANGELOG.md)
- **Documentation** : [docs/index.md](docs/index.md)

## Notes

Cette version marque une refonte complète de ThreadX avec une architecture 3 couches stricte, permettant une meilleure scalabilité, testabilité et maintenabilité. Toutes les opérations longues sont désormais asynchrones pour une meilleure expérience utilisateur.

Pour toute question ou problème, veuillez ouvrir une issue sur GitHub.
