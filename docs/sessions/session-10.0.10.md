2025-02-14 - Ajout d'un conftest pytest pour exposer le dossier src aux imports threadx lors des tests.
2025-02-14 - Consolidation du chargeur TOML (validation chemins/GPU, migration legacy, CLI) et ajustement ConfigurationError pour fiabiliser les tests config.
2025-02-14 - Correction de l'initialisation immuable des timeframes en synchronisant SUPPORTED_TIMEFRAMES et SUPPORTED_TF.
2025-02-14 - Rétablissement des erreurs de configuration (kwargs), de la validation des chemins et du parser CLI sans doublons.
- 2025-10-10 07:33 - indicators.engine: Stabilisation de l'import et ajout test fumée.
- 2025-10-10 07:44 - indicators/backtest: Stabilisation engine, cache TTL/LRU, timeouts.
- 2025-10-10 07:56 - data_manager: Gestion headless du démarrage Data Manager.
- 2025-10-10 08:07 - data_manager: Bootstrap imports et arrêt propre.
