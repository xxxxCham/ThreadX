# 🔍 Rapport d'Audit ThreadX - Analyse Complète

**Date:** 2025-10-17 00:39:45

---

## 📊 Résumé Exécutif

- **Fichiers analysés:** 124
- **Lignes de code:** 42,087
- **Fonctions:** 1079
- **Classes:** 159
- **Duplication:** 8.9%
- **Problèmes détectés:** 990

### 🎯 Score de Qualité Global: **0.0/10**

🚨 **Qualité préoccupante** - Action immédiate requise

## 🚨 Répartition par Sévérité

| Sévérité | Nombre | Pourcentage |
|----------|--------|-------------|
| 🔴 Critical | 1 | 0.1% |
| 🟠 High | 7 | 0.7% |
| 🟡 Medium | 820 | 82.8% |
| 🟢 Low | 162 | 16.4% |

## 📁 Répartition par Catégorie

| Catégorie | Nombre | Description |
|-----------|--------|-------------|
| Logic | 19 | Erreurs logiques de trading |
| Duplication | 753 | Duplication de code |
| Structural | 216 | Problèmes structurels |
| Security | 0 | Vulnérabilités de sécurité |
| Performance | 2 | Problèmes de performance |

## 🔴 Problèmes de Sévérité CRITICAL

### 1. Erreur de syntaxe: invalid non-printable character U+FEFF

**Fichier:** `D:\ThreadX\tests\phase_a\test_udfi_contract.py:1`

**Catégorie:** structural

**Recommandation:** Corriger l'erreur de syntaxe immédiatement

---

## 🟠 Problèmes de Sévérité HIGH

### 1. Fonction avec trop de paramètres - risque d'overfitting

**Fichier:** `D:\ThreadX\src\threadx\backtest\engine.py:1`

**Catégorie:** logic

**Recommandation:** Réduire le nombre de paramètres. Utiliser walk-forward ou cross-validation

---

### 2. Backtest sans validation out-of-sample apparente

**Fichier:** `D:\ThreadX\src\threadx\backtest\engine.py:1`

**Catégorie:** logic

**Recommandation:** Implémenter train/test split ou walk-forward validation

---

### 3. Fonction avec trop de paramètres - risque d'overfitting

**Fichier:** `D:\ThreadX\src\threadx\backtest\performance.py:1`

**Catégorie:** logic

**Recommandation:** Réduire le nombre de paramètres. Utiliser walk-forward ou cross-validation

---

### 4. Backtest sans validation out-of-sample apparente

**Fichier:** `D:\ThreadX\src\threadx\backtest\performance.py:1`

**Catégorie:** logic

**Recommandation:** Implémenter train/test split ou walk-forward validation

---

### 5. Fonction avec trop de paramètres - risque d'overfitting

**Fichier:** `D:\ThreadX\src\threadx\backtest\sweep.py:1`

**Catégorie:** logic

**Recommandation:** Réduire le nombre de paramètres. Utiliser walk-forward ou cross-validation

---

### 6. Backtest sans validation out-of-sample apparente

**Fichier:** `D:\ThreadX\src\threadx\backtest\sweep.py:1`

**Catégorie:** logic

**Recommandation:** Implémenter train/test split ou walk-forward validation

---

### 7. Backtest sans validation out-of-sample apparente

**Fichier:** `D:\ThreadX\src\threadx\backtest\__init__.py:1`

**Catégorie:** logic

**Recommandation:** Implémenter train/test split ou walk-forward validation

---

## 🟡 Problèmes de Sévérité MEDIUM

### 1. Fonction '_validate_inputs' trop complexe (complexité: 11)

**Fichier:** `D:\ThreadX\src\threadx\backtest\engine.py:424`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 2. Fonction '_generate_trading_signals' trop complexe (complexité: 18)

**Fichier:** `D:\ThreadX\src\threadx\backtest\engine.py:464`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 3. Fonction '_simulate_trades' trop complexe (complexité: 17)

**Fichier:** `D:\ThreadX\src\threadx\backtest\engine.py:577`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 4. Fonction 'summarize' trop complexe (complexité: 13)

**Fichier:** `D:\ThreadX\src\threadx\backtest\performance.py:818`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 5. Fonction 'run_grid' trop complexe (complexité: 20)

**Fichier:** `D:\ThreadX\src\threadx\backtest\sweep.py:315`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 6. Fonction 'benchmark_multi_gpu_balance' trop complexe (complexité: 11)

**Fichier:** `D:\ThreadX\src\threadx\benchmarks\run_auto_profile.py:35`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 7. Fonction 'benchmark_dynamic_decision' trop complexe (complexité: 20)

**Fichier:** `D:\ThreadX\src\threadx\benchmarks\run_auto_profile.py:173`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 8. Fonction 'validate_data' trop complexe (complexité: 13)

**Fichier:** `D:\ThreadX\src\threadx\bridge\controllers.py:446`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 9. Fonction 'ingest_binance' trop complexe (complexité: 11)

**Fichier:** `D:\ThreadX\src\threadx\data\ingest.py:691`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 10. Fonction 'download_ohlcv_1m' trop complexe (complexité: 12)

**Fichier:** `D:\ThreadX\src\threadx\data\ingest.py:88`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 11. Fonction 'verify_resample_consistency' trop complexe (complexité: 20)

**Fichier:** `D:\ThreadX\src\threadx\data\ingest.py:243`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 12. Fonction '_calculate_missing_ranges' trop complexe (complexité: 11)

**Fichier:** `D:\ThreadX\src\threadx\data\ingest.py:477`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 13. Fonction '_process_symbol_complete' trop complexe (complexité: 12)

**Fichier:** `D:\ThreadX\src\threadx\data\ingest.py:557`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 14. Fonction 'normalize_ohlcv' trop complexe (complexité: 11)

**Fichier:** `D:\ThreadX\src\threadx\data\io.py:182`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 15. Fonction 'read_frame' trop complexe (complexité: 15)

**Fichier:** `D:\ThreadX\src\threadx\data\io.py:300`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 16. Fonction 'write_frame' trop complexe (complexité: 11)

**Fichier:** `D:\ThreadX\src\threadx\data\io.py:373`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 17. Fonction 'download_ohlcv' trop complexe (complexité: 13)

**Fichier:** `D:\ThreadX\src\threadx\data\loader.py:171`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 18. Fonction 'resample_from_1m' trop complexe (complexité: 13)

**Fichier:** `D:\ThreadX\src\threadx\data\resample.py:180`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 19. Fonction 'make_synth_ohlcv' trop complexe (complexité: 11)

**Fichier:** `D:\ThreadX\src\threadx\data\synth.py:49`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 20. Fonction 'merge_and_rank_tokens' trop complexe (complexité: 12)

**Fichier:** `D:\ThreadX\src\threadx\data\tokens.py:215`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 21. Fonction 'fetch_ohlcv' trop complexe (complexité: 17)

**Fichier:** `D:\ThreadX\src\threadx\data\tokens.py:471`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 22. Fonction 'assert_udfi' trop complexe (complexité: 19)

**Fichier:** `D:\ThreadX\src\threadx\data\udfi_contract.py:113`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 23. Fonction 'run_unified_diversity' trop complexe (complexité: 16)

**Fichier:** `D:\ThreadX\src\threadx\data\unified_diversity_pipeline.py:397`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 24. Fonction '_compute_diversity_metrics' trop complexe (complexité: 14)

**Fichier:** `D:\ThreadX\src\threadx\data\unified_diversity_pipeline.py:601`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 25. Fonction 'run_diversity_mode' trop complexe (complexité: 15)

**Fichier:** `D:\ThreadX\src\threadx\data\unified_diversity_pipeline.py:743`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 26. Fonction 'validate_bank_integrity' trop complexe (complexité: 11)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bank.py:1275`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 27. Fonction 'compute' trop complexe (complexité: 13)

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:252`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 28. Fonction '_extract_unique_indicators' trop complexe (complexité: 13)

**Fichier:** `D:\ThreadX\src\threadx\optimization\engine.py:261`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 29. Classe 'ParametricOptimizationUI' trop grande (28 méthodes)

**Fichier:** `D:\ThreadX\src\threadx\optimization\ui.py:29`

**Catégorie:** structural

**Recommandation:** Considérer de diviser en classes plus petites

---

### 30. Fonction '__post_init__' trop complexe (complexité: 12)

**Fichier:** `D:\ThreadX\src\threadx\strategy\bb_atr.py:115`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 31. Fonction 'generate_signals' trop complexe (complexité: 12)

**Fichier:** `D:\ThreadX\src\threadx\strategy\bb_atr.py:327`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 32. Fonction 'backtest' trop complexe (complexité: 26)

**Fichier:** `D:\ThreadX\src\threadx\strategy\bb_atr.py:453`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 33. Fonction 'generate_signals_batch_gpu' trop complexe (complexité: 11)

**Fichier:** `D:\ThreadX\src\threadx\strategy\gpu_examples.py:123`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 34. Fonction 'backtest_monte_carlo_gpu' trop complexe (complexité: 12)

**Fichier:** `D:\ThreadX\src\threadx\strategy\gpu_examples.py:265`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 35. Fonction 'from_trades_and_equity' trop complexe (complexité: 20)

**Fichier:** `D:\ThreadX\src\threadx\strategy\model.py:440`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 36. Fonction 'register_callbacks' trop complexe (complexité: 77)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:62`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 37. Fonction 'poll_bridge_events' trop complexe (complexité: 22)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:140`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 38. Fonction 'submit_optimization_sweep' trop complexe (complexité: 11)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:594`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 39. Fonction 'download_and_validate_data' trop complexe (complexité: 16)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:732`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 40. Fonction 'plot_drawdown' trop complexe (complexité: 16)

**Fichier:** `D:\ThreadX\src\threadx\ui\charts.py:202`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 41. Fonction 'download_worker' trop complexe (complexité: 12)

**Fichier:** `D:\ThreadX\src\threadx\ui\downloads.py:339`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 42. Classe 'SweepOptimizationPage' trop grande (25 méthodes)

**Fichier:** `D:\ThreadX\src\threadx\ui\sweep.py:40`

**Catégorie:** structural

**Recommandation:** Considérer de diviser en classes plus petites

---

### 43. Fonction '_apply_config_to_ui' trop complexe (complexité: 13)

**Fichier:** `D:\ThreadX\src\threadx\ui\sweep.py:771`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 44. Fonction 'check_queues' trop complexe (complexité: 11)

**Fichier:** `D:\ThreadX\src\threadx\ui\sweep.py:856`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 45. Fonction 'render_trades_table' trop complexe (complexité: 12)

**Fichier:** `D:\ThreadX\src\threadx\ui\tables.py:43`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 46. Fonction 'render_metrics_table' trop complexe (complexité: 11)

**Fichier:** `D:\ThreadX\src\threadx\ui\tables.py:173`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 47. Fonction 'batch_generator' trop complexe (complexité: 16)

**Fichier:** `D:\ThreadX\src\threadx\utils\batching.py:82`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 48. Fonction 'batch_process' trop complexe (complexité: 14)

**Fichier:** `D:\ThreadX\src\threadx\utils\batching.py:288`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 49. Fonction 'cached' trop complexe (complexité: 23)

**Fichier:** `D:\ThreadX\src\threadx\utils\cache.py:528`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 50. Fonction 'decorator' trop complexe (complexité: 16)

**Fichier:** `D:\ThreadX\src\threadx\utils\cache.py:645`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 51. Fonction 'measure_throughput' trop complexe (complexité: 11)

**Fichier:** `D:\ThreadX\src\threadx\utils\timing.py:196`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 52. Fonction 'decorator' trop complexe (complexité: 11)

**Fichier:** `D:\ThreadX\src\threadx\utils\timing.py:234`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 53. Fonction 'wrapper' trop complexe (complexité: 11)

**Fichier:** `D:\ThreadX\src\threadx\utils\timing.py:236`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 54. Fonction 'run' trop complexe (complexité: 15)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\backtest_cmd.py:32`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 55. Fonction 'validate' trop complexe (complexité: 12)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\data_cmd.py:33`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 56. Fonction 'build' trop complexe (complexité: 14)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\indicators_cmd.py:32`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 57. Fonction 'sweep' trop complexe (complexité: 13)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\optimize_cmd.py:32`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 58. Fonction '_parse_gpu_name' trop complexe (complexité: 13)

**Fichier:** `D:\ThreadX\src\threadx\utils\gpu\device_manager.py:80`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 59. Fonction 'xp' trop complexe (complexité: 11)

**Fichier:** `D:\ThreadX\src\threadx\utils\gpu\device_manager.py:294`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 60. Fonction '_compute_chunk' trop complexe (complexité: 16)

**Fichier:** `D:\ThreadX\src\threadx\utils\gpu\multi_gpu.py:381`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 61. Fonction 'validate' trop complexe (complexité: 34)

**Fichier:** `D:\ThreadX\src\threadx\utils\gpu\vector_checks.py:98`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 62. Fonction 'check_forbidden_operations' trop complexe (complexité: 11)

**Fichier:** `D:\ThreadX\tests\test_architecture_separation.py:120`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 63. Fonction 'test_no_io_in_ui_modules' trop complexe (complexité: 15)

**Fichier:** `D:\ThreadX\tests\test_callbacks_contracts.py:77`

**Catégorie:** structural

**Recommandation:** Refactoriser en fonctions plus petites. Cible: <10

---

### 64. Chargement de données sans vérification de valeurs manquantes

**Fichier:** `D:\ThreadX\src\threadx\data\ingest.py:1`

**Catégorie:** logic

**Recommandation:** Ajouter: assert not df.isnull().any().any() ou df.fillna()

---

### 65. Chargement de données sans vérification de valeurs manquantes

**Fichier:** `D:\ThreadX\src\threadx\data\legacy_adapter.py:1`

**Catégorie:** logic

**Recommandation:** Ajouter: assert not df.isnull().any().any() ou df.fillna()

---

### 66. Chargement de données sans vérification de valeurs manquantes

**Fichier:** `D:\ThreadX\src\threadx\data\synth.py:1`

**Catégorie:** logic

**Recommandation:** Ajouter: assert not df.isnull().any().any() ou df.fillna()

---

### 67. Chargement de données sans vérification de valeurs manquantes

**Fichier:** `D:\ThreadX\src\threadx\data\tokens.py:1`

**Catégorie:** logic

**Recommandation:** Ajouter: assert not df.isnull().any().any() ou df.fillna()

---

### 68. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\engine.py:268`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 69. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\engine.py:956`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 70. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\engine.py:269`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 71. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\engine.py:957`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 72. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\engine.py:270`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 73. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\engine.py:958`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 74. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\engine.py:271`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 75. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\engine.py:959`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 76. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\performance.py:405`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 77. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\performance.py:496`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 78. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\performance.py:406`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 79. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\performance.py:497`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 80. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\performance.py:407`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 81. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\performance.py:498`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 82. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\performance.py:707`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 83. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\performance.py:779`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 84. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\performance.py:904`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 85. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\sweep.py:227`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 86. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\performance.py:905`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 87. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\sweep.py:228`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 88. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\performance.py:906`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 89. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\sweep.py:229`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 90. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\performance.py:907`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 91. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\sweep.py:230`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 92. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\performance.py:908`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 93. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\sweep.py:231`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 94. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\performance.py:909`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 95. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\sweep.py:232`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 96. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\performance.py:910`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 97. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\sweep.py:233`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 98. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\performance.py:911`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 99. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\sweep.py:234`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 100. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\performance.py:912`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 101. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\sweep.py:235`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 102. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\sweep.py:205`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 103. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\sweep.py:449`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 104. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\sweep.py:206`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 105. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\sweep.py:450`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 106. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\sweep.py:207`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 107. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\sweep.py:451`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 108. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\sweep.py:208`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 109. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\sweep.py:452`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 110. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\sweep.py:209`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 111. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\sweep.py:453`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 112. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\sweep.py:210`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 113. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\backtest\sweep.py:454`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 114. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\benchmarks\run_auto_profile.py:89`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 115. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\benchmarks\run_auto_profile.py:121`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 116. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\benchmarks\run_auto_profile.py:90`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 117. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\benchmarks\run_auto_profile.py:122`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 118. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\benchmarks\run_auto_profile.py:252`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 119. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\benchmarks\run_auto_profile.py:311`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 120. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\benchmarks\run_auto_profile.py:365`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 121. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\benchmarks\run_auto_profile.py:258`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 122. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\benchmarks\run_auto_profile.py:317`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 123. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\benchmarks\run_auto_profile.py:371`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 124. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\benchmarks\run_auto_profile.py:259`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 125. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\benchmarks\run_auto_profile.py:318`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 126. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\benchmarks\run_auto_profile.py:372`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 127. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\benchmarks\run_auto_profile.py:260`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 128. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\benchmarks\run_auto_profile.py:319`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 129. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\benchmarks\run_auto_profile.py:373`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 130. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\benchmarks\run_auto_profile.py:261`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 131. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\benchmarks\run_auto_profile.py:320`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 132. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\benchmarks\run_auto_profile.py:374`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 133. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\benchmarks\run_auto_profile.py:262`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 134. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\benchmarks\run_auto_profile.py:321`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 135. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\benchmarks\run_auto_profile.py:375`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 136. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\benchmarks\run_auto_profile.py:263`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 137. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\benchmarks\run_auto_profile.py:322`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 138. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\benchmarks\run_auto_profile.py:376`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 139. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\async_coordinator.py:51`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 140. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\__init__.py:72`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 141. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\async_coordinator.py:58`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 142. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\__init__.py:59`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 143. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\async_coordinator.py:59`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 144. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\__init__.py:60`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 145. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\async_coordinator.py:60`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 146. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\__init__.py:61`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 147. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\async_coordinator.py:61`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 148. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\__init__.py:62`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 149. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\async_coordinator.py:62`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 150. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\__init__.py:63`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 151. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\async_coordinator.py:63`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 152. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\__init__.py:64`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 153. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\async_coordinator.py:64`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 154. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\__init__.py:65`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 155. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\async_coordinator.py:654`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 156. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\async_coordinator.py:698`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 157. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\async_coordinator.py:744`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 158. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\controllers.py:12`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 159. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\models.py:10`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 160. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\__init__.py:22`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 161. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\controllers.py:13`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 162. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\models.py:11`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 163. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\__init__.py:23`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 164. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\controllers.py:153`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 165. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\__init__.py:25`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 166. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\controllers.py:154`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 167. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\__init__.py:26`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 168. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\validation.py:18`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 169. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\validation.py:43`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 170. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\validation.py:66`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 171. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\validation.py:95`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 172. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\validation.py:19`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 173. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\validation.py:44`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 174. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\validation.py:67`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 175. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\validation.py:96`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 176. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\validation.py:20`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 177. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\bridge\validation.py:68`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 178. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\config\loaders.py:362`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 179. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\config\__init__.py:15`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 180. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\config\loaders.py:363`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 181. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\config\__init__.py:16`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 182. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\config\loaders.py:364`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 183. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\config\__init__.py:17`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 184. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\config\loaders.py:365`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 185. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\config\__init__.py:18`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 186. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\config\settings.py:46`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 187. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_exception_handling.py:88`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 188. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\config\settings.py:47`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 189. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_exception_handling.py:89`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 190. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\config\settings.py:48`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 191. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_exception_handling.py:90`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 192. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\data\legacy_adapter.py:103`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 193. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\data\loader.py:99`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 194. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\data\legacy_adapter.py:104`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 195. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\data\loader.py:100`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 196. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bank.py:381`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 197. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\testing\mocks.py:59`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 198. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bank.py:382`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 199. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bank.py:934`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 200. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\testing\mocks.py:60`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 201. Bloc de code dupliqué (5 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bank.py:383`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 202. Bloc de code dupliqué (5 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bank.py:935`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 203. Bloc de code dupliqué (5 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bank.py:1125`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 204. Bloc de code dupliqué (5 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bank.py:1186`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 205. Bloc de code dupliqué (5 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\testing\mocks.py:61`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 206. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bank.py:384`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 207. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bank.py:936`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 208. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\testing\mocks.py:62`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 209. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bank.py:385`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 210. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bank.py:937`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 211. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bank.py:534`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 212. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bank.py:1217`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 213. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bank.py:584`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 214. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bank.py:618`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 215. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bank.py:585`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 216. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bank.py:619`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 217. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bank.py:586`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 218. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bank.py:620`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 219. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bank.py:1126`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 220. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bank.py:1187`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 221. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bank.py:1127`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 222. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bank.py:1188`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 223. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bank.py:1128`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 224. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bank.py:1189`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 225. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bank.py:1476`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 226. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bank.py:1495`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 227. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bank.py:1477`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 228. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bank.py:1496`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 229. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bank.py:1478`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 230. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bank.py:1497`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 231. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bollinger.py:36`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 232. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bollinger.py:560`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 233. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bollinger.py:63`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 234. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:64`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 235. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bollinger.py:64`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 236. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:65`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 237. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bollinger.py:193`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 238. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:209`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 239. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bollinger.py:194`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 240. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:210`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 241. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bollinger.py:195`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 242. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:211`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 243. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bollinger.py:196`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 244. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:212`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 245. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bollinger.py:283`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 246. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:318`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 247. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bollinger.py:284`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 248. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:319`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 249. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bollinger.py:398`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 250. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:479`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 251. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bollinger.py:416`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 252. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bollinger.py:485`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 253. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bollinger.py:630`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 254. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:737`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 255. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bollinger.py:631`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 256. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:738`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 257. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bollinger.py:632`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 258. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:739`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 259. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bollinger.py:633`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 260. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:740`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 261. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\bollinger.py:634`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 262. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:741`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 263. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:37`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 264. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:675`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 265. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:334`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 266. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:405`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 267. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:335`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 268. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:406`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 269. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:336`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 270. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:407`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 271. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:447`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 272. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:517`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 273. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:448`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 274. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:518`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 275. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\optimization\engine.py:989`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 276. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\optimization\ui.py:443`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 277. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\optimization\engine.py:990`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 278. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\optimization\ui.py:444`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 279. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\optimization\ui.py:693`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 280. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\sweep.py:794`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 281. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\strategy\bb_atr.py:586`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 282. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\strategy\bb_atr.py:594`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 283. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:343`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 284. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:434`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 285. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:344`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 286. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:435`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 287. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:524`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 288. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:628`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 289. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:345`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 290. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:436`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 291. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:346`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 292. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:437`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 293. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:347`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 294. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:438`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 295. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:348`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 296. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:439`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 297. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:349`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 298. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:440`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 299. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:382`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 300. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:473`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 301. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:562`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 302. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:677`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 303. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:383`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 304. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:474`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 305. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:563`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 306. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:678`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 307. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:523`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 308. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:627`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 309. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:525`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 310. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:629`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 311. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:526`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 312. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:630`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 313. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:527`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 314. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:631`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 315. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:528`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 316. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:632`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 317. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:529`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 318. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:633`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 319. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:778`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 320. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:794`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 321. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:925`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 322. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:779`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 323. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:795`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 324. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:807`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 325. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:844`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 326. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:926`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 327. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:938`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 328. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:780`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 329. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:796`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 330. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:808`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 331. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:845`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 332. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:927`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 333. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:939`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 334. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:806`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 335. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:843`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 336. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:937`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 337. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:935`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 338. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\callbacks.py:1039`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 339. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\charts.py:135`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 340. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\charts.py:304`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 341. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\charts.py:136`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 342. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\charts.py:305`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 343. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\charts.py:137`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 344. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\charts.py:306`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 345. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\charts.py:138`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 346. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\charts.py:307`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 347. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\charts.py:180`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 348. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\charts.py:368`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 349. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\charts.py:625`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 350. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\charts.py:181`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 351. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\charts.py:369`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 352. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\charts.py:626`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 353. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\charts.py:182`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 354. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\charts.py:370`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 355. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\charts.py:627`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 356. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\charts.py:377`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 357. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\charts.py:634`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 358. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\data_manager.py:14`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 359. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\downloads.py:16`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 360. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\data_manager.py:15`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 361. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\downloads.py:17`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 362. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\data_manager.py:100`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 363. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\downloads.py:126`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 364. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\data_manager.py:101`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 365. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\downloads.py:127`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 366. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\data_manager.py:102`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 367. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\downloads.py:128`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 368. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\data_manager.py:103`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 369. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\downloads.py:129`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 370. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\data_manager.py:104`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 371. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\downloads.py:130`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 372. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\data_manager.py:105`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 373. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\downloads.py:131`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 374. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\data_manager.py:106`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 375. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\downloads.py:132`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 376. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\downloads.py:257`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 377. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\sweep.py:409`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 378. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\downloads.py:426`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 379. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\downloads.py:449`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 380. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\downloads.py:537`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 381. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\sweep.py:890`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 382. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\downloads.py:538`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 383. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\sweep.py:891`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 384. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\downloads.py:539`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 385. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\sweep.py:892`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 386. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\layout.py:152`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 387. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\layout.py:177`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 388. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\layout.py:202`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 389. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\layout.py:153`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 390. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\layout.py:178`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 391. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\layout.py:203`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 392. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\layout.py:154`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 393. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\layout.py:179`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 394. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\layout.py:204`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 395. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\layout.py:275`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 396. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\layout.py:311`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 397. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\layout.py:281`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 398. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\layout.py:317`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 399. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\layout.py:291`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 400. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\layout.py:330`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 401. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\layout.py:292`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 402. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\layout.py:331`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 403. Bloc de code dupliqué (7 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\layout.py:293`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 404. Bloc de code dupliqué (7 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\layout.py:332`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 405. Bloc de code dupliqué (7 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:198`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 406. Bloc de code dupliqué (7 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:241`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 407. Bloc de code dupliqué (7 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:287`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 408. Bloc de code dupliqué (7 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:261`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 409. Bloc de code dupliqué (7 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:293`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 410. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\layout.py:294`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 411. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\layout.py:333`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 412. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\layout.py:295`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 413. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\layout.py:334`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 414. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\layout.py:296`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 415. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\layout.py:335`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 416. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\tables.py:261`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 417. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\tables.py:308`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 418. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\tables.py:262`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 419. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\tables.py:309`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 420. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\utils\batching.py:114`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 421. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\utils\batching.py:323`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 422. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\utils\cache.py:149`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 423. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\utils\cache.py:304`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 424. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\utils\cache.py:156`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 425. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\utils\cache.py:311`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 426. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\utils\cache.py:184`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 427. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\utils\cache.py:352`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 428. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\utils\cache.py:185`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 429. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\utils\cache.py:353`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 430. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\utils\cache.py:186`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 431. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\utils\cache.py:354`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 432. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\utils\cache.py:215`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 433. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\utils\cache.py:380`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 434. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\utils\cache.py:237`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 435. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\utils\cache.py:425`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 436. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\utils\cache.py:238`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 437. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\utils\cache.py:426`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 438. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\utils\timing.py:284`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 439. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\utils\timing.py:499`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 440. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\utils\timing.py:285`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 441. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\utils\timing.py:500`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 442. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\backtest_cmd.py:19`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 443. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\data_cmd.py:20`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 444. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\indicators_cmd.py:19`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 445. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\optimize_cmd.py:19`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 446. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\backtest_cmd.py:20`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 447. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\data_cmd.py:21`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 448. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\indicators_cmd.py:20`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 449. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\optimize_cmd.py:20`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 450. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\backtest_cmd.py:21`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 451. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\data_cmd.py:22`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 452. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\indicators_cmd.py:21`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 453. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\optimize_cmd.py:21`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 454. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\backtest_cmd.py:58`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 455. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\data_cmd.py:48`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 456. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\data_cmd.py:146`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 457. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\indicators_cmd.py:59`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 458. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\indicators_cmd.py:161`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 459. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\optimize_cmd.py:64`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 460. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\backtest_cmd.py:59`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 461. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\data_cmd.py:49`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 462. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\data_cmd.py:147`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 463. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\indicators_cmd.py:60`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 464. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\indicators_cmd.py:162`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 465. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\optimize_cmd.py:65`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 466. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\backtest_cmd.py:60`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 467. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\data_cmd.py:50`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 468. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\data_cmd.py:148`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 469. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\indicators_cmd.py:61`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 470. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\indicators_cmd.py:163`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 471. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\optimize_cmd.py:66`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 472. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\backtest_cmd.py:107`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 473. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\data_cmd.py:95`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 474. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\indicators_cmd.py:106`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 475. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\cli\commands\optimize_cmd.py:123`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 476. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:58`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 477. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:58`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 478. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:59`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 479. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:59`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 480. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:80`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 481. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\data_manager.py:125`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 482. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\indicators_panel.py:51`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 483. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:80`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 484. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:81`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 485. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\data_manager.py:126`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 486. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\indicators_panel.py:52`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 487. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:81`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 488. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:82`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 489. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\data_manager.py:127`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 490. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\indicators_panel.py:53`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 491. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:82`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 492. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:83`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 493. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\data_manager.py:128`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 494. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\indicators_panel.py:54`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 495. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:83`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 496. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:84`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 497. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\data_manager.py:129`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 498. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\indicators_panel.py:55`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 499. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:84`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 500. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:85`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 501. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\data_manager.py:130`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 502. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\indicators_panel.py:56`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 503. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:85`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 504. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:86`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 505. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\indicators_panel.py:57`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 506. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:86`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 507. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:87`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 508. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\indicators_panel.py:58`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 509. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:87`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 510. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:88`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 511. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:88`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 512. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:107`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 513. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\indicators_panel.py:100`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 514. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:145`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 515. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:214`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 516. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:146`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 517. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:215`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 518. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:175`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 519. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:193`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 520. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:176`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 521. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:194`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 522. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:177`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 523. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:195`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 524. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:178`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 525. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:196`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 526. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:179`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 527. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:197`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 528. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:292`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 529. Bloc de code dupliqué (22 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:199`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 530. Bloc de code dupliqué (22 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:201`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 531. Bloc de code dupliqué (22 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:242`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 532. Bloc de code dupliqué (22 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:244`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 533. Bloc de code dupliqué (22 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:246`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 534. Bloc de code dupliqué (22 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:288`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 535. Bloc de code dupliqué (22 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:290`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 536. Bloc de code dupliqué (22 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:292`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 537. Bloc de code dupliqué (22 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:294`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 538. Bloc de code dupliqué (22 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\data_manager.py:170`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 539. Bloc de code dupliqué (22 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\data_manager.py:298`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 540. Bloc de code dupliqué (22 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\data_manager.py:300`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 541. Bloc de code dupliqué (22 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\indicators_panel.py:164`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 542. Bloc de code dupliqué (22 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\indicators_panel.py:182`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 543. Bloc de code dupliqué (22 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:139`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 544. Bloc de code dupliqué (22 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:191`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 545. Bloc de code dupliqué (22 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:262`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 546. Bloc de code dupliqué (22 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:264`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 547. Bloc de code dupliqué (22 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:266`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 548. Bloc de code dupliqué (22 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:294`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 549. Bloc de code dupliqué (22 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:296`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 550. Bloc de code dupliqué (22 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:298`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 551. Bloc de code dupliqué (15 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:200`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 552. Bloc de code dupliqué (15 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:243`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 553. Bloc de code dupliqué (15 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:245`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 554. Bloc de code dupliqué (15 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:289`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 555. Bloc de code dupliqué (15 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:291`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 556. Bloc de code dupliqué (15 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:293`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 557. Bloc de code dupliqué (15 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:295`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 558. Bloc de code dupliqué (15 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\data_manager.py:299`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 559. Bloc de code dupliqué (15 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\data_manager.py:301`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 560. Bloc de code dupliqué (15 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\indicators_panel.py:183`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 561. Bloc de code dupliqué (15 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:263`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 562. Bloc de code dupliqué (15 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:265`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 563. Bloc de code dupliqué (15 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:295`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 564. Bloc de code dupliqué (15 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:297`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 565. Bloc de code dupliqué (15 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:299`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 566. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:217`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 567. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:262`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 568. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:234`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 569. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:218`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 570. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:263`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 571. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:235`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 572. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:219`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 573. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:264`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 574. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:236`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 575. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:220`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 576. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:265`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 577. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:237`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 578. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:221`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 579. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:266`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 580. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:238`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 581. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:232`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 582. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:278`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 583. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:238`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 584. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:284`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 585. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:258`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 586. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:239`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 587. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:285`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 588. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:259`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 589. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:240`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 590. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:286`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 591. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:260`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 592. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\backtest_panel.py:296`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 593. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\data_manager.py:217`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 594. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\data_manager.py:302`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 595. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\indicators_panel.py:114`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 596. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\indicators_panel.py:184`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 597. Bloc de code dupliqué (6 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:300`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 598. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\data_manager.py:154`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 599. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\data_manager.py:168`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 600. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\data_manager.py:216`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 601. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\indicators_panel.py:113`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 602. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\data_manager.py:329`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 603. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\indicators_panel.py:197`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 604. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:97`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 605. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:146`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 606. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:98`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 607. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:147`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 608. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:99`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 609. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:148`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 610. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:100`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 611. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:149`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 612. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:108`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 613. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:122`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 614. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:109`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 615. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:123`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 616. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:110`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 617. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:124`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 618. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:160`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 619. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:175`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 620. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:111`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 621. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:125`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 622. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:161`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 623. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:176`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 624. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:112`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 625. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:126`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 626. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:162`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 627. Bloc de code dupliqué (4 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:177`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 628. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:113`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 629. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:163`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 630. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:114`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 631. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:164`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 632. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:127`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 633. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:178`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 634. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:128`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 635. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:179`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 636. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:138`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 637. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:190`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 638. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:157`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 639. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:172`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 640. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:158`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 641. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:173`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 642. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:159`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 643. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:174`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 644. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\ui\components\optimization_panel.py:189`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 645. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\utils\gpu\multi_gpu.py:40`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 646. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\utils\gpu\__init__.py:12`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 647. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\utils\gpu\multi_gpu.py:41`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 648. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\utils\gpu\__init__.py:13`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 649. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\utils\gpu\multi_gpu.py:42`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 650. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\utils\gpu\__init__.py:14`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 651. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\utils\gpu\multi_gpu.py:43`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 652. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\utils\gpu\__init__.py:15`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 653. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\utils\gpu\vector_checks.py:390`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 654. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\utils\gpu\vector_checks.py:428`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 655. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\utils\gpu\vector_checks.py:460`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 656. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\src\threadx\utils\gpu\vector_checks.py:550`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 657. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_accessibility_theming.py:59`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 658. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_layout_smoke.py:117`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 659. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_accessibility_theming.py:60`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 660. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_layout_smoke.py:118`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 661. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\tests\test_accessibility_theming.py:61`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 662. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\tests\test_layout_smoke.py:119`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 663. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\tests\test_layout_smoke.py:145`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 664. Bloc de code dupliqué (16 occurrences)

**Fichier:** `D:\ThreadX\tests\test_accessibility_theming.py:62`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 665. Bloc de code dupliqué (16 occurrences)

**Fichier:** `D:\ThreadX\tests\test_accessibility_theming.py:89`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 666. Bloc de code dupliqué (16 occurrences)

**Fichier:** `D:\ThreadX\tests\test_accessibility_theming.py:116`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 667. Bloc de code dupliqué (16 occurrences)

**Fichier:** `D:\ThreadX\tests\test_accessibility_theming.py:147`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 668. Bloc de code dupliqué (16 occurrences)

**Fichier:** `D:\ThreadX\tests\test_accessibility_theming.py:176`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 669. Bloc de code dupliqué (16 occurrences)

**Fichier:** `D:\ThreadX\tests\test_accessibility_theming.py:207`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 670. Bloc de code dupliqué (16 occurrences)

**Fichier:** `D:\ThreadX\tests\test_accessibility_theming.py:220`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 671. Bloc de code dupliqué (16 occurrences)

**Fichier:** `D:\ThreadX\tests\test_accessibility_theming.py:247`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 672. Bloc de code dupliqué (16 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_backtest.py:154`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 673. Bloc de code dupliqué (16 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_backtest.py:180`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 674. Bloc de code dupliqué (16 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_data_indicators.py:193`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 675. Bloc de code dupliqué (16 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_data_indicators.py:216`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 676. Bloc de code dupliqué (16 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_optimization.py:141`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 677. Bloc de code dupliqué (16 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_optimization.py:166`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 678. Bloc de code dupliqué (16 occurrences)

**Fichier:** `D:\ThreadX\tests\test_layout_smoke.py:120`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 679. Bloc de code dupliqué (16 occurrences)

**Fichier:** `D:\ThreadX\tests\test_layout_smoke.py:146`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 680. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_accessibility_theming.py:63`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 681. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_layout_smoke.py:121`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 682. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_accessibility_theming.py:64`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 683. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_layout_smoke.py:122`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 684. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_accessibility_theming.py:65`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 685. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_layout_smoke.py:123`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 686. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_accessibility_theming.py:66`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 687. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_layout_smoke.py:124`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 688. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_backtest.py:150`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 689. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_optimization.py:137`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 690. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_backtest.py:151`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 691. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_optimization.py:138`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 692. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_backtest.py:152`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 693. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_optimization.py:139`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 694. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_backtest.py:153`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 695. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_optimization.py:140`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 696. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_backtest.py:155`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 697. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_optimization.py:142`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 698. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_backtest.py:156`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 699. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_optimization.py:143`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 700. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_backtest.py:174`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 701. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_optimization.py:160`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 702. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_backtest.py:175`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 703. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_optimization.py:161`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 704. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_backtest.py:176`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 705. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_optimization.py:162`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 706. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_backtest.py:177`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 707. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_optimization.py:163`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 708. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_backtest.py:178`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 709. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_optimization.py:164`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 710. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_backtest.py:179`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 711. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_optimization.py:165`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 712. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_backtest.py:181`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 713. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_optimization.py:167`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 714. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_backtest.py:182`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 715. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_optimization.py:168`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 716. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_backtest.py:183`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 717. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_optimization.py:169`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 718. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_backtest.py:184`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 719. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_components_optimization.py:170`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 720. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:7`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 721. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:7`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 722. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:13`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 723. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:14`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 724. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:14`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 725. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:15`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 726. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:15`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 727. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:16`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 728. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:16`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 729. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:17`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 730. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:62`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 731. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:65`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 732. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:68`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 733. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:83`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 734. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:74`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 735. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:89`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 736. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:75`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 737. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:90`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 738. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:76`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 739. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:91`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 740. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:104`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 741. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:115`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 742. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:105`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 743. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:116`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 744. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:106`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 745. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:117`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 746. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:107`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 747. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:118`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 748. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:108`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 749. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:119`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 750. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:109`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 751. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:120`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 752. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:110`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 753. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:121`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 754. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:120`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 755. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:153`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 756. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:121`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 757. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:154`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 758. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:122`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 759. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:155`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 760. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:123`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 761. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:156`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 762. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:124`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 763. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:157`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 764. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:147`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 765. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:215`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 766. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:148`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 767. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:216`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 768. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:149`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 769. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:217`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 770. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:150`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 771. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:218`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 772. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:166`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 773. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:313`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 774. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:172`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 775. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:319`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 776. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:199`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 777. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:434`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 778. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:232`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 779. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:525`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 780. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:233`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 781. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:526`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 782. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:234`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 783. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:527`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 784. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:235`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 785. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:528`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 786. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:253`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 787. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:559`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 788. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:254`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 789. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:560`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 790. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_clean.py:262`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 791. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_loaders.py:478`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 792. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_config_improvements.py:89`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 793. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_final_complet.py:151`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 794. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_dispatch_logic.py:113`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 795. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_dispatch_logic.py:124`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 796. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\tests\test_token_diversity_manager_option_b.py:91`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 797. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\tests\test_token_diversity_manager_option_b.py:155`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 798. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\tests\test_token_diversity_manager_option_b.py:422`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 799. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_token_diversity_manager_option_b.py:92`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 800. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_token_diversity_manager_option_b.py:423`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 801. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_token_diversity_manager_option_b.py:424`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 802. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_token_diversity_manager_option_b.py:439`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 803. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_token_diversity_manager_option_b.py:425`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 804. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_token_diversity_manager_option_b.py:440`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 805. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_token_diversity_manager_option_b.py:426`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 806. Bloc de code dupliqué (2 occurrences)

**Fichier:** `D:\ThreadX\tests\test_token_diversity_manager_option_b.py:441`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 807. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\tests\phase_a\test_udfi_contract.py:56`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 808. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\tests\phase_a\test_udfi_contract.py:78`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 809. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\tests\phase_a\test_udfi_contract.py:97`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 810. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\tests\phase_a\test_udfi_contract.py:57`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 811. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\tests\phase_a\test_udfi_contract.py:79`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 812. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\tests\phase_a\test_udfi_contract.py:98`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 813. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\tests\phase_a\test_udfi_contract.py:58`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 814. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\tests\phase_a\test_udfi_contract.py:80`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 815. Bloc de code dupliqué (3 occurrences)

**Fichier:** `D:\ThreadX\tests\phase_a\test_udfi_contract.py:99`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 816. Bloc de code dupliqué (5 occurrences)

**Fichier:** `D:\ThreadX\tests\phase_a\test_udfi_contract.py:59`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 817. Bloc de code dupliqué (5 occurrences)

**Fichier:** `D:\ThreadX\tests\phase_a\test_udfi_contract.py:81`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 818. Bloc de code dupliqué (5 occurrences)

**Fichier:** `D:\ThreadX\tests\phase_a\test_udfi_contract.py:100`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 819. Bloc de code dupliqué (5 occurrences)

**Fichier:** `D:\ThreadX\tests\phase_a\test_udfi_contract.py:116`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

### 820. Bloc de code dupliqué (5 occurrences)

**Fichier:** `D:\ThreadX\tests\phase_a\test_udfi_contract.py:148`

**Catégorie:** duplication

**Recommandation:** Extraire dans une fonction réutilisable

---

## 🟢 Problèmes de Sévérité LOW

### 1. Fonction publique 'get_xp_module' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\backtest\engine.py:69`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 2. Fonction publique 'measure_throughput' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\backtest\engine.py:49`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 3. Fonction publique 'track_memory' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\backtest\engine.py:55`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 4. Fonction publique 'get_xp_module' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\backtest\engine.py:77`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 5. Fonction publique 'decorator' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\backtest\engine.py:50`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 6. Fonction publique 'decorator' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\backtest\engine.py:56`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 7. Fonction publique 'test_func' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\benchmarks\run_auto_profile.py:70`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 8. Fonction publique 'validate_benchmark_config' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\benchmarks\run_backtests.py:74`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 9. Fonction publique 'load_config' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\benchmarks\run_backtests.py:84`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 10. Fonction publique 'run_backtest_benchmark' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\benchmarks\run_backtests.py:107`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 11. Fonction publique 'main' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\benchmarks\run_backtests.py:226`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 12. Fonction publique 'user_message' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\config\errors.py:19`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 13. Fonction publique 'load_config_dict' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\config\loaders.py:22`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 14. Fonction publique 'load_settings' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\config\loaders.py:297`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 15. Fonction publique 'get_settings' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\config\loaders.py:326`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 16. Fonction publique 'print_config' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\config\loaders.py:333`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 17. Fonction publique 'get_section' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\config\loaders.py:77`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 18. Fonction publique 'get_value' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\config\loaders.py:80`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 19. Fonction publique 'validate_config' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\config\loaders.py:86`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 20. Fonction publique 'create_settings' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\config\loaders.py:164`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 21. Fonction publique 'load_config' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\config\loaders.py:275`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 22. Fonction publique 'create_cli_parser' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\config\loaders.py:283`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 23. Fonction publique 'process_symbol' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\data\ingest.py:893`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 24. Fonction publique 'DataFrameSchema' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\data\io.py:36`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 25. Fonction publique 'Column' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\data\io.py:39`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 26. Fonction publique 'gt' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\data\io.py:43`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 27. Fonction publique 'ge' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\data\io.py:46`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 28. Fonction publique 'normalize_ohlcv' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\data\resample.py:49`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 29. Fonction publique 'normalize_ohlcv' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\data\synth.py:25`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 30. Fonction publique 'fetch_ohlcv' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\data\tokens.py:732`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 31. Fonction publique 'compute_diversity_metrics' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\data\tokens.py:735`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 32. Fonction publique 'register_udfi' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\data\udfi_contract.py:63`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 33. Fonction publique 'get_udfi' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\data\udfi_contract.py:67`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 34. Fonction publique 'list_udfi' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\data\udfi_contract.py:71`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 35. Fonction publique 'use' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\bollinger.py:81`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 36. Fonction publique 'getDeviceCount' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\bollinger.py:86`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 37. Fonction publique 'Device' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\bollinger.py:93`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 38. Fonction publique 'asarray' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\bollinger.py:102`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 39. Fonction publique 'asnumpy' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\bollinger.py:106`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 40. Fonction publique 'convolve' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\bollinger.py:110`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 41. Fonction publique 'ones' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\bollinger.py:118`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 42. Fonction publique 'zeros_like' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\bollinger.py:122`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 43. Fonction publique 'sqrt' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\bollinger.py:126`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 44. Fonction publique 'concatenate' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\bollinger.py:130`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 45. Fonction publique 'full' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\bollinger.py:134`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 46. Fonction publique 'std' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\bollinger.py:138`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 47. Fonction publique 'cpu_func' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\gpu_integration.py:263`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 48. Fonction publique 'gpu_func' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\gpu_integration.py:268`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 49. Fonction publique 'compute_fn' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\gpu_integration.py:379`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 50. Fonction publique 'cpu_func' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\gpu_integration.py:285`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 51. Fonction publique 'gpu_func' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\gpu_integration.py:288`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 52. Fonction publique 'cpu_func' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\gpu_integration.py:299`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 53. Fonction publique 'gpu_func' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\gpu_integration.py:302`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 54. Fonction publique 'use' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:82`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 55. Fonction publique 'getDeviceCount' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:87`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 56. Fonction publique 'getDeviceProperties' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:91`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 57. Fonction publique 'memGetInfo' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:95`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 58. Fonction publique 'Device' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:102`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 59. Fonction publique 'asarray' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:111`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 60. Fonction publique 'asnumpy' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:115`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 61. Fonction publique 'convolve' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:119`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 62. Fonction publique 'ones' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:126`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 63. Fonction publique 'zeros_like' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:130`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 64. Fonction publique 'concatenate' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:134`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 65. Fonction publique 'full' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:138`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 66. Fonction publique 'array' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:142`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 67. Fonction publique 'abs' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:146`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 68. Fonction publique 'maximum' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:150`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 69. Fonction publique 'exp' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:154`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 70. Fonction publique 'validate_cli_config' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\optimization\run.py:44`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 71. Fonction publique 'build_scenario_spec' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\optimization\run.py:96`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 72. Fonction publique 'run_sweep' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\optimization\run.py:142`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 73. Fonction publique 'main' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\optimization\run.py:219`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 74. Fonction publique 'default' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\strategy\model.py:675`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 75. Fonction publique 'pause' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\testing\mocks.py:293`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 76. Fonction publique 'resume' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\testing\mocks.py:296`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 77. Fonction publique 'stop' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\testing\mocks.py:299`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 78. Fonction publique 'reset' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\testing\mocks.py:302`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 79. Fonction publique 'check_interruption' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\testing\mocks.py:306`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 80. Fonction publique 'get_logger' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\ui\charts.py:55`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 81. Fonction publique 'emit' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\ui\data_manager.py:469`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 82. Fonction publique 'progress_callback' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\ui\data_manager.py:311`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 83. Fonction publique 'progress_callback' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\ui\sweep.py:637`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 84. Fonction publique 'get_logger' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\ui\tables.py:39`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 85. Fonction publique 'get_logger' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\batching.py:25`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 86. Fonction publique 'decorator' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\cache.py:645`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 87. Fonction publique 'get_logger' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\cache.py:44`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 88. Fonction publique 'check_floats' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\cache.py:487`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 89. Fonction publique 'wrapper' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\cache.py:672`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 90. Fonction publique 'cache_stats' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\cache.py:724`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 91. Fonction publique 'cache_clear' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\cache.py:732`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 92. Fonction publique 'cache_info' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\cache.py:741`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 93. Fonction publique 'decorator' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\timing.py:234`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 94. Fonction publique 'decorator' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\timing.py:333`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 95. Fonction publique 'decorator' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\timing.py:409`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 96. Fonction publique 'get_logger' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\timing.py:35`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 97. Fonction publique 'wrapper' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\timing.py:236`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 98. Fonction publique 'wrapper' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\timing.py:335`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 99. Fonction publique 'is_gpu_backend' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\xp.py:146`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 100. Fonction publique 'get_backend_name' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\xp.py:150`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 101. Fonction publique 'to_device' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\xp.py:168`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 102. Fonction publique 'to_host' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\xp.py:181`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 103. Fonction publique 'asnumpy' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\xp.py:188`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 104. Fonction publique 'ascupy' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\xp.py:192`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 105. Fonction publique 'ensure_array_type' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\xp.py:201`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 106. Fonction publique 'memory_pool_info' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\xp.py:214`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 107. Fonction publique 'clear_memory_pool' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\xp.py:231`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 108. Fonction publique 'device_synchronize' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\xp.py:278`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 109. Fonction publique 'get_array_info' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\xp.py:284`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 110. Fonction publique 'success' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\gpu\multi_gpu.py:143`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 111. Fonction publique 'convert_numpy' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\gpu\profile_persistence.py:162`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 112. Fonction publique 'get_logger' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\gpu\vector_checks.py:26`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 113. Fonction publique 'performance_context' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\timing\__init__.py:125`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 114. Fonction publique 'decorator' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\timing\__init__.py:151`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 115. Fonction publique 'decorator' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\timing\__init__.py:212`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 116. Fonction publique 'wrapper' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\timing\__init__.py:113`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 117. Fonction publique 'wrapper' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\timing\__init__.py:153`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 118. Fonction publique 'wrapper' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\timing\__init__.py:214`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 119. Fonction publique 'combined_measurement' sans docstring

**Fichier:** `D:\ThreadX\src\threadx\utils\timing\__init__.py:419`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 120. Fonction publique 'test_validate_cli_config_accepts_value_and_values' sans docstring

**Fichier:** `D:\ThreadX\tests\tests_test_config_schema.py:20`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 121. Fonction publique 'test_validate_cli_config_rejects_plain_list_for_param_block' sans docstring

**Fichier:** `D:\ThreadX\tests\tests_test_config_schema.py:26`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 122. Fonction publique 'test_validate_cli_config_rejects_unknown_keys_in_param_block' sans docstring

**Fichier:** `D:\ThreadX\tests\tests_test_config_schema.py:34`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 123. Fonction publique 'test_run_sweep_signature_includes_config_path' sans docstring

**Fichier:** `D:\ThreadX\tests\tests_test_run_api.py:9`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 124. Fonction publique 'test_main_passes_config_path' sans docstring

**Fichier:** `D:\ThreadX\tests\tests_test_run_api.py:18`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 125. Fonction publique 'fake_run_sweep' sans docstring

**Fichier:** `D:\ThreadX\tests\tests_test_run_api.py:47`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 126. Fonction publique 'fake_parse_args' sans docstring

**Fichier:** `D:\ThreadX\tests\tests_test_run_api.py:56`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 127. Fonction publique 'test_generate_param_grid_accepts_value_and_values' sans docstring

**Fichier:** `D:\ThreadX\tests\tests_test_scenarios_grid.py:6`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 128. Fonction publique 'test_generate_param_grid_rejects_unknown_shape' sans docstring

**Fichier:** `D:\ThreadX\tests\tests_test_scenarios_grid.py:26`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 129. Fonction publique 'test_generate_param_grid_rejects_non_mapping' sans docstring

**Fichier:** `D:\ThreadX\tests\tests_test_scenarios_grid.py:32`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 130. Fonction publique 'find_navbar' sans docstring

**Fichier:** `D:\ThreadX\tests\test_accessibility_theming.py:59`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 131. Fonction publique 'find_headers' sans docstring

**Fichier:** `D:\ThreadX\tests\test_accessibility_theming.py:85`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 132. Fonction publique 'find_cards' sans docstring

**Fichier:** `D:\ThreadX\tests\test_accessibility_theming.py:112`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 133. Fonction publique 'find_buttons' sans docstring

**Fichier:** `D:\ThreadX\tests\test_accessibility_theming.py:143`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 134. Fonction publique 'find_graphs' sans docstring

**Fichier:** `D:\ThreadX\tests\test_accessibility_theming.py:172`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 135. Fonction publique 'find_inputs' sans docstring

**Fichier:** `D:\ThreadX\tests\test_accessibility_theming.py:203`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 136. Fonction publique 'find_labels' sans docstring

**Fichier:** `D:\ThreadX\tests\test_accessibility_theming.py:216`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 137. Fonction publique 'find_loading' sans docstring

**Fichier:** `D:\ThreadX\tests\test_accessibility_theming.py:243`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 138. Fonction publique 'main' sans docstring

**Fichier:** `D:\ThreadX\tests\test_adaptation.py:9`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 139. Fonction publique 'find_tabs' sans docstring

**Fichier:** `D:\ThreadX\tests\test_components_backtest.py:150`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 140. Fonction publique 'find_grid_components' sans docstring

**Fichier:** `D:\ThreadX\tests\test_components_backtest.py:174`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 141. Fonction publique 'find_rows' sans docstring

**Fichier:** `D:\ThreadX\tests\test_components_data_indicators.py:189`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 142. Fonction publique 'find_cols' sans docstring

**Fichier:** `D:\ThreadX\tests\test_components_data_indicators.py:212`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 143. Fonction publique 'find_tabs' sans docstring

**Fichier:** `D:\ThreadX\tests\test_components_optimization.py:137`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 144. Fonction publique 'find_grid_components' sans docstring

**Fichier:** `D:\ThreadX\tests\test_components_optimization.py:160`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 145. Fonction publique 'test_generate_param_grid_accepts_values_and_value' sans docstring

**Fichier:** `D:\ThreadX\tests\test_config_contract.py:51`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 146. Fonction publique 'test_validate_cli_config_ok_on_mapping_values' sans docstring

**Fichier:** `D:\ThreadX\tests\test_config_contract.py:113`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 147. Fonction publique 'run_grid' sans docstring

**Fichier:** `D:\ThreadX\tests\test_config_contract.py:142`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 148. Fonction publique 'find_navbar' sans docstring

**Fichier:** `D:\ThreadX\tests\test_layout_smoke.py:117`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 149. Fonction publique 'find_footer' sans docstring

**Fichier:** `D:\ThreadX\tests\test_layout_smoke.py:143`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 150. Fonction publique 'test_pipeline' sans docstring

**Fichier:** `D:\ThreadX\tests\test_pipeline.py:11`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 151. Fonction publique 'run_ohlcv' sans docstring

**Fichier:** `D:\ThreadX\tests\test_token_diversity_manager_option_b.py:366`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 152. Fonction publique 'run_with_indicators' sans docstring

**Fichier:** `D:\ThreadX\tests\test_token_diversity_manager_option_b.py:392`

**Catégorie:** structural

**Recommandation:** Ajouter une docstring décrivant le comportement

---

### 153. Pas de vérification de timestamps dupliqués

**Fichier:** `D:\ThreadX\src\threadx\data\ingest.py:1`

**Catégorie:** logic

**Recommandation:** Ajouter: assert not df.duplicated().any()

---

### 154. Pas de vérification de timestamps dupliqués

**Fichier:** `D:\ThreadX\src\threadx\data\legacy_adapter.py:1`

**Catégorie:** logic

**Recommandation:** Ajouter: assert not df.duplicated().any()

---

### 155. Pas de vérification de timestamps dupliqués

**Fichier:** `D:\ThreadX\src\threadx\data\loader.py:1`

**Catégorie:** logic

**Recommandation:** Ajouter: assert not df.duplicated().any()

---

### 156. Pas de vérification de timestamps dupliqués

**Fichier:** `D:\ThreadX\src\threadx\data\resample.py:1`

**Catégorie:** logic

**Recommandation:** Ajouter: assert not df.duplicated().any()

---

### 157. Pas de vérification de timestamps dupliqués

**Fichier:** `D:\ThreadX\src\threadx\data\synth.py:1`

**Catégorie:** logic

**Recommandation:** Ajouter: assert not df.duplicated().any()

---

### 158. Pas de vérification de timestamps dupliqués

**Fichier:** `D:\ThreadX\src\threadx\data\tokens.py:1`

**Catégorie:** logic

**Recommandation:** Ajouter: assert not df.duplicated().any()

---

### 159. Pas de vérification de timestamps dupliqués

**Fichier:** `D:\ThreadX\src\threadx\data\udfi_contract.py:1`

**Catégorie:** logic

**Recommandation:** Ajouter: assert not df.duplicated().any()

---

### 160. Pas de vérification de timestamps dupliqués

**Fichier:** `D:\ThreadX\src\threadx\data\unified_diversity_pipeline.py:1`

**Catégorie:** logic

**Recommandation:** Ajouter: assert not df.duplicated().any()

---

### 161. Opérations GPU sans vérification de shape

**Fichier:** `D:\ThreadX\src\threadx\indicators\bollinger.py:1`

**Catégorie:** performance

**Recommandation:** Utiliser utils/gpu/vector_checks.py

---

### 162. Opérations GPU sans vérification de shape

**Fichier:** `D:\ThreadX\src\threadx\indicators\xatr.py:1`

**Catégorie:** performance

**Recommandation:** Utiliser utils/gpu/vector_checks.py

---

## 🎯 Plan d'Action Recommandé

### 🔴 URGENT: 1 problèmes critiques
**Action immédiate requise** - Ces problèmes peuvent causer des pertes financières

### 🟠 PRIORITAIRE: 7 problèmes haute priorité
**À traiter dans les 48h** - Impact significatif sur la fiabilité

## 💡 Recommandations Générales

1. **Installer les outils d'analyse:**
   ```bash
   pip install pylint flake8 mypy bandit black
   ```

2. **Configurer pre-commit hooks** pour prévenir les régressions

3. **Augmenter la couverture de tests** pour validation continue

4. **Implémenter CI/CD** avec vérifications automatiques

---

*Généré par ThreadX Audit System*
