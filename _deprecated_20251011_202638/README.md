# Fichiers Deprecated - 2025-10-11 20:26:38

## Raison

Ces fichiers ont été déplacés car ils sont obsolètes et redondants avec
la nouvelle architecture consolidée dans `src/threadx/data/`.

## Fichiers déplacés

Voir ANALYSE_EVOLUTION_DATA_MANAGEMENT.md pour détails complets.

### Principaux fichiers:

- **auto_data_sync.py** → Remplacé par sync_data_smart.py
- **unified_data_historique_with_indicators.py** → Fonctions dans src/threadx/data/
- **token_diversity_manager/** → Consolidé dans TokenManager
- **validate_data_structures.py** → Tests dans tests/

## Restauration

Si besoin de restaurer:
```powershell
Move-Item _deprecated_20251011_202638\* . -Force
```

## Suppression définitive

Après validation (tests OK):
```powershell
Remove-Item -Recurse -Force _deprecated_20251011_202638
```

## Voir aussi

- ANALYSE_EVOLUTION_DATA_MANAGEMENT.md (analyse complète)
- ANALYSE_REDONDANCE_TOKENS.md (plan consolidation tokens)
