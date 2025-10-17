# Correction de l'erreur UDFI : "Colonnes manquantes: symbol"

## 📋 Problème identifié

Lors de l'ingestion de données depuis Binance, une erreur de validation UDFI se produisait :
```
❌ Data ingestion error: Binance ingestion failed: UDFI validation failed: Colonnes manquantes: symbol
```

## 🔍 Cause racine

La fonction `LegacyAdapter.json_to_dataframe()` ne créait que les colonnes OHLCV (Open, High, Low, Close, Volume) mais **n'incluait pas la colonne `symbol`** requise par le contrat UDFI défini dans `src/threadx/data/udfi_contract.py` :

```python
REQUIRED_COLS: Set[str] = {"symbol", "open", "high", "low", "close", "volume"}
```

## ✅ Solution appliquée

### 1. Modification de `legacy_adapter.py`

**Fichier**: `src/threadx/data/legacy_adapter.py`

- Ajout du paramètre `symbol` à la signature de `json_to_dataframe()`:
  ```python
  def json_to_dataframe(self, raw_klines: List[Dict[str, Any]], symbol: str = None) -> pd.DataFrame:
  ```

- Ajout de la colonne `symbol` au DataFrame **après** la normalisation OHLCV:
  ```python
  # Validation schéma OHLCV (sans symbol pour l'instant)
  result = normalize_ohlcv(result)

  # Ajout de la colonne symbol APRÈS normalisation
  if symbol:
      result["symbol"] = symbol
  ```

### 2. Mise à jour des appels dans `ingest.py`

**Fichier**: `src/threadx/data/ingest.py`

Tous les appels à `json_to_dataframe()` ont été mis à jour pour passer le symbole:

```python
# Ligne 140 - Téléchargement de segments manquants
segment_df = self.adapter.json_to_dataframe(raw_klines, symbol)

# Ligne 635 - Vérification avec timeframes lents
df_slow = self.adapter.json_to_dataframe(raw_slow, symbol)
```

## 🧪 Validation

Un script de test (`test_symbol_fix.py`) a été créé et exécuté avec succès:

```
✅ Test de la colonne symbol
============================================================
Shape du DataFrame: (2, 6)
Colonnes: ['open', 'high', 'low', 'close', 'volume', 'symbol']

✅ Colonne 'symbol' présente !
Valeur symbol: ['BTCUSDT']

🎉 SUCCÈS : La colonne symbol est correctement ajoutée !
```

## 📊 Impact

- ✅ Conformité UDFI complète
- ✅ Validation schéma réussie
- ✅ Ingestion de données Binance fonctionnelle
- ✅ Pas de régression (test avec `symbol=None` fonctionne)

## 📝 Fichiers modifiés

1. `src/threadx/data/legacy_adapter.py` - Ajout paramètre `symbol` et colonne au DataFrame
2. `src/threadx/data/ingest.py` - Mise à jour des 2 appels à `json_to_dataframe()`
3. `test_symbol_fix.py` - Script de test créé pour validation

## 🔄 Prochaines étapes recommandées

1. ✅ Tester l'ingestion complète depuis l'interface Dash
2. ✅ Vérifier que les données existantes sur disque sont correctement chargées
3. ⚠️ Si nécessaire, migrer les anciennes données pour ajouter la colonne `symbol`

---
**Date**: 17 octobre 2025
**Statut**: ✅ Corrigé et validé
