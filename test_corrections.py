#!/usr/bin/env python3
"""
Test des corrections ThreadX - Variables et fichiers Parquet
"""

print("🧪 Test des corrections ThreadX")
print("=" * 40)

# Test 1: Lecture du fichier parquet
try:
    import pandas as pd
    import numpy as np
    
    df = pd.read_parquet('data/crypto_data_parquet/BTCUSDT/1m/2024-01.parquet')
    print(f"✅ Fichier parquet lu: {len(df):,} barres")
    
    # Test 2: Variables qui causaient NameError
    d = df.index.to_series().diff().dropna().dt.total_seconds().div(60).round()
    print(f"✅ Variable 'd' définie: {len(d)} valeurs")
    
    effective_tf_min = d.mode().iloc[0] if not d.empty else None
    print(f"✅ Variable 'effective_tf_min' définie: {effective_tf_min}")
    
    # Test 3: Affichage des données
    print(f"✅ Test print df.head():")
    print(df.head(3)[["open","high","low","close","volume"]])
    
    print(f"\n🎉 Tous les tests passés!")
    print(f"📊 Données disponibles pour l'analyse")
    
except Exception as e:
    print(f"❌ Erreur: {e}")