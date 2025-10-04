#!/usr/bin/env python3
"""
Script d'analyse des données parquet ThreadX
Corrige les erreurs de variables non définies
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_parquet_data():
    """Analyse complète des données parquet."""
    print("📊 ThreadX - Analyse des données parquet")
    print("=" * 50)
    
    # Charger les données
    try:
        parquet_path = 'data/crypto_data_parquet/BTCUSDT/1m/2024-01.parquet'
        df = pd.read_parquet(parquet_path)
        print(f"✅ Données chargées: {len(df):,} barres")
        print(f"📅 Période: {df.index[0]} → {df.index[-1]}")
        
    except FileNotFoundError as e:
        print(f"❌ Fichier non trouvé: {e}")
        return None
    
    # Analyse du timeframe effectif
    print("\n🔍 Analyse du timeframe...")
    d = df.index.to_series().diff().dropna().dt.total_seconds().div(60).round()
    effective_tf_min = d.mode().iloc[0] if not d.empty else None
    print(f"Timeframe effectif (min): {effective_tf_min}")
    
    # Statistiques des données
    print(f"\nUnique timeframes dans les données:")
    tf_counts = d.value_counts().head(5)
    for tf, count in tf_counts.items():
        print(f"  {tf:3.0f}min: {count:,} occurrences")
    
    # Aperçu des données OHLCV
    print(f"\n📈 Aperçu des données OHLCV:")
    print(df.head(3)[["open","high","low","close","volume"]])
    
    # Statistiques de prix
    print(f"\n💰 Statistiques de prix:")
    print(f"  Open   : {df['open'].min():.2f} - {df['open'].max():.2f}")
    print(f"  High   : {df['high'].min():.2f} - {df['high'].max():.2f}")
    print(f"  Low    : {df['low'].min():.2f} - {df['low'].max():.2f}")
    print(f"  Close  : {df['close'].min():.2f} - {df['close'].max():.2f}")
    print(f"  Volume : {df['volume'].min():.2f} - {df['volume'].max():.2f}")
    
    # Vérification cohérence OHLC
    print(f"\n🔧 Vérification cohérence OHLC:")
    high_ok = (df['high'] >= df['open']).all() and (df['high'] >= df['close']).all()
    low_ok = (df['low'] <= df['open']).all() and (df['low'] <= df['close']).all()
    
    print(f"  High >= Open/Close: {'✅' if high_ok else '❌'}")
    print(f"  Low <= Open/Close:  {'✅' if low_ok else '❌'}")
    
    if not high_ok or not low_ok:
        print("⚠️ Données OHLC incohérentes détectées!")
    
    # Détection des gaps
    print(f"\n🕳️ Analyse des gaps temporels:")
    expected_freq_min = 1  # 1min attendu
    gaps = d[d > expected_freq_min]
    
    if len(gaps) > 0:
        print(f"  {len(gaps)} gaps détectés:")
        for idx, gap_min in gaps.head(5).items():
            print(f"    {idx}: gap de {gap_min:.0f}min")
    else:
        print("  ✅ Aucun gap détecté")
    
    # Variables disponibles pour l'utilisateur
    print(f"\n🔬 Variables disponibles:")
    print(f"  df                : DataFrame principal ({len(df)} lignes)")
    print(f"  d                 : Série des différences temporelles")
    print(f"  effective_tf_min  : Timeframe effectif = {effective_tf_min}")
    print(f"  gaps              : Gaps temporels détectés")
    
    return {
        'df': df,
        'd': d,
        'effective_tf_min': effective_tf_min,
        'gaps': gaps,
        'parquet_path': parquet_path
    }

def interactive_mode():
    """Mode interactif avec variables pré-définies."""
    print("\n🎮 Mode interactif activé!")
    print("Variables disponibles: df, d, effective_tf_min, gaps")
    print("Tapez 'exit()' pour quitter\n")
    
    # Analyser et récupérer les variables
    vars_dict = analyze_parquet_data()
    
    if vars_dict is None:
        print("❌ Impossible de charger les données")
        return
    
    # Ajouter les variables au namespace global
    globals().update(vars_dict)
    
    # Exemples de commandes
    print("💡 Exemples de commandes à tester:")
    print("  df.head()")
    print("  d.describe()")
    print("  print('Effective TF (min):', effective_tf_min)")
    print("  df[['open','high','low','close','volume']].describe()")
    print()

if __name__ == "__main__":
    # Mode analyse automatique
    result = analyze_parquet_data()
    
    if result:
        print(f"\n✅ Analyse terminée avec succès!")
        print(f"📁 Fichier analysé: {result['parquet_path']}")
        
        # Proposer le mode interactif
        print(f"\n🎯 Pour utiliser les variables en mode interactif:")
        print(f"   python -i analyze_data.py")
    else:
        print(f"\n❌ Échec de l'analyse")