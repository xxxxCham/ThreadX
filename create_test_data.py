#!/usr/bin/env python3
"""
Script de création de données parquet de test pour ThreadX
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_btc_test_data():
    """Créer des données BTCUSDT de test."""
    print("🔧 Création des données BTCUSDT...")
    
    # Période de test : janvier 2024
    dates = pd.date_range('2024-01-01', '2024-01-31 23:59:00', freq='1min')
    n = len(dates)
    
    # Prix de base BTC ~45k
    np.random.seed(42)  # Reproductible
    base_price = 45000
    
    # Génération OHLCV cohérente
    close_prices = base_price + np.cumsum(np.random.randn(n) * 0.001) * 1000
    
    df = pd.DataFrame({
        'open': np.roll(close_prices, 1),  # Open = close précédent
        'high': close_prices * (1 + np.random.uniform(0, 0.005, n)),
        'low': close_prices * (1 - np.random.uniform(0, 0.005, n)),
        'close': close_prices,
        'volume': np.random.uniform(10, 1000, n)
    }, index=dates)
    
    # Premier open = premier close
    df.loc[df.index[0], 'open'] = df.loc[df.index[0], 'close']
    
    # Cohérence OHLC : H >= max(O,C), L <= min(O,C)
    df['high'] = np.maximum.reduce([df['open'], df['high'], df['close']])
    df['low'] = np.minimum.reduce([df['open'], df['low'], df['close']])
    
    # Créer le répertoire
    output_dir = Path('data/crypto_data_parquet/BTCUSDT/1m')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder
    output_path = output_dir / '2024-01.parquet'
    df.to_parquet(output_path)
    
    print(f"✅ Créé: {output_path}")
    print(f"📊 Données: {len(df):,} barres")
    print(f"💰 Prix range: {df['close'].min():.0f} - {df['close'].max():.0f}")
    print(f"📈 Volume moyen: {df['volume'].mean():.1f}")
    
    return df

if __name__ == "__main__":
    df = create_btc_test_data()
    
    # Test de lecture
    print("\n🧪 Test de lecture...")
    df_test = pd.read_parquet('data/crypto_data_parquet/BTCUSDT/1m/2024-01.parquet')
    print(f"✅ Lecture OK: {len(df_test)} barres")
    print(df_test.head(3)[["open","high","low","close","volume"]])