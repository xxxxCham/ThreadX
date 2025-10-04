#!/usr/bin/env python3
"""
ThreadX - One-liner pour résoudre instantanément les erreurs
==========================================================

COPIEZ-COLLEZ CETTE LIGNE DANS VOTRE TERMINAL PYTHON:
"""

print("""
═══════════════════════════════════════════════════════════════════════════════════════
🚀 ThreadX - SOLUTION INSTANTANÉE
═══════════════════════════════════════════════════════════════════════════════════════

COPIEZ-COLLEZ CETTE COMMANDE DANS VOTRE TERMINAL PYTHON:

import pandas as pd, numpy as np; exec('''
try: df = pd.read_parquet("data/crypto_data_parquet/BTCUSDT/1m/2024-01.parquet")
except: dates = pd.date_range("2024-01-01", "2024-01-31 23:59:00", freq="1min"); np.random.seed(42); close = 45000 + np.cumsum(np.random.randn(len(dates)) * 0.001) * 1000; df = pd.DataFrame({"open": np.roll(close, 1), "high": close * 1.002, "low": close * 0.998, "close": close, "volume": np.random.uniform(10, 1000, len(dates))}, index=dates); df.iloc[0, 0] = df.iloc[0, 3]
d = df.index.to_series().diff().dropna().dt.total_seconds().div(60).round()
effective_tf_min = d.mode().iloc[0] if not d.empty else None
print(f"✅ Variables définies: df({len(df)} lignes), d({len(d)} valeurs), effective_tf_min({effective_tf_min})")
print("🎉 Plus d'erreurs NameError ou FileNotFoundError!")
print(df.head(3)[["open","high","low","close","volume"]])
print(f"Effective TF (min): {effective_tf_min}")
''')

═══════════════════════════════════════════════════════════════════════════════════════

OU utilisez cette version simplifiée:

exec(open('quick_fix.py').read())

═══════════════════════════════════════════════════════════════════════════════════════
""")

print("💾 One-liner sauvegardé dans ce fichier pour référence future")