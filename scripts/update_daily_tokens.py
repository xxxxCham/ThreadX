#!/usr/bin/env python3
"""
Mise à jour quotidienne des top tokens et leurs données OHLCV.
À exécuter chaque matin avant le début du trading.

Usage:
    python update_daily_tokens.py

    Options:
        --tokens 150        Nombre de tokens (défaut: 100)
        --timeframes 1h,4h  Timeframes à mettre à jour (défaut: 1h,4h)
        --days 365          Jours d'historique (défaut: 365)
"""

import sys
from pathlib import Path
from datetime import datetime
import argparse

# Import modules consolidés (direct import pour éviter dépendances config)
sys.path.insert(0, str(Path(__file__).parent))

import importlib.util

# Import TokenManager
spec = importlib.util.spec_from_file_location(
    "tokens", Path(__file__).parent / "src" / "threadx" / "data" / "tokens.py"
)
tokens_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tokens_module)
TokenManager = tokens_module.TokenManager

# Import BinanceDataLoader
spec = importlib.util.spec_from_file_location(
    "loader", Path(__file__).parent / "src" / "threadx" / "data" / "loader.py"
)
loader_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader_module)
BinanceDataLoader = loader_module.BinanceDataLoader


def update_daily_tokens(num_tokens=100, timeframes=None, days_history=365):
    """
    Mise à jour quotidienne complète des tokens.

    Args:
        num_tokens: Nombre de tokens à récupérer
        timeframes: Liste des timeframes à mettre à jour
        days_history: Nombre de jours d'historique
    """

    if timeframes is None:
        timeframes = ["1h", "4h"]

    print("=" * 70)
    print(
        f"📅 MISE À JOUR QUOTIDIENNE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print("=" * 70)

    # ÉTAPE 1: Mise à jour liste tokens top N
    print(f"\n📝 ÉTAPE 1/3: Récupération top {num_tokens} tokens...")
    print("-" * 70)

    token_mgr = TokenManager(
        cache_path=Path("data/crypto_data_json/tokens_top100.json")
    )

    # Récupérer top N avec données fraîches
    top_tokens = token_mgr.get_top_tokens(
        limit=num_tokens,
        usdc_only=True,
        force_refresh=True,  # Force rafraîchissement quotidien
    )

    print(f"✅ {len(top_tokens)} tokens USDC sélectionnés")
    print(f"   Top 10: {', '.join(top_tokens[:10])}")

    # ÉTAPE 2: Téléchargement données OHLCV
    print(f"\n\n📥 ÉTAPE 2/3: Téléchargement OHLCV ({days_history} jours)...")
    print("-" * 70)

    loader = BinanceDataLoader(
        json_cache_dir=Path("data/crypto_data_json"),
        parquet_cache_dir=Path("data/crypto_data_parquet"),
    )

    total_downloaded = 0
    total_failed = 0

    for tf in timeframes:
        print(f"\n⏱️  Timeframe: {tf}")

        # Callback progression
        def progress_callback(pct, done, total):
            if done % 20 == 0 or done == total:
                print(f"   [{done}/{total}] {pct:.1f}%")

        # Téléchargement parallèle
        results = loader.download_multiple(
            symbols=top_tokens,
            interval=tf,
            days_history=days_history,
            max_workers=4,
            progress_callback=progress_callback,
        )

        downloaded = len(results)
        failed = len(top_tokens) - downloaded
        total_downloaded += downloaded
        total_failed += failed

        print(f"   ✅ {downloaded}/{len(top_tokens)} téléchargés")
        if failed > 0:
            print(f"   ⚠️  {failed} échecs")

    # ÉTAPE 3: Résumé et statistiques
    print(f"\n\n📊 ÉTAPE 3/3: Résumé")
    print("-" * 70)

    # Statistiques stockage
    json_dir = Path("data/crypto_data_json")
    parquet_dir = Path("data/crypto_data_parquet")

    json_files = len(list(json_dir.glob("*.json"))) if json_dir.exists() else 0
    parquet_files = (
        len(list(parquet_dir.glob("*.parquet"))) if parquet_dir.exists() else 0
    )

    # Taille fichiers
    def get_dir_size(directory):
        total = 0
        for file in Path(directory).rglob("*"):
            if file.is_file():
                total += file.stat().st_size
        return total / (1024 * 1024)  # MB

    json_size = get_dir_size(json_dir) if json_dir.exists() else 0
    parquet_size = get_dir_size(parquet_dir) if parquet_dir.exists() else 0

    print(f"📁 Tokens:           {len(top_tokens)}")
    print(f"📁 Timeframes:       {', '.join(timeframes)}")
    print(f"📁 Fichiers JSON:    {json_files} ({json_size:.1f} MB)")
    print(f"📁 Fichiers Parquet: {parquet_files} ({parquet_size:.1f} MB)")
    print(f"📥 Téléchargés:      {total_downloaded}")
    if total_failed > 0:
        print(f"⚠️  Échecs:          {total_failed}")

    print(f"\n✅ Mise à jour quotidienne terminée!")

    return {
        "tokens_count": len(top_tokens),
        "timeframes": timeframes,
        "downloaded": total_downloaded,
        "failed": total_failed,
        "json_files": json_files,
        "parquet_files": parquet_files,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Mise à jour quotidienne top tokens + OHLCV"
    )
    parser.add_argument(
        "--tokens", type=int, default=100, help="Nombre de tokens (défaut: 100)"
    )
    parser.add_argument(
        "--timeframes",
        type=str,
        default="1h,4h",
        help="Timeframes séparés par virgule (défaut: 1h,4h)",
    )
    parser.add_argument(
        "--days", type=int, default=365, help="Jours d'historique (défaut: 365)"
    )

    args = parser.parse_args()

    timeframes = [tf.strip() for tf in args.timeframes.split(",")]

    stats = update_daily_tokens(
        num_tokens=args.tokens, timeframes=timeframes, days_history=args.days
    )

    print("\n" + "=" * 70)
    print("✅ MISE À JOUR QUOTIDIENNE COMPLÈTE")
    print("=" * 70)


if __name__ == "__main__":
    main()
