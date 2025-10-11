#!/usr/bin/env python3
"""
Scan multi-tokens: analyse tous les tokens top N avec filtres personnalisés.

Usage:
    python scan_all_tokens.py
    python scan_all_tokens.py --tokens 50 --timeframe 4h --rsi-oversold 35
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import direct modules consolidés
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

# Import indicators
spec = importlib.util.spec_from_file_location(
    "indicators_np",
    Path(__file__).parent / "src" / "threadx" / "indicators" / "indicators_np.py",
)
indicators_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(indicators_module)


def analyze_single_token(symbol, loader, timeframe, days):
    """Analyse un token et retourne métriques clés."""

    try:
        # Télécharger données
        df = loader.download_ohlcv(
            symbol=symbol, interval=timeframe, days_history=days, force_update=False
        )

        if df is None or len(df) < 50:
            return None

        # Calculer indicateurs essentiels
        df["ema_9"] = indicators_module.ema_np(df["close"].values, 9)
        df["ema_21"] = indicators_module.ema_np(df["close"].values, 21)
        df["rsi"] = indicators_module.rsi_np(df["close"].values, 14)

        macd = indicators_module.macd_np(df["close"].values)
        df["macd"], df["macd_signal"] = macd[0], macd[1]

        # Dernière bougie
        last = df.iloc[-1]

        # Score tendance (0-100)
        trend_score = 0
        if last["ema_9"] > last["ema_21"]:
            trend_score += 50
        if last["close"] > last["ema_9"]:
            trend_score += 25
        if last["macd"] > last["macd_signal"]:
            trend_score += 25

        # Volume 24h
        volume_24h = df["volume"].tail(24).sum()

        return {
            "symbol": symbol,
            "price": last["close"],
            "ema_9": last["ema_9"],
            "ema_21": last["ema_21"],
            "rsi": last["rsi"],
            "macd": last["macd"],
            "macd_signal": last["macd_signal"],
            "volume_24h": volume_24h,
            "trend_score": trend_score,
            "timestamp": last.name,
        }

    except Exception as e:
        print(f"   ⚠️  {symbol}: {str(e)}")
        return None


def scan_all_tokens(
    num_tokens=100,
    timeframe="1h",
    days_history=7,
    rsi_oversold=30,
    rsi_overbought=70,
    min_volume=1_000_000,
):
    """
    Scan tous les top tokens et filtre selon critères.

    Args:
        num_tokens: Nombre de tokens à scanner
        timeframe: Timeframe analyse
        days_history: Jours d'historique
        rsi_oversold: Seuil RSI survendu
        rsi_overbought: Seuil RSI suracheté
        min_volume: Volume 24h minimum
    """

    print("=" * 70)
    print("🔍 SCAN MULTI-TOKENS")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tokens: {num_tokens}")
    print(f"Timeframe: {timeframe}")
    print(f"Historique: {days_history} jours")

    # ÉTAPE 1: Récupérer liste tokens
    print("\n📝 ÉTAPE 1/3: Récupération tokens...")
    print("-" * 70)

    token_mgr = TokenManager(
        cache_path=Path("data/crypto_data_json/tokens_top100.json")
    )

    symbols = token_mgr.get_top_tokens(
        limit=num_tokens, usdc_only=True, force_refresh=False
    )

    print(f"✅ {len(symbols)} tokens sélectionnés")

    # ÉTAPE 2: Analyse parallèle
    print("\n🔧 ÉTAPE 2/3: Analyse parallèle...")
    print("-" * 70)

    loader = BinanceDataLoader(
        json_cache_dir=Path("data/crypto_data_json"),
        parquet_cache_dir=Path("data/crypto_data_parquet"),
    )

    results = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                analyze_single_token, symbol, loader, timeframe, days_history
            ): symbol
            for symbol in symbols
        }

        for i, future in enumerate(as_completed(futures), 1):
            symbol = futures[future]

            if i % 10 == 0:
                print(f"   [{i}/{len(symbols)}] {(i/len(symbols)*100):.0f}%")

            result = future.result()
            if result:
                results.append(result)

    print(f"✅ {len(results)}/{len(symbols)} tokens analysés")

    # ÉTAPE 3: Filtrage et classement
    print("\n📊 ÉTAPE 3/3: Filtrage et classement...")
    print("-" * 70)

    # Filtre volume
    filtered = [r for r in results if r["volume_24h"] >= min_volume]
    print(f"   Volume > ${min_volume:,}: {len(filtered)} tokens")

    # Catégories
    oversold = [r for r in filtered if r["rsi"] < rsi_oversold]
    overbought = [r for r in filtered if r["rsi"] > rsi_overbought]
    bullish = [r for r in filtered if r["trend_score"] >= 75]
    bearish = [r for r in filtered if r["trend_score"] <= 25]

    print(f"   RSI < {rsi_oversold} (survendu): {len(oversold)}")
    print(f"   RSI > {rsi_overbought} (suracheté): {len(overbought)}")
    print(f"   Tendance haussière forte: {len(bullish)}")
    print(f"   Tendance baissière forte: {len(bearish)}")

    # Affichage top opportunités
    print("\n" + "=" * 70)
    print("🎯 TOP OPPORTUNITÉS (RSI survendu + volume)")
    print("=" * 70)

    if oversold:
        # Trier par volume décroissant
        top_oversold = sorted(oversold, key=lambda x: x["volume_24h"], reverse=True)[
            :10
        ]

        print(
            f"\n{'Symbol':<12} {'Prix':<12} {'RSI':<8} {'Trend':<8} "
            f"{'Volume 24h':<15}"
        )
        print("-" * 70)

        for r in top_oversold:
            print(
                f"{r['symbol']:<12} "
                f"${r['price']:<11.4f} "
                f"{r['rsi']:<7.2f} "
                f"{r['trend_score']:<7.0f} "
                f"${r['volume_24h']:>14,.0f}"
            )
    else:
        print("   Aucun token en zone de survente")

    # Top haussiers
    print("\n" + "=" * 70)
    print("🚀 TOP TENDANCES HAUSSIÈRES (score >= 75)")
    print("=" * 70)

    if bullish:
        top_bullish = sorted(bullish, key=lambda x: x["trend_score"], reverse=True)[:10]

        print(
            f"\n{'Symbol':<12} {'Prix':<12} {'RSI':<8} {'Trend':<8} "
            f"{'Volume 24h':<15}"
        )
        print("-" * 70)

        for r in top_bullish:
            print(
                f"{r['symbol']:<12} "
                f"${r['price']:<11.4f} "
                f"{r['rsi']:<7.2f} "
                f"{r['trend_score']:<7.0f} "
                f"${r['volume_24h']:>14,.0f}"
            )
    else:
        print("   Aucune tendance haussière forte")

    # Sauvegarde résultats
    output_path = Path("data/exports/scan_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import csv

    with open(output_path, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print(f"\n💾 Résultats sauvegardés: {output_path}")

    print("\n" + "=" * 70)
    print("✅ SCAN MULTI-TOKENS TERMINÉ")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="Scan multi-tokens avec filtres")
    parser.add_argument(
        "--tokens", type=int, default=100, help="Nombre de tokens (défaut: 100)"
    )
    parser.add_argument(
        "--timeframe", type=str, default="1h", help="Timeframe (défaut: 1h)"
    )
    parser.add_argument(
        "--days", type=int, default=7, help="Jours d'historique (défaut: 7)"
    )
    parser.add_argument(
        "--rsi-oversold", type=float, default=30, help="Seuil RSI survendu (défaut: 30)"
    )
    parser.add_argument(
        "--rsi-overbought",
        type=float,
        default=70,
        help="Seuil RSI suracheté (défaut: 70)",
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        default=1_000_000,
        help="Volume 24h minimum (défaut: 1000000)",
    )

    args = parser.parse_args()

    results = scan_all_tokens(
        num_tokens=args.tokens,
        timeframe=args.timeframe,
        days_history=args.days,
        rsi_oversold=args.rsi_oversold,
        rsi_overbought=args.rsi_overbought,
        min_volume=args.min_volume,
    )

    print(f"\n✅ {len(results)} tokens analysés avec succès!")


if __name__ == "__main__":
    main()
