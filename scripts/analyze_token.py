#!/usr/bin/env python3
"""
Analyse technique complète d'un token unique avec tous les indicateurs.

Usage:
    python analyze_token.py BTCUSDC
    python analyze_token.py ETHUSDC --timeframe 15m --days 30
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
from datetime import datetime

# Import direct des modules consolidés
sys.path.insert(0, str(Path(__file__).parent))

import importlib.util

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


def analyze_token(symbol, timeframe="1h", days_history=30):
    """
    Analyse technique complète d'un token.

    Args:
        symbol: Symbol du token (ex: BTCUSDC)
        timeframe: Timeframe (1m, 5m, 15m, 1h, 4h...)
        days_history: Nombre de jours d'historique

    Returns:
        DataFrame avec toutes les données + indicateurs
    """

    print("=" * 70)
    print(f"📊 ANALYSE TECHNIQUE: {symbol}")
    print("=" * 70)
    print(f"Timeframe: {timeframe}")
    print(f"Historique: {days_history} jours")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ÉTAPE 1: Téléchargement données
    print("\n📥 ÉTAPE 1/3: Téléchargement données...")
    print("-" * 70)

    loader = BinanceDataLoader(
        json_cache_dir=Path("data/crypto_data_json"),
        parquet_cache_dir=Path("data/crypto_data_parquet"),
    )

    df = loader.download_ohlcv(
        symbol=symbol, interval=timeframe, days_history=days_history, force_update=False
    )

    if df is None or len(df) == 0:
        print(f"❌ Impossible de récupérer données pour {symbol}")
        return None

    print(f"✅ {len(df)} bougies récupérées")
    print(f"   Période: {df.index[0]} → {df.index[-1]}")
    print(f"   Prix: ${df['close'].iloc[-1]:.2f}")
    print(f"   Volume 24h: ${df['volume'].tail(24).sum():,.0f}")

    # ÉTAPE 2: Calcul indicateurs
    print("\n🔧 ÉTAPE 2/3: Calcul indicateurs techniques...")
    print("-" * 70)

    # EMA
    print("   ⚙️  EMA 9/21/50/200...")
    df["ema_9"] = indicators_module.ema_np(df["close"].values, 9)
    df["ema_21"] = indicators_module.ema_np(df["close"].values, 21)
    df["ema_50"] = indicators_module.ema_np(df["close"].values, 50)
    df["ema_200"] = indicators_module.ema_np(df["close"].values, 200)

    # RSI
    print("   ⚙️  RSI 14...")
    df["rsi"] = indicators_module.rsi_np(df["close"].values, period=14)

    # Bollinger Bands
    print("   ⚙️  Bollinger Bands 20/2...")
    bb_result = indicators_module.boll_np(df["close"].values, period=20, num_std=2)
    df["bb_upper"], df["bb_middle"], df["bb_lower"] = bb_result

    # MACD
    print("   ⚙️  MACD 12/26/9...")
    macd_result = indicators_module.macd_np(df["close"].values)
    df["macd"], df["macd_signal"], df["macd_hist"] = macd_result

    # ATR
    print("   ⚙️  ATR 14...")
    df["atr"] = indicators_module.atr_np(
        df["high"].values, df["low"].values, df["close"].values, period=14
    )

    # VWAP
    print("   ⚙️  VWAP...")
    df["vwap"] = indicators_module.vwap_np(
        df["high"].values, df["low"].values, df["close"].values, df["volume"].values
    )

    # OBV
    print("   ⚙️  OBV...")
    df["obv"] = indicators_module.obv_np(df["close"].values, df["volume"].values)

    # Vortex
    print("   ⚙️  Vortex 14...")
    vortex_result = indicators_module.vortex_df(df, period=14)
    df["vi_plus"] = vortex_result["VI+"]
    df["vi_minus"] = vortex_result["VI-"]

    print("✅ Tous les indicateurs calculés")

    # ÉTAPE 3: Analyse de la dernière bougie
    print("\n📈 ÉTAPE 3/3: Analyse dernière bougie...")
    print("-" * 70)

    last = df.iloc[-1]

    print(f"\n💰 PRIX:")
    print(f"   Open:   ${last['open']:.4f}")
    print(f"   High:   ${last['high']:.4f}")
    print(f"   Low:    ${last['low']:.4f}")
    print(f"   Close:  ${last['close']:.4f}")
    print(f"   Volume: {last['volume']:,.0f}")

    print(f"\n📊 TENDANCE (EMA):")
    print(f"   EMA 9:   ${last['ema_9']:.4f}")
    print(f"   EMA 21:  ${last['ema_21']:.4f}")
    print(f"   EMA 50:  ${last['ema_50']:.4f}")
    print(f"   EMA 200: ${last['ema_200']:.4f}")

    # Détection tendance
    if last["ema_9"] > last["ema_21"] > last["ema_50"] > last["ema_200"]:
        print(f"   🟢 Tendance: HAUSSIÈRE FORTE")
    elif last["ema_9"] > last["ema_21"]:
        print(f"   🟢 Tendance: Haussière")
    elif last["ema_9"] < last["ema_21"]:
        print(f"   🔴 Tendance: Baissière")
    else:
        print(f"   🟡 Tendance: Neutre")

    print(f"\n📉 MOMENTUM:")
    print(f"   RSI:        {last['rsi']:.2f}")

    if last["rsi"] > 70:
        print(f"   🔴 RSI: SURACHETÉ")
    elif last["rsi"] < 30:
        print(f"   🟢 RSI: SURVENDU")
    else:
        print(f"   🟡 RSI: Neutre")

    print(f"\n   MACD:       {last['macd']:.4f}")
    print(f"   Signal:     {last['macd_signal']:.4f}")
    print(f"   Histogram: {last['macd_hist']:.4f}")

    if last["macd"] > last["macd_signal"]:
        print(f"   🟢 MACD: Signal haussier")
    else:
        print(f"   🔴 MACD: Signal baissier")

    print(f"\n📍 SUPPORTS/RÉSISTANCES:")
    print(f"   BB Upper:  ${last['bb_upper']:.4f}")
    print(f"   BB Middle: ${last['bb_middle']:.4f}")
    print(f"   BB Lower:  ${last['bb_lower']:.4f}")
    print(f"   VWAP:      ${last['vwap']:.4f}")

    bb_position = (last["close"] - last["bb_lower"]) / (
        last["bb_upper"] - last["bb_lower"]
    )
    print(f"   Position BB: {bb_position*100:.1f}%")

    print(f"\n⚡ VOLATILITÉ:")
    print(f"   ATR:       ${last['atr']:.4f}")
    print(f"   ATR %:     {(last['atr']/last['close'])*100:.2f}%")

    print(f"\n🔄 VORTEX:")
    print(f"   VI+:       {last['vi_plus']:.4f}")
    print(f"   VI-:       {last['vi_minus']:.4f}")

    if last["vi_plus"] > last["vi_minus"]:
        print(f"   🟢 Vortex: Mouvement haussier")
    else:
        print(f"   🔴 Vortex: Mouvement baissier")

    print(f"\n📊 VOLUME:")
    print(f"   OBV:       {last['obv']:,.0f}")

    # Sauvegarde résultats
    output_path = Path("data/exports") / f"{symbol}_{timeframe}_analysis.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)

    print(f"\n💾 Données sauvegardées: {output_path}")

    print("\n" + "=" * 70)
    print("✅ ANALYSE COMPLÈTE")
    print("=" * 70)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Analyse technique complète d'un token"
    )
    parser.add_argument("symbol", type=str, help="Symbol du token (ex: BTCUSDC)")
    parser.add_argument(
        "--timeframe", type=str, default="1h", help="Timeframe (défaut: 1h)"
    )
    parser.add_argument(
        "--days", type=int, default=30, help="Jours d'historique (défaut: 30)"
    )

    args = parser.parse_args()

    df = analyze_token(
        symbol=args.symbol.upper(), timeframe=args.timeframe, days_history=args.days
    )

    if df is not None:
        print(f"\n✅ Analyse de {args.symbol} terminée!")
        print(f"   {len(df)} bougies analysées")
        print(f"   {df.columns.size} colonnes (prix + {df.columns.size - 6} ind.)")
    else:
        print(f"\n❌ Échec analyse de {args.symbol}")
        sys.exit(1)


if __name__ == "__main__":
    main()
