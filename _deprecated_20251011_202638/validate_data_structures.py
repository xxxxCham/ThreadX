#!/usr/bin/env python3
"""
Script de validation des structures de données ThreadX
======================================================

Ce script vérifie que les DataFrames OHLCV et les indicateurs
respectent les formats requis par ThreadX.

Usage:
    python validate_data_structures.py [--file path/to/data.parquet]
    python validate_data_structures.py --generate-sample
    python validate_data_structures.py --test-indicators
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Union

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from threadx.indicators.bank import IndicatorBank
    from threadx.backtest.engine import BacktestEngine

    THREADX_AVAILABLE = True
except ImportError:
    print("⚠️ ThreadX non disponible, tests basiques uniquement")
    THREADX_AVAILABLE = False


def validate_ohlcv_dataframe(df: pd.DataFrame, name: str = "DataFrame") -> bool:
    """
    Valide qu'un DataFrame respecte le format OHLCV ThreadX.

    Args:
        df: DataFrame à valider
        name: Nom pour les messages d'erreur

    Returns:
        bool: True si valide, False sinon
    """
    print(f"\n🔍 Validation {name}")
    print("=" * 50)

    errors = []
    warnings = []

    # 1. Vérification colonnes requises
    required_cols = ["open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Colonnes manquantes: {missing_cols}")
    else:
        print("✅ Colonnes requises présentes")

    # 2. Vérification index
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        errors.append("Index doit être datetime64")
    else:
        print("✅ Index datetime valide")

        # Vérification timezone
        if df.index.tz is None:
            warnings.append("Index sans timezone (recommandé: UTC)")
        else:
            print(f"✅ Timezone: {df.index.tz}")

    # 3. Vérification types de données
    if "open" in df.columns and not pd.api.types.is_numeric_dtype(df["open"]):
        errors.append("Colonne 'open' doit être numérique")
    if "high" in df.columns and not pd.api.types.is_numeric_dtype(df["high"]):
        errors.append("Colonne 'high' doit être numérique")
    if "low" in df.columns and not pd.api.types.is_numeric_dtype(df["low"]):
        errors.append("Colonne 'low' doit être numérique")
    if "close" in df.columns and not pd.api.types.is_numeric_dtype(df["close"]):
        errors.append("Colonne 'close' doit être numérique")
    if "volume" in df.columns and not pd.api.types.is_numeric_dtype(df["volume"]):
        errors.append("Colonne 'volume' doit être numérique")

    if not errors:
        print("✅ Types de données corrects")

    # 4. Vérification logique OHLC
    if all(col in df.columns for col in ["open", "high", "low", "close"]):
        # High >= max(open, close)
        high_valid = (df["high"] >= df[["open", "close"]].max(axis=1)).all()
        if not high_valid:
            invalid_count = (~(df["high"] >= df[["open", "close"]].max(axis=1))).sum()
            errors.append(f"High < max(open,close) pour {invalid_count} lignes")

        # Low <= min(open, close)
        low_valid = (df["low"] <= df[["open", "close"]].min(axis=1)).all()
        if not low_valid:
            invalid_count = (~(df["low"] <= df[["open", "close"]].min(axis=1))).sum()
            errors.append(f"Low > min(open,close) pour {invalid_count} lignes")

        if high_valid and low_valid:
            print("✅ Logique OHLC respectée")

    # 5. Vérification valeurs positives
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            if (df[col] <= 0).any():
                negative_count = (df[col] <= 0).sum()
                warnings.append(
                    f"Colonne '{col}' contient {negative_count} valeurs <= 0"
                )

    # 6. Vérification NaN
    nan_cols = []
    for col in required_cols:
        if col in df.columns and df[col].isna().any():
            nan_count = df[col].isna().sum()
            nan_cols.append(f"{col}({nan_count})")

    if nan_cols:
        warnings.append(f"Valeurs NaN dans: {', '.join(nan_cols)}")
    else:
        print("✅ Aucune valeur NaN")

    # 7. Statistiques
    print(f"\n📊 Statistiques:")
    print(f"   Lignes: {len(df):,}")
    print(f"   Colonnes: {len(df.columns)}")
    if len(df) > 0:
        print(f"   Période: {df.index[0]} → {df.index[-1]}")
        if len(df) > 1:
            freq = pd.infer_freq(df.index)
            print(f"   Fréquence: {freq or 'Non uniforme'}")

    # Résumé
    print(f"\n📋 Résumé de validation:")
    if errors:
        print("❌ ERREURS:")
        for error in errors:
            print(f"   • {error}")

    if warnings:
        print("⚠️ AVERTISSEMENTS:")
        for warning in warnings:
            print(f"   • {warning}")

    if not errors and not warnings:
        print("✅ DataFrame parfaitement conforme ThreadX")
        return True
    elif not errors:
        print("✅ DataFrame valide avec avertissements mineurs")
        return True
    else:
        print("❌ DataFrame NON CONFORME")
        return False


def generate_sample_data(n_points: int = 1000) -> pd.DataFrame:
    """
    Génère un DataFrame OHLCV d'exemple conforme ThreadX.

    Args:
        n_points: Nombre de points de données

    Returns:
        pd.DataFrame: Données OHLCV valides
    """
    print(f"\n🔧 Génération de {n_points:,} points de données d'exemple")

    # Génération déterministe
    np.random.seed(42)

    # Dates
    dates = pd.date_range(
        "2024-01-01 00:00:00", periods=n_points, freq="1min", tz="UTC"
    )

    # Prix avec trend réaliste
    base_price = 50000.0
    trend = np.linspace(0, 5000, n_points)  # Trend haussier
    noise = np.random.randn(n_points).cumsum() * 100  # Random walk
    volatility = 200 * (
        1 + 0.3 * np.sin(np.arange(n_points) / 1440)
    )  # Volatilité cyclique

    close = base_price + trend + noise

    # OHLC réaliste
    intrabar_range = np.abs(np.random.randn(n_points)) * volatility * 0.5
    high = close + intrabar_range
    low = close - intrabar_range

    # Open = close précédent avec gap occasionnel
    open_price = np.roll(close, 1)
    open_price[0] = close[0]

    # Gaps aléatoires (5% de chance)
    gap_mask = np.random.random(n_points) < 0.05
    gap_size = np.random.randn(n_points) * volatility * 0.2
    open_price[gap_mask] += gap_size[gap_mask]

    # Ajustement OHLC pour respecter la logique
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    # Volume réaliste
    base_volume = 5000
    volume_noise = np.random.lognormal(0, 0.5, n_points)
    volume = (base_volume * volume_noise).astype(int)

    df = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )

    print(f"✅ Données générées:")
    print(f"   Prix moyen: ${close.mean():.2f}")
    print(f"   Volatilité: {close.std():.2f}")
    print(f"   Volume moyen: {volume.mean():,.0f}")

    return df


def test_indicators(df: pd.DataFrame) -> bool:
    """
    Test les indicateurs ThreadX avec les données fournies.

    Args:
        df: DataFrame OHLCV

    Returns:
        bool: True si tous les tests passent
    """
    if not THREADX_AVAILABLE:
        print("⚠️ ThreadX non disponible, skip test indicateurs")
        return False

    print(f"\n🧪 Test des indicateurs ThreadX")
    print("=" * 50)

    try:
        bank = IndicatorBank()

        # Test Bollinger Bands
        print("🔵 Test Bollinger Bands...")
        bb_result = bank.ensure(
            "bollinger", {"period": 20, "std": 2.0}, df, symbol="TEST", timeframe="1m"
        )

        if isinstance(bb_result, tuple) and len(bb_result) == 3:
            upper, middle, lower = bb_result
            print(f"   ✅ Format: Tuple[Series, Series, Series]")
            print(f"   ✅ Taille: {len(upper)} points")
            print(f"   ✅ Bande sup moyenne: {np.mean(upper):.2f}")
            print(f"   ✅ Bande inf moyenne: {np.mean(lower):.2f}")
        else:
            print(f"   ❌ Format incorrect: {type(bb_result)}")
            return False

        # Test ATR
        print("🟠 Test ATR...")
        atr_result = bank.ensure(
            "atr", {"period": 14}, df, symbol="TEST", timeframe="1m"
        )

        if isinstance(atr_result, (np.ndarray, pd.Series)):
            print(f"   ✅ Format: {type(atr_result).__name__}")
            print(f"   ✅ Taille: {len(atr_result)} points")
            print(f"   ✅ ATR moyen: {np.mean(atr_result):.4f}")
        else:
            print(f"   ❌ Format incorrect: {type(atr_result)}")
            return False

        # Test intégration BacktestEngine
        print("🚀 Test BacktestEngine...")
        engine = BacktestEngine()

        indicators = {"bollinger": bb_result, "atr": atr_result}

        strategy_params = {
            "entry_z": 2.0,
            "k_sl": 1.5,
            "leverage": 3.0,
            "initial_capital": 10000.0,
            "fees_bps": 10.0,
            "slip_bps": 5.0,
        }

        result = engine.run(
            df_1m=df,
            indicators=indicators,
            params=strategy_params,
            symbol="TEST",
            timeframe="1m",
            seed=42,
        )

        print(f"   ✅ Backtest exécuté")
        print(f"   ✅ Trades: {len(result.trades)}")
        print(f"   ✅ Equity finale: ${result.equity.iloc[-1]:,.2f}")
        print(f"   ✅ Métadonnées: {list(result.meta.keys())}")

        print("\n✅ Tous les tests indicateurs réussis")
        return True

    except Exception as e:
        print(f"\n❌ Erreur test indicateurs: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description="Validation des structures de données ThreadX"
    )
    parser.add_argument("--file", type=str, help="Fichier Parquet à valider")
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Génère et sauvegarde un échantillon",
    )
    parser.add_argument(
        "--test-indicators", action="store_true", help="Test les indicateurs ThreadX"
    )
    parser.add_argument(
        "--points", type=int, default=1000, help="Nombre de points pour l'échantillon"
    )

    args = parser.parse_args()

    print("🧪 ThreadX Data Structure Validator")
    print("=" * 60)

    success = True

    if args.generate_sample:
        # Génération échantillon
        df_sample = generate_sample_data(args.points)

        # Validation de l'échantillon
        if validate_ohlcv_dataframe(df_sample, "Échantillon généré"):
            # Sauvegarde
            output_file = PROJECT_ROOT / f"sample_ohlcv_{args.points}pts.parquet"
            df_sample.to_parquet(output_file)
            print(f"\n💾 Échantillon sauvegardé: {output_file}")

            # Test indicateurs si demandé
            if args.test_indicators:
                success = test_indicators(df_sample) and success
        else:
            success = False

    elif args.file:
        # Validation fichier spécifique
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ Fichier non trouvé: {file_path}")
            return False

        try:
            df = pd.read_parquet(file_path)
            success = validate_ohlcv_dataframe(df, f"Fichier {file_path.name}")

            if success and args.test_indicators:
                success = test_indicators(df) and success

        except Exception as e:
            print(f"❌ Erreur lecture fichier: {e}")
            success = False

    else:
        # Mode par défaut : génère un petit échantillon et teste
        print("Mode par défaut : génération et test d'un petit échantillon")
        df_sample = generate_sample_data(100)

        success = validate_ohlcv_dataframe(df_sample, "Échantillon test")

        if success and THREADX_AVAILABLE:
            success = test_indicators(df_sample) and success

    # Résumé final
    print("\n" + "=" * 60)
    if success:
        print("🎉 Validation RÉUSSIE - Données conformes ThreadX")
        return True
    else:
        print("💥 Validation ÉCHOUÉE - Voir erreurs ci-dessus")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
