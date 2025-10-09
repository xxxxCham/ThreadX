#!/usr/bin/env python3
"""
Exemple concret d'utilisation TokenDiversityManager Option B
===========================================================

Démonstration complète du pipeline unifié :
1. Configuration & initialisation
2. Données prix seules (OHLCV)
3. Prix + indicateurs simples
4. Pipeline complet avec cache TTL
5. Export multi-formats
6. Intégration UI/CLI
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Path setup
sys.path.insert(0, str(Path(__file__).parent / "src"))

from threadx.data.providers.token_diversity import (
    TokenDiversityManager,
    IndicatorSpec,
    PriceSourceSpec,
)


def example_1_ohlcv_only():
    """Exemple 1 : Données OHLCV seules."""
    print("\n📊 Exemple 1 : OHLCV seul")
    print("-" * 40)

    manager = TokenDiversityManager()

    df, meta = manager.prepare_dataframe(
        market="BTCUSDT",
        timeframe="1h",
        start="2023-01-01",
        end="2023-01-15",  # 2 semaines
        indicators=[],  # Pas d'indicateurs
        price_source=PriceSourceSpec(name="stub", params={}),
    )

    print(f"✅ DataFrame OHLCV obtenu :")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    print(
        f"   Period: {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}"
    )
    print(f"   Index type: {type(df.index)} (tz: {df.index.tz})")
    print(f"   Execution: {meta['execution_time_ms']:.1f}ms")
    print(f"   Cache hit: {meta['cache_hit']}")

    # Validation OHLCV
    print(f"\n📈 Validation OHLCV :")
    print(
        f"   OHLC constraints OK: {(df['high'] >= df[['open', 'close']].max(axis=1)).all()}"
    )
    print(f"   Volume positive: {(df['volume'] > 0).all()}")
    print(f"   No NaN: {df.isna().sum().sum() == 0}")

    return df, meta


def example_2_with_indicators():
    """Exemple 2 : OHLCV + indicateurs RSI et Bollinger Bands."""
    print("\n🔧 Exemple 2 : OHLCV + Indicateurs")
    print("-" * 45)

    manager = TokenDiversityManager()

    indicators = [
        IndicatorSpec(name="rsi", params={"window": 14}),
        IndicatorSpec(name="bbands", params={"window": 20, "n_std": 2.0}),
    ]

    try:
        df, meta = manager.prepare_dataframe(
            market="ETHUSDT",
            timeframe="4h",
            start="2023-02-01",
            end="2023-03-01",  # 1 mois
            indicators=indicators,
            price_source=PriceSourceSpec(name="stub", params={}),
            seed=42,  # Reproductibilité
        )

        print(f"✅ DataFrame avec indicateurs :")
        print(f"   Rows: {len(df)}")
        print(f"   Total columns: {len(df.columns)}")

        # Séparation colonnes
        ohlcv_cols = [col for col in df.columns if not col.startswith("ind_")]
        indicator_cols = [col for col in df.columns if col.startswith("ind_")]

        print(f"   OHLCV columns: {ohlcv_cols}")
        print(f"   Indicator columns: {indicator_cols}")
        print(f"   Indicators computed: {meta['indicators_count']}")
        print(f"   Execution: {meta['execution_time_ms']:.1f}ms")
        print(f"   Coverage: {meta['coverage_pct']:.1f}%")

        # Analyse indicateurs
        if indicator_cols:
            print(f"\n📊 Analyse indicateurs :")
            for col in indicator_cols:
                values = df[col].dropna()
                if len(values) > 0:
                    print(
                        f"   {col}: min={values.min():.2f}, max={values.max():.2f}, mean={values.mean():.2f}"
                    )

        return df, meta

    except ImportError:
        print("⚠️  IndicatorBank non disponible, indicateurs ignorés")
        # Fallback OHLCV seul
        df, meta = manager.prepare_dataframe(
            market="ETHUSDT",
            timeframe="4h",
            start="2023-02-01",
            end="2023-03-01",
            indicators=[],
            price_source=PriceSourceSpec(name="stub", params={}),
        )
        print(f"✅ Fallback OHLCV : {len(df)} rows")
        return df, meta


def example_3_cache_performance():
    """Exemple 3 : Démonstration cache TTL et performance."""
    print("\n⚡ Exemple 3 : Cache TTL & Performance")
    print("-" * 45)

    manager = TokenDiversityManager()

    params = {
        "market": "SOLUSDT",
        "timeframe": "1h",
        "start": "2023-03-01",
        "end": "2023-03-15",
        "indicators": [
            IndicatorSpec(name="rsi", params={"window": 14}),
        ],
        "price_source": PriceSourceSpec(name="stub", params={}),
        "cache_ttl_sec": 30,  # 30 secondes TTL
    }

    # Premier appel (cache miss)
    print("🔄 Premier appel (cache miss)...")
    import time

    start_time = time.time()
    df1, meta1 = manager.prepare_dataframe(**params)
    first_latency = (time.time() - start_time) * 1000

    print(f"   Latency: {first_latency:.1f}ms")
    print(f"   Cache hit: {meta1['cache_hit']}")
    print(f"   Rows: {len(df1)}")

    # Deuxième appel immédiat (cache hit)
    print("\n🏃 Deuxième appel immédiat (cache hit)...")
    start_time = time.time()
    df2, meta2 = manager.prepare_dataframe(**params)
    second_latency = (time.time() - start_time) * 1000

    print(f"   Latency: {second_latency:.1f}ms")
    print(f"   Cache hit: {meta2['cache_hit']}")
    print(f"   Speedup: {first_latency/second_latency:.1f}x")

    # Vérification identité
    are_identical = df1.equals(df2)
    print(f"   DataFrames identiques: {are_identical}")

    # Stats manager
    stats = manager.get_stats()
    print(f"\n📈 Stats manager :")
    print(f"   Total calls: {stats['prepare_calls']}")
    print(f"   Cache hits: {stats['cache_hits']}")
    print(f"   Cache hit rate: {stats['cache_hit_rate']:.2%}")
    print(f"   Cache size: {stats['cache_size']}")

    return df1, meta1


def example_4_export_formats():
    """Exemple 4 : Export multi-formats (CSV, Parquet)."""
    print("\n💾 Exemple 4 : Export Multi-Formats")
    print("-" * 42)

    manager = TokenDiversityManager()

    # Données avec plusieurs indicateurs
    indicators = [
        IndicatorSpec(name="rsi", params={"window": 14}),
        IndicatorSpec(name="atr", params={"window": 14}),
    ]

    try:
        df, meta = manager.prepare_dataframe(
            market="ADAUSDT",
            timeframe="15m",
            start="2023-04-01",
            end="2023-04-15",  # 2 semaines
            indicators=indicators,
            price_source=PriceSourceSpec(name="stub", params={}),
        )

        # Création répertoire export
        export_dir = Path("exports")
        export_dir.mkdir(exist_ok=True)

        # Export CSV
        csv_path = export_dir / "adausdt_15m_indicators.csv"
        df.to_csv(csv_path, index=True)
        csv_size = csv_path.stat().st_size / 1024  # KB
        print(f"✅ CSV exporté : {csv_path} ({csv_size:.1f} KB)")

        # Export Parquet (plus efficace)
        parquet_path = export_dir / "adausdt_15m_indicators.parquet"
        df.to_parquet(parquet_path, compression="snappy", engine="pyarrow")
        parquet_size = parquet_path.stat().st_size / 1024  # KB
        print(f"✅ Parquet exporté : {parquet_path} ({parquet_size:.1f} KB)")

        # Comparaison tailles
        compression_ratio = csv_size / parquet_size
        print(f"📊 Compression ratio: {compression_ratio:.1f}x (Parquet vs CSV)")

        # Métadonnées d'export
        meta_path = export_dir / "adausdt_15m_metadata.json"
        import json

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"✅ Métadonnées : {meta_path}")

        return df, export_dir

    except ImportError:
        print("⚠️  Export limité (IndicatorBank non disponible)")
        return None, None


def example_5_ui_integration_hook():
    """Exemple 5 : Hook d'intégration UI (pagination/échantillonnage)."""
    print("\n🖥️  Exemple 5 : Intégration UI")
    print("-" * 38)

    manager = TokenDiversityManager()

    def get_dataframe_for_ui(
        market: str,
        timeframe: str,
        days_back: int = 30,
        max_rows: int = 1000,
        sample_indicators: bool = True,
    ):
        """Hook UI : données récentes avec pagination."""

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        indicators = []
        if sample_indicators:
            indicators = [
                IndicatorSpec(name="rsi", params={"window": 14}),
            ]

        try:
            df, meta = manager.prepare_dataframe(
                market=market,
                timeframe=timeframe,
                start=start_date,
                end=end_date,
                indicators=indicators,
                price_source=PriceSourceSpec(name="stub", params={}),
                cache_ttl_sec=300,  # 5min cache pour UI
            )

            # Pagination si nécessaire
            if len(df) > max_rows:
                # Échantillonnage uniforme
                step = len(df) // max_rows
                df_sampled = df.iloc[::step].head(max_rows)
                print(
                    f"📱 UI pagination : {len(df)} → {len(df_sampled)} rows (step={step})"
                )
                df = df_sampled

            return df, meta

        except Exception as e:
            print(f"❌ Erreur UI hook : {e}")
            return pd.DataFrame(), {}

    # Test hook UI
    df_ui, meta_ui = get_dataframe_for_ui(
        market="BTCUSDT",
        timeframe="1h",
        days_back=7,  # 1 semaine
        max_rows=500,
        sample_indicators=True,
    )

    if not df_ui.empty:
        print(f"✅ Données UI prêtes :")
        print(f"   Rows: {len(df_ui)} (optimisé pour UI)")
        print(f"   Columns: {len(df_ui.columns)}")
        print(f"   Latest: {df_ui.index[-1].strftime('%Y-%m-%d %H:%M')}")
        print(f"   Cache hit: {meta_ui.get('cache_hit', False)}")
        print(f"   Latency: {meta_ui.get('execution_time_ms', 0):.1f}ms")

    return df_ui, meta_ui


def main():
    """Démonstration complète des exemples."""
    print("🚀 TokenDiversityManager Option B - Exemples Concrets")
    print("=" * 60)

    try:
        # Exemples séquentiels
        example_1_ohlcv_only()
        example_2_with_indicators()
        example_3_cache_performance()
        example_4_export_formats()
        example_5_ui_integration_hook()

        print("\n🎯 Tous les exemples exécutés avec succès !")
        print("\nFichiers générés :")
        print("  - exports/adausdt_15m_indicators.csv")
        print("  - exports/adausdt_15m_indicators.parquet")
        print("  - exports/adausdt_15m_metadata.json")

        print("\n📋 Prochaines étapes :")
        print("  1. Intégrer dans ThreadX UI (Data Manager)")
        print("  2. Connecter sources externes (Binance, etc.)")
        print("  3. Optimiser cache TTL selon usage")
        print("  4. Benchmarker sur données réelles")

    except Exception as e:
        print(f"\n❌ Erreur durant les exemples : {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
