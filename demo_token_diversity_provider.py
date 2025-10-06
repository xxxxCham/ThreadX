#!/usr/bin/env python3
"""
Démonstration du TokenDiversityDataSource
==========================================

Script montrant l'utilisation complète du provider de données ThreadX
pour les tokens organisés par groupes de diversité.
"""

import logging
from threadx.data.providers.token_diversity import (
    TokenDiversityDataSource,
    create_default_config,
)
from threadx.data.errors import DataNotFoundError, UnsupportedTimeframeError

# Configuration logging pour demo
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def demo_provider_usage():
    """Démonstration complète du provider."""
    print("🚀 Démonstration TokenDiversityDataSource")
    print("=" * 50)

    # 1. Configuration par défaut
    print("\n📋 Configuration par défaut")
    config = create_default_config()
    print(f"Groupes configurés: {list(config.groups.keys())}")
    print(f"Total symboles: {len(config.symbols)}")
    print(f"Timeframes supportés: {config.supported_tf}")

    # 2. Initialisation provider
    print("\n🔧 Initialisation provider")
    provider = TokenDiversityDataSource(config)

    # 3. Énumération symboles
    print("\n📊 Énumération des symboles")

    # Tous les symboles
    all_symbols = provider.list_symbols()
    print(f"Tous symboles ({len(all_symbols)}): {all_symbols[:5]}...")

    # Par groupes
    for group_name in ["L1", "L2", "DeFi"]:
        group_symbols = provider.list_symbols(group_name)
        print(f"Groupe {group_name}: {group_symbols}")

    # 4. Récupération données
    print("\n📈 Récupération de données OHLCV")

    # Exemples avec différents symboles/timeframes
    test_cases = [
        ("BTCUSDT", "1h"),
        ("ETHUSDT", "4h"),
        ("ARBUSDT", "1d"),
        ("UNIUSDT", "M1"),  # Test alias
    ]

    for symbol, timeframe in test_cases:
        try:
            df = provider.get_frame(symbol, timeframe)
            print(
                f"✅ {symbol}@{timeframe}: {len(df)} bars, "
                f"période {df.index[0].strftime('%Y-%m-%d')} → "
                f"{df.index[-1].strftime('%Y-%m-%d')}"
            )

            # Aperçu des données
            print(
                f"   Prix moyen: {df['close'].mean():.2f}, "
                f"Volume total: {df['volume'].sum():.0f}"
            )

        except (DataNotFoundError, UnsupportedTimeframeError) as e:
            print(f"❌ {symbol}@{timeframe}: {e.__class__.__name__}: {e}")

    # 5. Test gestion d'erreurs
    print("\n⚠️  Tests de gestion d'erreurs")

    error_cases = [
        ("INVALIDUSDT", "1h", "Symbole inexistant"),
        ("BTCUSDT", "2m", "Timeframe non supporté"),
        ("NONEXISTENT", "7h", "Symbole et TF invalides"),
    ]

    for symbol, timeframe, description in error_cases:
        try:
            df = provider.get_frame(symbol, timeframe)
            print(f"⚠️  {description}: Inattendu, devrait échouer")
        except (DataNotFoundError, UnsupportedTimeframeError) as e:
            print(f"✅ {description}: {e.__class__.__name__}")
        except Exception as e:
            print(f"❌ {description}: Erreur inattendue: {e}")

    # 6. Analyse des groupes
    print("\n🏷️  Analyse des groupes de diversité")

    for group_name, symbols in config.groups.items():
        print(f"\n{group_name} ({len(symbols)} tokens):")

        # Test premier symbole du groupe
        if symbols:
            first_symbol = symbols[0]
            try:
                df = provider.get_frame(first_symbol, "1h")
                avg_price = df["close"].mean()
                volatility = df["close"].std() / avg_price * 100
                print(
                    f"  📊 {first_symbol}: Prix moy={avg_price:.2f}, "
                    f"Volatilité={volatility:.1f}%"
                )
            except Exception as e:
                print(f"  ❌ {first_symbol}: Erreur: {e}")

    print("\n✅ Démonstration terminée avec succès !")


def demo_custom_configuration():
    """Démonstration avec configuration personnalisée."""
    print("\n" + "=" * 50)
    print("🔧 Configuration personnalisée")
    print("=" * 50)

    from threadx.data.providers.token_diversity import TokenDiversityConfig

    # Configuration custom pour trading focus
    custom_config = TokenDiversityConfig(
        groups={
            "MajorPairs": ["BTCUSDT", "ETHUSDT"],
            "Altcoins": ["ADAUSDT", "DOTUSDT", "LINKUSDT"],
            "Memes": ["DOGEUSDT", "SHIBUSDT"],
        },
        symbols=[
            "BTCUSDT",
            "ETHUSDT",
            "ADAUSDT",
            "DOTUSDT",
            "LINKUSDT",
            "DOGEUSDT",
            "SHIBUSDT",
        ],
        supported_tf=("1m", "15m", "1h", "4h"),  # TF réduit
    )

    provider = TokenDiversityDataSource(custom_config)

    print(f"Configuration custom: {len(custom_config.symbols)} symboles")
    print(f"TF supportés: {custom_config.supported_tf}")

    # Test quelques récupérations
    for group in ["MajorPairs", "Altcoins"]:
        symbols = provider.list_symbols(group)
        print(f"\n{group}: {symbols}")

        for symbol in symbols[:2]:  # Premiers 2 symboles
            try:
                df = provider.get_frame(symbol, "1h")
                print(f"  ✅ {symbol}: {len(df)} bars")
            except Exception as e:
                print(f"  ❌ {symbol}: {e}")


if __name__ == "__main__":
    try:
        demo_provider_usage()
        demo_custom_configuration()
    except KeyboardInterrupt:
        print("\n\n⏹️  Démonstration interrompue")
    except Exception as e:
        print(f"\n💥 Erreur durant démonstration: {e}")
        raise
