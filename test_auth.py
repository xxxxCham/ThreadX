#!/usr/bin/env python3
"""
ThreadX Authentication Test Script
=================================

Script de test pour vérifier la configuration d'authentification ThreadX.
Vérifie les variables d'environnement, teste les connexions API, et fournit
des diagnostics détaillés.

Usage:
    python test_auth.py
    python test_auth.py --verbose
    python test_auth.py --test-binance
    python test_auth.py --test-coingecko

Author: ThreadX Framework
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Ajouter le chemin src pour les imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from threadx.config.auth import AuthManager, get_auth_manager
    from threadx.data.client import DataClient, get_data_client
    from threadx.config.settings import get_settings
    THREADX_AVAILABLE = True
except ImportError as e:
    print(f"❌ Erreur import ThreadX: {e}")
    print("💡 Assurez-vous que ThreadX est installé: pip install -e .")
    THREADX_AVAILABLE = False


def check_environment_variables():
    """Vérifie les variables d'environnement."""
    print("\n🔍 VÉRIFICATION DES VARIABLES D'ENVIRONNEMENT")
    print("=" * 60)
    
    # Variables importantes
    important_vars = {
        'BINANCE_API_KEY': 'Clé API Binance',
        'BINANCE_API_SECRET': 'Secret API Binance',
        'COINGECKO_API_KEY': 'Clé API CoinGecko (optionnelle)',
        'ALPHA_VANTAGE_API_KEY': 'Clé API Alpha Vantage (optionnelle)',
        'POLYGON_API_KEY': 'Clé API Polygon (optionnelle)',
        'FINNHUB_API_KEY': 'Clé API Finnhub (optionnelle)'
    }
    
    found_vars = 0
    for var_name, description in important_vars.items():
        value = os.getenv(var_name)
        if value:
            # Masquer les clés sensibles
            if len(value) > 8:
                masked_value = f"{value[:4]}...{value[-4:]}"
            else:
                masked_value = "***"
            
            print(f"✅ {var_name:20s}: {masked_value} ({description})")
            found_vars += 1
        else:
            print(f"❌ {var_name:20s}: Non définie ({description})")
    
    print(f"\n📊 Résumé: {found_vars}/{len(important_vars)} variables configurées")
    
    # Vérifications spéciales
    binance_complete = os.getenv('BINANCE_API_KEY') and os.getenv('BINANCE_API_SECRET')
    if binance_complete:
        print("✅ Configuration Binance complète (clé + secret)")
    elif os.getenv('BINANCE_API_KEY') or os.getenv('BINANCE_API_SECRET'):
        print("⚠️ Configuration Binance incomplète (clé OU secret manquant)")
    else:
        print("❌ Configuration Binance absente")
    
    return found_vars > 0


def test_auth_manager():
    """Teste le gestionnaire d'authentification."""
    if not THREADX_AVAILABLE:
        print("❌ ThreadX non disponible - impossible de tester AuthManager")
        return False
    
    print("\n🔐 TEST DU GESTIONNAIRE D'AUTHENTIFICATION")
    print("=" * 60)
    
    try:
        auth = get_auth_manager()
        print("✅ AuthManager initialisé avec succès")
        
        # Test validation des credentials
        validation_results = auth.validate_all_credentials()
        print(f"\n📋 Validation des credentials:")
        
        for provider, is_valid in validation_results.items():
            status = "✅ Valide" if is_valid else "❌ Invalide/Manquant"
            print(f"  - {provider.capitalize():15s}: {status}")
        
        # Test récupération credentials Binance
        if auth.has_binance_credentials():
            print("\n🔑 Test récupération credentials Binance:")
            try:
                api_key, api_secret = auth.get_binance_credentials()
                print(f"  ✅ API Key: {api_key[:8]}...")
                print(f"  ✅ Secret: {api_secret[:8]}...")
            except Exception as e:
                print(f"  ❌ Erreur récupération: {e}")
        
        # Test URLs et rate limits
        print(f"\n🌐 URLs et limites:")
        for provider in ['binance', 'coingecko']:
            url = auth.get_api_url(provider)
            rate_limit = auth.get_rate_limit(provider)
            print(f"  - {provider.capitalize():15s}: {url} ({rate_limit} req/min)")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur AuthManager: {e}")
        return False


def test_data_client():
    """Teste le client de données."""
    if not THREADX_AVAILABLE:
        print("❌ ThreadX non disponible - impossible de tester DataClient")
        return False
    
    print("\n📡 TEST DU CLIENT DE DONNÉES")
    print("=" * 60)
    
    try:
        client = get_data_client()
        print("✅ DataClient initialisé avec succès")
        
        # Test des connexions
        print("\n🧪 Test des connexions API:")
        connections = client.test_all_connections()
        
        success_count = 0
        for provider, status in connections.items():
            status_text = "✅ OK" if status else "❌ Échec"
            print(f"  - {provider.capitalize():15s}: {status_text}")
            if status:
                success_count += 1
        
        total_count = len(connections)
        success_rate = success_count / total_count * 100
        
        print(f"\n📊 Taux de succès: {success_rate:.1f}% ({success_count}/{total_count})")
        
        return success_count > 0
        
    except Exception as e:
        print(f"❌ Erreur DataClient: {e}")
        return False


def test_binance_api(verbose=False):
    """Teste spécifiquement l'API Binance."""
    if not THREADX_AVAILABLE:
        print("❌ ThreadX non disponible")
        return False
    
    print("\n🟡 TEST SPÉCIFIQUE BINANCE API")
    print("=" * 60)
    
    try:
        client = get_data_client()
        
        # Test téléchargement données OHLCV
        print("📥 Test téléchargement données BTCUSDC 1d...")
        df = client.get_binance_klines("BTCUSDC", "1d", days=7)
        
        print(f"✅ Données téléchargées: {len(df)} lignes")
        print(f"📅 Période: {df.index[0]} → {df.index[-1]}")
        print(f"💰 Prix actuel: ${df['close'].iloc[-1]:,.2f}")
        
        if verbose:
            print(f"\n📊 Aperçu des données:")
            print(df.head())
            
            print(f"\n📈 Statistiques:")
            print(f"  - Prix min: ${df['close'].min():,.2f}")
            print(f"  - Prix max: ${df['close'].max():,.2f}")
            print(f"  - Volume moyen: {df['volume'].mean():,.0f}")
        
        # Test ticker 24h
        print(f"\n📊 Test ticker 24h...")
        tickers = client.get_binance_24hr_ticker()
        
        print(f"✅ Tickers reçus: {len(tickers)} symboles USDC")
        if tickers and verbose:
            print("🏆 Top 5 par volume:")
            for i, ticker in enumerate(tickers[:5], 1):
                print(f"  {i}. {ticker['symbol']:8s} - Volume: ${ticker['volume']:,.0f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur test Binance: {e}")
        return False


def test_coingecko_api(verbose=False):
    """Teste spécifiquement l'API CoinGecko."""
    if not THREADX_AVAILABLE:
        print("❌ ThreadX non disponible")
        return False
    
    print("\n🦎 TEST SPÉCIFIQUE COINGECKO API")
    print("=" * 60)
    
    try:
        client = get_data_client()
        
        # Test récupération top coins
        print("📥 Test récupération top 20 coins...")
        coins = client.get_coingecko_coins(limit=20)
        
        print(f"✅ Coins téléchargés: {len(coins)}")
        
        if verbose and coins:
            print(f"\n🏆 Top 10 par market cap:")
            for i, coin in enumerate(coins[:10], 1):
                print(f"  {i:2d}. {coin['symbol']:8s} - {coin['name'][:20]:<20s} "
                      f"${coin['price']:>10.2f} (#{coin['market_cap_rank']})")
        
        # Vérifier les données importantes
        btc_data = next((coin for coin in coins if coin['symbol'] == 'BTC'), None)
        if btc_data:
            print(f"\n₿ Bitcoin Info:")
            print(f"  - Prix: ${btc_data['price']:,.2f}")
            print(f"  - Market Cap Rank: #{btc_data['market_cap_rank']}")
            print(f"  - Volume 24h: ${btc_data['volume']:,.0f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur test CoinGecko: {e}")
        return False


def generate_diagnostic_report():
    """Génère un rapport de diagnostic complet."""
    print("\n📋 GÉNÉRATION DU RAPPORT DE DIAGNOSTIC")
    print("=" * 60)
    
    report = {
        "timestamp": str(pd.Timestamp.now()),
        "environment": {
            "python_version": sys.version,
            "platform": sys.platform,
            "threadx_available": THREADX_AVAILABLE
        },
        "environment_variables": {},
        "authentication": {},
        "connectivity": {}
    }
    
    # Variables d'environnement
    env_vars = ['BINANCE_API_KEY', 'BINANCE_API_SECRET', 'COINGECKO_API_KEY']
    for var in env_vars:
        value = os.getenv(var)
        report["environment_variables"][var] = {
            "present": value is not None,
            "length": len(value) if value else 0
        }
    
    # Tests d'authentification
    if THREADX_AVAILABLE:
        try:
            auth = get_auth_manager()
            report["authentication"] = auth.validate_all_credentials()
        except Exception as e:
            report["authentication"]["error"] = str(e)
        
        # Tests de connectivité
        try:
            client = get_data_client()
            report["connectivity"] = client.test_all_connections()
        except Exception as e:
            report["connectivity"]["error"] = str(e)
    
    # Sauvegarder le rapport
    report_file = Path("threadx_diagnostic_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Rapport sauvegardé: {report_file}")
    return report


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Test d'authentification ThreadX")
    parser.add_argument("--verbose", "-v", action="store_true", help="Mode verbeux")
    parser.add_argument("--test-binance", action="store_true", help="Test spécifique Binance")
    parser.add_argument("--test-coingecko", action="store_true", help="Test spécifique CoinGecko")
    parser.add_argument("--generate-report", action="store_true", help="Générer rapport diagnostic")
    
    args = parser.parse_args()
    
    print("🔐 THREADX AUTHENTICATION TEST SCRIPT")
    print("=" * 60)
    
    # Import pandas pour les timestamps si disponible
    try:
        import pandas as pd
    except ImportError:
        print("⚠️ Pandas non disponible - timestamps simplifiés")
        pd = None
    
    # Tests de base
    env_ok = check_environment_variables()
    
    if THREADX_AVAILABLE:
        auth_ok = test_auth_manager()
        data_ok = test_data_client()
    else:
        auth_ok = data_ok = False
    
    # Tests spécifiques
    if args.test_binance:
        test_binance_api(args.verbose)
    
    if args.test_coingecko:
        test_coingecko_api(args.verbose)
    
    # Rapport diagnostic
    if args.generate_report:
        generate_diagnostic_report()
    
    # Résumé final
    print("\n📊 RÉSUMÉ FINAL")
    print("=" * 60)
    
    if env_ok and auth_ok and data_ok:
        print("🎉 ✅ Tous les tests sont passés avec succès!")
        print("💡 ThreadX est prêt à utiliser avec authentification complète.")
    elif env_ok and not THREADX_AVAILABLE:
        print("⚠️ Variables d'environnement configurées mais ThreadX non installé")
        print("💡 Installez ThreadX: pip install -e .")
    elif not env_ok:
        print("❌ Variables d'environnement manquantes")
        print("💡 Consultez le fichier .env.example pour la configuration")
    else:
        print("⚠️ Configuration partielle détectée")
        print("💡 Vérifiez les messages d'erreur ci-dessus")
    
    print(f"\n🔗 Liens utiles:")
    print(f"  - Configuration: .env.example")
    print(f"  - Documentation Binance: https://binance-docs.github.io/apidocs/")
    print(f"  - Documentation CoinGecko: https://www.coingecko.com/en/api/documentation")


if __name__ == "__main__":
    main()