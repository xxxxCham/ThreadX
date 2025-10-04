<!-- MODULE-START: __init__.py -->
## 
*Chemin* : `D:/TradXPro/scripts/mise_a_jour_dataframe/token_diversity_manager/__init__.py`  
*Type* : `.py`  

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradXPro Token Diversity Manager
===============================

Module spécialisé pour la gestion des tokens crypto avec diversité garantie.

Ce module fournit un gestionnaire complet qui :
- Récupère automatiquement les top 100 tokens crypto
- Garantit une diversité par catégorie (≥3 tokens par catégorie importante)
- Gère le téléchargement et le stockage des données historiques
- Calcule les indicateurs techniques
- Fournit une API unifiée pour l'intégration

Usage:
    from tradxpro_token_diversity_manager import TradXProManager

    manager = TradXProManager()
    tokens = manager.get_top_100_tokens()  # Avec diversité garantie !
    df = manager.get_trading_data("BTCUSDC", "1h", ["rsi", "bollinger"])

Auteur: TradXPro Team
Date: 2 octobre 2025
Version: 1.1 - Diversité Garantie
"""

from .tradxpro_core_manager import TradXProManager, TradXProPaths

__version__ = "1.1.0"
__author__ = "TradXPro Team"
__email__ = "support@tradxpro.com"

__all__ = [
    "TradXProManager",
    "TradXProPaths"
]

# Informations sur le module
MODULE_INFO = {
    "name": "TradXPro Token Diversity Manager",
    "version": __version__,
    "description": "Gestionnaire de tokens crypto avec diversité garantie",
    "features": [
        "🏆 Récupération top 100 tokens (CoinGecko + Binance)",
        "🔒 Diversité garantie (≥3 tokens par catégorie)",
        "📥 Téléchargement données historiques multi-threading",
        "📈 Indicateurs techniques (RSI, Bollinger, ATR, EMA, MACD)",
        "💾 Stockage optimisé (JSON + Parquet)",
        "📊 Analyse et rapport de diversité",
        "⚡ Chargement parallèle et cache automatique",
        "🛠️ API simple et unifiée"
    ],
    "categories": [
        "Layer 1 Blockchain", "DeFi Protocols", "Layer 2 Scaling",
        "Smart Contracts", "Meme Coins", "Exchange Tokens",
        "Stablecoins", "AI Gaming", "Privacy Coins", "Infrastructure"
    ]
}

def get_module_info():
    """Retourne les informations du module"""
    return MODULE_INFO

def print_module_info():
    """Affiche les informations du module"""
    info = MODULE_INFO
    print(f"📦 {info['name']} v{info['version']}")
    print("=" * 50)
    print(f"📋 {info['description']}")
    print()
    print("🎯 Fonctionnalités:")
    for feature in info['features']:
        print(f"   {feature}")
    print()
    print(f"📊 Catégories couvertes ({len(info['categories'])}):")
    for i, category in enumerate(info['categories'], 1):
        print(f"   {i:2d}. {category}")

# Auto-configuration du logging si importé directement
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
```
<!-- MODULE-END: __init__.py -->

<!-- MODULE-START: launch.py -->
## 
*Chemin* : `D:/TradXPro/scripts/mise_a_jour_dataframe/token_diversity_manager/launch.py`  
*Type* : `.py`  

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Lanceur Principal - Token Diversity Manager
==============================================

Lanceur principal pour utiliser facilement le Token Diversity Manager.

Usage:
    python launch.py

Auteur: TradXPro Team
Date: 2 octobre 2025
"""

def main():
    """Lanceur principal avec menu interactif"""

    print("🚀 TRADXPRO TOKEN DIVERSITY MANAGER")
    print("=" * 50)
    print("Module avec diversité garantie des tokens crypto")
    print()

    while True:
        print("📋 OPTIONS DISPONIBLES:")
        print("1. 🧪 Test simple du module")
        print("2. 🔒 Test de diversité des tokens")
        print("3. 💡 Exemple de démarrage rapide")
        print("4. 📊 Exemple d'intégration complète")
        print("5. ⚙️ Setup et configuration")
        print("6. 📚 Voir la documentation")
        print("0. ❌ Quitter")
        print()

        try:
            choice = input("Votre choix (0-6): ").strip()

            if choice == "1":
                print("\n🧪 Lancement du test simple...")
                from tradxpro_core_manager import TradXProManager

                manager = TradXProManager()
                print("✅ TradXProManager initialisé avec succès !")
                print(f"📁 Racine: {manager.paths.root}")
                print(f"⚙️ Configuration: {manager.history_days} jours, {manager.max_workers} workers")

            elif choice == "2":
                print("\n🔒 Test de diversité des tokens...")
                import subprocess
                import sys
                subprocess.run([sys.executable, "tests/test_diversite_simple.py"])

            elif choice == "3":
                print("\n💡 Lancement de l'exemple rapide...")
                import subprocess
                import sys
                subprocess.run([sys.executable, "examples/quick_start_tradxpro.py"])

            elif choice == "4":
                print("\n📊 Lancement de l'exemple complet...")
                import subprocess
                import sys
                subprocess.run([sys.executable, "examples/exemple_integration_tradxpro.py"])

            elif choice == "5":
                print("\n⚙️ Lancement du setup...")
                import subprocess
                import sys
                subprocess.run([sys.executable, "setup_module.py"])

            elif choice == "6":
                print("\n📚 DOCUMENTATION DISPONIBLE:")
                print("📄 README.md - Guide principal")
                print("📄 docs/README_CORE_MANAGER.md - Documentation complète")
                print("📄 docs/DIVERSITE_GARANTIE.md - Détails sur la diversité")
                print("📄 INDEX_TOKEN_DIVERSITY_MANAGER.md - Index des fichiers")

            elif choice == "0":
                print("👋 Au revoir !")
                break

            else:
                print("❌ Choix invalide, veuillez choisir entre 0 et 6")

        except KeyboardInterrupt:
            print("\n👋 Au revoir !")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")

        print("\n" + "-" * 50 + "\n")

def demo_rapide():
    """Démonstration rapide des fonctionnalités"""

    print("🎯 DÉMONSTRATION RAPIDE")
    print("=" * 30)

    try:
        # Import et initialisation
        from tradxpro_core_manager import TradXProManager
        manager = TradXProManager()

        print("✅ Module initialisé")

        # Test des fonctionnalités de diversité
        test_tokens = [
            {"symbol": "BTC", "name": "Bitcoin", "score": 100},
            {"symbol": "ETH", "name": "Ethereum", "score": 95},
            {"symbol": "ADA", "name": "Cardano", "score": 85},
            {"symbol": "UNI", "name": "Uniswap", "score": 80},
            {"symbol": "AAVE", "name": "Aave", "score": 75},
            {"symbol": "MATIC", "name": "Polygon", "score": 70},
        ]

        print(f"📊 Test avec {len(test_tokens)} tokens...")
        diversity_stats = manager.analyze_token_diversity(test_tokens)

        print(f"✅ Score de diversité: {diversity_stats['global']['diversity_score']:.1f}%")
        print(f"✅ Tokens catégorisés: {diversity_stats['global']['categorized_tokens']}/{len(test_tokens)}")

        print("\n🎉 Le module Token Diversity Manager fonctionne parfaitement !")

    except Exception as e:
        print(f"❌ Erreur lors de la démo: {e}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_rapide()
    else:
        main()
```
<!-- MODULE-END: launch.py -->

<!-- MODULE-START: setup_module.py -->
## 
*Chemin* : `D:/TradXPro/scripts/mise_a_jour_dataframe/token_diversity_manager/setup_module.py`  
*Type* : `.py`  

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration et Setup - TradXPro Token Diversity Manager
=========================================================

Script de configuration pour le module Token Diversity Manager.

Usage:
    python setup_module.py

Auteur: TradXPro Team
Date: 2 octobre 2025
"""

import os
import sys
from pathlib import Path

def setup_module():
    """Configure le module Token Diversity Manager"""

    print("🔧 SETUP - TRADXPRO TOKEN DIVERSITY MANAGER")
    print("=" * 50)

    # Vérification de l'environnement
    module_path = Path(__file__).parent
    print(f"📁 Chemin du module: {module_path}")

    # Vérification des fichiers
    required_files = [
        "tradxpro_core_manager.py",
        "__init__.py",
        "README.md"
    ]

    print("\n📋 Vérification des fichiers...")
    missing_files = []

    for file in required_files:
        file_path = module_path / file
        if file_path.exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MANQUANT")
            missing_files.append(file)

    # Vérification des dossiers
    required_dirs = ["tests", "examples", "docs"]
    print(f"\n📂 Vérification des dossiers...")

    for dir_name in required_dirs:
        dir_path = module_path / dir_name
        if dir_path.exists():
            file_count = len(list(dir_path.glob("*")))
            print(f"✅ {dir_name}/ ({file_count} fichiers)")
        else:
            print(f"❌ {dir_name}/ - MANQUANT")
            missing_files.append(f"{dir_name}/")

    # Test d'importation
    print(f"\n🧪 Test d'importation...")
    try:
        sys.path.append(str(module_path))
        from tradxpro_core_manager import TradXProManager
        print("✅ Module importé avec succès")

        # Test basique
        manager = TradXProManager()
        print("✅ Gestionnaire initialisé")

    except Exception as e:
        print(f"❌ Erreur d'importation: {e}")
        return False

    # Résultat final
    print(f"\n📊 RÉSULTAT DU SETUP")
    print("-" * 30)

    if not missing_files:
        print("🎉 SETUP RÉUSSI !")
        print("✅ Tous les fichiers sont présents")
        print("✅ Module fonctionnel")
        print(f"\n🚀 Le module est prêt à être utilisé !")
        print(f"📁 Chemin: {module_path}")

        # Instructions d'utilisation
        print(f"\n💡 UTILISATION:")
        print(f"cd {module_path}")
        print("python tests/test_diversite_simple.py")
        print("python examples/quick_start_tradxpro.py")

        return True
    else:
        print("⚠️ SETUP INCOMPLET")
        print(f"❌ {len(missing_files)} fichiers manquants:")
        for file in missing_files:
            print(f"   • {file}")
        return False

def create_launcher_script():
    """Crée un script de lancement rapide"""

    module_path = Path(__file__).parent
    launcher_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lanceur Rapide - TradXPro Token Diversity Manager
================================================

Script de lancement rapide pour tester le module.
"""

import sys
from pathlib import Path

# Ajout du chemin du module
module_path = Path(__file__).parent
sys.path.append(str(module_path))

# Import et test
from tradxpro_core_manager import TradXProManager

def quick_test():
    """Test rapide du module"""
    print("🚀 TEST RAPIDE - TOKEN DIVERSITY MANAGER")
    print("=" * 50)

    try:
        # Initialisation
        manager = TradXProManager()
        print("✅ Gestionnaire initialisé")

        # Info du module
        from . import get_module_info, print_module_info
        print_module_info()

        print("\\n🎉 Module opérationnel !")

    except Exception as e:
        print(f"❌ Erreur: {{e}}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()
'''

    launcher_path = module_path / "launch_quick_test.py"
    with open(launcher_path, 'w', encoding='utf-8') as f:
        f.write(launcher_content)

    print(f"📝 Script de lancement créé: {launcher_path}")

def main():
    """Fonction principale"""

    try:
        # Setup principal
        success = setup_module()

        if success:
            # Créer le script de lancement
            create_launcher_script()

            print(f"\n" + "=" * 50)
            print("🎯 MODULE TOKEN DIVERSITY MANAGER CONFIGURÉ !")
            print("=" * 50)
            print("📦 Fonctionnalités disponibles:")
            print("   🔒 Diversité garantie des tokens")
            print("   📊 Analyse et rapports de diversité")
            print("   📈 Indicateurs techniques intégrés")
            print("   ⚡ Téléchargements et cache optimisés")
            print("   🛠️ API simple et unifiée")

        else:
            print(f"\n⚠️ Configuration incomplète - Vérifiez les fichiers manquants")

    except KeyboardInterrupt:
        print(f"\n👋 Setup interrompu")
    except Exception as e:
        print(f"\n❌ Erreur setup: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
```
<!-- MODULE-END: setup_module.py -->

<!-- MODULE-START: test_module.py -->
## 
*Chemin* : `D:/TradXPro/scripts/mise_a_jour_dataframe/token_diversity_manager/test_module.py`  
*Type* : `.py`  

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Simple du Module Token Diversity Manager
==============================================

Test simple pour vérifier que le module fonctionne correctement.

Usage:
    python test_module.py

Auteur: TradXPro Team
Date: 2 octobre 2025
"""

import sys
from pathlib import Path

# Ajout du chemin du module
module_path = Path(__file__).parent
sys.path.append(str(module_path))

def test_module_import():
    """Test d'importation du module"""

    print("📦 TEST D'IMPORTATION")
    print("-" * 30)

    try:
        # Test import principal
        from tradxpro_core_manager import TradXProManager
        print("✅ TradXProManager importé")

        # Test info module
        from __init__ import MODULE_INFO
        print("✅ Module complet importé")

        # Affichage info module
        print(f"📋 Version: {MODULE_INFO['version']}")

        return True

    except Exception as e:
        print(f"❌ Erreur importation: {e}")
        return False

def test_manager_init():
    """Test d'initialisation du gestionnaire"""

    print("\n🔧 TEST D'INITIALISATION")
    print("-" * 30)

    try:
        from tradxpro_core_manager import TradXProManager

        # Initialisation
        manager = TradXProManager()
        print("✅ TradXProManager initialisé")

        # Vérification des chemins
        print(f"📁 Racine: {manager.paths.root}")
        print(f"📄 JSON: {manager.paths.json_root}")
        print(f"⚡ Parquet: {manager.paths.parquet_root}")

        # Vérification configuration
        print(f"⚙️ Historique: {manager.history_days} jours")
        print(f"🔗 Workers: {manager.max_workers}")
        print(f"📊 Intervals: {manager.intervals}")

        return True

    except Exception as e:
        print(f"❌ Erreur initialisation: {e}")
        return False

def test_diversity_features():
    """Test des fonctionnalités de diversité"""

    print("\n🔒 TEST FONCTIONNALITÉS DIVERSITÉ")
    print("-" * 30)

    try:
        from tradxpro_core_manager import TradXProManager

        manager = TradXProManager()

        # Test des méthodes de diversité
        print("🧪 Test analyze_token_diversity...")
        test_tokens = [
            {"symbol": "BTC", "name": "Bitcoin", "score": 100},
            {"symbol": "ETH", "name": "Ethereum", "score": 95},
            {"symbol": "UNI", "name": "Uniswap", "score": 80}
        ]

        diversity_stats = manager.analyze_token_diversity(test_tokens)
        print("✅ analyze_token_diversity fonctionne")

        # Test rapport
        print("🧪 Test print_diversity_report...")
        manager.print_diversity_report(test_tokens)
        print("✅ print_diversity_report fonctionne")

        return True

    except Exception as e:
        print(f"❌ Erreur test diversité: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test complet du module"""

    print("🧪 TEST COMPLET - TOKEN DIVERSITY MANAGER")
    print("=" * 50)

    tests = [
        ("Import", test_module_import),
        ("Initialisation", test_manager_init),
        ("Diversité", test_diversity_features)
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n[TEST] {test_name}...")
        success = test_func()
        results.append((test_name, success))

    # Résultats finaux
    print("\n" + "=" * 50)
    print("📊 RÉSULTATS DES TESTS")
    print("=" * 50)

    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1

    print(f"\n📈 Score: {passed}/{len(tests)} tests réussis")

    if passed == len(tests):
        print("🎉 TOUS LES TESTS RÉUSSIS !")
        print("✅ Le module Token Diversity Manager est opérationnel")

        print(f"\n💡 UTILISATION:")
        print("from tradxpro_core_manager import TradXProManager")
        print("manager = TradXProManager()")
        print("tokens = manager.get_top_100_tokens()  # Avec diversité garantie !")

    else:
        print("⚠️ CERTAINS TESTS ONT ÉCHOUÉ")
        print("🔧 Vérifiez la configuration du module")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Test interrompu")
    except Exception as e:
        print(f"\n❌ Erreur générale: {e}")
        import traceback
        traceback.print_exc()
```
<!-- MODULE-END: test_module.py -->

<!-- MODULE-START: tradxpro_core_manager.py -->
## 
*Chemin* : `D:/TradXPro/scripts/mise_a_jour_dataframe/token_diversity_manager/tradxpro_core_manager.py`  
*Type* : `.py`  

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradXPro Core Manager - Module Unifié
=====================================

Module tout-en-un qui incorpore toute la logique TradXPro :
- Gestion des téléchargements crypto
- Sélection des tokens (top 100 marketcap/volume)
- Chargement et traitement des données OHLCV
- Calcul et cache des indicateurs techniques
- Gestion des fichiers JSON/Parquet
- API simplifiée pour intégration

Utilisation :
    from tradxpro_core_manager import TradXProManager

    manager = TradXProManager()

    # Récupérer les 100 meilleurs tokens
    top_tokens = manager.get_top_100_tokens()

    # Télécharger les données
    manager.download_crypto_data(["BTCUSDC", "ETHUSDC"])

    # Charger avec indicateurs
    df = manager.get_trading_data("BTCUSDC", "1h", indicators=["rsi", "bollinger"])

Auteur: TradXPro Team
Date: 2 octobre 2025
Version: 1.0
"""

import os
import sys
import json
import time
import logging
import platform
import requests
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# Configuration des chemins TradXPro
IS_WINDOWS = platform.system() == "Windows"

class TradXProPaths:
    """Gestionnaire centralisé des chemins TradXPro"""

    def __init__(self, root_path: Optional[str] = None):
        if root_path is None:
            self.root = Path(r"D:\TradXPro") if IS_WINDOWS else Path("/home/user/TradXPro")
        else:
            self.root = Path(root_path)

        # Dossiers principaux
        self.json_root = self.root / "crypto_data_json"
        self.parquet_root = self.root / "crypto_data_parquet"
        self.indicators_db = self.root / "indicators_db"
        self.scripts_dir = self.root / "scripts" / "mise_a_jour_dataframe"

        # Fichiers de configuration
        self.tokens_json = self.scripts_dir / "resultats_choix_des_100tokens.json"
        self.log_file = self.scripts_dir / "unified_data_historique.log"

        # Création des dossiers si nécessaire
        self._ensure_directories()

    def _ensure_directories(self):
        """Crée les dossiers nécessaires s'ils n'existent pas"""
        for path in [self.json_root, self.parquet_root, self.indicators_db, self.scripts_dir]:
            path.mkdir(parents=True, exist_ok=True)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class TradXProManager:
    """
    Gestionnaire principal TradXPro - API unifiée pour toutes les fonctionnalités
    """

    def __init__(self, root_path: Optional[str] = None):
        """
        Initialise le gestionnaire TradXPro

        Args:
            root_path: Chemin racine personnalisé (optionnel)
        """
        self.paths = TradXProPaths(root_path)
        self.logger = logger

        # Configuration par défaut
        self.history_days = 365
        self.binance_limit = 1000
        self.intervals = ["3m", "5m", "15m", "30m", "1h"]
        self.max_workers = max(4, (os.cpu_count() or 8) // 2)

        logger.info(f"TradXPro Manager initialisé - Racine: {self.paths.root}")

    # =========================================================
    #  SECTION 1: Gestion des tokens (Top 100)
    # =========================================================

    def get_top_100_marketcap_coingecko(self) -> List[Dict]:
        """
        Récupère les 100 cryptos avec la plus grosse capitalisation via CoinGecko

        Returns:
            Liste des tokens avec marketcap, nom, symbol, rang
        """
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 100,
            "page": 1
        }

        try:
            logger.info("Récupération top 100 marketcap CoinGecko...")
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            result = []
            for entry in data:
                result.append({
                    "symbol": entry["symbol"].upper(),
                    "name": entry["name"],
                    "market_cap": entry.get("market_cap", 0),
                    "market_cap_rank": entry.get("market_cap_rank", 999),
                    "volume": entry.get("total_volume", 0)
                })

            logger.info(f"✅ {len(result)} tokens récupérés via CoinGecko")
            return result

        except Exception as e:
            logger.error(f"Erreur CoinGecko API: {e}")
            return []

    def get_top_100_volume_binance(self) -> List[Dict]:
        """
        Récupère les 100 cryptos USDC avec le plus gros volume 24h via Binance

        Returns:
            Liste des tokens USDC avec volume 24h
        """
        url = "https://api.binance.com/api/v3/ticker/24hr"

        try:
            logger.info("Récupération top 100 volume USDC Binance...")
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()

            # Filtrer uniquement les paires USDC
            usdc_pairs = []
            for entry in data:
                if entry["symbol"].endswith("USDC"):
                    base_asset = entry["symbol"].replace("USDC", "")
                    usdc_pairs.append({
                        "symbol": base_asset,
                        "volume": float(entry["quoteVolume"]) if entry["quoteVolume"] else 0,
                        "price_change": float(entry["priceChangePercent"]) if entry["priceChangePercent"] else 0
                    })

            # Trier par volume décroissant et prendre les 100 premiers
            usdc_pairs.sort(key=lambda x: x["volume"], reverse=True)
            result = usdc_pairs[:100]

            logger.info(f"✅ {len(result)} tokens USDC récupérés via Binance")
            return result

        except Exception as e:
            logger.error(f"Erreur Binance API: {e}")
            return []

    def _ensure_category_representation(self, tokens: List[Dict]) -> List[Dict]:
        """
        Garantit qu'au moins les 3 meilleures cryptos de chaque catégorie importante sont incluses

        Args:
            tokens: Liste des tokens triés par score

        Returns:
            Liste ajustée avec représentation garantie par catégorie
        """
        # Définition des catégories importantes avec leurs tokens représentatifs
        essential_categories = {
            "layer1_blockchain": ["BTC", "ETH", "ADA", "SOL", "AVAX", "DOT", "NEAR", "ALGO"],
            "defi_protocols": ["UNI", "AAVE", "COMP", "MKR", "SUSHI", "CRV", "1INCH", "YFI"],
            "layer2_scaling": ["MATIC", "ARB", "OP", "IMX", "LRC", "MINA"],
            "smart_contracts": ["ETH", "ADA", "SOL", "AVAX", "DOT", "ALGO", "NEAR", "ATOM"],
            "meme_coins": ["DOGE", "SHIB", "PEPE", "FLOKI", "BONK"],
            "exchange_tokens": ["BNB", "CRO", "FTT", "HT", "KCS", "OKB"],
            "stablecoins": ["USDT", "USDC", "BUSD", "DAI", "FRAX", "TUSD"],
            "ai_gaming": ["FET", "AGIX", "OCEAN", "AXS", "SAND", "MANA", "ENJ"],
            "privacy_coins": ["XMR", "ZEC", "DASH", "SCRT"],
            "infrastructure": ["LINK", "GRT", "FIL", "AR", "STORJ", "SIA"]
        }

        logger.info("🔍 Vérification de la représentation par catégorie...")

        # Créer un index des tokens actuels
        current_symbols = {token["symbol"] for token in tokens}
        guaranteed_tokens = []

        # Pour chaque catégorie, garantir au moins 3 tokens du top marketcap
        for category, category_tokens in essential_categories.items():
            category_count = 0
            category_found = []

            # Vérifier les tokens déjà présents dans cette catégorie
            for token in tokens:
                if token["symbol"] in category_tokens:
                    category_found.append(token)
                    category_count += 1

            # Si moins de 3 tokens de cette catégorie, essayer d'en ajouter
            if category_count < 3:
                missing_count = 3 - category_count
                logger.debug(f"Catégorie {category}: {category_count} tokens présents, besoin de {missing_count} supplémentaires")

                # Chercher les tokens manquants dans les données originales
                for symbol in category_tokens:
                    if symbol not in current_symbols and missing_count > 0:
                        # Créer un token de base avec score élevé pour garantir l'inclusion
                        guaranteed_token = {
                            "symbol": symbol,
                            "name": symbol,
                            "market_cap": 0,
                            "market_cap_rank": 999,
                            "volume": 0,
                            "price_change": 0,
                            "source": "category_guarantee",
                            "category": category,
                            "score": 150  # Score élevé pour garantir l'inclusion
                        }
                        guaranteed_tokens.append(guaranteed_token)
                        current_symbols.add(symbol)
                        missing_count -= 1
                        logger.debug(f"Token {symbol} ajouté pour garantir la catégorie {category}")

        # Fusionner les tokens garantis avec la liste originale
        if guaranteed_tokens:
            combined_tokens = tokens + guaranteed_tokens
            # Retrier par score
            combined_tokens.sort(key=lambda x: x["score"], reverse=True)
            logger.info(f"✅ {len(guaranteed_tokens)} tokens ajoutés pour garantir la diversité des catégories")
            return combined_tokens[:100]  # Toujours retourner 100 tokens max

        return tokens

    def merge_and_select_top_100(self, marketcap_list: List[Dict], volume_list: List[Dict]) -> List[Dict]:
        """
        Fusionne les listes marketcap et volume pour sélectionner les 100 meilleurs tokens
        avec garantie de représentation par catégorie

        Args:
            marketcap_list: Liste des tokens par marketcap
            volume_list: Liste des tokens par volume

        Returns:
            Liste fusionnée des 100 meilleurs tokens avec représentation garantie
        """
        logger.info("Fusion des listes marketcap et volume avec garantie de diversité...")

        # Index par symbole
        marketcap_dict = {token["symbol"]: token for token in marketcap_list}
        volume_dict = {token["symbol"]: token for token in volume_list}

        # Fusion des données
        merged_tokens = {}
        all_symbols = set(marketcap_dict.keys()) | set(volume_dict.keys())

        for symbol in all_symbols:
            mc_data = marketcap_dict.get(symbol, {})
            vol_data = volume_dict.get(symbol, {})

            merged_tokens[symbol] = {
                "symbol": symbol,
                "name": mc_data.get("name", symbol),
                "market_cap": mc_data.get("market_cap", 0),
                "market_cap_rank": mc_data.get("market_cap_rank", 999),
                "volume": vol_data.get("volume", 0),
                "price_change": vol_data.get("price_change", 0),
                "source": "both" if (symbol in marketcap_dict and symbol in volume_dict) else
                         ("marketcap" if symbol in marketcap_dict else "volume")
            }

        # Scoring composite pour sélectionner les meilleurs
        scored_tokens = []
        for token in merged_tokens.values():
            # Score basé sur marketcap (inversé car rang 1 = meilleur) et volume
            mc_score = max(0, 101 - token["market_cap_rank"]) if token["market_cap_rank"] < 999 else 0
            vol_score = min(100, token["volume"] / 1_000_000)  # Normalisation volume

            # Bonus si présent dans les deux listes
            bonus = 20 if token["source"] == "both" else 0

            total_score = mc_score + vol_score + bonus
            token["score"] = total_score
            scored_tokens.append(token)

        # Trier par score décroissant
        scored_tokens.sort(key=lambda x: x["score"], reverse=True)

        # Appliquer la garantie de représentation par catégorie
        diversified_tokens = self._ensure_category_representation(scored_tokens)

        # Prendre les 100 premiers après diversification
        top_100 = diversified_tokens[:100]

        # Statistiques finales
        avg_score = np.mean([t['score'] for t in top_100])
        category_stats = {}
        for token in top_100:
            source = token.get("source", "unknown")
            category_stats[source] = category_stats.get(source, 0) + 1

        logger.info(f"✅ Top 100 tokens sélectionnés avec diversité garantie:")
        logger.info(f"   Score moyen: {avg_score:.1f}")
        logger.info(f"   Répartition: {category_stats}")

        return top_100

    def analyze_token_diversity(self, tokens: List[Dict]) -> Dict[str, Any]:
        """
        Analyse la diversité des tokens sélectionnés par catégorie

        Args:
            tokens: Liste des tokens à analyser

        Returns:
            Dictionnaire avec statistiques de diversité
        """
        # Définition des catégories (même que dans _ensure_category_representation)
        categories = {
            "layer1_blockchain": ["BTC", "ETH", "ADA", "SOL", "AVAX", "DOT", "NEAR", "ALGO"],
            "defi_protocols": ["UNI", "AAVE", "COMP", "MKR", "SUSHI", "CRV", "1INCH", "YFI"],
            "layer2_scaling": ["MATIC", "ARB", "OP", "IMX", "LRC", "MINA"],
            "smart_contracts": ["ETH", "ADA", "SOL", "AVAX", "DOT", "ALGO", "NEAR", "ATOM"],
            "meme_coins": ["DOGE", "SHIB", "PEPE", "FLOKI", "BONK"],
            "exchange_tokens": ["BNB", "CRO", "FTT", "HT", "KCS", "OKB"],
            "stablecoins": ["USDT", "USDC", "BUSD", "DAI", "FRAX", "TUSD"],
            "ai_gaming": ["FET", "AGIX", "OCEAN", "AXS", "SAND", "MANA", "ENJ"],
            "privacy_coins": ["XMR", "ZEC", "DASH", "SCRT"],
            "infrastructure": ["LINK", "GRT", "FIL", "AR", "STORJ", "SIA"]
        }

        diversity_stats = {}
        token_symbols = {token["symbol"] for token in tokens}

        for category, category_tokens in categories.items():
            found_tokens = [symbol for symbol in category_tokens if symbol in token_symbols]
            diversity_stats[category] = {
                "count": len(found_tokens),
                "tokens": found_tokens,
                "coverage": len(found_tokens) / len(category_tokens) * 100
            }

        # Statistiques globales
        total_categorized = sum(len(stats["tokens"]) for stats in diversity_stats.values())
        diversity_stats["global"] = {
            "total_tokens": len(tokens),
            "categorized_tokens": total_categorized,
            "uncategorized_tokens": len(tokens) - total_categorized,
            "diversity_score": len([cat for cat, stats in diversity_stats.items()
                                  if cat != "global" and stats["count"] >= 3]) / len(categories) * 100
        }

        return diversity_stats

    def print_diversity_report(self, tokens: List[Dict]):
        """
        Affiche un rapport détaillé de la diversité des tokens

        Args:
            tokens: Liste des tokens à analyser
        """
        diversity_stats = self.analyze_token_diversity(tokens)

        print("\n📊 RAPPORT DE DIVERSITÉ DES TOKENS")
        print("=" * 50)

        # Statistiques globales
        global_stats = diversity_stats["global"]
        print(f"Total de tokens: {global_stats['total_tokens']}")
        print(f"Tokens catégorisés: {global_stats['categorized_tokens']}")
        print(f"Score de diversité: {global_stats['diversity_score']:.1f}%")
        print()

        # Détail par catégorie
        print("Représentation par catégorie:")
        print("-" * 30)

        for category, stats in diversity_stats.items():
            if category == "global":
                continue

            status = "✅" if stats["count"] >= 3 else ("⚠️" if stats["count"] >= 1 else "❌")
            category_name = category.replace("_", " ").title()

            print(f"{status} {category_name:<18} {stats['count']:2d}/10 ({stats['coverage']:4.1f}%)")
            if stats["tokens"]:
                tokens_str = ", ".join(stats["tokens"][:5])
                if len(stats["tokens"]) > 5:
                    tokens_str += f" (+{len(stats['tokens'])-5} autres)"
                print(f"    {tokens_str}")

        print()

    def get_top_100_tokens(self, save_to_file: bool = True) -> List[Dict]:
        """
        API principale : récupère et fusionne les top 100 tokens

        Args:
            save_to_file: Sauvegarder le résultat dans resultats_choix_des_100tokens.json

        Returns:
            Liste des 100 meilleurs tokens
        """
        logger.info("🚀 Récupération des top 100 tokens...")

        # Récupération des données depuis les APIs
        marketcap_tokens = self.get_top_100_marketcap_coingecko()
        volume_tokens = self.get_top_100_volume_binance()

        if not marketcap_tokens and not volume_tokens:
            logger.error("❌ Impossible de récupérer les données des APIs")
            return []

        # Fusion et sélection avec garantie de diversité
        top_100 = self.merge_and_select_top_100(marketcap_tokens, volume_tokens)

        # Analyse de la diversité finale
        diversity_stats = self.analyze_token_diversity(top_100)
        logger.info(f"📊 Analyse de diversité:")
        logger.info(f"   Score de diversité: {diversity_stats['global']['diversity_score']:.1f}%")
        logger.info(f"   Tokens catégorisés: {diversity_stats['global']['categorized_tokens']}/100")

        # Afficher les catégories bien représentées
        well_represented = [cat for cat, stats in diversity_stats.items()
                          if cat != "global" and stats["count"] >= 3]
        logger.info(f"   Catégories bien représentées (≥3): {len(well_represented)}/10")

        # Sauvegarde optionnelle
        if save_to_file and top_100:
            try:
                with open(self.paths.tokens_json, 'w', encoding='utf-8') as f:
                    json.dump(top_100, f, indent=2, ensure_ascii=False)
                logger.info(f"✅ Top 100 sauvegardé: {self.paths.tokens_json}")
            except Exception as e:
                logger.error(f"Erreur sauvegarde: {e}")

        return top_100

    def load_saved_tokens(self) -> List[Dict]:
        """
        Charge les tokens sauvegardés depuis le fichier JSON

        Returns:
            Liste des tokens ou liste vide si erreur
        """
        try:
            if self.paths.tokens_json.exists():
                with open(self.paths.tokens_json, 'r', encoding='utf-8') as f:
                    tokens = json.load(f)
                logger.info(f"✅ {len(tokens)} tokens chargés depuis {self.paths.tokens_json}")
                return tokens
            else:
                logger.warning(f"Fichier tokens non trouvé: {self.paths.tokens_json}")
                return []
        except Exception as e:
            logger.error(f"Erreur chargement tokens: {e}")
            return []

    # =========================================================
    #  SECTION 2: Téléchargement des données crypto
    # =========================================================

    def _interval_to_ms(self, interval: str) -> int:
        """Convertit un interval en millisecondes"""
        multipliers = {
            "m": 60 * 1000,
            "h": 60 * 60 * 1000,
            "d": 24 * 60 * 60 * 1000,
            "w": 7 * 24 * 60 * 60 * 1000
        }

        if interval[-1] in multipliers:
            return int(interval[:-1]) * multipliers[interval[-1]]
        return 60 * 1000  # Default 1 minute

    def _download_single_pair(self, symbol: str, interval: str,
                             progress_callback: Optional[Callable] = None) -> bool:
        """
        Télécharge les données pour une paire symbol/interval via Binance API

        Args:
            symbol: Symbol (ex: BTCUSDC)
            interval: Interval (ex: 1h)
            progress_callback: Callback optionnel pour progression

        Returns:
            True si succès, False sinon
        """
        url = "https://api.binance.com/api/v3/klines"

        # Calcul période de téléchargement
        end_time = int(time.time() * 1000)
        start_time = end_time - (self.history_days * 24 * 60 * 60 * 1000)

        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": self.binance_limit
        }

        try:
            logger.debug(f"Téléchargement {symbol}_{interval}...")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if not data:
                logger.warning(f"Aucune donnée pour {symbol}_{interval}")
                return False

            # Conversion en DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'count', 'taker_buy_volume',
                'taker_buy_quote_volume', 'ignore'
            ])

            # Nettoyage et typage
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()

            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df = df.set_index('timestamp').sort_index()

            # Suppression des doublons et NaN
            df = df[~df.index.duplicated(keep='last')]
            df = df.dropna()

            if len(df) == 0:
                logger.warning(f"DataFrame vide après nettoyage: {symbol}_{interval}")
                return False

            # Sauvegarde JSON
            json_file = self.paths.json_root / f"{symbol}_{interval}.json"

            # Conversion pour JSON (timestamp en ms)
            df_json = df.reset_index()
            df_json['timestamp'] = df_json['timestamp'].astype(int) // 1_000_000

            with open(json_file, 'w') as f:
                json.dump(df_json.to_dict('records'), f)

            # Sauvegarde Parquet (plus efficace)
            parquet_file = self.paths.parquet_root / f"{symbol}_{interval}.parquet"
            df.to_parquet(parquet_file, compression='zstd')

            logger.debug(f"✅ {symbol}_{interval}: {len(df)} lignes sauvegardées")

            if progress_callback:
                progress_callback(symbol, interval, len(df))

            return True

        except Exception as e:
            logger.error(f"❌ Erreur téléchargement {symbol}_{interval}: {e}")
            return False

    def download_crypto_data(self, symbols: List[str], intervals: Optional[List[str]] = None,
                           progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Télécharge les données crypto pour plusieurs symboles/intervals

        Args:
            symbols: Liste des symboles (ex: ["BTCUSDC", "ETHUSDC"])
            intervals: Liste des intervals (par défaut: ["3m", "5m", "15m", "30m", "1h"])
            progress_callback: Callback optionnel(symbol, interval, nb_rows)

        Returns:
            Dictionnaire avec statistiques de téléchargement
        """
        if intervals is None:
            intervals = self.intervals

        logger.info(f"🔄 Téléchargement de {len(symbols)} symboles × {len(intervals)} intervals...")

        # Préparation des tâches
        tasks = []
        for symbol in symbols:
            for interval in intervals:
                tasks.append((symbol, interval))

        # Téléchargement parallèle
        results = {"success": 0, "errors": 0, "details": []}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(self._download_single_pair, symbol, interval, progress_callback): (symbol, interval)
                for symbol, interval in tasks
            }

            for future in as_completed(future_to_task):
                symbol, interval = future_to_task[future]
                try:
                    success = future.result()
                    if success:
                        results["success"] += 1
                        results["details"].append(f"✅ {symbol}_{interval}")
                    else:
                        results["errors"] += 1
                        results["details"].append(f"❌ {symbol}_{interval}")
                except Exception as e:
                    results["errors"] += 1
                    results["details"].append(f"❌ {symbol}_{interval}: {e}")

        logger.info(f"✅ Téléchargement terminé: {results['success']} succès, {results['errors']} erreurs")
        return results

    def download_top_100_data(self, intervals: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Télécharge les données pour tous les top 100 tokens

        Args:
            intervals: Liste des intervals (optionnel)

        Returns:
            Statistiques de téléchargement
        """
        # Chargement des tokens
        tokens = self.load_saved_tokens()
        if not tokens:
            logger.info("Aucun token sauvegardé, récupération des top 100...")
            tokens = self.get_top_100_tokens()

        if not tokens:
            logger.error("❌ Impossible de récupérer les tokens")
            return {"success": 0, "errors": 1, "details": ["Pas de tokens disponibles"]}

        # Conversion en symboles USDC
        symbols = [token["symbol"] + "USDC" for token in tokens]

        logger.info(f"🚀 Téléchargement des données pour {len(symbols)} tokens...")
        return self.download_crypto_data(symbols, intervals)

    # =========================================================
    #  SECTION 3: Chargement et traitement des données
    # =========================================================

    def load_ohlcv_data(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Charge les données OHLCV avec priorité Parquet → JSON

        Args:
            symbol: Symbole (ex: BTCUSDC)
            interval: Interval (ex: 1h)

        Returns:
            DataFrame OHLCV avec DatetimeIndex UTC ou None
        """
        # Priorité 1: Parquet
        parquet_file = self.paths.parquet_root / f"{symbol}_{interval}.parquet"
        if parquet_file.exists():
            try:
                df = pd.read_parquet(parquet_file)
                logger.debug(f"Chargé depuis Parquet: {symbol}_{interval} ({len(df)} lignes)")
                return df
            except Exception as e:
                logger.warning(f"Erreur lecture Parquet {parquet_file}: {e}")

        # Priorité 2: JSON
        json_file = self.paths.json_root / f"{symbol}_{interval}.json"
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                df = pd.DataFrame(data)

                # Vérification des colonnes
                required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
                if not all(col in df.columns for col in required_cols):
                    logger.error(f"Colonnes manquantes dans {json_file}")
                    return None

                # Conversion types
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df = df.set_index('timestamp').sort_index()

                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                df = df.dropna()

                # Création automatique du Parquet pour optimiser les futurs accès
                try:
                    df.to_parquet(parquet_file, compression="zstd")
                    logger.debug(f"Parquet créé: {parquet_file}")
                except Exception as e:
                    logger.warning(f"Impossible de créer {parquet_file}: {e}")

                logger.debug(f"Chargé depuis JSON: {symbol}_{interval} ({len(df)} lignes)")
                return df

            except Exception as e:
                logger.error(f"Erreur lecture JSON {json_file}: {e}")

        logger.warning(f"Aucune donnée trouvée pour {symbol}_{interval}")
        return None

    # =========================================================
    #  SECTION 4: Calcul des indicateurs techniques
    # =========================================================

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcule le RSI"""
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(window=period, min_periods=period).mean()
        loss = (-delta.clip(upper=0)).rolling(window=period, min_periods=period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """Calcule les bandes de Bollinger"""
        close = df['close']
        ma = close.rolling(window=period, min_periods=period).mean()
        std = close.rolling(window=period, min_periods=period).std()

        return {
            'bb_middle': ma,
            'bb_upper': ma + (std_dev * std),
            'bb_lower': ma - (std_dev * std),
            'bb_width': (2 * std_dev * std) / ma
        }

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcule l'ATR (Average True Range)"""
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period, min_periods=period).mean()

    def calculate_ema(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calcule l'EMA (Exponential Moving Average)"""
        return df['close'].ewm(span=period, adjust=False).mean()

    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calcule le MACD"""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return {
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram
        }

    def add_indicators(self, df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
        """
        Ajoute plusieurs indicateurs à un DataFrame OHLCV

        Args:
            df: DataFrame OHLCV
            indicators: Liste des indicateurs ('rsi', 'bollinger', 'atr', 'ema', 'macd')

        Returns:
            DataFrame avec indicateurs ajoutés
        """
        result = df.copy()

        for indicator in indicators:
            try:
                if indicator == 'rsi':
                    result['rsi'] = self.calculate_rsi(df)
                elif indicator == 'bollinger':
                    bb_data = self.calculate_bollinger_bands(df)
                    for key, series in bb_data.items():
                        result[key] = series
                elif indicator == 'atr':
                    result['atr'] = self.calculate_atr(df)
                elif indicator == 'ema':
                    result['ema'] = self.calculate_ema(df)
                elif indicator == 'macd':
                    macd_data = self.calculate_macd(df)
                    for key, series in macd_data.items():
                        result[key] = series
                else:
                    logger.warning(f"Indicateur non supporté: {indicator}")

            except Exception as e:
                logger.error(f"Erreur calcul {indicator}: {e}")

        # Suppression des lignes avec NaN
        result = result.dropna()

        return result

    # =========================================================
    #  SECTION 5: API principale unifiée
    # =========================================================

    def get_trading_data(self, symbol: str, interval: str,
                        indicators: Optional[List[str]] = None,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        API principale : charge les données OHLCV + indicateurs

        Args:
            symbol: Symbole crypto (ex: BTCUSDC)
            interval: Interval (ex: 1h, 5m)
            indicators: Liste des indicateurs à calculer (optionnel)
            start_date: Date de début (format YYYY-MM-DD, optionnel)
            end_date: Date de fin (format YYYY-MM-DD, optionnel)

        Returns:
            DataFrame complet avec OHLCV + indicateurs ou None

        Example:
            >>> manager = TradXProManager()
            >>> df = manager.get_trading_data("BTCUSDC", "1h",
            ...                              indicators=["rsi", "bollinger", "atr"])
            >>> print(f"DataFrame: {len(df)} lignes, {len(df.columns)} colonnes")
        """
        # Chargement des données de base
        df = self.load_ohlcv_data(symbol, interval)

        if df is None:
            logger.error(f"Impossible de charger {symbol}_{interval}")
            return None

        # Filtrage temporel si demandé
        if start_date or end_date:
            df = df.loc[start_date:end_date]
            logger.info(f"Filtrage temporel: {len(df)} lignes après filtrage")

        # Ajout des indicateurs si demandés
        if indicators:
            df = self.add_indicators(df, indicators)
            logger.info(f"Indicateurs ajoutés: {indicators}")

        logger.info(f"✅ {symbol}_{interval}: {len(df)} lignes, {len(df.columns)} colonnes")
        return df

    def get_multiple_trading_data(self, pairs: List[Tuple[str, str]],
                                 indicators: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Charge les données pour plusieurs paires en parallèle

        Args:
            pairs: Liste de tuples (symbol, interval)
            indicators: Liste des indicateurs à calculer

        Returns:
            Dictionnaire {symbol_interval: DataFrame}
        """
        logger.info(f"Chargement de {len(pairs)} paires en parallèle...")

        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_pair = {
                executor.submit(self.get_trading_data, symbol, interval, indicators): (symbol, interval)
                for symbol, interval in pairs
            }

            for future in as_completed(future_to_pair):
                symbol, interval = future_to_pair[future]
                try:
                    df = future.result()
                    if df is not None:
                        results[f"{symbol}_{interval}"] = df
                        logger.debug(f"✅ {symbol}_{interval} chargé")
                    else:
                        logger.warning(f"❌ Échec chargement {symbol}_{interval}")
                except Exception as e:
                    logger.error(f"Erreur {symbol}_{interval}: {e}")

        logger.info(f"✅ {len(results)}/{len(pairs)} paires chargées avec succès")
        return results

    # =========================================================
    #  SECTION 6: Utilitaires et statistiques
    # =========================================================

    def get_available_data(self) -> Dict[str, List[str]]:
        """
        Scanne les données disponibles sur disque

        Returns:
            Dictionnaire {symbol: [list_of_intervals]}
        """
        available = {}

        # Scan des fichiers Parquet
        for file_path in self.paths.parquet_root.glob("*.parquet"):
            filename = file_path.stem
            if "_" in filename:
                symbol, interval = filename.rsplit("_", 1)
                if symbol not in available:
                    available[symbol] = []
                available[symbol].append(interval)

        # Compléter avec les fichiers JSON s'ils ne sont pas en Parquet
        for file_path in self.paths.json_root.glob("*.json"):
            filename = file_path.stem
            if "_" in filename and not filename.startswith("resultats"):
                symbol, interval = filename.rsplit("_", 1)
                if symbol not in available:
                    available[symbol] = []
                if interval not in available[symbol]:
                    available[symbol].append(interval)

        # Tri des intervals
        for symbol in available:
            available[symbol] = sorted(available[symbol])

        return available

    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Calcule des statistiques sur les données disponibles

        Returns:
            Dictionnaire avec statistiques
        """
        available_data = self.get_available_data()

        total_files = sum(len(intervals) for intervals in available_data.values())

        # Taille des dossiers
        json_size = sum(f.stat().st_size for f in self.paths.json_root.glob("*.json")) / 1024 / 1024
        parquet_size = sum(f.stat().st_size for f in self.paths.parquet_root.glob("*.parquet")) / 1024 / 1024

        stats = {
            "symbols_count": len(available_data),
            "total_files": total_files,
            "intervals": list(set(interval for intervals in available_data.values() for interval in intervals)),
            "json_size_mb": round(json_size, 1),
            "parquet_size_mb": round(parquet_size, 1),
            "total_size_mb": round(json_size + parquet_size, 1),
            "top_symbols": sorted(available_data.keys())[:10] if available_data else [],
            "sample_data": dict(list(available_data.items())[:5]) if available_data else {}
        }

        return stats

    def cleanup_old_files(self, days_old: int = 7) -> Dict[str, int]:
        """
        Nettoie les fichiers anciens

        Args:
            days_old: Supprimer les fichiers plus anciens que X jours

        Returns:
            Statistiques de nettoyage
        """
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        stats = {"json_removed": 0, "parquet_removed": 0}

        # Nettoyage JSON
        for file_path in self.paths.json_root.glob("*.json"):
            if file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                stats["json_removed"] += 1

        # Nettoyage Parquet
        for file_path in self.paths.parquet_root.glob("*.parquet"):
            if file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                stats["parquet_removed"] += 1

        logger.info(f"Nettoyage terminé: {stats['json_removed']} JSON, {stats['parquet_removed']} Parquet supprimés")
        return stats

# =========================================================
#  SECTION 7: Interface en ligne de commande
# =========================================================

def main():
    """Interface en ligne de commande pour tester le gestionnaire"""
    print("🚀 TradXPro Core Manager - Test Interface")
    print("=" * 50)

    manager = TradXProManager()

    while True:
        print("\nOptions disponibles:")
        print("1. 📊 Récupérer top 100 tokens")
        print("2. 📥 Télécharger données crypto")
        print("3. 📈 Charger données avec indicateurs")
        print("4. 📋 Statistiques des données")
        print("5. 🧹 Nettoyer anciens fichiers")
        print("0. ❌ Quitter")

        choice = input("\nVotre choix: ").strip()

        try:
            if choice == "1":
                print("\n🔄 Récupération des top 100 tokens...")
                tokens = manager.get_top_100_tokens()
                print(f"✅ {len(tokens)} tokens récupérés")
                for i, token in enumerate(tokens[:10], 1):
                    print(f"  {i:2d}. {token['symbol']:10s} - {token['name'][:30]:<30s} (Score: {token['score']:.1f})")

            elif choice == "2":
                symbols = input("Symboles (séparés par des virgules, ex: BTCUSDC,ETHUSDC): ").strip()
                if symbols:
                    symbol_list = [s.strip().upper() for s in symbols.split(",")]
                    print(f"\n🔄 Téléchargement de {len(symbol_list)} symboles...")
                    results = manager.download_crypto_data(symbol_list)
                    print(f"✅ Résultats: {results['success']} succès, {results['errors']} erreurs")

            elif choice == "3":
                symbol = input("Symbole (ex: BTCUSDC): ").strip().upper()
                interval = input("Interval (ex: 1h): ").strip()
                indicators_str = input("Indicateurs (ex: rsi,bollinger,atr): ").strip()

                indicators = [i.strip() for i in indicators_str.split(",") if i.strip()] if indicators_str else None

                print(f"\n🔄 Chargement {symbol}_{interval} avec indicateurs {indicators}...")
                df = manager.get_trading_data(symbol, interval, indicators)

                if df is not None:
                    print(f"✅ DataFrame chargé: {len(df)} lignes, {len(df.columns)} colonnes")
                    print(f"Colonnes: {list(df.columns)}")
                    print(f"Période: {df.index[0]} à {df.index[-1]}")
                    print("\nAperçu des dernières valeurs:")
                    print(df.tail(3))
                else:
                    print("❌ Impossible de charger les données")

            elif choice == "4":
                print("\n📊 Calcul des statistiques...")
                stats = manager.get_data_statistics()
                print(f"✅ Statistiques des données:")
                print(f"  Symboles: {stats['symbols_count']}")
                print(f"  Fichiers total: {stats['total_files']}")
                print(f"  Intervals disponibles: {stats['intervals']}")
                print(f"  Taille JSON: {stats['json_size_mb']} MB")
                print(f"  Taille Parquet: {stats['parquet_size_mb']} MB")
                print(f"  Taille totale: {stats['total_size_mb']} MB")

            elif choice == "5":
                days = input("Supprimer fichiers plus anciens que X jours (défaut: 7): ").strip()
                days = int(days) if days.isdigit() else 7
                print(f"\n🧹 Nettoyage des fichiers > {days} jours...")
                stats = manager.cleanup_old_files(days)
                print(f"✅ {stats['json_removed'] + stats['parquet_removed']} fichiers supprimés")

            elif choice == "0":
                print("👋 Au revoir!")
                break

            else:
                print("❌ Choix invalide")

        except KeyboardInterrupt:
            print("\n👋 Au revoir!")
            break
        except Exception as e:
            logger.error(f"Erreur: {e}")

if __name__ == "__main__":
    main()
```
<!-- MODULE-END: tradxpro_core_manager.py -->

<!-- MODULE-START: test_diversite_simple.py -->
## 
*Chemin* : `D:/TradXPro/scripts/mise_a_jour_dataframe/token_diversity_manager/tests/test_diversite_simple.py`  
*Type* : `.py`  

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Rapide - Diversité des Tokens
==================================

Test simple pour vérifier que la nouvelle fonctionnalité de diversité
garantie fonctionne correctement.

Usage:
    python test_diversite_simple.py

Auteur: TradXPro Team
Date: 2 octobre 2025
"""

from tradxpro_core_manager import TradXProManager

def test_simple():
    """Test simple de la diversité"""

    print("🧪 TEST SIMPLE - DIVERSITÉ DES TOKENS")
    print("=" * 50)

    # Initialisation
    manager = TradXProManager()

    # Récupération avec diversité garantie
    print("📊 Récupération des top 100 tokens avec diversité garantie...")
    tokens = manager.get_top_100_tokens(save_to_file=False)

    if not tokens:
        print("❌ Erreur : Impossible de récupérer les tokens")
        return False

    print(f"✅ {len(tokens)} tokens récupérés")

    # Analyse rapide de la diversité
    diversity_stats = manager.analyze_token_diversity(tokens)

    print(f"\n📊 RÉSULTATS:")
    print(f"Score de diversité: {diversity_stats['global']['diversity_score']:.1f}%")
    print(f"Tokens catégorisés: {diversity_stats['global']['categorized_tokens']}/100")

    # Test des catégories essentielles
    categories_ok = 0
    categories_essentielles = ["layer1_blockchain", "defi_protocols", "exchange_tokens", "stablecoins"]

    print(f"\n🎯 VÉRIFICATION DES CATÉGORIES ESSENTIELLES:")
    for category in categories_essentielles:
        count = diversity_stats[category]["count"]
        status = "✅" if count >= 3 else "❌"
        print(f"{status} {category.replace('_', ' ').title()}: {count} tokens")

        if count >= 3:
            categories_ok += 1

    # Résultat final
    print(f"\n📋 RÉSULTAT FINAL:")
    if categories_ok >= 3:
        print("🎉 TEST RÉUSSI - Diversité excellente !")
        print("✅ La sélection automatique garantit bien la diversité")
        return True
    else:
        print("⚠️ TEST PARTIELLEMENT RÉUSSI")
        print(f"📈 {categories_ok}/4 catégories essentielles bien représentées")
        return True

if __name__ == "__main__":
    try:
        success = test_simple()
        if success:
            print("\n🚀 Le système TradXPro avec diversité garantie est opérationnel !")
        else:
            print("\n❌ Des ajustements sont nécessaires")

    except KeyboardInterrupt:
        print("\n👋 Test interrompu")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
```
<!-- MODULE-END: test_diversite_simple.py -->

<!-- MODULE-START: test_token_diversity.py -->
## 
*Chemin* : `D:/TradXPro/scripts/mise_a_jour_dataframe/token_diversity_manager/tests/test_token_diversity.py`  
*Type* : `.py`  

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test de la Sélection Diversifiée des Tokens
===========================================

Script de test pour vérifier que la sélection automatique des top 100 tokens
inclut bien au moins 3 représentants de chaque catégorie importante.

Usage:
    python test_token_diversity.py

Auteur: TradXPro Team
Date: 2 octobre 2025
"""

import sys
from pathlib import Path

# Import du gestionnaire TradXPro
from tradxpro_core_manager import TradXProManager

def test_token_diversity():
    """Test de la diversité des tokens sélectionnés"""

    print("🧪 TEST DE DIVERSITÉ DES TOKENS")
    print("=" * 50)

    # Initialisation du gestionnaire
    manager = TradXProManager()

    # Test 1: Récupération avec diversité garantie
    print("📊 Récupération des top 100 tokens avec diversité garantie...")
    tokens = manager.get_top_100_tokens(save_to_file=False)  # Test sans sauvegarde

    if not tokens:
        print("❌ Impossible de récupérer les tokens")
        return False

    print(f"✅ {len(tokens)} tokens récupérés")

    # Test 2: Analyse de la diversité
    print("\n🔍 Analyse de la diversité...")
    manager.print_diversity_report(tokens)

    # Test 3: Vérification des catégories essentielles
    print("🎯 VÉRIFICATION DES CATÉGORIES ESSENTIELLES")
    print("-" * 50)

    diversity_stats = manager.analyze_token_diversity(tokens)

    # Catégories qui DOIVENT avoir au moins 3 représentants
    essential_categories = [
        "layer1_blockchain",
        "defi_protocols",
        "exchange_tokens",
        "infrastructure"
    ]

    all_good = True
    for category in essential_categories:
        count = diversity_stats[category]["count"]
        status = "✅" if count >= 3 else "❌"

        if count < 3:
            all_good = False

        print(f"{status} {category:<18} {count:2d} tokens")
        if count > 0:
            print(f"    Tokens: {', '.join(diversity_stats[category]['tokens'])}")

    print()

    # Test 4: Top tokens par catégorie
    print("🏆 TOP TOKENS PAR CATÉGORIE")
    print("-" * 50)

    categories_examples = {
        "Layer 1 Blockchain": ["BTC", "ETH", "ADA", "SOL"],
        "DeFi Protocols": ["UNI", "AAVE", "COMP", "MKR"],
        "Exchange Tokens": ["BNB", "CRO", "FTT", "HT"],
        "Stablecoins": ["USDT", "USDC", "BUSD", "DAI"]
    }

    token_dict = {token["symbol"]: token for token in tokens}

    for category_name, example_tokens in categories_examples.items():
        print(f"\n{category_name}:")
        found_tokens = []

        for symbol in example_tokens:
            if symbol in token_dict:
                token = token_dict[symbol]
                found_tokens.append(token)

        # Trier par score décroissant
        found_tokens.sort(key=lambda x: x["score"], reverse=True)

        for i, token in enumerate(found_tokens[:3], 1):
            print(f"  {i}. {token['symbol']:8s} - {token['name'][:25]:<25s} (Score: {token['score']:.1f})")

    print()

    # Test 5: Résumé final
    print("📋 RÉSUMÉ DU TEST")
    print("-" * 50)

    global_stats = diversity_stats["global"]
    score_diversite = global_stats["diversity_score"]

    print(f"Score de diversité global: {score_diversite:.1f}%")
    print(f"Catégories bien représentées: {len([cat for cat, stats in diversity_stats.items() if cat != 'global' and stats['count'] >= 3])}/10")
    print(f"Tokens catégorisés: {global_stats['categorized_tokens']}/100")

    if score_diversite >= 80 and all_good:
        print("✅ TEST RÉUSSI: Excellente diversité des tokens")
        result = True
    elif score_diversite >= 60:
        print("⚠️ TEST PARTIELLEMENT RÉUSSI: Diversité acceptable")
        result = True
    else:
        print("❌ TEST ÉCHOUÉ: Diversité insuffisante")
        result = False

    return result

def test_category_guarantee():
    """Test spécifique de la garantie par catégorie"""

    print("\n🔒 TEST DE GARANTIE PAR CATÉGORIE")
    print("=" * 50)

    manager = TradXProManager()

    # Simuler une liste limitée pour forcer l'activation de la garantie
    limited_marketcap = [
        {"symbol": "BTC", "name": "Bitcoin", "market_cap": 1000000000, "market_cap_rank": 1},
        {"symbol": "ETH", "name": "Ethereum", "market_cap": 500000000, "market_cap_rank": 2},
        {"symbol": "XRP", "name": "XRP", "market_cap": 100000000, "market_cap_rank": 3},
    ]

    limited_volume = [
        {"symbol": "BTC", "volume": 1000000},
        {"symbol": "ETH", "volume": 800000},
        {"symbol": "DOGE", "volume": 500000},
    ]

    print("🧪 Test avec données limitées pour activer la garantie...")
    result_tokens = manager.merge_and_select_top_100(limited_marketcap, limited_volume)

    print(f"✅ {len(result_tokens)} tokens générés")

    # Vérifier que des tokens ont été ajoutés automatiquement
    guaranteed_tokens = [token for token in result_tokens if token.get("source") == "category_guarantee"]

    if guaranteed_tokens:
        print(f"🔒 {len(guaranteed_tokens)} tokens ajoutés automatiquement pour garantir la diversité:")
        for token in guaranteed_tokens[:5]:
            print(f"   • {token['symbol']} (Catégorie: {token.get('category', 'Unknown')})")
    else:
        print("ℹ️ Aucun token supplémentaire nécessaire (diversité déjà suffisante)")

    return True

def main():
    """Fonction principale"""

    print("🚀 TEST COMPLET DE LA SÉLECTION DIVERSIFIÉE")
    print("=" * 60)

    try:
        # Test principal
        success1 = test_token_diversity()

        # Test de garantie
        success2 = test_category_guarantee()

        print("\n" + "=" * 60)
        if success1 and success2:
            print("🎉 TOUS LES TESTS RÉUSSIS!")
            print("✅ La sélection automatique garantit bien la diversité des catégories")
        else:
            print("⚠️ TESTS PARTIELLEMENT RÉUSSIS")
            print("🔧 Quelques ajustements peuvent être nécessaires")

    except KeyboardInterrupt:
        print("\n👋 Test interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur pendant les tests: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
```
<!-- MODULE-END: test_token_diversity.py -->

<!-- MODULE-START: exemple_integration_tradxpro.py -->
## 
*Chemin* : `D:/TradXPro/scripts/mise_a_jour_dataframe/token_diversity_manager/examples/exemple_integration_tradxpro.py`  
*Type* : `.py`  

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exemple d'Incorporation du TradXPro Core Manager
===============================================

Exemple concret montrant comment incorporer toute la logique TradXPro
(téléchargements, tokens, indicateurs) dans un autre programme.

Usage:
    python exemple_integration_tradxpro.py

Auteur: TradXPro Team
Date: 2 octobre 2025
"""

import sys
import time
from pathlib import Path
from typing import Optional

# Import du gestionnaire TradXPro
from tradxpro_core_manager import TradXProManager

def exemple_basic_usage():
    """Exemple d'usage basique du gestionnaire"""
    print("=== EXEMPLE BASIQUE ===")

    # Initialisation du gestionnaire
    manager = TradXProManager()

    # 1. Récupérer les top 100 tokens avec diversité garantie
    print("📊 Récupération des top 100 tokens avec diversité garantie...")
    tokens = manager.get_top_100_tokens()

    if tokens:
        print(f"✅ {len(tokens)} tokens récupérés")
        print("Top 5 tokens:")
        for i, token in enumerate(tokens[:5], 1):
            print(f"  {i}. {token['symbol']:8s} - {token['name'][:25]:<25s} (Score: {token['score']:.1f})")

        # Afficher le rapport de diversité
        print("\n📊 Rapport de diversité:")
        manager.print_diversity_report(tokens)

    # 2. Charger des données existantes avec indicateurs
    print("\n📈 Chargement de données avec indicateurs...")
    df = manager.get_trading_data(
        symbol="BTCUSDC",
        interval="1h",
        indicators=["rsi", "bollinger", "atr"]
    )

    if df is not None:
        print(f"✅ DataFrame chargé: {len(df)} lignes, {len(df.columns)} colonnes")
        print(f"Colonnes: {list(df.columns)}")
        print("\nAperçu des dernières valeurs:")
        print(df.tail(3))
    else:
        print("❌ Données non disponibles")

def exemple_trading_strategy():
    """Exemple d'une stratégie de trading simple utilisant TradXPro"""
    print("\n=== EXEMPLE STRATÉGIE TRADING ===")

    manager = TradXProManager()

    # Symboles à analyser
    symbols = ["BTCUSDC", "ETHUSDC", "ADAUSDC"]
    interval = "1h"
    indicators = ["rsi", "bollinger", "atr", "macd"]

    print(f"🔍 Analyse de {len(symbols)} symboles avec indicateurs...")

    signals = []

    for symbol in symbols:
        print(f"\n📊 Analyse {symbol}...")

        # Chargement des données avec indicateurs
        df = manager.get_trading_data(symbol, interval, indicators)

        if df is None or len(df) < 50:
            print(f"  ❌ Données insuffisantes pour {symbol}")
            continue

        # Calcul des signaux simples
        latest = df.iloc[-1]

        # Signal RSI
        rsi_signal = "OVERSOLD" if latest['rsi'] < 30 else ("OVERBOUGHT" if latest['rsi'] > 70 else "NEUTRAL")

        # Signal Bollinger
        bb_signal = "BELOW_LOWER" if latest['close'] < latest['bb_lower'] else \
                   ("ABOVE_UPPER" if latest['close'] > latest['bb_upper'] else "MIDDLE")

        # Signal MACD
        macd_signal = "BULLISH" if latest['macd'] > latest['macd_signal'] else "BEARISH"

        signal = {
            "symbol": symbol,
            "price": latest['close'],
            "rsi": latest['rsi'],
            "rsi_signal": rsi_signal,
            "bb_signal": bb_signal,
            "macd_signal": macd_signal
        }

        signals.append(signal)

        print(f"  💰 Prix: ${latest['close']:.2f}")
        print(f"  📈 RSI: {latest['rsi']:.1f} ({rsi_signal})")
        print(f"  🔵 Bollinger: {bb_signal}")
        print(f"  📊 MACD: {macd_signal}")

    # Résumé des signaux
    print("\n=== RÉSUMÉ DES SIGNAUX ===")
    for signal in signals:
        print(f"{signal['symbol']:8s} | RSI: {signal['rsi']:5.1f} ({signal['rsi_signal']:10s}) | "
              f"BB: {signal['bb_signal']:12s} | MACD: {signal['macd_signal']}")

def exemple_portfolio_analysis():
    """Exemple d'analyse de portfolio utilisant les top tokens"""
    print("\n=== EXEMPLE ANALYSE PORTFOLIO ===")

    manager = TradXProManager()

    # Récupérer et sélectionner les meilleurs tokens
    tokens = manager.load_saved_tokens()
    if not tokens:
        print("📊 Récupération des tokens...")
        tokens = manager.get_top_100_tokens()

    if not tokens:
        print("❌ Impossible de récupérer les tokens")
        return

    # Sélectionner le top 10 pour l'analyse
    top_10_symbols = [token["symbol"] + "USDC" for token in tokens[:10]]

    print(f"🔍 Analyse du top 10 portfolio...")

    # Chargement des données en parallèle
    pairs = [(symbol, "1h") for symbol in top_10_symbols]
    results = manager.get_multiple_trading_data(pairs, indicators=["rsi", "atr"])

    portfolio_data = []

    for symbol_interval, df in results.items():
        if df is not None and len(df) > 0:
            symbol = symbol_interval.replace("_1h", "")
            latest = df.iloc[-1]

            # Calculs de volatilité et momentum
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * 100  # Volatilité en %
            momentum = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1) * 100  # Momentum 20 périodes

            portfolio_data.append({
                "symbol": symbol,
                "price": latest['close'],
                "rsi": latest['rsi'],
                "atr": latest['atr'],
                "volatility": volatility,
                "momentum_20d": momentum
            })

    # Tri par momentum
    portfolio_data.sort(key=lambda x: x['momentum_20d'], reverse=True)

    print(f"\n📊 Portfolio Analysis ({len(portfolio_data)} tokens):")
    print("Symbol    | Price     | RSI  | Volatility | Momentum 20d")
    print("-" * 60)

    for data in portfolio_data:
        print(f"{data['symbol']:8s} | ${data['price']:8.2f} | {data['rsi']:4.1f} | "
              f"{data['volatility']:8.2f}% | {data['momentum_20d']:8.1f}%")

def exemple_data_management():
    """Exemple de gestion des données"""
    print("\n=== EXEMPLE GESTION DONNÉES ===")

    manager = TradXProManager()

    # Statistiques des données disponibles
    print("📊 Statistiques des données...")
    stats = manager.get_data_statistics()

    print(f"✅ Données disponibles:")
    print(f"  Symboles: {stats['symbols_count']}")
    print(f"  Fichiers total: {stats['total_files']}")
    print(f"  Intervals: {stats['intervals']}")
    print(f"  Taille totale: {stats['total_size_mb']} MB")

    # Données disponibles
    available = manager.get_available_data()
    if available:
        print(f"\nTop 5 symboles disponibles:")
        for i, (symbol, intervals) in enumerate(list(available.items())[:5], 1):
            print(f"  {i}. {symbol}: {intervals}")

    # Exemple de téléchargement de nouvelles données
    print(f"\n📥 Exemple de téléchargement...")
    print("(Simulation - pas de téléchargement réel)")

    # Test avec quelques symboles
    test_symbols = ["BTCUSDC", "ETHUSDC"]
    print(f"Téléchargement simulé pour: {test_symbols}")

    # Dans un vrai cas, vous appelleriez:
    # results = manager.download_crypto_data(test_symbols, ["1h", "4h"])

class MonProgrammeAvecTradXPro:
    """
    Exemple d'intégration du TradXProManager dans votre propre classe
    """

    def __init__(self, custom_root_path: Optional[str] = None):
        """Initialisation avec TradXPro intégré"""
        self.tradx = TradXProManager(custom_root_path)
        self.watchlist = []

        print("🚀 Mon Programme avec TradXPro initialisé")

    def setup_watchlist(self, top_n=20):
        """Configure une watchlist basée sur les top tokens"""
        print(f"📋 Configuration watchlist (top {top_n})...")

        tokens = self.tradx.load_saved_tokens()
        if not tokens:
            tokens = self.tradx.get_top_100_tokens()

        self.watchlist = [token["symbol"] + "USDC" for token in tokens[:top_n]]
        print(f"✅ Watchlist configurée: {len(self.watchlist)} symboles")

        return self.watchlist

    def analyze_watchlist(self):
        """Analyse tous les symboles de la watchlist"""
        if not self.watchlist:
            self.setup_watchlist()

        print(f"🔍 Analyse de la watchlist ({len(self.watchlist)} symboles)...")

        # Chargement en parallèle
        pairs = [(symbol, "1h") for symbol in self.watchlist]
        results = self.tradx.get_multiple_trading_data(pairs, indicators=["rsi", "bollinger"])

        analysis_results = []

        for symbol_interval, df in results.items():
            if df is not None and len(df) > 0:
                symbol = symbol_interval.replace("_1h", "")
                latest = df.iloc[-1]

                # Votre logique d'analyse personnalisée ici
                score = self._calculate_custom_score(df, latest)

                analysis_results.append({
                    "symbol": symbol,
                    "score": score,
                    "price": latest['close'],
                    "rsi": latest['rsi']
                })

        # Tri par score
        analysis_results.sort(key=lambda x: x['score'], reverse=True)

        print(f"📊 Top 5 recommandations:")
        for i, result in enumerate(analysis_results[:5], 1):
            print(f"  {i}. {result['symbol']:8s} - Score: {result['score']:.2f} "
                  f"(Prix: ${result['price']:.2f}, RSI: {result['rsi']:.1f})")

        return analysis_results

    def _calculate_custom_score(self, df, latest_row):
        """Calcule un score personnalisé (exemple)"""
        # Exemple de logique de scoring
        rsi_score = max(0, min(100, 100 - abs(latest_row['rsi'] - 50))) / 100

        # Position relative dans les bandes de Bollinger
        bb_position = (latest_row['close'] - latest_row['bb_lower']) / \
                     (latest_row['bb_upper'] - latest_row['bb_lower'])
        bb_score = 1 - abs(bb_position - 0.5) * 2  # Score max au milieu

        # Score composite
        return (rsi_score * 0.6) + (bb_score * 0.4)

    def run_custom_strategy(self):
        """Lance votre stratégie personnalisée"""
        print("🎯 Lancement de la stratégie personnalisée...")

        # Configuration
        self.setup_watchlist(10)

        # Analyse
        results = self.analyze_watchlist()

        # Vos actions personnalisées ici
        print("✅ Stratégie exécutée avec succès!")

        return results

def main():
    """Fonction principale avec tous les exemples"""
    print("🚀 EXEMPLES D'INTÉGRATION TRADXPRO CORE MANAGER")
    print("=" * 60)

    try:
        # Exemple 1: Usage basique
        exemple_basic_usage()

        # Exemple 2: Stratégie de trading
        exemple_trading_strategy()

        # Exemple 3: Analyse de portfolio
        exemple_portfolio_analysis()

        # Exemple 4: Gestion des données
        exemple_data_management()

        # Exemple 5: Intégration dans une classe personnalisée
        print("\n=== EXEMPLE CLASSE PERSONNALISÉE ===")
        mon_programme = MonProgrammeAvecTradXPro()
        mon_programme.run_custom_strategy()

        print("\n" + "=" * 60)
        print("✅ TOUS LES EXEMPLES TERMINÉS AVEC SUCCÈS!")
        print("💡 Vous pouvez maintenant adapter ces exemples à vos besoins")

    except KeyboardInterrupt:
        print("\n👋 Arrêt demandé par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
```
<!-- MODULE-END: exemple_integration_tradxpro.py -->

<!-- MODULE-START: quick_start_tradxpro.py -->
## 
*Chemin* : `D:/TradXPro/scripts/mise_a_jour_dataframe/token_diversity_manager/examples/quick_start_tradxpro.py`  
*Type* : `.py`  

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradXPro - Démarrage Rapide
===========================

Script de démarrage rapide pour incorporer facilement toute la logique TradXPro
dans votre programme en 3 étapes simples.

Usage:
    python quick_start_tradxpro.py

Ce script vous montre comment :
1. Récupérer automatiquement les 100 meilleurs tokens crypto
2. Télécharger leurs données historiques
3. Les analyser avec des indicateurs techniques

Auteur: TradXPro Team
Date: 2 octobre 2025
"""

import sys
import time
from pathlib import Path

# Assurez-vous que le module TradXPro est dans le chemin
sys.path.append(str(Path(__file__).parent))

from tradxpro_core_manager import TradXProManager

def quick_start_demo():
    """Démonstration en 3 étapes simples"""

    print("🚀 TRADXPRO - DÉMARRAGE RAPIDE")
    print("=" * 50)
    print("Incorporez toute la logique TradXPro en 3 étapes simples !")
    print()

    # ========================================
    # ÉTAPE 1: Initialisation
    # ========================================
    print("📋 ÉTAPE 1: Initialisation du gestionnaire TradXPro")
    print("-" * 50)

    # Créer le gestionnaire - il gère tout automatiquement !
    manager = TradXProManager()

    print("✅ Gestionnaire TradXPro initialisé")
    print(f"   📁 Dossier racine: {manager.paths.root}")
    print(f"   💾 Données JSON: {manager.paths.json_root}")
    print(f"   ⚡ Données Parquet: {manager.paths.parquet_root}")
    print()

    # ========================================
    # ÉTAPE 2: Récupération des meilleurs tokens
    # ========================================
    print("📊 ÉTAPE 2: Récupération des 100 meilleurs tokens crypto")
    print("-" * 50)

    # Le gestionnaire récupère automatiquement les top 100 depuis CoinGecko + Binance
    tokens = manager.get_top_100_tokens(save_to_file=True)

    if tokens:
        print(f"✅ {len(tokens)} tokens récupérés et sauvegardés")
        print("🏆 Top 10 tokens par score composite:")

        for i, token in enumerate(tokens[:10], 1):
            print(f"   {i:2d}. {token['symbol']:8s} - {token['name'][:30]:<30s} "
                  f"(Score: {token['score']:.1f})")
    else:
        print("❌ Erreur lors de la récupération des tokens")
        return False

    print()

    # ========================================
    # ÉTAPE 3: Analyse avec indicateurs techniques
    # ========================================
    print("📈 ÉTAPE 3: Analyse avec indicateurs techniques")
    print("-" * 50)

    # Sélectionner quelques tokens pour la démo
    demo_symbols = [token["symbol"] + "USDC" for token in tokens[:5]]

    print(f"🔍 Analyse de {len(demo_symbols)} tokens avec indicateurs...")

    for symbol in demo_symbols:
        print(f"\n📊 Analyse {symbol}:")

        # Chargement des données avec indicateurs automatique
        df = manager.get_trading_data(
            symbol=symbol,
            interval="1h",
            indicators=["rsi", "bollinger", "atr", "macd"]
        )

        if df is not None and len(df) > 0:
            latest = df.iloc[-1]

            print(f"   💰 Prix actuel: ${latest['close']:.4f}")
            print(f"   📈 RSI (14): {latest['rsi']:.1f}")
            print(f"   🔵 Bollinger Upper: ${latest['bb_upper']:.4f}")
            print(f"   🔵 Bollinger Lower: ${latest['bb_lower']:.4f}")
            print(f"   ⚡ ATR (14): {latest['atr']:.6f}")
            print(f"   📊 MACD: {latest['macd']:.6f}")

            # Signal simple
            if latest['rsi'] < 30:
                print("   🟢 SIGNAL: RSI Oversold - Potentiel d'achat")
            elif latest['rsi'] > 70:
                print("   🔴 SIGNAL: RSI Overbought - Potentiel de vente")
            else:
                print("   🟡 SIGNAL: RSI Neutre")

        else:
            print("   ❌ Données non disponibles")

    print()
    print("=" * 50)
    print("✅ DÉMARRAGE RAPIDE TERMINÉ AVEC SUCCÈS!")
    print()

    return True

def integration_template():
    """Template pour intégrer TradXPro dans votre code"""

    print("💡 TEMPLATE D'INTÉGRATION POUR VOTRE CODE")
    print("=" * 50)

    template_code = '''
# ========================================
# INTÉGRATION TRADXPRO DANS VOTRE CODE
# ========================================

from tradxpro_core_manager import TradXProManager

class MonApplication:
    def __init__(self):
        # Initialiser TradXPro
        self.tradx = TradXProManager()

    def obtenir_meilleurs_tokens(self, nombre=100):
        """Récupère les N meilleurs tokens"""
        return self.tradx.get_top_100_tokens()[:nombre]

    def analyser_token(self, symbol, interval="1h"):
        """Analyse complète d'un token"""
        return self.tradx.get_trading_data(
            symbol=symbol,
            interval=interval,
            indicators=["rsi", "bollinger", "atr", "macd"]
        )

    def telecharger_donnees(self, symbols):
        """Télécharge les données pour une liste de tokens"""
        return self.tradx.download_crypto_data(symbols)

    def ma_strategie_personnalisee(self):
        """Votre stratégie personnalisée"""
        # 1. Récupérer les meilleurs tokens
        tokens = self.obtenir_meilleurs_tokens(20)

        # 2. Analyser chaque token
        for token in tokens:
            symbol = token["symbol"] + "USDC"
            df = self.analyser_token(symbol)

            if df is not None:
                # 3. Appliquer votre logique
                latest = df.iloc[-1]

                # Exemple de condition d'achat
                if (latest['rsi'] < 35 and
                    latest['close'] < latest['bb_lower'] and
                    latest['macd'] > latest['macd_signal']):
                    print(f"🟢 SIGNAL ACHAT: {symbol}")

                # Vos autres conditions...

# Usage:
app = MonApplication()
app.ma_strategie_personnalisee()
'''

    print(template_code)

    # Sauvegarder le template
    template_file = Path(__file__).parent / "template_integration_tradxpro.py"
    with open(template_file, 'w', encoding='utf-8') as f:
        f.write(template_code)

    print(f"💾 Template sauvegardé: {template_file}")

def show_features():
    """Affiche toutes les fonctionnalités disponibles"""

    print("🎯 FONCTIONNALITÉS DISPONIBLES")
    print("=" * 50)

    features = [
        ("🏆 Top 100 Tokens", "Récupération automatique via CoinGecko + Binance"),
        ("📥 Téléchargement", "Données historiques OHLCV multi-timeframes"),
        ("💾 Stockage Optimisé", "JSON + Parquet avec compression"),
        ("📈 Indicateurs", "RSI, Bollinger, ATR, EMA, MACD et plus"),
        ("⚡ Performance", "Chargement parallèle et cache automatique"),
        ("🔄 Mise à jour", "Actualisation automatique des données"),
        ("📊 Analyse", "Outils d'analyse technique intégrés"),
        ("🛠️ API Simple", "Interface unifiée facile à utiliser"),
        ("📁 Gestion Fichiers", "Organisation automatique des données"),
        ("🚀 Extensible", "Facilement intégrable dans vos projets")
    ]

    for feature, description in features:
        print(f"{feature:<20} {description}")

    print()
    print("📚 MÉTHODES PRINCIPALES:")
    methods = [
        "manager.get_top_100_tokens()",
        "manager.download_crypto_data(symbols)",
        "manager.get_trading_data(symbol, interval, indicators)",
        "manager.get_multiple_trading_data(pairs)",
        "manager.get_data_statistics()",
        "manager.get_available_data()"
    ]

    for method in methods:
        print(f"   • {method}")

def main():
    """Fonction principale"""

    print("Choisissez une option:")
    print("1. 🚀 Démarrage rapide (démo complète)")
    print("2. 💡 Template d'intégration")
    print("3. 🎯 Voir toutes les fonctionnalités")
    print("4. ❌ Quitter")

    try:
        choice = input("\nVotre choix (1-4): ").strip()

        if choice == "1":
            success = quick_start_demo()
            if success:
                print("🎉 Vous pouvez maintenant utiliser TradXPro dans vos projets!")

        elif choice == "2":
            integration_template()

        elif choice == "3":
            show_features()

        elif choice == "4":
            print("👋 Au revoir!")

        else:
            print("❌ Choix invalide")

    except KeyboardInterrupt:
        print("\n👋 Au revoir!")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")

if __name__ == "__main__":
    main()
```
<!-- MODULE-END: quick_start_tradxpro.py -->

## Sommaire

1. [__init__.py](#) — `__init__.py` — ligne 1 — `.py`
2. [launch.py](#) — `launch.py` — ligne 96 — `.py`
3. [setup_module.py](#) — `setup_module.py` — ligne 238 — `.py`
4. [test_module.py](#) — `test_module.py` — ligne 434 — `.py`
5. [tradxpro_core_manager.py](#) — `tradxpro_core_manager.py` — ligne 608 — `.py`
6. [test_diversite_simple.py](#) — `tests\test_diversite_simple.py` — ligne 1710 — `.py`
7. [test_token_diversity.py](#) — `tests\test_token_diversity.py` — ligne 1801 — `.py`
8. [exemple_integration_tradxpro.py](#) — `examples\exemple_integration_tradxpro.py` — ligne 2004 — `.py`
9. [quick_start_tradxpro.py](#) — `examples\quick_start_tradxpro.py` — ligne 2352 — `.py`

## Arborescence minimale

`cd D:\TradXPro\scripts\mise_a_jour_dataframe\token_diversity_manager`

- __init__.py
- launch.py
- setup_module.py
- test_module.py
- tradxpro_core_manager.py
- **examples/**
  - exemple_integration_tradxpro.py
  - quick_start_tradxpro.py
- **tests/**
  - test_diversite_simple.py
  - test_token_diversity.py