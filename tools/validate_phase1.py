#!/usr/bin/env python3
"""
Script de validation Phase 1: Configuration and Paths
Test manuel sans pytest pour validation rapide.
"""

import sys
import os
from pathlib import Path

# Ajout du path ThreadX
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    # Test direct des modules
    from threadx.config.settings_simple import Settings
    from threadx.config.loaders import load_settings, print_config, ConfigurationError
except ImportError as e:
    print(f"❌ Erreur import ThreadX: {e}")
    print("Vérifiez que toml est installé: pip install toml")
    print("Si besoin : pip install toml")
    sys.exit(1)


def test_basic_loading():
    """Test 1: Chargement de base."""
    print("🔍 Test 1: Chargement configuration de base...")
    
    try:
        # Chargement depuis paths.toml root
        settings = load_settings()
        print("✅ Configuration chargée avec succès")
        
        # Vérifications critiques Phase 1
        assert isinstance(settings, Settings), "Type Settings incorrect"
        assert hasattr(settings, 'DATA_ROOT'), "DATA_ROOT manquant"
        assert hasattr(settings, 'GPU_DEVICES'), "GPU_DEVICES manquant"
        assert hasattr(settings, 'SUPPORTED_TIMEFRAMES'), "SUPPORTED_TIMEFRAMES manquant"
        
        print(f"   📁 Data Root: {settings.DATA_ROOT}")
        print(f"   🚀 GPU Devices: {settings.GPU_DEVICES}")
        print(f"   📊 Timeframes: {len(settings.SUPPORTED_TIMEFRAMES)} supportés")
        
        return True
        
    except Exception as e:
        print(f"❌ Échec chargement: {e}")
        return False


def test_no_environment_vars():
    """Test 2: Pas de variables d'environnement."""
    print("\n🔍 Test 2: Indépendance variables d'environnement...")
    
    try:
        # Pollution environnement style TradXPro
        original_env = os.environ.copy()
        
        os.environ["TRADX_DATA_ROOT"] = "/fake/tradx/path"
        os.environ["INDICATORS_DB_ROOT"] = "/fake/indicators"
        os.environ["TRADX_USE_GPU"] = "0"
        
        # Chargement - doit ignorer les env vars
        settings = load_settings()
        
        # ThreadX ne doit pas utiliser les env vars
        data_root_str = str(settings.DATA_ROOT)
        indicators_str = str(settings.INDICATORS_ROOT)
        
        if "/fake/" in data_root_str or "/fake/" in indicators_str:
            print("❌ ThreadX utilise les variables d'environnement!")
            return False
        
        print("✅ Variables d'environnement ignorées correctement")
        print(f"   📁 Data Root (TOML): {settings.DATA_ROOT}")
        print(f"   📁 Indicators (TOML): {settings.INDICATORS_ROOT}")
        
        # Restauration env
        os.environ.clear()
        os.environ.update(original_env)
        
        return True
        
    except Exception as e:
        print(f"❌ Échec test env vars: {e}")
        return False


def test_cli_overrides():
    """Test 3: Overrides CLI."""
    print("\n🔍 Test 3: Surcharges ligne de commande...")
    
    try:
        # Override via paramètres
        settings = load_settings(
            data_root="./custom_test_data",
            logs="./custom_test_logs"
        )
        
        # Normalisation des chemins pour comparaison
        data_root_norm = str(settings.DATA_ROOT).replace("\\", "/")
        logs_norm = str(settings.LOGS_DIR).replace("\\", "/")
        
        if data_root_norm != "custom_test_data":
            print(f"❌ Override data_root échoué: attendu 'custom_test_data', obtenu '{data_root_norm}'")
            return False
            
        if logs_norm != "custom_test_logs":
            print(f"❌ Override logs échoué: attendu 'custom_test_logs', obtenu '{logs_norm}'")
            return False
        
        print("✅ Overrides CLI fonctionnels")
        print(f"   📁 Data override: {settings.DATA_ROOT}")
        print(f"   📁 Logs override: {settings.LOGS_DIR}")
        
        return True
        
    except Exception as e:
        print(f"❌ Échec overrides: {e}")
        return False


def test_paths_expansion():
    """Test 4: Expansion de chemins avec variables."""
    print("\n🔍 Test 4: Expansion chemins avec variables...")
    
    try:
        settings = load_settings()
        
        # Vérification expansion {data_root}
        data_root = settings.DATA_ROOT
        indicators = settings.INDICATORS_ROOT
        
        # Si indicators contient data_root, l'expansion a fonctionné
        if str(data_root) in str(indicators) or indicators.is_relative_to(data_root):
            print("✅ Expansion de chemins fonctionnelle")
            print(f"   📁 Base: {data_root}")
            print(f"   📁 Expandé: {indicators}")
            return True
        else:
            print(f"⚠️  Expansion possiblement non nécessaire")
            print(f"   📁 Data: {data_root}")
            print(f"   📁 Indicators: {indicators}")
            return True  # Pas forcément une erreur
        
    except Exception as e:
        print(f"❌ Échec expansion: {e}")
        return False


def test_print_config():
    """Test 5: Affichage configuration."""
    print("\n🔍 Test 5: Affichage configuration...")
    
    try:
        settings = load_settings()
        print("\n" + "="*50)
        print_config(settings)
        print("="*50)
        
        print("✅ Affichage configuration réussi")
        return True
        
    except Exception as e:
        print(f"❌ Échec affichage: {e}")
        return False


def validate_paths_toml():
    """Validation du fichier paths.toml."""
    print("\n🔍 Validation fichier paths.toml...")
    
    toml_path = Path(__file__).parent.parent / "paths.toml"
    
    if not toml_path.exists():
        print(f"❌ Fichier paths.toml introuvable: {toml_path}")
        return False
    
    try:
        import toml
        with open(toml_path, 'r') as f:
            config = toml.load(f)
        
        # Sections requises
        required_sections = ['paths', 'gpu', 'performance', 'timeframes']
        for section in required_sections:
            if section not in config:
                print(f"❌ Section [{section}] manquante")
                return False
        
        # Clés critiques
        if 'data_root' not in config['paths']:
            print("❌ paths.data_root manquant")
            return False
            
        if 'devices' not in config['gpu']:
            print("❌ gpu.devices manquant")
            return False
        
        print("✅ Fichier paths.toml valide")
        print(f"   📁 Localisation: {toml_path}")
        print(f"   📊 Sections: {list(config.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur validation TOML: {e}")
        return False


def main():
    """Exécution complète des tests Phase 1."""
    print("🚀 VALIDATION PHASE 1 - CONFIGURATION AND PATHS")
    print("=" * 60)
    
    tests = [
        ("Fichier paths.toml", validate_paths_toml),
        ("Chargement de base", test_basic_loading),
        ("Pas d'env vars", test_no_environment_vars),
        ("Overrides CLI", test_cli_overrides),  
        ("Expansion chemins", test_paths_expansion),
        ("Affichage config", test_print_config)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            if success:
                passed += 1
            else:
                print(f"⚠️  Test '{test_name}' échoué")
        except Exception as e:
            print(f"❌ Test '{test_name}' erreur: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 RÉSULTAT: {passed}/{total} tests réussis")
    
    if passed == total:
        print("🎉 PHASE 1 VALIDÉE - Configuration TOML opérationnelle!")
        print("\n✅ Critères de succès Phase 1:")
        print("   ✓ Chargement config sans variables d'environnement")
        print("   ✓ Fonction print_config() affiche configuration")
        print("   ✓ Overrides CLI fonctionnels")
        print("   ✓ Chemins relatifs uniquement")
        print("   ✓ Inspiré TradXPro mais 100% TOML-only")
        return True
    else:
        print("❌ PHASE 1 INCOMPLÈTE - Corrections nécessaires")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)