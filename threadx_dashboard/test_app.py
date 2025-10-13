"""
Script de test pour ThreadX Dashboard
====================================

Ce script teste le lancement de l'application et les fonctionnalités de base.
"""

import sys
import os
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Teste l'importation de tous les modules."""
    print("🧪 Test des imports...")

    try:
        import config

        print("✅ config.py importé avec succès")

        from utils.auth import auth_manager

        print("✅ utils.auth importé avec succès")

        from utils.helpers import setup_logging

        print("✅ utils.helpers importé avec succès")

        from components.navbar import create_navbar

        print("✅ components.navbar importé avec succès")

        from components.sidebar import create_sidebar

        print("✅ components.sidebar importé avec succès")

        from pages.home import create_home_layout

        print("✅ pages.home importé avec succès")

        return True

    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        return False


def test_config():
    """Teste la configuration."""
    print("\n🔧 Test de la configuration...")

    try:
        from config import THEME, AUTH_ENABLED, HOST, PORT

        print(f"✅ Thème chargé: {len(THEME)} couleurs")
        print(f"✅ Authentification: {'Activée' if AUTH_ENABLED else 'Désactivée'}")
        print(f"✅ Serveur: {HOST}:{PORT}")

        return True

    except Exception as e:
        print(f"❌ Erreur de configuration: {e}")
        return False


def test_auth():
    """Teste l'authentification."""
    print("\n🔐 Test de l'authentification...")

    try:
        from utils.auth import auth_manager
        from flask import Flask

        # Créer un contexte Flask temporaire pour les tests
        temp_app = Flask(__name__)
        temp_app.secret_key = "test-secret-key"

        with temp_app.test_request_context():
            # Test de connexion avec des identifiants valides
            success = auth_manager.login("admin", "admin123")
            if success:
                print("✅ Connexion réussie avec admin/admin123")

                # Test de vérification d'authentification
                if auth_manager.is_authenticated():
                    print("✅ Vérification d'authentification OK")
                else:
                    print("❌ Vérification d'authentification échoué")

                # Test de déconnexion
                auth_manager.logout()
                print("✅ Déconnexion réussie")

            else:
                print("❌ Connexion échouée")

            return success

    except Exception as e:
        print(f"❌ Erreur d'authentification: {e}")
        return False


def test_components():
    """Teste la création des composants."""
    print("\n🧩 Test des composants...")

    try:
        from components.navbar import create_navbar
        from components.sidebar import create_sidebar
        from pages.home import create_home_layout

        navbar = create_navbar()
        print("✅ Navbar créée")

        sidebar = create_sidebar()
        print("✅ Sidebar créée")

        home = create_home_layout()
        print("✅ Page d'accueil créée")

        return True

    except Exception as e:
        print(f"❌ Erreur de création des composants: {e}")
        return False


def test_app_creation():
    """Teste la création de l'application Dash."""
    print("\n🚀 Test de création de l'application...")

    try:
        # Import de l'app (sans la lancer)
        import app

        if hasattr(app, "app") and hasattr(app, "server"):
            print("✅ Application Dash créée")
            print("✅ Serveur Flask configuré")
            return True
        else:
            print("❌ Application mal configurée")
            return False

    except Exception as e:
        print(f"❌ Erreur de création d'application: {e}")
        return False


def main():
    """Fonction principale de test."""
    print("🎯 ThreadX Dashboard - Tests de validation\n")

    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Authentification", test_auth),
        ("Composants", test_components),
        ("Application", test_app_creation),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} échoué: {e}")
            results.append((test_name, False))

    # Résumé
    print("\n" + "=" * 50)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 50)

    passed = 0
    for test_name, result in results:
        status = "✅ PASSÉ" if result else "❌ ÉCHEC"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1

    print(f"\n🎯 Score: {passed}/{len(tests)} tests réussis")

    if passed == len(tests):
        print("\n🎉 Tous les tests sont passés ! L'application est prête.")
        print("\n💡 Pour lancer l'application:")
        print("   cd threadx_dashboard")
        print("   python app.py")
    else:
        print("\n⚠️  Certains tests ont échoué. Vérifiez la configuration.")

    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
