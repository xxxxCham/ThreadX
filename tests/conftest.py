"""Configuration pytest pour les tests ThreadX."""
import os


def pytest_configure():
    """Configure l'environnement de test."""
    # Évite d'afficher le warning Pandera dans les tests
    os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
