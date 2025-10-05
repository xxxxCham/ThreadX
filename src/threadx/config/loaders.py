"""
ThreadX Configuration Loaders - Phase 1
Chargement et validation de la configuration TOML.
Remplace les env vars de TradXPro par un système centralisé.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

import toml

from .settings import Settings

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Erreur de configuration ThreadX."""

    pass


class PathValidationError(Exception):
    """Erreur de validation de chemin."""

    pass


class TOMLConfigLoader:
    """
    Chargeur de configuration TOML pour ThreadX.

    Remplace le système dispersé de TradXPro:
    - core/indicators_db.py: load_config() + env vars
    - perf_manager.py: PerfConfig hardcodé
    - Chemins absolus éparpillés
    """

    DEFAULT_CONFIG_NAME = "paths.toml"

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialise le loader de configuration.

        Args:
            config_path: Chemin vers fichier TOML. Si None, cherche paths.toml
                        dans le répertoire courant puis parent.
        """
        self.config_path = self._find_config_file(config_path)
        self.config_data: Dict[str, Any] = {}
        self._load_config()

    def _find_config_file(self, provided_path: Optional[Union[str, Path]]) -> Path:
        """
        Trouve le fichier de configuration TOML.
        Inspiré de la logique de recherche de TradXPro mais plus robuste.
        """
        if provided_path:
            path = Path(provided_path)
            if path.exists():
                logger.info(f"Configuration trouvée: {path}")
                return path
            else:
                raise ConfigurationError(f"Fichier config spécifié introuvable: {path}")

        # Recherche automatique: répertoire courant puis parent
        search_paths = [
            Path.cwd() / self.DEFAULT_CONFIG_NAME,
            Path.cwd().parent / self.DEFAULT_CONFIG_NAME,
            Path(__file__).parent.parent.parent.parent
            / self.DEFAULT_CONFIG_NAME,  # ThreadX root
        ]

        for path in search_paths:
            if path.exists():
                logger.info(f"Configuration auto-détectée: {path}")
                return path

        raise ConfigurationError(
            f"Fichier {self.DEFAULT_CONFIG_NAME} introuvable dans:\n"
            + "\n".join(f"  - {p}" for p in search_paths)
        )

    def _load_config(self) -> None:
        """Charge le fichier TOML avec gestion d'erreurs."""
        try:
            logger.debug(f"Chargement configuration: {self.config_path}")
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config_data = toml.load(f)
            logger.info(f"Configuration chargée: {len(self.config_data)} sections")
        except Exception as e:
            raise ConfigurationError(f"Erreur lecture {self.config_path}: {e}")

    def get_section(self, section_name: str) -> Dict[str, Any]:
        """Récupère une section de la configuration."""
        if section_name not in self.config_data:
            logger.warning(f"Section [{section_name}] manquante, retour dict vide")
            return {}
        return self.config_data[section_name]

    def get_value(self, section: str, key: str, default: Any = None) -> Any:
        """Récupère une valeur spécifique."""
        section_data = self.get_section(section)
        return section_data.get(key, default)

    def expand_paths(self, section_name: str = "paths") -> Dict[str, Path]:
        """
        Expanse les chemins avec substitution de variables.
        Remplace la logique de chemins absolus de TradXPro.
        """
        paths_config = self.get_section(section_name)
        expanded = {}

        # Premier passage: chemins de base
        for key, value in paths_config.items():
            if isinstance(value, str):
                expanded[key] = Path(value)

        # Deuxième passage: substitution variables
        for key, path in expanded.items():
            path_str = str(path)
            if "{" in path_str:
                # Substitution simple: {data_root} -> valeur de data_root
                for var_key, var_path in expanded.items():
                    placeholder = f"{{{var_key}}}"
                    if placeholder in path_str:
                        path_str = path_str.replace(placeholder, str(var_path))
                expanded[key] = Path(path_str)

        return expanded

    def validate_config(self) -> List[str]:
        """
        Valide la configuration et retourne les erreurs.
        Plus strict que TradXPro pour éviter les problèmes runtime.
        """
        errors = []

        # Sections requises
        required_sections = ["paths", "gpu", "performance", "timeframes"]
        for section in required_sections:
            if section not in self.config_data:
                errors.append(f"Section [{section}] manquante")

        # Validation paths
        paths_config = self.get_section("paths")
        if "data_root" not in paths_config:
            errors.append("paths.data_root requis")

        # Validation GPU
        gpu_config = self.get_section("gpu")
        if gpu_config.get("devices") and not isinstance(gpu_config["devices"], list):
            errors.append("gpu.devices doit être une liste")

        # Validation timeframes
        tf_config = self.get_section("timeframes")
        if tf_config.get("supported") and not isinstance(tf_config["supported"], list):
            errors.append("timeframes.supported doit être une liste")

        return errors

    def create_settings(self, **overrides) -> Settings:
        """
        Crée une instance Settings à partir de la config TOML.
        Permet des overrides pour CLI args.
        """
        # Validation préalable
        errors = self.validate_config()
        if errors:
            raise ConfigurationError(
                f"Configuration invalide:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        # Expansion des chemins
        expanded_paths = self.expand_paths()

        # Construction des paramètres
        paths_config = self.get_section("paths")
        gpu_config = self.get_section("gpu")
        perf_config = self.get_section("performance")
        indicators_config = self.get_section("indicators")
        tf_config = self.get_section("timeframes")
        logging_config = self.get_section("logging")
        security_config = self.get_section("security")
        backtest_config = self.get_section("backtest")
        ui_config = self.get_section("ui")

        # Phase 1: Version simplifiée - seulement les champs de settings_simple.py
        return Settings(
            # Paths
            DATA_ROOT=Path(
                overrides.get("data_root", expanded_paths.get("data_root", "./data"))
            ),
            INDICATORS_ROOT=Path(
                overrides.get(
                    "indicators", expanded_paths.get("indicators", "./data/indicators")
                )
            ),
            LOGS_DIR=Path(overrides.get("logs", expanded_paths.get("logs", "./logs"))),
            # GPU
            GPU_DEVICES=gpu_config.get("devices", ["5090", "2060"]),
            GPU_LOAD_BALANCE=gpu_config.get(
                "load_balance", {"5090": 0.75, "2060": 0.25}
            ),
            # Performance
            TARGET_TASKS_PER_MIN=perf_config.get("target_tasks_per_min", 2500),
            # Timeframes
            SUPPORTED_TIMEFRAMES=tuple(
                tf_config.get(
                    "supported",
                    [
                        "1m",
                        "3m",
                        "5m",
                        "15m",
                        "30m",
                        "1h",
                        "2h",
                        "4h",
                        "6h",
                        "8h",
                        "12h",
                        "1d",
                    ],
                )
            ),
            # Logging
            LOG_LEVEL=logging_config.get("level", "INFO"),
            # Security
            ALLOW_ABSOLUTE_PATHS=security_config.get("allow_absolute_paths", False),
        )

    def load_config(self, overrides: Optional[Dict[str, Any]] = None) -> Settings:
        """
        Méthode de compatibilité pour les tests.
        Charge la config et retourne un objet Settings.
        """
        if overrides is None:
            overrides = {}
        return self.create_settings(**overrides)


def load_settings(
    config_path: Optional[Union[str, Path]] = None, **overrides
) -> Settings:
    """
    Fonction utilitaire pour charger les settings ThreadX.

    Args:
        config_path: Chemin vers paths.toml (optionnel)
        **overrides: Surcharges CLI (ex: data_root="./custom")

    Returns:
        Instance Settings configurée

    Example:
        # Chargement standard
        settings = load_settings()

        # Avec overrides CLI
        settings = load_settings(data_root="./custom_data", logs="./custom_logs")
    """
    loader = TOMLConfigLoader(config_path)
    return loader.create_settings(**overrides)


# Cache global pour get_settings
_cached_settings: Optional[Settings] = None


def get_settings(force_reload: bool = False) -> Settings:
    """
    Récupère les settings avec cache global (singleton pattern).
    Compatible avec les tests existants.
    """
    global _cached_settings
    if _cached_settings is None or force_reload:
        _cached_settings = load_settings()
    return _cached_settings


def print_config(settings: Optional[Settings] = None) -> None:
    """
    Affiche la configuration actuelle pour debugging.
    Utile pour valider la Phase 1.
    """
    if settings is None:
        settings = load_settings()

    print("=" * 60)
    print("📋 CONFIGURATION THREADX - PHASE 1")
    print("=" * 60)

    # Paths - Phase 1 (seulement champs existants)
    print("\n📁 CHEMINS:")
    print(f"  Data Root:    {settings.DATA_ROOT}")
    print(f"  Indicators:   {settings.INDICATORS_ROOT}")
    print(f"  Logs:         {settings.LOGS_DIR}")

    # GPU
    print(f"\n🚀 GPU:")
    print(f"  Devices:      {settings.GPU_DEVICES}")
    print(f"  Load Balance: {settings.GPU_LOAD_BALANCE}")

    # Performance
    print(f"\n⚡ PERFORMANCE:")
    print(f"  Target Tasks/min: {settings.TARGET_TASKS_PER_MIN}")

    # Timeframes
    print(f"\n📊 TIMEFRAMES:")
    print(f"  Supportés:    {', '.join(settings.SUPPORTED_TIMEFRAMES)}")

    # Sécurité
    print(f"\n🔒 SÉCURITÉ:")
    print(f"  Chemins absolus:  {settings.ALLOW_ABSOLUTE_PATHS}")
    print(f"  Log Level:        {settings.LOG_LEVEL}")

    print("=" * 60)


# CLI Argument Parser pour overrides
def create_argument_parser():
    """Crée le parser CLI pour surcharger la config TOML."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ThreadX - Framework de backtesting crypto",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python -m threadx --data-root ./custom_data
  python -m threadx --config ./custom_paths.toml
  python -m threadx --gpu-devices 5090 2060 --log-level DEBUG
        """,
    )

    # Configuration
    parser.add_argument(
        "--config", type=str, help="Chemin vers fichier paths.toml personnalisé"
    )

    # Chemins
    parser.add_argument("--data-root", type=str, help="Répertoire racine des données")
    parser.add_argument("--indicators", type=str, help="Répertoire cache indicateurs")
    parser.add_argument("--logs", type=str, help="Répertoire logs")

    # GPU
    parser.add_argument(
        "--gpu-devices", nargs="+", help="Liste devices GPU (ex: 5090 2060)"
    )
    parser.add_argument("--disable-gpu", action="store_true", help="Désactiver GPU")

    # Performance
    parser.add_argument("--max-workers", type=int, help="Nombre max workers parallèles")
    parser.add_argument("--batch-size", type=int, help="Taille batch vectorisation")

    # Logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Niveau de log",
    )

    return parser


if __name__ == "__main__":
    # Test CLI
    parser = create_argument_parser()
    args = parser.parse_args()

    # Conversion args vers dict overrides
    overrides = {k: v for k, v in vars(args).items() if v is not None and k != "config"}

    # Chargement avec overrides
    settings = load_settings(
        args.config if hasattr(args, "config") else None, **overrides
    )

    # Affichage
    print_config(settings)
