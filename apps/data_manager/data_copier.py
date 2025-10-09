"""
ThreadX Data Copier - Utilitaire de copie des données locales
Copie les données depuis les sources externes vers l'espace de travail ThreadX
"""

import shutil
import json
from pathlib import Path
from typing import Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ThreadXDataCopier:
    """Utilitaire pour copier les données dans l'espace de travail ThreadX"""

    def __init__(self, threadx_root: str = "D:\\ThreadX"):
        self.threadx_root = Path(threadx_root)
        self.data_root = self.threadx_root / "data"

        # Sources de données découvertes
        self.sources = {
            "indicators_cache": Path("D:\\ThreadX\\indicators_cache"),
            "trading_data": Path("d:\\Tools\\dataframe\\parquet"),
            # Ajouter d'autres sources si découvertes
        }

        # Destinations dans ThreadX
        self.destinations = {
            "indicators_cache": self.data_root / "raw" / "indicators_cache",
            "trading_data": self.data_root / "raw" / "trading_json",
        }

    def setup_data_structure(self) -> bool:
        """Créer la structure de données ThreadX"""
        logger.info("📁 Création de la structure de données ThreadX...")

        directories = [
            self.data_root,
            self.data_root / "raw",
            self.data_root / "processed",
            self.data_root / "indicators",
            self.data_root / "raw" / "indicators_cache",
            self.data_root / "raw" / "trading_json",
            self.data_root / "cache",
            self.data_root / "exports",
        ]

        try:
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"✅ Créé: {directory}")

            # Créer un README dans data/
            self._create_data_readme()

            logger.info("✅ Structure de données créée avec succès")
            return True

        except Exception as e:
            logger.error(f"❌ Erreur création structure: {e}")
            return False

    def copy_indicators_cache(self) -> Dict[str, Any]:
        """Copier le cache d'indicateurs ThreadX existant"""
        source = self.sources["indicators_cache"]
        destination = self.destinations["indicators_cache"]

        logger.info("📊 Copie du cache d'indicateurs...")
        logger.info(f"   Source: {source}")
        logger.info(f"   Destination: {destination}")

        result: Dict[str, Any] = {
            "success": False,
            "files_copied": 0,
            "total_size_mb": 0.0,
            "error": None,
        }

        if not source.exists():
            result["error"] = f"Source inexistante: {source}"
            logger.warning(result["error"])
            return result

        try:
            # Nettoyer la destination si elle existe
            if destination.exists():
                logger.info("🧹 Nettoyage destination existante...")
                shutil.rmtree(destination)

            # Copier récursivement
            shutil.copytree(source, destination)

            # Calculer les statistiques
            stats = self._calculate_directory_stats(destination)
            result.update(stats)
            result["success"] = True

            logger.info(
                f"✅ Cache copié: {result['files_copied']} fichiers, "
                f"{result['total_size_mb']:.1f} MB"
            )

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ Erreur copie cache: {e}")

        return result

    def copy_trading_data(self) -> Dict[str, Any]:
        """Copier les données de trading JSON"""
        source = self.sources["trading_data"]
        destination = self.destinations["trading_data"]

        logger.info("💰 Copie des données de trading...")
        logger.info(f"   Source: {source}")
        logger.info(f"   Destination: {destination}")

        result: Dict[str, Any] = {
            "success": False,
            "files_copied": 0,
            "total_size_mb": 0.0,
            "error": None,
            "symbols": [],
            "timeframes": [],
        }

        if not source.exists():
            result["error"] = f"Source inexistante: {source}"
            logger.warning(result["error"])
            return result

        try:
            # Créer la destination
            destination.mkdir(parents=True, exist_ok=True)

            # Copier les fichiers JSON
            symbols = set()
            timeframes = set()
            files_copied = 0
            total_size = 0

            for file_path in source.glob("*.json"):
                dest_file = destination / file_path.name
                shutil.copy2(file_path, dest_file)

                # Parser le nom pour extraire symbol/timeframe
                name_parts = file_path.stem.split("_")
                if len(name_parts) >= 2:
                    symbols.add(name_parts[0])
                    timeframes.add(name_parts[1])

                files_copied += 1
                total_size += file_path.stat().st_size

                if files_copied % 50 == 0:
                    logger.info(f"   📥 {files_copied} fichiers copiés...")

            result.update(
                {
                    "success": True,
                    "files_copied": files_copied,
                    "total_size_mb": total_size / (1024 * 1024),
                    "symbols": sorted(list(symbols)),
                    "timeframes": sorted(list(timeframes)),
                }
            )

            logger.info(
                f"✅ Trading data copiée: {files_copied} fichiers, "
                f"{result['total_size_mb']:.1f} MB"
            )
            logger.info(
                f"   📈 {len(symbols)} symboles, " f"{len(timeframes)} timeframes"
            )

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ Erreur copie trading: {e}")

        return result

    def copy_all_data(self) -> Dict[str, Any]:
        """Copier toutes les données disponibles"""
        logger.info("🚀 Début de la copie complète des données...")

        # Créer la structure
        if not self.setup_data_structure():
            return {"success": False, "error": "Échec création structure"}

        # Copier chaque source
        results: Dict[str, Any] = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "total_files": 0,
            "total_size_mb": 0.0,
            "sources": {},
        }

        # Cache d'indicateurs
        cache_result = self.copy_indicators_cache()
        results["sources"]["indicators_cache"] = cache_result
        if cache_result["success"]:
            results["total_files"] += cache_result["files_copied"]
            results["total_size_mb"] += cache_result["total_size_mb"]

        # Données de trading
        trading_result = self.copy_trading_data()
        results["sources"]["trading_data"] = trading_result
        if trading_result["success"]:
            results["total_files"] += trading_result["files_copied"]
            results["total_size_mb"] += trading_result["total_size_mb"]

        # Vérifier le succès global
        results["success"] = any(
            source["success"] for source in results["sources"].values()
        )

        if results["success"]:
            logger.info("🎉 Copie terminée avec succès!")
            logger.info(f"   📊 Total: {results['total_files']} fichiers")
            logger.info(f"   💾 Taille: {results['total_size_mb']:.1f} MB")

            # Créer un rapport de copie
            self._create_copy_report(results)
        else:
            logger.error("❌ Échec de la copie globale")

        return results

    def _calculate_directory_stats(self, directory: Path) -> Dict[str, Any]:
        """Calculer les statistiques d'un répertoire"""
        stats = {"files_copied": 0, "total_size_mb": 0.0, "directories": 0}

        for item in directory.rglob("*"):
            if item.is_file():
                stats["files_copied"] += 1
                stats["total_size_mb"] += item.stat().st_size / (1024 * 1024)
            elif item.is_dir():
                stats["directories"] += 1

        return stats

    def _create_data_readme(self):
        """Créer un README dans le dossier data/"""
        readme_content = """# ThreadX Data Directory

## 📁 Structure
```
data/
├── raw/                    # Données brutes copiées (NE PAS MODIFIER)
│   ├── indicators_cache/   # Cache ThreadX existant
│   └── trading_json/       # Données de trading JSON
├── processed/              # Données traitées par ThreadX
├── indicators/             # IndicatorBank final
├── cache/                  # Cache de calcul
└── exports/                # Exports utilisateur
```

## ⚠️ Important
- **NE PAS COMMITTER** ces données sur Git
- Les données raw/ sont des **copies de sauvegarde**
- Les sources originales restent intactes
- Utiliser ThreadX Data Manager pour la gestion

## 🔄 Mise à jour
Pour recopier les données depuis les sources:
```bash
python -m apps.data_manager.data_copier
```

## 📊 Dernière copie
Voir `copy_report.json` pour les détails de la dernière copie.
"""

        readme_path = self.data_root / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)

    def _create_copy_report(self, results: Dict[str, Any]):
        """Créer un rapport de copie"""
        report_path = self.data_root / "copy_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"📋 Rapport de copie créé: {report_path}")

    def get_local_data_info(self) -> Dict[str, Any]:
        """Obtenir des informations sur les données locales copiées"""
        info = {
            "data_root": str(self.data_root),
            "exists": self.data_root.exists(),
            "sources_available": [],
        }

        if info["exists"]:
            for name, path in self.destinations.items():
                if path.exists():
                    stats = self._calculate_directory_stats(path)
                    info["sources_available"].append(
                        {
                            "name": name,
                            "path": str(path),
                            "files": stats["files_copied"],
                            "size_mb": round(stats["total_size_mb"], 2),
                        }
                    )

        return info


def main():
    """Point d'entrée principal"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print("🚀 ThreadX Data Copier")
    print("=" * 50)

    copier = ThreadXDataCopier()

    # Vérifier l'état actuel
    info = copier.get_local_data_info()
    print(f"📁 Répertoire data: {info['data_root']}")
    print(f"   Existe: {'✅' if info['exists'] else '❌'}")

    if info["sources_available"]:
        print("📊 Sources déjà copiées:")
        for source in info["sources_available"]:
            print(
                f"   - {source['name']}: {source['files']} fichiers "
                f"({source['size_mb']} MB)"
            )

    # Demander confirmation pour la copie
    print("\n" + "=" * 50)
    print("⚠️  ATTENTION: Cette opération va copier les données")
    print("   dans l'espace de travail ThreadX local.")
    print("   Les données ne seront PAS commitées sur Git.")

    response = input("\n🤔 Procéder à la copie? (y/N): ").lower().strip()

    if response in ("y", "yes", "oui"):
        print("\n🔄 Début de la copie...")
        results = copier.copy_all_data()

        if results["success"]:
            print("\n🎉 Copie terminée avec succès!")
            print(f"📊 {results['total_files']} fichiers copiés")
            print(f"💾 {results['total_size_mb']:.1f} MB")
        else:
            print("\n❌ Échec de la copie. Voir les logs pour plus de détails.")
    else:
        print("\n⏹️  Copie annulée.")


if __name__ == "__main__":
    main()
