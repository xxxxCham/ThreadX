"""
ThreadX Data Discovery - Scanner pour données réelles découvertes
Scanner optimisé pour les structures de données spécifiques trouvées
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RealDataScanner:
    """Scanner pour les vraies données découvertes de l'utilisateur"""

    def __init__(self):
        # Chemins réels découverts
        self.discovered_paths = {
            "indicators_cache": Path("D:\\ThreadX\\indicators_cache"),
            "trading_data": Path("d:\\Tools\\dataframe\\parquet"),
            "hypothetical_g": Path("g:\\indicators_db"),
            "hypothetical_i": Path("i:\\indicators_db"),
        }

    def scan_real_data(self) -> Dict[str, any]:
        """Scan complet des données réelles découvertes"""
        results = {
            "total_files": 0,
            "total_size_mb": 0.0,
            "data_sources": {},
            "summary": {},
        }

        logger.info("🔍 Début du scan des données réelles...")

        # 1. Scanner le cache ThreadX existant
        if self.discovered_paths["indicators_cache"].exists():
            cache_info = self._scan_indicators_cache()
            results["data_sources"]["indicators_cache"] = cache_info

        # 2. Scanner les données de trading
        if self.discovered_paths["trading_data"].exists():
            trading_info = self._scan_trading_data()
            results["data_sources"]["trading_data"] = trading_info

        # 3. Scanner les chemins hypothétiques si ils existent
        for name, path in [
            ("g_drive", self.discovered_paths["hypothetical_g"]),
            ("i_drive", self.discovered_paths["hypothetical_i"]),
        ]:
            if path.exists():
                hyp_info = self._scan_hypothetical_path(path)
                results["data_sources"][name] = hyp_info

        # Calculer les totaux
        self._calculate_totals(results)

        return results

    def _scan_indicators_cache(self) -> Dict[str, any]:
        """Scanner D:\\ThreadX\\indicators_cache\\"""
        logger.info("📊 Scan du cache d'indicateurs ThreadX...")

        cache_path = self.discovered_paths["indicators_cache"]
        info = {
            "path": str(cache_path),
            "type": "ThreadX Indicators Cache",
            "indicators": {},
            "total_files": 0,
            "total_size_mb": 0.0,
            "status": "ThreadX Ready",  # Déjà au bon format !
        }

        try:
            for indicator_dir in cache_path.iterdir():
                if indicator_dir.is_dir():
                    indicator_name = indicator_dir.name
                    indicator_info = {
                        "files": [],
                        "parquet_files": 0,
                        "meta_files": 0,
                        "size_mb": 0.0,
                    }

                    for file_path in indicator_dir.iterdir():
                        if file_path.is_file():
                            file_info = self._analyze_cache_file(file_path)
                            indicator_info["files"].append(file_info)

                            if file_path.suffix == ".parquet":
                                indicator_info["parquet_files"] += 1
                            elif file_path.suffix == ".meta":
                                indicator_info["meta_files"] += 1

                            indicator_info["size_mb"] += file_info["size_mb"]

                    info["indicators"][indicator_name] = indicator_info
                    info["total_files"] += len(indicator_info["files"])
                    info["total_size_mb"] += indicator_info["size_mb"]

        except Exception as e:
            logger.error(f"Erreur scan cache: {e}")
            info["error"] = str(e)

        logger.info(
            f"✅ Cache scanné: {info['total_files']} fichiers, "
            f"{len(info['indicators'])} indicateurs"
        )
        return info

    def _scan_trading_data(self) -> Dict[str, any]:
        """Scanner d:\\Tools\\dataframe\\parquet\\"""
        logger.info("💰 Scan des données de trading...")

        trading_path = self.discovered_paths["trading_data"]
        info = {
            "path": str(trading_path),
            "type": "Trading Data JSON",
            "symbols": {},
            "timeframes": set(),
            "total_files": 0,
            "total_size_mb": 0.0,
            "status": "Raw Data - Needs Processing",
        }

        try:
            for file_path in trading_path.iterdir():
                if file_path.is_file() and file_path.suffix == ".json":
                    file_info = self._analyze_trading_file(file_path)
                    if file_info:
                        symbol = file_info["symbol"]
                        timeframe = file_info["timeframe"]

                        if symbol not in info["symbols"]:
                            info["symbols"][symbol] = {
                                "timeframes": {},
                                "total_size_mb": 0.0,
                            }

                        info["symbols"][symbol]["timeframes"][timeframe] = file_info
                        info["symbols"][symbol]["total_size_mb"] += file_info["size_mb"]
                        info["timeframes"].add(timeframe)
                        info["total_files"] += 1
                        info["total_size_mb"] += file_info["size_mb"]

        except Exception as e:
            logger.error(f"Erreur scan trading: {e}")
            info["error"] = str(e)

        # Convertir set en list pour JSON
        info["timeframes"] = sorted(list(info["timeframes"]))

        logger.info(
            f"✅ Trading scanné: {info['total_files']} fichiers, "
            f"{len(info['symbols'])} symboles, "
            f"{len(info['timeframes'])} timeframes"
        )
        return info

    def _analyze_cache_file(self, file_path: Path) -> Dict[str, any]:
        """Analyser un fichier du cache ThreadX"""
        stat = file_path.stat()

        # Parser le nom de fichier cache ThreadX
        # Format: indicator_BENCHMARK_DEVICE_SIZE_TIMEFRAME_HASH1_HASH2.ext
        name_parts = file_path.stem.split("_")

        return {
            "filename": file_path.name,
            "size_mb": stat.st_size / (1024 * 1024),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "extension": file_path.suffix,
            "parsed_info": {
                "indicator": name_parts[0] if name_parts else "unknown",
                "benchmark": name_parts[1] if len(name_parts) > 1 else None,
                "device": name_parts[2] if len(name_parts) > 2 else None,
                "size": name_parts[3] if len(name_parts) > 3 else None,
                "timeframe": name_parts[4] if len(name_parts) > 4 else None,
            },
        }

    def _analyze_trading_file(self, file_path: Path) -> Optional[Dict[str, any]]:
        """Analyser un fichier de données de trading"""
        try:
            # Parser le nom: SYMBOL_TIMEFRAME_12months.json
            name_parts = file_path.stem.split("_")
            if len(name_parts) < 3:
                return None

            symbol = name_parts[0]
            timeframe = name_parts[1]
            period = name_parts[2] if len(name_parts) > 2 else "12months"

            stat = file_path.stat()

            # Tentative d'analyse rapide du contenu JSON
            row_count = None
            date_range = None
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    # Lire juste le début pour éviter de charger tout
                    data = json.load(f)
                    if isinstance(data, list) and data:
                        row_count = len(data)
                        # Essayer d'extraire la plage de dates
                        if isinstance(data[0], dict):
                            first_date = data[0].get("timestamp") or data[0].get("date")
                            last_date = data[-1].get("timestamp") or data[-1].get(
                                "date"
                            )
                            if first_date and last_date:
                                date_range = (first_date, last_date)
            except:
                pass  # Ignorer les erreurs d'analyse rapide

            return {
                "filename": file_path.name,
                "symbol": symbol,
                "timeframe": timeframe,
                "period": period,
                "size_mb": stat.st_size / (1024 * 1024),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "row_count": row_count,
                "date_range": date_range,
            }

        except Exception as e:
            logger.debug(f"Erreur analyse {file_path}: {e}")
            return None

    def _scan_hypothetical_path(self, path: Path) -> Dict[str, any]:
        """Scanner un chemin hypothétique (g:\ ou i:\)"""
        logger.info(f"🔍 Scan de {path}...")

        info = {
            "path": str(path),
            "type": "Hypothetical Indicators DB",
            "exists": True,
            "total_files": 0,
            "total_size_mb": 0.0,
            "structure": "To be analyzed",
        }

        # TODO: Implémenter le scan si les chemins existent
        return info

    def _calculate_totals(self, results: Dict[str, any]) -> None:
        """Calculer les totaux globaux"""
        total_files = 0
        total_size = 0.0

        for source_info in results["data_sources"].values():
            total_files += source_info.get("total_files", 0)
            total_size += source_info.get("total_size_mb", 0.0)

        results["total_files"] = total_files
        results["total_size_mb"] = total_size

        # Résumé intelligent
        cache_ready = "indicators_cache" in results["data_sources"]
        trading_available = "trading_data" in results["data_sources"]

        results["summary"] = {
            "cache_ready": cache_ready,
            "trading_available": trading_available,
            "integration_strategy": self._recommend_strategy(
                cache_ready, trading_available
            ),
            "priority": self._determine_priority(results["data_sources"]),
        }

    def _recommend_strategy(self, cache_ready: bool, trading_available: bool) -> str:
        """Recommander une stratégie d'intégration"""
        if cache_ready and trading_available:
            return "HYBRID: Utiliser cache existant + convertir trading data"
        elif cache_ready:
            return "CACHE_FIRST: Exploiter le cache ThreadX existant"
        elif trading_available:
            return "CONVERT_TRADING: Convertir les données de trading en indicateurs"
        else:
            return "SCAN_MORE: Rechercher d'autres sources de données"

    def _determine_priority(self, sources: Dict[str, any]) -> List[str]:
        """Déterminer l'ordre de priorité des sources"""
        priority = []

        # Cache ThreadX = priorité absolue (déjà prêt)
        if "indicators_cache" in sources:
            priority.append("indicators_cache")

        # Données de trading = priorité secondaire (conversion nécessaire)
        if "trading_data" in sources:
            priority.append("trading_data")

        return priority


def create_real_data_summary() -> Dict[str, any]:
    """Créer un résumé des données réelles pour démonstration"""
    return {
        "discovered_paths": {
            "indicators_cache": "D:\\ThreadX\\indicators_cache",
            "trading_data": "d:\\Tools\\dataframe\\parquet",
        },
        "cache_analysis": {
            "bollinger": {
                "files": ["bollinger_BENCH_CPU_100000_1m_*.parquet", "*.meta"],
                "status": "ThreadX Ready",
            },
            "atr": {
                "files": ["atr_BENCH_ATR_CPU_10000_1m_*.meta"],
                "status": "ThreadX Ready",
            },
        },
        "trading_analysis": {
            "total_symbols": "300+",
            "timeframes": ["3m", "5m", "15m", "30m", "1h"],
            "period": "12 months",
            "total_size": "~2-3 GB",
            "major_symbols": ["BTC", "ETH", "SOL", "BNB", "DOGE"],
            "status": "Raw OHLCV - Needs Conversion",
        },
        "integration_plan": {
            "phase1": "Utiliser cache ThreadX existant",
            "phase2": "Convertir données trading en indicateurs",
            "phase3": "Unifier dans IndicatorBank ThreadX",
            "estimated_time": "2-3 heures",
        },
    }


if __name__ == "__main__":
    # Test du scanner
    scanner = RealDataScanner()
    results = scanner.scan_real_data()

    print("🎯 RÉSULTATS DU SCAN RÉEL:")
    print(
        f"📊 Total: {results['total_files']} fichiers, {results['total_size_mb']:.1f} MB"
    )

    for name, info in results["data_sources"].items():
        print(f"\n📁 {name.upper()}:")
        print(f"   Path: {info['path']}")
        print(f"   Type: {info['type']}")
        print(f"   Files: {info['total_files']}")
        print(f"   Size: {info['total_size_mb']:.1f} MB")
        print(f"   Status: {info['status']}")
