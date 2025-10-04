"""
ThreadX Data Registry Module - Phase 2
Scanner rapide des données disponibles avec checksums et inventory.

Remplace et améliore:
- TradXPro analyze_crypto_data_availability (lent, lecture complète)
- Checksums manuels dispersés

Nouveautés:
- Scan O(n) sur nombre fichiers (pas de lecture Parquet)
- Checksums streamés avec algo configurable
- Convention processed/{symbol}/{timeframe}.parquet
- Intégration config ThreadX paths
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import time

# Config ThreadX
try:
    from threadx.config import load_settings
    _settings = load_settings()
    DATA_ROOT = _settings.DATA_ROOT if _settings else Path("./data")
except ImportError:
    DATA_ROOT = Path("./data")

logger = logging.getLogger(__name__)

__all__ = [
    "dataset_exists",
    "scan_symbols", 
    "scan_timeframes",
    "quick_inventory",
    "file_checksum",
    "RegistryError"
]

# ============================================================================
# EXCEPTIONS REGISTRY
# ============================================================================

class RegistryError(Exception):
    """Erreur lors des opérations registry."""
    pass


# ============================================================================
# CHEMINS ET CONVENTIONS
# ============================================================================

def _get_data_root(root: Optional[Union[Path, str]] = None) -> Path:
    """Retourne racine data avec fallback config ThreadX."""
    if root is not None:
        return Path(root)
    
    # Utilise config ThreadX ou fallback
    return Path(DATA_ROOT)


def _get_processed_root(root: Optional[Union[Path, str]] = None) -> Path:
    """Retourne dossier processed/ depuis racine data."""
    data_root = _get_data_root(root)
    return data_root / "processed"


def _build_dataset_path(symbol: str, timeframe: str, root: Optional[Union[Path, str]] = None) -> Path:
    """
    Construit chemin dataset selon convention ThreadX.
    
    Convention: processed/{symbol}/{timeframe}.parquet
    
    Args:
        symbol: Symbole trading (ex: BTCUSDC)
        timeframe: Timeframe (ex: 1m, 15m, 1h)
        root: Racine data optionnelle
        
    Returns:
        Path vers fichier dataset
    """
    processed_root = _get_processed_root(root)
    return processed_root / symbol / f"{timeframe}.parquet"


# ============================================================================
# EXISTENCE ET SCAN RAPIDES
# ============================================================================

def dataset_exists(symbol: str, timeframe: str, root: Optional[Union[Path, str]] = None) -> bool:
    """
    Vérifie existence d'un dataset OHLCV.
    
    Critères:
    - Fichier existe
    - Taille > 0 octets (évite fichiers corrompus)
    
    Args:
        symbol: Symbole trading 
        timeframe: Timeframe
        root: Racine data optionnelle
        
    Returns:
        True si dataset valide existe
    """
    dataset_path = _build_dataset_path(symbol, timeframe, root)
    
    try:
        # Vérification existence + taille
        return dataset_path.exists() and dataset_path.stat().st_size > 0
    except (OSError, PermissionError) as e:
        logger.debug(f"Erreur accès dataset {dataset_path}: {e}")
        return False


def scan_symbols(root: Optional[Union[Path, str]] = None) -> List[str]:
    """
    Scan rapide des symboles disponibles.
    
    Liste les dossiers sous processed/ représentant des symboles.
    
    Args:
        root: Racine data optionnelle
        
    Returns:
        Liste symboles triés (ex: ["ADAUSDC", "BTCUSDC", "ETHUSDC"])
    """
    processed_root = _get_processed_root(root)
    
    if not processed_root.exists():
        logger.debug(f"Dossier processed inexistant: {processed_root}")
        return []
    
    symbols = []
    
    try:
        for item in processed_root.iterdir():
            if item.is_dir():
                # Nom dossier = symbole
                symbols.append(item.name)
                
        logger.debug(f"Scan symboles: {len(symbols)} trouvés sous {processed_root}")
        return sorted(symbols)
        
    except (OSError, PermissionError) as e:
        logger.error(f"Erreur scan symboles: {e}")
        raise RegistryError(f"Impossible scanner symboles: {e}")


def scan_timeframes(symbol: str, root: Optional[Union[Path, str]] = None) -> List[str]:
    """
    Scan timeframes disponibles pour un symbole.
    
    Liste les fichiers .parquet sous processed/{symbol}/.
    
    Args:
        symbol: Symbole à scanner
        root: Racine data optionnelle
        
    Returns:
        Liste timeframes triés (ex: ["1m", "15m", "1h", "4h"])
    """
    processed_root = _get_processed_root(root)
    symbol_dir = processed_root / symbol
    
    if not symbol_dir.exists() or not symbol_dir.is_dir():
        logger.debug(f"Dossier symbole inexistant: {symbol_dir}")
        return []
    
    timeframes = []
    
    try:
        for file_path in symbol_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() == ".parquet":
                # Nom fichier sans extension = timeframe
                timeframe = file_path.stem
                timeframes.append(timeframe)
                
        logger.debug(f"Scan timeframes {symbol}: {len(timeframes)} trouvés")
        return sorted(timeframes)
        
    except (OSError, PermissionError) as e:
        logger.error(f"Erreur scan timeframes {symbol}: {e}")
        raise RegistryError(f"Impossible scanner timeframes {symbol}: {e}")


def quick_inventory(root: Optional[Union[Path, str]] = None) -> Dict[str, List[str]]:
    """
    Inventaire rapide complet: {symbol: [timeframes...]}.
    
    Remplace TradXPro analyze_crypto_data_availability avec performance O(n).
    Aucune lecture de fichier Parquet - scan filesystem seulement.
    
    Args:
        root: Racine data optionnelle
        
    Returns:
        Dict {symbol: [timeframes...]} trié
        
    Raises:
        RegistryError: Erreur scan filesystem
    """
    start_time = time.perf_counter()
    
    try:
        symbols = scan_symbols(root)
        
        if not symbols:
            logger.warning("Aucun symbole trouvé - inventaire vide")
            return {}
        
        inventory = {}
        
        for symbol in symbols:
            timeframes = scan_timeframes(symbol, root)
            if timeframes:  # Seulement symboles avec données
                inventory[symbol] = timeframes
        
        elapsed = time.perf_counter() - start_time
        total_datasets = sum(len(tfs) for tfs in inventory.values())
        
        logger.info(
            f"Inventaire rapide: {len(inventory)} symboles, "
            f"{total_datasets} datasets en {elapsed:.3f}s"
        )
        
        return inventory
        
    except Exception as e:
        logger.error(f"Échec inventaire rapide: {e}")
        raise RegistryError(f"Inventaire impossible: {e}")


# ============================================================================
# CHECKSUMS STREAMÉS
# ============================================================================

def file_checksum(
    path: Union[Path, str], 
    algo: str = "md5", 
    chunk_size: int = 1 << 20  # 1MB chunks
) -> str:
    """
    Calcul checksum streamé (évite chargement RAM complet).
    
    Inspiré pattern TradXPro mais avec gestion d'erreurs robuste.
    
    Args:
        path: Chemin fichier à hasher
        algo: Algorithme hash ("md5", "sha1", "sha256", "blake2b")
        chunk_size: Taille chunks lecture (défaut 1MB)
        
    Returns:
        Hash hexadécimal (ex: "a1b2c3d4e5f6...")
        
    Raises:
        RegistryError: Fichier introuvable ou erreur lecture
    """
    path_obj = Path(path)
    
    if not path_obj.exists():
        raise RegistryError(f"Fichier introuvable pour checksum: {path_obj}")
    
    if not path_obj.is_file():
        raise RegistryError(f"Chemin n'est pas un fichier: {path_obj}")
    
    # Validation algorithme
    try:
        hasher = hashlib.new(algo)
    except ValueError as e:
        raise RegistryError(f"Algorithme hash invalide '{algo}': {e}")
    
    # Lecture streamée
    try:
        with open(path_obj, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        
        checksum = hasher.hexdigest()
        logger.debug(f"Checksum {algo} {path_obj.name}: {checksum[:16]}...")
        
        return checksum
        
    except (OSError, PermissionError) as e:
        raise RegistryError(f"Erreur lecture fichier {path_obj}: {e}")


# ============================================================================
# UTILITAIRES BONUS
# ============================================================================

def dataset_info(symbol: str, timeframe: str, root: Optional[Union[Path, str]] = None) -> Optional[Dict]:
    """
    Informations détaillées sur un dataset (taille, checksum, timestamps).
    
    Args:
        symbol: Symbole
        timeframe: Timeframe  
        root: Racine data
        
    Returns:
        Dict avec infos ou None si dataset inexistant
    """
    if not dataset_exists(symbol, timeframe, root):
        return None
    
    dataset_path = _build_dataset_path(symbol, timeframe, root)
    
    try:
        stat_info = dataset_path.stat()
        
        info = {
            "symbol": symbol,
            "timeframe": timeframe,
            "path": str(dataset_path),
            "size_bytes": stat_info.st_size,
            "size_mb": round(stat_info.st_size / (1024 * 1024), 2),
            "modified_time": stat_info.st_mtime,
            "checksum_md5": file_checksum(dataset_path, "md5")
        }
        
        return info
        
    except Exception as e:
        logger.error(f"Erreur info dataset {symbol}/{timeframe}: {e}")
        return None


def cleanup_empty_datasets(root: Optional[Union[Path, str]] = None, dry_run: bool = True) -> Dict[str, int]:
    """
    Nettoyage fichiers vides ou corrompus.
    
    Args:
        root: Racine data
        dry_run: Si True, liste seulement (pas de suppression)
        
    Returns:
        Stats {"found": n, "removed": n}
    """
    processed_root = _get_processed_root(root)
    
    if not processed_root.exists():
        return {"found": 0, "removed": 0}
    
    empty_files = []
    
    # Scan récursif fichiers .parquet
    for parquet_file in processed_root.rglob("*.parquet"):
        if parquet_file.is_file():
            try:
                if parquet_file.stat().st_size == 0:
                    empty_files.append(parquet_file)
            except OSError:
                continue
    
    stats = {"found": len(empty_files), "removed": 0}
    
    if empty_files:
        logger.warning(f"Fichiers vides détectés: {len(empty_files)}")
        
        if not dry_run:
            for empty_file in empty_files:
                try:
                    empty_file.unlink()
                    stats["removed"] += 1
                    logger.debug(f"Supprimé: {empty_file}")
                except OSError as e:
                    logger.error(f"Échec suppression {empty_file}: {e}")
    
    return stats
