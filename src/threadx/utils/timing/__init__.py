"""
ThreadX Timing and Performance Monitoring - Phase 9
===================================================

Décorateurs et utilitaires pour mesure de performance et monitoring.
Compatible CPU/GPU avec synchronisation device-appropriate.

Features:
- @measure_throughput pour calcul de débit
- @track_memory pour monitoring RAM/VRAM
- Support GPU avec synchronisation CUDA
- Logs structurés avec métriques

Author: ThreadX Framework  
Version: Phase 9 - Timing Utils
"""

import time
import functools
import logging
from typing import Any, Callable, Dict, Optional
import threading
import gc

from threadx.utils.log import get_logger

logger = get_logger(__name__)

# Thread-local storage pour éviter conflicts
_local = threading.local()


def measure_throughput(name: Optional[str] = None):
    """
    Décorateur pour mesurer le débit d'une opération.
    
    Calcule éléments/seconde et log automatiquement les performances.
    Support GPU avec synchronisation CUDA si disponible.
    
    Args:
        name: Nom custom pour les logs (sinon nom fonction)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            operation_name = name or func.__name__
            
            # Démarrage timing
            start_time = time.perf_counter()
            
            # Garbage collection avant mesure
            gc.collect()
            
            try:
                # Exécution
                result = func(*args, **kwargs)
                
                # Synchronisation GPU si nécessaire
                _sync_gpu_if_needed()
                
                # Fin timing
                end_time = time.perf_counter()
                duration = end_time - start_time
                
                # Calcul throughput (estimation basique)
                throughput = _estimate_throughput(result, duration)
                
                # Logging
                if throughput:
                    logger.debug(f"⚡ {operation_name}: {duration:.3f}s, {throughput:.1f} items/sec")
                    
                    # Warning si throughput faible
                    if throughput < 1000:
                        logger.warning(f"⚠️ {operation_name}: throughput faible ({throughput:.1f} items/sec)")
                else:
                    logger.debug(f"⚡ {operation_name}: {duration:.3f}s")
                
                return result
                
            except Exception as e:
                logger.error(f"❌ {operation_name}: erreur pendant mesure: {e}")
                raise
        
        return wrapper
    return decorator


def track_memory(name: Optional[str] = None):
    """
    Décorateur pour tracking de mémoire (RAM/VRAM).
    
    Mesure la consommation mémoire avant/après et log peak usage.
    Support GPU avec monitoring VRAM si CuPy disponible.
    
    Args:
        name: Nom custom pour les logs (sinon nom fonction)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            operation_name = name or func.__name__
            
            # Mémoire avant
            memory_before = _get_memory_usage()
            
            try:
                result = func(*args, **kwargs)
                
                # Mémoire après  
                memory_after = _get_memory_usage()
                
                # Calcul consommation
                ram_delta = memory_after.get("ram", 0) - memory_before.get("ram", 0)
                vram_delta = memory_after.get("vram", 0) - memory_before.get("vram", 0)
                
                # Logging
                if vram_delta > 0:
                    logger.debug(f"💾 {operation_name}: RAM +{ram_delta:.1f}MB, VRAM +{vram_delta:.1f}MB")
                else:
                    logger.debug(f"💾 {operation_name}: RAM +{ram_delta:.1f}MB")
                
                # Warning si consommation élevée
                if ram_delta > 1000:  # > 1GB
                    logger.warning(f"⚠️ {operation_name}: consommation RAM élevée ({ram_delta:.1f}MB)")
                
                return result
                
            except Exception as e:
                logger.error(f"❌ {operation_name}: erreur pendant tracking mémoire: {e}")
                raise
        
        return wrapper
    return decorator


def _sync_gpu_if_needed():
    """Synchronise GPU si CuPy est utilisé."""
    try:
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
    except ImportError:
        pass  # Pas de GPU, pas de sync nécessaire
    except Exception:
        pass  # Erreur GPU, ignore


def _estimate_throughput(result: Any, duration: float) -> Optional[float]:
    """
    Estime le throughput basé sur le résultat.
    
    Args:
        result: Résultat de la fonction
        duration: Durée d'exécution en secondes
        
    Returns:
        Throughput en items/sec ou None si non calculable
    """
    if duration <= 0:
        return None
    
    items_count = None
    
    # Estimation basée sur le type de résultat
    if hasattr(result, '__len__'):
        items_count = len(result)
    elif hasattr(result, 'size'):
        items_count = getattr(result, 'size', None)
    elif hasattr(result, 'shape'):
        shape = getattr(result, 'shape', None)
        if shape:
            items_count = 1
            for dim in shape:
                items_count *= dim
    
    if items_count and items_count > 0:
        return items_count / duration
    
    return None


def _get_memory_usage() -> Dict[str, float]:
    """
    Retourne l'usage mémoire actuel (RAM/VRAM).
    
    Returns:
        Dict {"ram": MB, "vram": MB}
    """
    memory_info = {"ram": 0.0, "vram": 0.0}
    
    # RAM usage (approximation)
    try:
        import psutil
        process = psutil.Process()
        memory_info["ram"] = process.memory_info().rss / (1024 * 1024)  # MB
    except ImportError:
        pass
    
    # VRAM usage (CuPy)
    try:
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        memory_info["vram"] = mempool.used_bytes() / (1024 * 1024)  # MB
    except ImportError:
        pass
    except Exception:
        pass
    
    return memory_info


# Fonctions utilitaires standalone
def time_operation(operation: Callable, *args, **kwargs) -> tuple[Any, float]:
    """
    Time une opération et retourne (résultat, durée).
    
    Args:
        operation: Function à timer
        *args, **kwargs: Arguments pour la fonction
        
    Returns:
        (result, duration_seconds)
    """
    start_time = time.perf_counter()
    
    try:
        result = operation(*args, **kwargs)
        _sync_gpu_if_needed()
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        return result, duration
        
    except Exception as e:
        end_time = time.perf_counter()
        duration = end_time - start_time
        logger.error(f"❌ Operation failed after {duration:.3f}s: {e}")
        raise


def benchmark_operation(
    operation: Callable,
    runs: int = 3,
    warmup: int = 1,
    *args,
    **kwargs
) -> Dict[str, float]:
    """
    Benchmark une opération avec plusieurs runs.
    
    Args:
        operation: Function à benchmarker
        runs: Nombre de runs de mesure
        warmup: Nombre de runs de warmup
        *args, **kwargs: Arguments pour la fonction
        
    Returns:
        Dict avec statistiques (mean, std, min, max)
    """
    times = []
    
    # Warmup runs
    for _ in range(warmup):
        try:
            _, _ = time_operation(operation, *args, **kwargs)
        except Exception:
            pass
    
    # Mesure runs
    for _ in range(runs):
        try:
            _, duration = time_operation(operation, *args, **kwargs)
            times.append(duration)
        except Exception as e:
            logger.warning(f"⚠️ Benchmark run failed: {e}")
            continue
    
    if not times:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "runs": 0}
    
    import numpy as np
    return {
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
        "runs": len(times)
    }