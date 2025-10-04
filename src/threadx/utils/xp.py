"""
ThreadX Utils Module - Phase 9
Device-Agnostic Computing Helpers (NumPy/CuPy).

Provides seamless CPU/GPU array operations with:
- Automatic backend selection (NumPy vs CuPy)
- Device detection and capability checking
- Memory transfer utilities (host ↔ device)
- Graceful fallback when GPU unavailable
- Integration with tools/check_env.py for hardware detection

Windows-first design, no environment variables, Settings/TOML configuration.
Designed to be imported by existing compute modules without API changes.
"""

import logging
from typing import Any, Optional, Union, Tuple, Dict, Callable
import numpy as np

# Import ThreadX logger - fallback to standard logging if not available
try:
    from threadx.utils.log import get_logger
except ImportError:
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)

# Import ThreadX Settings - fallback if not available
try:
    from threadx.config.settings import load_settings
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False

# Import check_env if available for hardware detection
try:
    from tools.check_env import check_gpu_availability, get_gpu_info
    CHECK_ENV_AVAILABLE = True
except ImportError:
    CHECK_ENV_AVAILABLE = False

# Try to import CuPy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


logger = get_logger(__name__)

# Global state for backend selection
_gpu_enabled = None
_gpu_devices_available = None


def _check_gpu_settings() -> bool:
    """Check if GPU is enabled in ThreadX settings."""
    if not SETTINGS_AVAILABLE:
        return True  # Default to allowing GPU if no settings
        
    try:
        settings = load_settings()
        return settings.ENABLE_GPU
    except Exception as e:
        logger.debug(f"Failed to load GPU settings: {e}")
        return True  # Default to allowing GPU


def _detect_gpu_capabilities() -> Dict[str, Any]:
    """
    Detect available GPU capabilities.
    
    Uses tools/check_env.py if available, otherwise basic CuPy detection.
    
    Returns
    -------
    dict
        GPU capability information.
    """
    capabilities = {
        'available': False,
        'devices': [],
        'memory_total': 0,
        'cuda_version': None,
        'driver_version': None
    }
    
    if not CUPY_AVAILABLE:
        logger.debug("CuPy not available - GPU support disabled")
        return capabilities
        
    if CHECK_ENV_AVAILABLE:
        try:
            # Use tools/check_env.py for detailed detection
            gpu_available = check_gpu_availability()
            if gpu_available:
                gpu_info = get_gpu_info()
                capabilities.update({
                    'available': True,
                    'devices': gpu_info.get('devices', []),
                    'memory_total': gpu_info.get('total_memory', 0),
                    'cuda_version': gpu_info.get('cuda_version'),
                    'driver_version': gpu_info.get('driver_version')
                })
                logger.info(f"GPU detected via check_env: {len(capabilities['devices'])} device(s)")
            else:
                logger.info("No GPU detected via check_env")
        except Exception as e:
            logger.debug(f"GPU detection via check_env failed: {e}")
            # Fall back to basic CuPy detection below
    
    # Fallback to basic CuPy detection if check_env unavailable
    if not capabilities['available']:
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count > 0:
                capabilities['available'] = True
                capabilities['devices'] = list(range(device_count))
                
                # Get basic info for first device
                with cp.cuda.Device(0):
                    meminfo = cp.cuda.runtime.memGetInfo()
                    free_mem, total_mem = meminfo
                    capabilities['memory_total'] = total_mem
                    
                logger.info(f"GPU detected via CuPy: {device_count} device(s)")
            else:
                logger.info("No GPU devices found via CuPy")
        except Exception as e:
            logger.debug(f"Basic GPU detection failed: {e}")
            
    return capabilities


def _initialize_gpu_state() -> None:
    """Initialize global GPU state."""
    global _gpu_enabled, _gpu_devices_available
    
    if _gpu_enabled is not None:
        return  # Already initialized
        
    # Check settings first
    gpu_allowed_by_settings = _check_gpu_settings()
    
    if not gpu_allowed_by_settings:
        logger.info("GPU disabled by ThreadX settings")
        _gpu_enabled = False
        _gpu_devices_available = []
        return
        
    if not CUPY_AVAILABLE:
        logger.info("GPU support unavailable: CuPy not installed")
        _gpu_enabled = False
        _gpu_devices_available = []
        return
        
    # Detect hardware capabilities
    capabilities = _detect_gpu_capabilities()
    
    if capabilities['available']:
        _gpu_enabled = True
        _gpu_devices_available = capabilities['devices']
        logger.info(
            f"GPU support enabled: {len(_gpu_devices_available)} device(s) available, "
            f"total memory: {capabilities['memory_total'] / (1024**3):.1f}GB"
        )
        
        # Log device details if available
        if capabilities.get('cuda_version'):
            logger.info(f"CUDA version: {capabilities['cuda_version']}")
        if capabilities.get('driver_version'):
            logger.info(f"Driver version: {capabilities['driver_version']}")
    else:
        _gpu_enabled = False
        _gpu_devices_available = []
        logger.info("GPU support disabled: no compatible devices found")


def gpu_available() -> bool:
    """
    Check if GPU acceleration is available and enabled.
    
    Considers both hardware availability and ThreadX settings.
    
    Returns
    -------
    bool
        True if GPU acceleration is available and enabled.
        
    Examples
    --------
    >>> if gpu_available():
    ...     print("GPU acceleration is available")
    ... else:
    ...     print("Using CPU fallback")
    """
    _initialize_gpu_state()
    return _gpu_enabled


def get_gpu_devices() -> list:
    """
    Get list of available GPU devices.
    
    Returns
    -------
    list
        List of available GPU device IDs.
    """
    _initialize_gpu_state()
    return _gpu_devices_available.copy()


def xp() -> Any:
    """
    Get the appropriate array library (NumPy or CuPy).
    
    Returns CuPy if GPU is available and enabled, otherwise NumPy.
    This is the main entry point for device-agnostic array operations.
    
    Returns
    -------
    module
        Either numpy or cupy module.
        
    Examples
    --------
    >>> import threadx.utils.xp as txp
    >>> xp = txp.xp()
    >>> 
    >>> # Works with both NumPy and CuPy
    >>> data = xp.array([1, 2, 3, 4, 5])
    >>> result = xp.mean(data)
    >>> 
    >>> # Convert back to host if needed
    >>> host_result = txp.to_host(result)
    
    Notes
    -----
    - Automatically selects best available backend
    - Consistent API between NumPy and CuPy
    - Performance: CuPy operations stay on GPU until explicitly moved
    - Memory: Be aware of GPU memory limitations for large arrays
    """
    _initialize_gpu_state()
    
    if _gpu_enabled:
        return cp
    else:
        return np


def to_device(array: Any, device_id: Optional[int] = None) -> Any:
    """
    Move array to GPU device (if GPU available).
    
    No-op if GPU unavailable or array already on correct device.
    
    Parameters
    ----------
    array : array-like
        Input array (NumPy array, CuPy array, or array-like).
    device_id : int, optional
        Target GPU device ID. If None, uses default device.
        
    Returns
    -------
    array-like
        Array on GPU device (if available) or unchanged array.
        
    Examples
    --------
    >>> import numpy as np
    >>> cpu_array = np.array([1, 2, 3])
    >>> gpu_array = to_device(cpu_array)  # Move to GPU if available
    >>> 
    >>> # Works with existing GPU arrays too
    >>> gpu_array2 = to_device(gpu_array)  # No-op if already on device
    """
    if not gpu_available():
        return array  # No-op if GPU unavailable
        
    try:
        # Handle different input types
        if hasattr(array, '__array__') or isinstance(array, (list, tuple)):
            # Convert to CuPy array
            if device_id is not None:
                with cp.cuda.Device(device_id):
                    return cp.asarray(array)
            else:
                return cp.asarray(array)
        else:
            # Already a CuPy array or compatible
            if device_id is not None and hasattr(array, 'device'):
                if array.device.id != device_id:
                    with cp.cuda.Device(device_id):
                        return cp.asarray(array)
            return array
            
    except Exception as e:
        logger.warning(f"Failed to move array to GPU device: {e}")
        return array  # Return original array on failure


def to_host(array: Any) -> Any:
    """
    Move array to CPU host memory.
    
    No-op if array is already on CPU or GPU unavailable.
    
    Parameters
    ----------
    array : array-like
        Input array (NumPy or CuPy array).
        
    Returns
    -------
    numpy.ndarray
        Array in CPU host memory.
        
    Examples
    --------
    >>> gpu_array = xp().array([1, 2, 3])  # Might be on GPU
    >>> cpu_array = to_host(gpu_array)     # Ensure it's on CPU
    >>> print(type(cpu_array))             # numpy.ndarray
    """
    if not gpu_available():
        return array  # Already on host if no GPU
        
    try:
        if hasattr(array, 'get'):
            # CuPy array - move to host
            return array.get()
        elif hasattr(array, '__array__'):
            # Already NumPy array or compatible
            return np.asarray(array)
        else:
            # Scalar or other type
            return array
            
    except Exception as e:
        logger.warning(f"Failed to move array to host: {e}")
        return array  # Return original array on failure


def device_synchronize() -> None:
    """
    Synchronize GPU device (wait for all operations to complete).
    
    No-op if GPU unavailable. Use before timing measurements.
    
    Examples
    --------
    >>> xp = xp()
    >>> result = xp.dot(a, b)  # Async GPU operation
    >>> device_synchronize()   # Wait for completion
    >>> # Now safe to measure timing
    """
    if not gpu_available():
        return
        
    try:
        cp.cuda.Stream.null.synchronize()
    except Exception as e:
        logger.debug(f"Device synchronization failed: {e}")


def get_array_info(array: Any) -> Dict[str, Any]:
    """
    Get information about an array (device, memory, etc.).
    
    Parameters
    ----------
    array : array-like
        Input array.
        
    Returns
    -------
    dict
        Array information including device location, shape, dtype, memory usage.
        
    Examples
    --------
    >>> array = xp().array([1, 2, 3])
    >>> info = get_array_info(array)
    >>> print(f"Device: {info['device']}, Memory: {info['memory_mb']:.1f}MB")
    """
    info = {
        'device': 'cpu',
        'shape': getattr(array, 'shape', None),
        'dtype': getattr(array, 'dtype', None),
        'memory_mb': 0.0,
        'is_contiguous': getattr(array, 'flags', {}).get('C_CONTIGUOUS', False)
    }
    
    try:
        # Determine device
        if hasattr(array, 'device'):
            info['device'] = f"gpu:{array.device.id}"
        elif hasattr(array, '__array__'):
            info['device'] = 'cpu'
            
        # Calculate memory usage
        if hasattr(array, 'nbytes'):
            info['memory_mb'] = array.nbytes / (1024 * 1024)
        elif hasattr(array, 'size') and hasattr(array, 'dtype'):
            info['memory_mb'] = array.size * array.dtype.itemsize / (1024 * 1024)
            
    except Exception as e:
        logger.debug(f"Failed to get complete array info: {e}")
        
    return info


def ensure_array_type(array: Any, dtype: Optional[Any] = None) -> Any:
    """
    Ensure array is the correct type for current backend.
    
    Converts to appropriate array type (NumPy or CuPy) and optionally
    changes dtype. Useful for ensuring compatibility before operations.
    
    Parameters
    ----------
    array : array-like
        Input array.
    dtype : numpy.dtype, optional
        Target data type.
        
    Returns
    -------
    array-like
        Array compatible with current backend.
        
    Examples
    --------
    >>> mixed_array = [1.0, 2.0, 3.0]  # Python list
    >>> proper_array = ensure_array_type(mixed_array, dtype=np.float32)
    >>> # Now guaranteed to be NumPy or CuPy array with float32 dtype
    """
    xp_module = xp()
    
    try:
        if dtype is not None:
            return xp_module.asarray(array, dtype=dtype)
        else:
            return xp_module.asarray(array)
    except Exception as e:
        logger.warning(f"Failed to ensure array type: {e}")
        return array


def memory_pool_info() -> Optional[Dict[str, Any]]:
    """
    Get GPU memory pool information.
    
    Returns memory usage statistics for GPU memory pools.
    Returns None if GPU unavailable.
    
    Returns
    -------
    dict or None
        Memory pool statistics or None if GPU unavailable.
        
    Examples
    --------
    >>> info = memory_pool_info()
    >>> if info:
    ...     print(f"GPU memory used: {info['used_mb']:.1f}MB")
    """
    if not gpu_available():
        return None
        
    try:
        mempool = cp.get_default_memory_pool()
        
        return {
            'used_bytes': mempool.used_bytes(),
            'total_bytes': mempool.total_bytes(),
            'used_mb': mempool.used_bytes() / (1024 * 1024),
            'total_mb': mempool.total_bytes() / (1024 * 1024),
            'fragmentation_ratio': (
                mempool.total_bytes() - mempool.used_bytes()
            ) / max(mempool.total_bytes(), 1)
        }
        
    except Exception as e:
        logger.debug(f"Failed to get memory pool info: {e}")
        return None


def clear_memory_pool() -> bool:
    """
    Clear GPU memory pool to free unused memory.
    
    Useful for managing GPU memory in long-running processes.
    
    Returns
    -------
    bool
        True if memory pool was cleared, False if GPU unavailable.
        
    Examples
    --------
    >>> # After processing large arrays
    >>> if clear_memory_pool():
    ...     print("GPU memory pool cleared")
    """
    if not gpu_available():
        return False
        
    try:
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        logger.debug("GPU memory pool cleared")
        return True
        
    except Exception as e:
        logger.debug(f"Failed to clear memory pool: {e}")
        return False


# Convenience functions for common patterns
def asnumpy(array: Any) -> np.ndarray:
    """Ensure array is a NumPy array (alias for to_host)."""
    return np.asarray(to_host(array))


def ascupy(array: Any) -> Any:
    """
    Ensure array is a CuPy array (if GPU available).
    
    Returns NumPy array if GPU unavailable.
    """
    if gpu_available():
        return to_device(array)
    else:
        return np.asarray(array)


# Performance measurement helpers
def benchmark_operation(
    operation: Callable,
    *args,
    warmup_runs: int = 3,
    benchmark_runs: int = 10,
    **kwargs
) -> Dict[str, float]:
    """
    Benchmark an operation on current backend.
    
    Performs warmup runs followed by timed benchmark runs.
    Includes device synchronization for accurate GPU timing.
    
    Parameters
    ----------
    operation : callable
        Function to benchmark.
    *args
        Arguments for the operation.
    warmup_runs : int, default 3
        Number of warmup runs.
    benchmark_runs : int, default 10
        Number of benchmark runs for timing.
    **kwargs
        Keyword arguments for the operation.
        
    Returns
    -------
    dict
        Benchmark results including mean, std, min, max times in seconds.
        
    Examples
    --------
    >>> import threadx.utils.xp as txp
    >>> xp = txp.xp()
    >>> 
    >>> def matrix_mult(a, b):
    ...     return xp.dot(a, b)
    >>> 
    >>> a = xp.random.random((1000, 1000))
    >>> b = xp.random.random((1000, 1000))
    >>> 
    >>> results = benchmark_operation(matrix_mult, a, b)
    >>> print(f"Mean time: {results['mean_sec']:.4f}s")
    """
    import time
    
    # Warmup runs
    for _ in range(warmup_runs):
        try:
            _ = operation(*args, **kwargs)
            device_synchronize()
        except Exception as e:
            logger.warning(f"Warmup run failed: {e}")
            
    # Benchmark runs
    times = []
    for _ in range(benchmark_runs):
        try:
            start_time = time.perf_counter()
            result = operation(*args, **kwargs)
            device_synchronize()  # Ensure GPU operations complete
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
        except Exception as e:
            logger.warning(f"Benchmark run failed: {e}")
            continue
            
    if not times:
        return {
            'mean_sec': 0.0,
            'std_sec': 0.0,
            'min_sec': 0.0,
            'max_sec': 0.0,
            'runs_completed': 0
        }
        
    return {
        'mean_sec': np.mean(times),
        'std_sec': np.std(times),
        'min_sec': np.min(times),
        'max_sec': np.max(times),
        'runs_completed': len(times)
    }


# =============================================================================
# Device-Agnostic Array Interface (Phase 10 Extension)
# =============================================================================

_current_backend = "numpy"

def configure_backend(use_gpu: Optional[bool] = None) -> str:
    """
    Configure le backend de calcul (NumPy/CuPy).
    
    Args:
        use_gpu: Force GPU usage (None=auto-detect)
        
    Returns:
        Backend configuré ("numpy" ou "cupy")
    """
    global _current_backend
    
    if use_gpu is None:
        # Auto-détection basée sur GPU disponibles
        _initialize_gpu_state()
        use_gpu = _gpu_enabled and _gpu_devices_available and len(_gpu_devices_available) > 0
    
    if use_gpu and CUPY_AVAILABLE and _gpu_enabled:
        _current_backend = "cupy"
        logger.debug("✅ Backend CuPy configuré")
    else:
        _current_backend = "numpy"  
        logger.debug("✅ Backend NumPy configuré")
    
    return _current_backend


def get_backend() -> str:
    """Retourne le backend actuel."""
    return _current_backend


def is_gpu_backend() -> bool:
    """Vérifie si on utilise le backend GPU."""
    return _current_backend == "cupy"


# Interface unifiée - functions principales
def asarray(a, dtype=None):
    """Convertit en array device-appropriate."""
    if _current_backend == "cupy" and CUPY_AVAILABLE and cp is not None:
        return cp.asarray(a, dtype=dtype)
    else:
        return np.asarray(a, dtype=dtype)


def zeros(shape, dtype=np.float64):
    """Crée un array de zéros."""
    if _current_backend == "cupy" and CUPY_AVAILABLE and cp is not None:
        return cp.zeros(shape, dtype=dtype)
    else:
        return np.zeros(shape, dtype=dtype)


def ones(shape, dtype=np.float64):
    """Crée un array de uns."""
    if _current_backend == "cupy" and CUPY_AVAILABLE and cp is not None:
        return cp.ones(shape, dtype=dtype)
    else:
        return np.ones(shape, dtype=dtype)
def where(condition, x, y):
    """Sélection conditionnelle."""
    if _current_backend == "cupy" and CUPY_AVAILABLE and cp is not None:
        return cp.where(condition, x, y)
    else:
        return np.where(condition, x, y)


def sum(a, axis=None):
    """Somme d'array."""
    if _current_backend == "cupy" and CUPY_AVAILABLE and cp is not None:
        result = cp.sum(a, axis=axis)
        if axis is None and hasattr(result, 'get'):
            return float(result.get())
        return result
    else:
        return np.sum(a, axis=axis)


def percentile(a, q, axis=None):
    """Percentile d'array."""
    if _current_backend == "cupy" and CUPY_AVAILABLE and cp is not None:
        result = cp.percentile(a, q, axis=axis)
        if axis is None and hasattr(result, 'get'):
            return float(result.get())
        return result
    else:
        return np.percentile(a, q, axis=axis)


def to_cpu(arr):
    """Convertit array vers CPU (NumPy)."""
    if hasattr(arr, 'get'):
        return arr.get()  # CuPy → NumPy
    else:
        return np.asarray(arr)  # Déjà NumPy


def to_gpu(arr):
    """Convertit array vers GPU si disponible."""
    if _current_backend == "cupy" and CUPY_AVAILABLE and cp is not None:
        return cp.asarray(arr)
    else:
        return np.asarray(arr)