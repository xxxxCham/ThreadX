"""
ThreadX Data Package - Phase 2
Module unifié pour I/O, resampling, registry et génération synthétique.
"""

from .io import (
    read_frame,
    write_frame, 
    normalize_ohlcv,
    OHLCV_SCHEMA,
    DataNotFoundError,
    FileValidationError, 
    SchemaMismatchError
)

from .resample import (
    resample_from_1m,
    resample_batch,
    TimeframeError,
    GapFillingError
)

from .registry import (
    dataset_exists,
    scan_symbols,
    scan_timeframes, 
    quick_inventory,
    file_checksum,
    RegistryError
)

from .synth import (
    make_synth_ohlcv,
    make_trending_ohlcv,
    make_volatile_ohlcv,
    SynthDataError
)

__all__ = [
    # I/O
    "read_frame", "write_frame", "normalize_ohlcv", "OHLCV_SCHEMA",
    "DataNotFoundError", "FileValidationError", "SchemaMismatchError",
    
    # Resampling  
    "resample_from_1m", "resample_batch",
    "TimeframeError", "GapFillingError",
    
    # Registry
    "dataset_exists", "scan_symbols", "scan_timeframes", "quick_inventory", "file_checksum",
    "RegistryError",
    
    # Synthétique
    "make_synth_ohlcv", "make_trending_ohlcv", "make_volatile_ohlcv", 
    "SynthDataError"
]