"""
ThreadX Configuration Settings - Phase 1 (Version Simplifiée)
Dataclass Settings pour remplacer les variables d'environnement TradXPro.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional


@dataclass(frozen=False)
class Settings:
    """
    Configuration centralisée ThreadX - Phase 1.
    Remplace toutes les variables d'environnement de TradXPro.
    
    Version simplifiée focalisée sur les besoins essentiels:
    - Chemins de données (remplace TRADX_DATA_ROOT)
    - Configuration GPU (remplace TRADX_USE_GPU)
    - Timeframes supportés
    - Logging basique
    """
    
    # === CHEMINS (remplace env vars TradXPro) ===
    DATA_ROOT: Path = Path("./data")
    INDICATORS_ROOT: Path = Path("./data/indicators") 
    LOGS_DIR: Path = Path("./logs")
    
    # === GPU (remplace TRADX_USE_GPU, etc.) ===
    GPU_DEVICES: Optional[List[str]] = None
    GPU_LOAD_BALANCE: Optional[Dict[str, float]] = None
    
    # === PERFORMANCE ===
    TARGET_TASKS_PER_MIN: int = 2500
    
    # === TIMEFRAMES ===
    SUPPORTED_TIMEFRAMES: Tuple[str, ...] = ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d")
    
    # === LOGGING ===
    LOG_LEVEL: str = "INFO"
    
    # === COMPATIBILITÉ TESTS ===
    RAW_JSON: Optional[str] = None
    PROCESSED: Optional[str] = None
    ENABLE_GPU: bool = True
    SUPPORTED_TF: Tuple[str, ...] = ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d")
    LOAD_BALANCE: Optional[Dict[str, float]] = None
    BASE_CURRENCY: str = "USD"
    INITIAL_CAPITAL: float = 10000.0
    READ_ONLY_DATA: bool = True
    DEFAULT_SIMULATIONS: int = 10000
    PRIMARY_UI: str = "tkinter"
    
    # === SÉCURITÉ ===
    ALLOW_ABSOLUTE_PATHS: bool = False
    
    def __post_init__(self):
        """Initialisation des valeurs par défaut."""
        if self.GPU_DEVICES is None:
            self.GPU_DEVICES = ["5090", "2060"]
        if self.GPU_LOAD_BALANCE is None:
            self.GPU_LOAD_BALANCE = {"5090": 0.75, "2060": 0.25}
        if self.LOAD_BALANCE is None:
            self.LOAD_BALANCE = self.GPU_LOAD_BALANCE