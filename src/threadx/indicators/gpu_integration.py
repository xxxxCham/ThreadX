"""
ThreadX Indicators GPU Integration - Phase 5
=============================================

Intégration de la distribution multi-GPU avec la couche d'indicateurs.

Permet d'accélérer les calculs d'indicateurs techniques (Bollinger Bands, ATR, etc.)
en utilisant automatiquement la répartition GPU/CPU optimale.

Usage:
    >>> # Calcul distribué d'indicateurs
    >>> from threadx.indicators import get_gpu_accelerated_bank
    >>> 
    >>> bank = get_gpu_accelerated_bank()
    >>> bb_upper, bb_middle, bb_lower = bank.bollinger_bands(
    ...     df, period=20, std_dev=2.0, use_gpu=True
    ... )
"""

import logging
from typing import Tuple, Optional, Union
import numpy as np
import pandas as pd

from threadx.utils.log import get_logger
from threadx.utils.gpu import get_default_manager, MultiGPUManager
from threadx.config.settings import S

logger = get_logger(__name__)


class GPUAcceleratedIndicatorBank:
    """
    Banque d'indicateurs avec accélération multi-GPU.
    
    Wraps les calculs d'indicateurs pour utiliser automatiquement
    la distribution multi-GPU quand disponible et bénéfique.
    """
    
    def __init__(self, gpu_manager: Optional[MultiGPUManager] = None):
        """
        Initialise la banque d'indicateurs GPU.
        
        Args:
            gpu_manager: Gestionnaire multi-GPU optionnel
                        Si None, utilise le gestionnaire par défaut
        """
        self.gpu_manager = gpu_manager or get_default_manager()
        self.min_samples_for_gpu = 1000  # Seuil pour utilisation GPU
        
        logger.info(f"Banque indicateurs GPU initialisée: "
                   f"{len(self.gpu_manager._gpu_devices)} GPU(s)")
    
    def _should_use_gpu(self, data_size: int, force_gpu: bool = False) -> bool:
        """
        Détermine si le GPU doit être utilisé pour ce calcul.
        
        Args:
            data_size: Taille des données
            force_gpu: Force l'utilisation GPU même si pas optimal
            
        Returns:
            True si GPU recommandé
        """
        if force_gpu:
            return len(self.gpu_manager._gpu_devices) > 0
        
        # Critères automatiques
        has_gpu = len(self.gpu_manager._gpu_devices) > 0
        sufficient_data = data_size >= self.min_samples_for_gpu
        
        return has_gpu and sufficient_data
    
    def _should_use_gpu_dynamic(self, data_size: int, *, profile_key: str, default_min: int = 1000) -> bool:
        """
        Détermine l'utilisation GPU avec seuil dynamique basé sur profil historique.
        
        Args:
            data_size: Taille des données
            profile_key: Clé de profil (indicateur+params)
            default_min: Seuil par défaut si pas de profil
            
        Returns:
            True si GPU recommandé selon profil
        """
        has_gpu = len(self.gpu_manager._gpu_devices) > 0
        if not has_gpu:
            return False
        
        # Chargement du profil GPU
        dynamic_threshold = self._load_gpu_threshold(profile_key, default_min)
        
        return data_size >= dynamic_threshold
    
    def _load_gpu_threshold(self, profile_key: str, default_min: int) -> int:
        """Charge le seuil GPU depuis le profil historique."""
        profile_path = Path("artifacts/profiles/gpu_thresholds.json")
        
        if profile_path.exists():
            try:
                with profile_path.open('r') as f:
                    profiles = json.load(f)
                
                if profile_key in profiles:
                    threshold = profiles[profile_key].get('optimal_threshold', default_min)
                    logger.debug(f"Seuil GPU chargé pour {profile_key}: {threshold}")
                    return threshold
            
            except Exception as e:
                logger.warning(f"Erreur chargement profil GPU {profile_key}: {e}")
        
        # Premier appel : profiling automatique
        logger.info(f"Profil GPU manquant pour {profile_key}, lancement du probing...")
        optimal_threshold = self._probe_cpu_gpu_once(profile_key, default_min)
        
        return optimal_threshold
    
    def _probe_cpu_gpu_once(self, profile_key: str, base_threshold: int) -> int:
        """
        Probe une seule fois le seuil optimal CPU vs GPU pour cette clé.
        
        Args:
            profile_key: Clé de profil
            base_threshold: Seuil de base
            
        Returns:
            Seuil optimal déterminé
        """
        logger.info(f"Probing GPU optimal pour {profile_key}")
        
        # Tailles de test
        test_sizes = [500, 1000, 2000, 5000, 10000]
        results = {}
        
        # Test dummy data
        for size in test_sizes:
            dummy_data = np.random.randn(size)
            
            # Timing CPU
            start_cpu = time.time()
            _ = self._dummy_compute_cpu(dummy_data)
            cpu_time = time.time() - start_cpu
            
            # Timing GPU (si disponible)
            gpu_time = float('inf')
            if len(self.gpu_manager._gpu_devices) > 0:
                try:
                    start_gpu = time.time()
                    _ = self._dummy_compute_gpu(dummy_data)
                    gpu_time = time.time() - start_gpu
                except Exception as e:
                    logger.debug(f"GPU compute échoué pour size={size}: {e}")
            
            # Ratio de performance
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0.0
            results[size] = {
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'speedup': speedup
            }
            
            logger.debug(f"Size {size}: CPU={cpu_time:.4f}s, GPU={gpu_time:.4f}s, "
                        f"speedup={speedup:.2f}x")
        
        # Détermination du seuil optimal
        optimal_threshold = base_threshold
        
        for size, metrics in results.items():
            if metrics['speedup'] >= 1.5:  # GPU 50% plus rapide
                optimal_threshold = size
                break
        
        # Sauvegarde du profil
        self._save_gpu_threshold_profile(profile_key, optimal_threshold, results)
        
        logger.info(f"Seuil optimal déterminé pour {profile_key}: {optimal_threshold}")
        
        return optimal_threshold
    
    def _dummy_compute_cpu(self, data: np.ndarray) -> np.ndarray:
        """Calcul dummy CPU pour probing."""
        return np.convolve(data, np.ones(20)/20, mode='same')
    
    def _dummy_compute_gpu(self, data: np.ndarray) -> np.ndarray:
        """Calcul dummy GPU pour probing."""
        # Utilisation du MultiGPUManager pour calcul distribué
        def compute_fn(x):
            return np.convolve(x, np.ones(20)/20, mode='same')
        
        return self.gpu_manager.distribute_workload(data, compute_fn)
    
    def _save_gpu_threshold_profile(self, profile_key: str, threshold: int, benchmark_results: Dict):
        """Sauvegarde le profil de seuil GPU."""
        profile_path = Path("artifacts/profiles/gpu_thresholds.json")
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Chargement des profils existants
        profiles = {}
        if profile_path.exists():
            try:
                with profile_path.open('r') as f:
                    profiles = json.load(f)
            except Exception as e:
                logger.warning(f"Erreur lecture profils existants: {e}")
        
        # Ajout du nouveau profil
        profiles[profile_key] = {
            'optimal_threshold': threshold,
            'benchmark_timestamp': time.time(),
            'benchmark_results': benchmark_results
        }
        
        # Sauvegarde
        try:
            with profile_path.open('w') as f:
                json.dump(profiles, f, indent=2)
            logger.debug(f"Profil GPU sauvegardé: {profile_key} → {threshold}")
        except Exception as e:
            logger.error(f"Erreur sauvegarde profil GPU: {e}")


def make_profile_key(indicator_name: str, params: Dict[str, Any]) -> str:
    """
    Génère une clé de profil stable pour un indicateur et ses paramètres.
    
    Args:
        indicator_name: Nom de l'indicateur
        params: Paramètres de l'indicateur
        
    Returns:
        Clé de profil unique et stable
        
    Example:
        >>> make_profile_key("bollinger", {"period": 20, "std": 2.0})
        'bollinger_period:20_std:2.0'
    """
    # Tri des paramètres pour ordre stable
    sorted_params = sorted(params.items())
    
    # Construction de la clé
    param_str = "_".join(f"{k}:{v}" for k, v in sorted_params)
    
    return f"{indicator_name}_{param_str}"
    
    def bollinger_bands(self,
                       data: pd.DataFrame,
                       period: int = 20,
                       std_dev: float = 2.0,
                       price_col: str = 'close',
                       use_gpu: Optional[bool] = None) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calcul GPU des Bollinger Bands.
        
        Args:
            data: DataFrame OHLCV avec colonne price_col
            period: Période de la moyenne mobile
            std_dev: Multiplicateur d'écart-type
            price_col: Colonne de prix à utiliser
            use_gpu: Force GPU (None=auto, True=force, False=CPU only)
            
        Returns:
            Tuple (upper_band, middle_band, lower_band)
            
        Example:
            >>> df = pd.DataFrame({'close': [100, 101, 99, 102, 98]})
            >>> upper, middle, lower = bank.bollinger_bands(df, period=3)
        """
        if price_col not in data.columns:
            raise ValueError(f"Colonne '{price_col}' non trouvée dans les données")
        
        data_size = len(data)
        use_gpu_decision = use_gpu if use_gpu is not None else self._should_use_gpu(data_size)
        
        logger.debug(f"Bollinger Bands: {data_size} échantillons, "
                    f"GPU={'activé' if use_gpu_decision else 'désactivé'}")
        
        prices = data[price_col].values
        
        if use_gpu_decision and data_size >= self.min_samples_for_gpu:
            # Version GPU distribuée
            return self._bollinger_bands_gpu(prices, period, std_dev, data.index)
        else:
            # Version CPU classique
            return self._bollinger_bands_cpu(prices, period, std_dev, data.index)
    
    def _bollinger_bands_gpu(self,
                            prices: np.ndarray,
                            period: int,
                            std_dev: float,
                            index: pd.Index) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calcul Bollinger Bands distribué sur GPU.
        
        Architecture:
        - Split des prix en chunks
        - Calcul moving average & std par chunk avec overlap
        - Merge avec gestion des bordures
        """
        def bb_compute_func(price_chunk):
            """Fonction vectorielle pour un chunk de prix."""
            if len(price_chunk) < period:
                # Chunk trop petit: moyenne simple
                ma = np.full_like(price_chunk, np.mean(price_chunk))
                std = np.full_like(price_chunk, np.std(price_chunk))
            else:
                # Moving average avec convolution
                weights = np.ones(period) / period
                ma = np.convolve(price_chunk, weights, mode='same')
                
                # Moving standard deviation
                squared_diff = (price_chunk - ma) ** 2
                variance = np.convolve(squared_diff, weights, mode='same')
                std = np.sqrt(variance)
            
            # Bandes de Bollinger
            upper = ma + std_dev * std
            middle = ma
            lower = ma - std_dev * std
            
            # Empilement pour retour
            return np.column_stack([upper, middle, lower])
        
        # Distribution GPU
        start_time = pd.Timestamp.now()
        
        try:
            # Reshape pour distribution (ajout dimension batch si nécessaire)
            if prices.ndim == 1:
                prices_2d = prices.reshape(-1, 1)
            else:
                prices_2d = prices
            
            result = self.gpu_manager.distribute_workload(
                prices_2d,
                bb_compute_func,
                seed=42
            )
            
            # Extraction des bandes
            upper_band = result[:, 0]
            middle_band = result[:, 1] 
            lower_band = result[:, 2]
            
            elapsed = (pd.Timestamp.now() - start_time).total_seconds()
            logger.info(f"Bollinger Bands GPU: {len(prices)} échantillons en {elapsed:.3f}s")
            
        except Exception as e:
            logger.warning(f"Erreur calcul GPU Bollinger Bands: {e}")
            logger.info("Fallback calcul CPU")
            return self._bollinger_bands_cpu(prices, period, std_dev, index)
        
        return (
            pd.Series(upper_band, index=index, name='bb_upper'),
            pd.Series(middle_band, index=index, name='bb_middle'),
            pd.Series(lower_band, index=index, name='bb_lower')
        )
    
    def _bollinger_bands_cpu(self,
                            prices: np.ndarray,
                            period: int,
                            std_dev: float,
                            index: pd.Index) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calcul CPU classique des Bollinger Bands."""
        # Rolling window avec pandas pour simplicité
        price_series = pd.Series(prices, index=index)
        
        middle_band = price_series.rolling(window=period, min_periods=1).mean()
        rolling_std = price_series.rolling(window=period, min_periods=1).std()
        
        upper_band = middle_band + std_dev * rolling_std
        lower_band = middle_band - std_dev * rolling_std
        
        # Nommage des séries
        upper_band.name = 'bb_upper'
        middle_band.name = 'bb_middle'
        lower_band.name = 'bb_lower'
        
        return upper_band, middle_band, lower_band
    
    def atr(self,
            data: pd.DataFrame,
            period: int = 14,
            use_gpu: Optional[bool] = None) -> pd.Series:
        """
        Calcul GPU de l'Average True Range (ATR).
        
        Args:
            data: DataFrame OHLCV avec colonnes 'high', 'low', 'close'
            period: Période pour le calcul ATR
            use_gpu: Force GPU (None=auto, True=force, False=CPU only)
            
        Returns:
            Série ATR
            
        Example:
            >>> df = pd.DataFrame({
            ...     'high': [102, 103, 101],
            ...     'low': [98, 99, 97],
            ...     'close': [100, 101, 99]
            ... })
            >>> atr_series = bank.atr(df, period=2)
        """
        required_cols = ['high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Colonnes manquantes: {missing_cols}")
        
        data_size = len(data)
        use_gpu_decision = use_gpu if use_gpu is not None else self._should_use_gpu(data_size)
        
        logger.debug(f"ATR: {data_size} échantillons, "
                    f"GPU={'activé' if use_gpu_decision else 'désactivé'}")
        
        if use_gpu_decision and data_size >= self.min_samples_for_gpu:
            return self._atr_gpu(data, period)
        else:
            return self._atr_cpu(data, period)
    
    def _atr_gpu(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calcul ATR distribué sur GPU."""
        def atr_compute_func(ohlc_chunk):
            """Calcul ATR vectorisé pour un chunk."""
            if len(ohlc_chunk) < 2:
                return np.zeros(len(ohlc_chunk))
            
            high = ohlc_chunk[:, 0]  # Colonne high
            low = ohlc_chunk[:, 1]   # Colonne low  
            close = ohlc_chunk[:, 2] # Colonne close
            
            # True Range calculation
            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]  # Premier élément
            
            tr1 = high - low
            tr2 = np.abs(high - prev_close)
            tr3 = np.abs(low - prev_close)
            
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # ATR (moyenne mobile du True Range)
            if len(true_range) < period:
                atr = np.full_like(true_range, np.mean(true_range))
            else:
                weights = np.ones(period) / period
                atr = np.convolve(true_range, weights, mode='same')
            
            return atr
        
        try:
            # Préparation données pour distribution
            ohlc_array = data[['high', 'low', 'close']].values
            
            result = self.gpu_manager.distribute_workload(
                ohlc_array,
                atr_compute_func,
                seed=42
            )
            
            return pd.Series(result, index=data.index, name='atr')
            
        except Exception as e:
            logger.warning(f"Erreur calcul GPU ATR: {e}")
            return self._atr_cpu(data, period)
    
    def _atr_cpu(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calcul ATR CPU classique."""
        high = data['high']
        low = data['low']
        close = data['close']
        prev_close = close.shift(1)
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR (moyenne mobile)
        atr = true_range.rolling(window=period, min_periods=1).mean()
        atr.name = 'atr'
        
        return atr
    
    def rsi(self,
            data: pd.DataFrame,
            period: int = 14,
            price_col: str = 'close',
            use_gpu: Optional[bool] = None) -> pd.Series:
        """
        Calcul GPU du Relative Strength Index (RSI).
        
        Args:  
            data: DataFrame avec colonne price_col
            period: Période RSI
            price_col: Colonne de prix
            use_gpu: Force GPU (None=auto)
            
        Returns:
            Série RSI (0-100)
        """
        if price_col not in data.columns:
            raise ValueError(f"Colonne '{price_col}' non trouvée")
        
        data_size = len(data)
        use_gpu_decision = use_gpu if use_gpu is not None else self._should_use_gpu(data_size)
        
        prices = data[price_col].values
        
        if use_gpu_decision and data_size >= self.min_samples_for_gpu:
            return self._rsi_gpu(prices, period, data.index)
        else:
            return self._rsi_cpu(prices, period, data.index)
    
    def _rsi_gpu(self, prices: np.ndarray, period: int, index: pd.Index) -> pd.Series:
        """Calcul RSI distribué sur GPU."""
        def rsi_compute_func(price_chunk):
            """Calcul RSI vectorisé."""
            if len(price_chunk) < 2:
                return np.full(len(price_chunk), 50.0)  # RSI neutre
            
            # Calcul des gains/pertes
            price_diff = np.diff(price_chunk, prepend=price_chunk[0])
            gains = np.where(price_diff > 0, price_diff, 0)
            losses = np.where(price_diff < 0, -price_diff, 0)
            
            # Moyennes des gains/pertes
            if len(gains) < period:
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
            else:
                weights = np.ones(period) / period
                avg_gain = np.convolve(gains, weights, mode='same')
                avg_loss = np.convolve(losses, weights, mode='same')
            
            # RSI calculation
            rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss!=0)
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        
        try:
            if prices.ndim == 1:
                prices_2d = prices.reshape(-1, 1)
            else:
                prices_2d = prices
            
            result = self.gpu_manager.distribute_workload(
                prices_2d,
                rsi_compute_func,
                seed=42
            )
            
            if result.ndim > 1:
                result = result.flatten()
            
            return pd.Series(result, index=index, name='rsi')
            
        except Exception as e:
            logger.warning(f"Erreur calcul GPU RSI: {e}")
            return self._rsi_cpu(prices, period, index)
    
    def _rsi_cpu(self, prices: np.ndarray, period: int, index: pd.Index) -> pd.Series:
        """Calcul RSI CPU classique."""
        price_series = pd.Series(prices, index=index)
        
        # Gains et pertes
        price_change = price_series.diff()
        gains = price_change.where(price_change > 0, 0)
        losses = -price_change.where(price_change < 0, 0)
        
        # Moyennes mobiles
        avg_gain = gains.rolling(window=period, min_periods=1).mean()
        avg_loss = losses.rolling(window=period, min_periods=1).mean()
        
        # RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi.name = 'rsi'
        
        return rsi
    
    def get_performance_stats(self) -> dict:
        """
        Récupère les statistiques de performance du gestionnaire GPU.
        
        Returns:
            Dict avec stats devices et balance
        """
        return {
            'gpu_manager_stats': self.gpu_manager.get_device_stats(),
            'current_balance': self.gpu_manager.device_balance,
            'min_samples_for_gpu': self.min_samples_for_gpu,
            'available_indicators': [
                'bollinger_bands', 'atr', 'rsi'
            ]
        }
    
    def optimize_balance(self, 
                        sample_data: pd.DataFrame,
                        runs: int = 3) -> dict:
        """
        Optimise automatiquement la balance GPU pour les indicateurs.
        
        Args:
            sample_data: Données représentatives pour benchmark
            runs: Nombre de runs pour moyenne
            
        Returns:
            Nouveaux ratios optimaux
        """
        logger.info("Optimisation balance GPU pour indicateurs...")
        
        # Utilisation des données pour profiling
        if 'close' in sample_data.columns:
            # Test avec Bollinger Bands (représentatif)
            old_min_samples = self.min_samples_for_gpu
            self.min_samples_for_gpu = 0  # Force GPU pour profiling
            
            try:
                # Benchmark sur échantillon
                optimal_ratios = self.gpu_manager.profile_auto_balance(
                    sample_size=min(len(sample_data), 50000),
                    runs=runs
                )
                
                # Application des nouveaux ratios
                self.gpu_manager.set_balance(optimal_ratios)
                
                logger.info(f"Balance optimisée: {optimal_ratios}")
                return optimal_ratios
                
            finally:
                self.min_samples_for_gpu = old_min_samples
        
        else:
            logger.warning("Données sans colonne 'close', optimisation ignorée")
            return self.gpu_manager.device_balance


# === Instance globale ===

_gpu_indicator_bank: Optional[GPUAcceleratedIndicatorBank] = None

def get_gpu_accelerated_bank() -> GPUAcceleratedIndicatorBank:
    """
    Récupère l'instance globale de la banque d'indicateurs GPU.
    
    Returns:
        Instance GPUAcceleratedIndicatorBank
        
    Example:
        >>> bank = get_gpu_accelerated_bank()
        >>> upper, middle, lower = bank.bollinger_bands(df, use_gpu=True)
    """
    global _gpu_indicator_bank
    
    if _gpu_indicator_bank is None:
        _gpu_indicator_bank = GPUAcceleratedIndicatorBank()
        logger.info("Banque indicateurs GPU globale créée")
    
    return _gpu_indicator_bank