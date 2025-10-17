# 📋 FICHE TECHNIQUE ThreadX - GPU & Monte Carlo
## Guide d'utilisation complète des composants

---

## 🎯 **PRÉSENTATION GÉNÉRALE**

ThreadX est un framework de backtesting haute performance pour les stratégies de trading, optimisé pour l'accélération GPU multi-cartes avec support Monte Carlo avancé.

### Configuration Matérielle Cible
- **GPU Principal** : RTX 5090 (32GB GDDR7, Architecture Blackwell)
- **GPU Secondaire** : RTX 2060 (6GB GDDR6)
- **Répartition de charge** : 70-80% sur RTX 5090, 20-30% sur RTX 2060
- **Accélération attendue** : 1.5-2x sur tâches vectorisées, >100x sur Monte Carlo

---

## 🔧 **1. INSTALLATION ET CONFIGURATION**

### Dépendances Principales
```bash
# Installation des dépendances GPU
pip install cupy-cuda12x==13.6.0
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121
pip install numba scipy numpy pandas

# Vérification de l'environnement
python tools/check_env.py
```

### Configuration TOML
```toml
# paths.toml
[gpu]
devices = ["5090", "2060"]
load_balance = {5080 = 0.75, 2060 = 0.25}
memory_threshold = 0.8
auto_fallback = true

[performance]
target_tasks_per_min = 2500
vectorization_batch_size = 10000
cache_ttl_sec = 3600
```

---

## 🚀 **2. CUPY - ACCÉLÉRATION GPU NUMPY**

### Utilisation de Base
```python
import cupy as cp
import numpy as np

# Conversion CPU → GPU
data_cpu = np.array([1, 2, 3, 4, 5])
data_gpu = cp.asarray(data_cpu)

# Opérations sur GPU
result_gpu = cp.sqrt(data_gpu * 2)
result_cpu = cp.asnumpy(result_gpu)  # Retour CPU
```

### Integration ThreadX
```python
# src/threadx/indicators/bollinger.py
def bollinger_cupy(close: pd.Series, period: int, std: float) -> pd.DataFrame:
    """Version GPU avec CuPy"""
    close_gpu = cp.asarray(close.values)
    
    # Calcul rolling mean sur GPU
    rolling_mean = cp.convolve(close_gpu, cp.ones(period)/period, mode='valid')
    
    # Calcul rolling std sur GPU
    rolling_std = cp.sqrt(cp.convolve(
        (close_gpu[:-period+1] - rolling_mean)**2, 
        cp.ones(period)/period, mode='valid'
    ))
    
    upper = rolling_mean + (std * rolling_std)
    lower = rolling_mean - (std * rolling_std)
    
    return pd.DataFrame({
        'upper': cp.asnumpy(upper),
        'middle': cp.asnumpy(rolling_mean),
        'lower': cp.asnumpy(lower)
    })
```

### Multi-GPU avec CuPy
```python
def distribute_cupy_compute(data_chunks, func):
    """Distribution sur plusieurs GPU"""
    results = []
    for i, chunk in enumerate(data_chunks):
        with cp.cuda.Device(i % 2):  # Alterne entre GPU 0 et 1
            result = func(cp.asarray(chunk))
            results.append(cp.asnumpy(result))
    return np.concatenate(results)
```

---

## ⚡ **3. NUMBA - KERNELS CUDA OPTIMISÉS**

### Kernel CUDA de Base
```python
from numba import cuda
import math

@cuda.jit
def bollinger_kernel(close, period, std_mult, upper, middle, lower):
    """Kernel optimisé pour Bollinger Bands"""
    i = cuda.grid(1)
    
    if i >= period and i < close.size:
        # Calcul moyenne mobile
        sum_val = 0.0
        for j in range(period):
            sum_val += close[i - period + j + 1]
        mean = sum_val / period
        
        # Calcul écart-type
        sum_sq = 0.0
        for j in range(period):
            diff = close[i - period + j + 1] - mean
            sum_sq += diff * diff
        std = math.sqrt(sum_sq / period)
        
        # Résultats
        middle[i] = mean
        upper[i] = mean + (std_mult * std)
        lower[i] = mean - (std_mult * std)

def compute_bollinger_gpu(close_data, period=20, std_mult=2.0):
    """Interface haut niveau"""
    n = len(close_data)
    
    # Allocation mémoire GPU
    d_close = cuda.to_device(close_data)
    d_upper = cuda.device_array(n, dtype=np.float64)
    d_middle = cuda.device_array(n, dtype=np.float64)
    d_lower = cuda.device_array(n, dtype=np.float64)
    
    # Configuration kernel
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    
    # Exécution
    bollinger_kernel[blocks_per_grid, threads_per_block](
        d_close, period, std_mult, d_upper, d_middle, d_lower
    )
    
    return {
        'upper': d_upper.copy_to_host(),
        'middle': d_middle.copy_to_host(),
        'lower': d_lower.copy_to_host()
    }
```

### Optimisations Numba
```python
@cuda.jit
def optimized_atr_kernel(high, low, close, period, atr_out):
    """ATR optimisé avec mémoire partagée"""
    i = cuda.grid(1)
    
    # Mémoire partagée pour réduire les accès globaux
    shared_data = cuda.shared.array(256, dtype=cuda.float64)
    tid = cuda.threadIdx.x
    
    if i < len(high) - 1:
        # True Range calculation
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1]) if i > 0 else tr1
        tr3 = abs(low[i] - close[i-1]) if i > 0 else tr1
        tr = max(tr1, max(tr2, tr3))
        
        shared_data[tid] = tr
        cuda.syncthreads()
        
        # ATR calculation (moyenne mobile)
        if i >= period:
            sum_tr = 0.0
            for j in range(period):
                if tid - j >= 0:
                    sum_tr += shared_data[tid - j]
            atr_out[i] = sum_tr / period
```

---

## 🎲 **4. MONTE CARLO - SIMULATIONS MASSIVES**

### Monte Carlo Simple
```python
@cuda.jit
def monte_carlo_kernel(states, n_simulations, n_steps, dt, mu, sigma, results):
    """Simulation Monte Carlo sur GPU"""
    i = cuda.grid(1)
    
    if i < n_simulations:
        # Générateur aléatoire (cuRAND)
        state = states[i]
        price = 100.0  # Prix initial
        
        for step in range(n_steps):
            # Génération nombre aléatoire normal
            random_val = cuda.random.xoroshiro128p_normal_float64(state, 0)
            
            # Mouvement brownien géométrique
            drift = mu * dt
            diffusion = sigma * math.sqrt(dt) * random_val
            price *= math.exp(drift + diffusion)
        
        results[i] = price

def run_monte_carlo_gpu(n_simulations=100000, n_steps=252, mu=0.1, sigma=0.2):
    """Interface Monte Carlo GPU"""
    # Initialisation générateurs aléatoires
    states = cuda.random.create_xoroshiro128p_states(n_simulations, seed=42)
    results = cuda.device_array(n_simulations, dtype=np.float64)
    
    # Configuration kernel
    threads_per_block = 256
    blocks_per_grid = (n_simulations + threads_per_block - 1) // threads_per_block
    
    # Exécution
    dt = 1.0 / 252  # Jour de trading
    monte_carlo_kernel[blocks_per_grid, threads_per_block](
        states, n_simulations, n_steps, dt, mu, sigma, results
    )
    
    return results.copy_to_host()
```

### Monte Carlo pour Backtesting
```python
def monte_carlo_strategy_validation(strategy_func, price_data, n_scenarios=10000):
    """Validation Monte Carlo d'une stratégie"""
    
    @cuda.jit
    def strategy_mc_kernel(prices, scenarios, returns, n_trades):
        i = cuda.grid(1)
        if i < scenarios:
            # Simulation prix alternatifs
            scenario_prices = prices.copy()
            state = cuda.random.create_xoroshiro128p_states(1, seed=i)[0]
            
            # Ajout bruit stochastique
            for j in range(len(scenario_prices)):
                noise = cuda.random.xoroshiro128p_normal_float64(state, 0) * 0.01
                scenario_prices[j] *= (1 + noise)
            
            # Application stratégie
            pnl = 0.0
            trades = 0
            # ... logique stratégie ...
            
            returns[i] = pnl
            n_trades[i] = trades
    
    # Exécution et analyse résultats
    results = cuda.device_array(n_scenarios, dtype=np.float64)
    trades_count = cuda.device_array(n_scenarios, dtype=np.int32)
    
    # ... configuration et lancement kernel ...
    
    return {
        'mean_return': np.mean(results),
        'std_return': np.std(results),
        'var_95': np.percentile(results, 5),
        'success_rate': np.mean(results > 0)
    }
```

---

## 🔄 **5. MULTI-GPU - DISTRIBUTION DE CHARGE**

### Gestionnaire Multi-GPU
```python
# src/threadx/utils/gpu/multi_gpu.py
import cupy as cp
from concurrent.futures import ThreadPoolExecutor

class MultiGPUManager:
    def __init__(self, devices=["5090", "2060"], balance={"5090": 0.75, "2060": 0.25}):
        self.devices = devices
        self.balance = balance
        self.device_ids = list(range(len(devices)))
    
    def distribute_workload(self, data, func):
        """Distribution proportionnelle des données"""
        total_size = len(data)
        chunks = []
        
        start_idx = 0
        for device, ratio in self.balance.items():
            chunk_size = int(total_size * ratio)
            end_idx = start_idx + chunk_size
            chunks.append(data[start_idx:end_idx])
            start_idx = end_idx
        
        # Traitement parallèle
        with ThreadPoolExecutor(max_workers=len(self.devices)) as executor:
            futures = []
            for i, chunk in enumerate(chunks):
                future = executor.submit(self._process_on_device, i, chunk, func)
                futures.append(future)
            
            results = [f.result() for f in futures]
        
        return np.concatenate(results)
    
    def _process_on_device(self, device_id, data, func):
        """Traitement sur un GPU spécifique"""
        with cp.cuda.Device(device_id):
            gpu_data = cp.asarray(data)
            result = func(gpu_data)
            return cp.asnumpy(result)

# Utilisation
manager = MultiGPUManager()
result = manager.distribute_workload(large_dataset, bollinger_cupy)
```

### Auto-Balancing Dynamique
```python
def auto_balance_profile():
    """Profil automatique des performances GPU"""
    benchmark_data = np.random.randn(100000)
    
    timings = {}
    for device_id in range(cp.cuda.runtime.getDeviceCount()):
        with cp.cuda.Device(device_id):
            start_time = time.time()
            
            # Benchmark standard
            gpu_data = cp.asarray(benchmark_data)
            result = cp.sqrt(gpu_data ** 2 + 1)
            cp.cuda.Stream.null.synchronize()
            
            timings[device_id] = time.time() - start_time
    
    # Calcul ratios optimaux
    total_speed = sum(1/t for t in timings.values())
    optimal_balance = {f"gpu_{i}": (1/t)/total_speed for i, t in timings.items()}
    
    return optimal_balance
```

---

## 📊 **6. INTÉGRATION THREADX**

### Classe Indicateur GPU
```python
# src/threadx/indicators/base_gpu.py
from abc import ABC, abstractmethod

class GPUIndicator(ABC):
    def __init__(self, use_gpu=True, multi_gpu=True):
        self.use_gpu = use_gpu and cp.cuda.is_available()
        self.multi_gpu = multi_gpu and cp.cuda.runtime.getDeviceCount() > 1
        self.gpu_manager = MultiGPUManager() if self.multi_gpu else None
    
    @abstractmethod
    def compute_cpu(self, data, **params):
        pass
    
    @abstractmethod  
    def compute_gpu(self, data, **params):
        pass
    
    def compute(self, data, **params):
        """Interface unifiée CPU/GPU"""
        if self.use_gpu:
            try:
                if self.multi_gpu and len(data) > 50000:
                    return self.gpu_manager.distribute_workload(
                        data, lambda chunk: self.compute_gpu(chunk, **params)
                    )
                else:
                    return self.compute_gpu(data, **params)
            except Exception as e:
                print(f"GPU failed, falling back to CPU: {e}")
                return self.compute_cpu(data, **params)
        else:
            return self.compute_cpu(data, **params)
```

### Backtesting Accéléré
```python
# src/threadx/backtest/gpu_engine.py
class GPUBacktestEngine:
    def __init__(self, use_monte_carlo=True, mc_scenarios=10000):
        self.use_monte_carlo = use_monte_carlo
        self.mc_scenarios = mc_scenarios
        self.gpu_manager = MultiGPUManager()
    
    def run_strategy_batch(self, strategies, data_dict):
        """Batch de stratégies en parallèle GPU"""
        results = {}
        
        # Préparation données GPU
        gpu_data = {symbol: cp.asarray(df.values) for symbol, df in data_dict.items()}
        
        for strategy_name, strategy in strategies.items():
            if self.use_monte_carlo:
                # Validation Monte Carlo
                mc_results = self.monte_carlo_validation(strategy, gpu_data)
                results[f"{strategy_name}_mc"] = mc_results
            
            # Backtest standard
            bt_results = self.gpu_backtest(strategy, gpu_data)
            results[strategy_name] = bt_results
        
        return results
    
    def gpu_backtest(self, strategy, gpu_data):
        """Backtest principal sur GPU"""
        # Calcul indicateurs en batch
        indicators = self.compute_indicators_batch(gpu_data)
        
        # Génération signaux (vectorisé)
        signals = strategy.generate_signals_gpu(indicators)
        
        # Calcul PnL vectorisé
        returns = self.compute_returns_gpu(signals, gpu_data)
        
        return {
            'total_return': float(cp.asnumpy(cp.sum(returns))),
            'sharpe_ratio': float(self.compute_sharpe_gpu(returns)),
            'max_drawdown': float(self.compute_drawdown_gpu(returns))
        }
```

---

## 🔧 **7. BONNES PRATIQUES**

### Gestion Mémoire GPU
```python
def gpu_memory_manager():
    """Monitoring et nettoyage mémoire GPU"""
    for device_id in range(cp.cuda.runtime.getDeviceCount()):
        with cp.cuda.Device(device_id):
            # Vérification utilisation
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            
            print(f"GPU {device_id}:")
            print(f"  Used memory: {mempool.used_bytes() / 1024**3:.2f} GB")
            print(f"  Free memory: {mempool.free_bytes() / 1024**3:.2f} GB")
            
            # Nettoyage si nécessaire
            if mempool.used_bytes() / mempool.total_bytes() > 0.8:
                mempool.free_all_blocks()
                pinned_mempool.free_all_blocks()
```

### Optimisations Performances
```python
# Pré-allocation mémoire
def preallocate_gpu_arrays(max_size=1000000):
    """Pré-allocation pour éviter les allocations répétées"""
    return {
        'temp_array_1': cp.zeros(max_size, dtype=cp.float64),
        'temp_array_2': cp.zeros(max_size, dtype=cp.float64),
        'results': cp.zeros(max_size, dtype=cp.float64)
    }

# Streams CUDA pour parallélisme
def async_gpu_compute(data_chunks):
    """Calculs asynchrones multi-streams"""
    streams = [cp.cuda.Stream() for _ in range(4)]
    results = []
    
    for i, chunk in enumerate(data_chunks):
        stream = streams[i % len(streams)]
        with stream:
            gpu_chunk = cp.asarray(chunk)
            result = cp.sqrt(gpu_chunk ** 2 + 1)  # Example computation
            results.append(result)
    
    # Synchronisation
    for stream in streams:
        stream.synchronize()
    
    return cp.concatenate(results)
```

---

## 📈 **8. BENCHMARKS ET MONITORING**

### Tests de Performance
```python
def benchmark_gpu_vs_cpu():
    """Comparaison performances CPU/GPU"""
    sizes = [1000, 10000, 100000, 1000000]
    results = {'size': [], 'cpu_time': [], 'gpu_time': [], 'speedup': []}
    
    for size in sizes:
        data = np.random.randn(size)
        
        # CPU
        start = time.time()
        cpu_result = np.sqrt(data ** 2 + 1)
        cpu_time = time.time() - start
        
        # GPU
        start = time.time()
        gpu_data = cp.asarray(data)
        gpu_result = cp.sqrt(gpu_data ** 2 + 1)
        cp.cuda.Stream.null.synchronize()
        gpu_time = time.time() - start
        
        speedup = cpu_time / gpu_time
        
        results['size'].append(size)
        results['cpu_time'].append(cpu_time)
        results['gpu_time'].append(gpu_time)
        results['speedup'].append(speedup)
        
        print(f"Size: {size:>8}, CPU: {cpu_time:.4f}s, GPU: {gpu_time:.4f}s, Speedup: {speedup:.2f}x")
    
    return results
```

### Monitoring en Temps Réel
```python
class GPUMonitor:
    def __init__(self):
        self.metrics = {'gpu_usage': [], 'memory_usage': [], 'timestamps': []}
    
    def start_monitoring(self, interval=1.0):
        """Monitoring continu des GPU"""
        import threading
        
        def monitor_loop():
            while self.monitoring:
                timestamp = time.time()
                
                for device_id in range(cp.cuda.runtime.getDeviceCount()):
                    with cp.cuda.Device(device_id):
                        # Utilisation mémoire
                        mempool = cp.get_default_memory_pool()
                        memory_usage = mempool.used_bytes() / mempool.total_bytes()
                        
                        self.metrics['memory_usage'].append(memory_usage)
                        self.metrics['timestamps'].append(timestamp)
                
                time.sleep(interval)
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        self.monitoring = False
        self.monitor_thread.join()
        return self.metrics
```

---

## 🚨 **9. GESTION D'ERREURS ET FALLBACK**

### Système de Fallback Robuste
```python
class RobustGPUCompute:
    def __init__(self, max_retries=3):
        self.max_retries = max_retries
    
    def safe_gpu_compute(self, func, data, **kwargs):
        """Exécution sécurisée avec fallback CPU"""
        
        # Tentative GPU
        for attempt in range(self.max_retries):
            try:
                if cp.cuda.is_available():
                    gpu_data = cp.asarray(data)
                    result = func(gpu_data, **kwargs)
                    return cp.asnumpy(result)
            except (cp.cuda.memory.OutOfMemoryError, RuntimeError) as e:
                print(f"GPU attempt {attempt + 1} failed: {e}")
                
                # Nettoyage mémoire
                if cp.cuda.is_available():
                    mempool = cp.get_default_memory_pool()
                    mempool.free_all_blocks()
                
                if attempt == self.max_retries - 1:
                    break
        
        # Fallback CPU
        print("Falling back to CPU computation")
        return self._cpu_fallback(func, data, **kwargs)
    
    def _cpu_fallback(self, func, data, **kwargs):
        """Version CPU de secours"""
        # Conversion fonction GPU → CPU
        cpu_func = self._convert_gpu_to_cpu_func(func)
        return cpu_func(data, **kwargs)
```

---

## ⚙️ **10. CONFIGURATION AVANCÉE**

### Script de Configuration Automatique
```python
# tools/setup_gpu_environment.py
def detect_and_configure_gpus():
    """Détection et configuration automatique des GPU"""
    
    if not cp.cuda.is_available():
        print("❌ CUDA non disponible")
        return {"gpu_enabled": False}
    
    gpu_count = cp.cuda.runtime.getDeviceCount()
    print(f"✅ {gpu_count} GPU(s) détecté(s)")
    
    gpu_info = {}
    for i in range(gpu_count):
        with cp.cuda.Device(i):
            props = cp.cuda.runtime.getDeviceProperties(i)
            memory_gb = props['totalGlobalMem'] / (1024**3)
            
            gpu_info[f"gpu_{i}"] = {
                'name': props['name'].decode(),
                'memory_gb': memory_gb,
                'compute_capability': f"{props['major']}.{props['minor']}"
            }
            
            print(f"  GPU {i}: {props['name'].decode()} - {memory_gb:.1f} GB")
    
    # Configuration optimale automatique
    optimal_config = auto_balance_profile()
    
    # Sauvegarde configuration
    config = {
        'gpu_enabled': True,
        'device_count': gpu_count,
        'device_info': gpu_info,
        'optimal_balance': optimal_config,
        'recommended_batch_size': min(100000, int(memory_gb * 10000))
    }
    
    with open('gpu_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    return config

if __name__ == "__main__":
    config = detect_and_configure_gpus()
    print(f"\n✅ Configuration sauvée dans gpu_config.json")
```

---

## 📝 **RÉSUMÉ DES COMMANDES ESSENTIELLES**

### Installation Rapide
```bash
# 1. Vérification environnement
python tools/check_env.py

# 2. Installation dépendances GPU
pip install -r requirements.txt

# 3. Configuration automatique
python tools/setup_gpu_environment.py

# 4. Test de performance
python tools/benchmarks_cpu_gpu.py
```

### Utilisation dans le Code
```python
# Import des modules GPU ThreadX
from threadx.utils.gpu.multi_gpu import MultiGPUManager
from threadx.indicators.bollinger import BollingerGPU
from threadx.backtest.gpu_engine import GPUBacktestEngine

# Configuration simple
manager = MultiGPUManager()
indicator = BollingerGPU(use_gpu=True, multi_gpu=True)
engine = GPUBacktestEngine(use_monte_carlo=True)

# Exécution
results = engine.run_strategy_batch(strategies, data)
```

---

## 🎯 **OBJECTIFS DE PERFORMANCE**

| Métrique | CPU Seul | RTX 2060 | RTX 5090 | Multi-GPU |
|----------|----------|----------|----------|-----------|
| **Tâches/min** | 1,000 | 1,500 | 2,000 | 2,500+ |
| **Monte Carlo** | 1,000 sim/s | 50,000 sim/s | 100,000 sim/s | 150,000+ sim/s |
| **Indicateurs** | 1x | 10x | 20x | 30x+ |
| **Backtests** | 1x | 5x | 15x | 25x+ |

---

Cette fiche technique vous donne tous les éléments pour maîtriser l'écosystème GPU de ThreadX, de l'installation à l'optimisation avancée. Chaque section inclut des exemples pratiques directement intégrables dans votre code.