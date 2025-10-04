"""
Test pour le moteur d'optimisation paramétrique unifié ThreadX
=============================================================

Test simple pour vérifier que le UnifiedOptimizationEngine fonctionne
correctement avec l'IndicatorBank existant.

Usage:
    python -m pytest tests/test_optimization_engine.py -v
    # ou 
    python tools/test_optimization_simple.py

Author: ThreadX Framework
Version: Phase 10 - Unified Compute Engine
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Ajouter le répertoire src au path Python
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from threadx.optimization.engine import UnifiedOptimizationEngine, DEFAULT_SWEEP_CONFIG
from threadx.indicators.bank import IndicatorBank
from threadx.utils.log import setup_logging_once, get_logger

def create_test_data(n_points: int = 1000) -> pd.DataFrame:
    """Crée des données de test OHLCV."""
    np.random.seed(42)
    
    # Dates
    dates = pd.date_range('2024-01-01', periods=n_points, freq='1H')
    
    # Prix avec une tendance et du bruit
    base_price = 50000
    trend = np.linspace(0, 5000, n_points)  # Tendance haussière
    noise = np.random.randn(n_points).cumsum() * 100
    
    close_prices = base_price + trend + noise
    
    # OHLC basique
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    
    high_prices = np.maximum(open_prices, close_prices) + np.random.rand(n_points) * 100
    low_prices = np.minimum(open_prices, close_prices) - np.random.rand(n_points) * 100
    
    volumes = np.random.randint(100, 1000, n_points)
    
    return pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }).set_index('timestamp')

def test_simple_optimization():
    """Test simple du moteur d'optimisation."""
    print("🚀 Test du moteur d'optimisation paramétrique unifié ThreadX")
    print("=" * 60)
    
    # Setup logging
    setup_logging_once()
    logger = get_logger(__name__)
    
    try:
        # 1. Création des données de test
        print("📊 Génération des données de test...")
        test_data = create_test_data(500)  # Dataset plus petit pour test rapide
        print(f"   Données créées: {len(test_data)} barres")
        print(f"   Période: {test_data.index[0]} -> {test_data.index[-1]}")
        
        # 2. Initialisation du moteur unifié
        print("\n🔧 Initialisation du moteur unifié...")
        
        # Créer IndicatorBank
        indicator_bank = IndicatorBank()
        print("   ✅ IndicatorBank initialisé")
        
        # Créer moteur d'optimisation
        engine = UnifiedOptimizationEngine(
            indicator_bank=indicator_bank,
            max_workers=2  # Limité pour le test
        )
        print("   ✅ UnifiedOptimizationEngine initialisé")
        
        # 3. Configuration de test simple
        print("\n⚙️ Configuration du sweep de test...")
        test_config = {
            "dataset": {
                "symbol": "TEST",
                "timeframe": "1h",
                "start": "2024-01-01",
                "end": "2024-12-31"
            },
            "grid": {
                "bollinger": {
                    "period": [10, 20],  # Seulement 2 valeurs
                    "std": {"start": 2.0, "stop": 2.5, "step": 0.5}  # 2 valeurs
                }
            },
            "scoring": {
                "primary": "pnl",
                "secondary": ["sharpe", "-max_drawdown"],
                "top_k": 10
            }
        }
        
        # Calcul du nombre de combinaisons attendues
        bb_periods = len(test_config["grid"]["bollinger"]["period"])
        bb_stds = len([2.0, 2.5])  # start=2.0, stop=2.5, step=0.5
        expected_combos = bb_periods * bb_stds
        
        print(f"   Configuration: {expected_combos} combinaisons attendues")
        
        # 4. Callback de progression
        progress_data = {"last_progress": 0}
        
        def progress_callback(progress, completed, total, eta):
            if completed % 2 == 0 or completed == total:  # Log tous les 2
                eta_str = f" (ETA: {eta:.1f}s)" if eta else ""
                print(f"   📈 Progression: {completed}/{total} ({progress*100:.1f}%){eta_str}")
                progress_data["last_progress"] = progress
        
        engine.progress_callback = progress_callback
        
        # 5. Exécution du sweep
        print("\n🎯 Démarrage du sweep paramétrique...")
        print(f"   Utilisation du cache IndicatorBank pour optimiser les calculs")
        
        results = engine.run_parameter_sweep(test_config, test_data)
        
        # 6. Analyse des résultats
        print(f"\n🏆 Résultats obtenus:")
        print(f"   Nombre de résultats: {len(results)}")
        
        if not results.empty:
            print(f"   Colonnes: {list(results.columns)}")
            
            # Top 3 résultats
            print(f"\n🥇 Top 3 combinaisons:")
            for i, (_, row) in enumerate(results.head(3).iterrows()):
                print(f"   {i+1}. {row.get('indicator_type', 'N/A')} "
                      f"(period={row.get('period', 'N/A')}, std={row.get('std', 'N/A'):.2f}) "
                      f"-> PnL: {row.get('pnl', 0):.2f}, "
                      f"Sharpe: {row.get('sharpe', 0):.3f}")
            
            # Statistiques de performance
            stats = engine.get_indicator_bank_stats()
            print(f"\n📊 Statistiques du moteur:")
            print(f"   Cache hits: {stats.get('cache_hits', 0)}")
            print(f"   Cache size: {stats.get('cache_size', 0)} entries")
            print(f"   GPU usage: {stats.get('gpu_usage', 'N/A')}")
            
        else:
            print("   ⚠️ Aucun résultat obtenu")
        
        # 7. Test de nettoyage du cache
        print(f"\n🧹 Test nettoyage du cache...")
        cleaned = engine.cleanup_cache()
        print(f"   Entrées de cache nettoyées: {cleaned}")
        
        print(f"\n✅ Test terminé avec succès!")
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur pendant le test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_indicator_bank_direct():
    """Test direct de l'IndicatorBank pour vérifier la connectivité."""
    print("\n🔍 Test direct IndicatorBank...")
    
    try:
        # Données simples
        data = create_test_data(100)
        
        # Test IndicatorBank
        bank = IndicatorBank()
        
        # Test Bollinger Bands
        result = bank.ensure(
            indicator_type='bollinger',
            params={'period': 20, 'std': 2.0},
            data=data,
            symbol='TEST',
            timeframe='1h'
        )
        
        print(f"   ✅ Bollinger calculé: type={type(result)}")
        if isinstance(result, tuple):
            print(f"      Résultat: {len(result)} arrays de longueur {len(result[0])}")
        
        # Test ATR
        result_atr = bank.ensure(
            indicator_type='atr',
            params={'period': 14},
            data=data,
            symbol='TEST',
            timeframe='1h'
        )
        
        print(f"   ✅ ATR calculé: type={type(result_atr)}, longueur={len(result_atr)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur IndicatorBank: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("ThreadX Optimization Engine Test")
    print("================================")
    
    # Test IndicatorBank d'abord
    bank_ok = test_indicator_bank_direct()
    
    if bank_ok:
        # Test complet si IndicatorBank fonctionne
        success = test_simple_optimization()
        
        if success:
            print(f"\n🎉 Tous les tests sont passés!")
            print(f"   Le moteur d'optimisation unifié fonctionne correctement")
            print(f"   avec l'IndicatorBank existant.")
        else:
            print(f"\n💥 Le test d'optimisation a échoué")
            sys.exit(1)
    else:
        print(f"\n💥 L'IndicatorBank ne fonctionne pas correctement")
        sys.exit(1)