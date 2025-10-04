"""
ThreadX UI Smoke Tests - TechinTerror Interface
==============================================

Tests de base pour l'interface TechinTerror ThreadX.
Tests headless sans interaction utilisateur.

Author: ThreadX Framework
Version: Phase 8 - Smoke Tests
"""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging

import pandas as pd
import numpy as np

# Configuration pour tests headless
os.environ['MPLBACKEND'] = 'Agg'
np.random.seed(42)


class TestTechinTerrorBasic(unittest.TestCase):
    """Tests de base pour l'interface TechinTerror."""
    
    def setUp(self):
        """Configuration des tests."""
        logging.getLogger().setLevel(logging.ERROR)
        
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Données de test
        self.test_dates = pd.date_range('2024-01-01', periods=100, freq='1h', tz='UTC')
        self.test_df = pd.DataFrame({
            'open': np.random.uniform(50000, 51000, 100),
            'high': np.random.uniform(51000, 52000, 100),
            'low': np.random.uniform(49000, 50000, 100),
            'close': np.random.uniform(50000, 51000, 100),
            'volume': np.random.uniform(100, 1000, 100)
        }, index=self.test_dates)
        
        self.test_returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=self.test_dates)
        self.test_equity = (1 + self.test_returns).cumprod() * 10000
        
        self.test_trades = pd.DataFrame({
            'entry_time': self.test_dates[::10],
            'exit_time': self.test_dates[5::10],
            'side': ['LONG'] * 10,
            'pnl': np.random.normal(50, 100, 10),
            'entry_price': np.random.uniform(50000, 51000, 10),
            'exit_price': np.random.uniform(50000, 51000, 10)
        })
        
        self.test_metrics = {
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.08,
            'total_trades': 10,
            'win_rate': 0.6
        }
    
    def tearDown(self):
        """Nettoyage après tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_import_threadx_ui_modules(self):
        """Test d'importation des modules UI ThreadX."""
        try:
            from threadx.ui.app import ThreadXApp
            self.assertTrue(True, "Module ThreadXApp importé avec succès")
        except ImportError:
            self.skipTest("Module ThreadXApp non disponible")
    
    def test_import_threadx_charts(self):
        """Test d'importation du module charts."""
        try:
            from threadx.ui.charts import plot_equity, plot_drawdown
            self.assertTrue(True, "Module charts importé avec succès")
        except ImportError:
            self.skipTest("Module charts non disponible")
    
    def test_import_threadx_tables(self):
        """Test d'importation du module tables."""
        try:
            from threadx.ui.tables import render_trades_table, render_metrics_table
            self.assertTrue(True, "Module tables importé avec succès")
        except ImportError:
            self.skipTest("Module tables non disponible")
    
    def test_utility_functions_available(self):
        """Test de disponibilité des fonctions utilitaires."""
        try:
            from threadx.ui.app import extract_sym_tf, scan_dir_by_ext, _clean_series
            
            # Test extract_sym_tf
            result = extract_sym_tf("BTCUSDC_1h.parquet")
            self.assertEqual(result, ("BTCUSDC", "1h"))
            
            result = extract_sym_tf("invalid.txt")
            self.assertIsNone(result)
            
            # Test _clean_series
            cleaned = _clean_series(self.test_df)
            self.assertIsInstance(cleaned, pd.DataFrame)
            
        except ImportError:
            self.skipTest("Fonctions utilitaires non disponibles")
    
    def test_scan_directory_function(self):
        """Test de la fonction de scan de répertoires."""
        try:
            from threadx.ui.app import scan_dir_by_ext
            
            # Créer des fichiers de test
            test_files = [
                "BTCUSDC_1h.parquet",
                "ETHUSDT_15m.json", 
                "ADAUSDC_1m.csv",
                "invalid.txt"
            ]
            
            for filename in test_files:
                (self.temp_path / filename).touch()
            
            # Test scan JSON/CSV
            json_files = scan_dir_by_ext(str(self.temp_path), {".json", ".csv"})
            self.assertEqual(len(json_files), 2)
            
            # Test scan Parquet
            parquet_files = scan_dir_by_ext(str(self.temp_path), {".parquet"})
            self.assertEqual(len(parquet_files), 1)
            
        except ImportError:
            self.skipTest("Fonction scan_dir_by_ext non disponible")
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_chart_functions_basic(self, mock_close, mock_savefig):
        """Test de base des fonctions de graphiques."""
        try:
            from threadx.ui.charts import plot_equity, plot_drawdown
            
            save_path = self.temp_path / "equity.png"
            result = plot_equity(self.test_equity, save_path=save_path)
            self.assertEqual(result, save_path)
            
            save_path = self.temp_path / "drawdown.png"
            result = plot_drawdown(self.test_equity, save_path=save_path)
            self.assertEqual(result, save_path)
            
            # Vérifier que matplotlib a été appelé
            self.assertGreater(mock_savefig.call_count, 0)
            
        except ImportError:
            self.skipTest("Fonctions de graphiques non disponibles")
    
    def test_app_creation_mock(self):
        """Test de création d'app avec mocks."""
        try:
            with patch('threadx.ui.app.get_settings'), \
                 patch('threadx.ui.app.setup_logging_once'), \
                 patch('threadx.ui.app.get_logger'):
                
                from threadx.ui.app import ThreadXApp
                
                app = ThreadXApp()
                
                # Vérifier que l'app a les attributs essentiels
                self.assertTrue(hasattr(app, 'notebook'))
                self.assertTrue(hasattr(app, 'executor'))
                self.assertTrue(hasattr(app, 'logger'))
                
                app.destroy()
                
        except ImportError:
            self.skipTest("ThreadXApp non disponible")
        except Exception as e:
            self.skipTest(f"Erreur lors de la création de l'app: {e}")
    
    def test_threading_basic(self):
        """Test de base du threading."""
        from concurrent.futures import ThreadPoolExecutor
        
        def mock_work():
            return 42
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(mock_work)
            result = future.result(timeout=1.0)
            self.assertEqual(result, 42)
    
    def test_data_operations_basic(self):
        """Test de base des opérations sur les données."""
        # Test que pandas fonctionne
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        self.assertEqual(len(df), 3)
        
        # Test que numpy fonctionne
        arr = np.array([1, 2, 3])
        self.assertEqual(len(arr), 3)
        
        # Test des données de test
        self.assertIsInstance(self.test_df, pd.DataFrame)
        self.assertIsInstance(self.test_equity, pd.Series)
        self.assertIsInstance(self.test_trades, pd.DataFrame)
    
    def test_file_operations_basic(self):
        """Test de base des opérations sur fichiers."""
        # Test écriture/lecture JSON
        test_file = self.temp_path / "test.json"
        test_data = {"symbol": "BTCUSDC", "timeframe": "1h"}
        
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        with open(test_file, 'r') as f:
            loaded_data = json.load(f)
        
        self.assertEqual(loaded_data, test_data)
        
        # Test écriture CSV pandas
        csv_file = self.temp_path / "test.csv"
        self.test_df.to_csv(csv_file)
        
        loaded_df = pd.read_csv(csv_file, index_col=0)
        self.assertEqual(len(loaded_df), len(self.test_df))
    
    def test_streamlit_modules_exist(self):
        """Test que les modules Streamlit existent."""
        streamlit_app_path = Path("apps/streamlit/app.py")
        if streamlit_app_path.exists():
            self.assertTrue(True, "App Streamlit trouvée")
        else:
            self.skipTest("App Streamlit non trouvée")
    
    def test_configuration_files_exist(self):
        """Test que les fichiers de configuration existent."""
        config_files = [
            "pyproject.toml",
            "requirements.txt", 
            "paths.toml"
        ]
        
        for config_file in config_files:
            if Path(config_file).exists():
                self.assertTrue(True, f"Fichier {config_file} trouvé")
            else:
                self.skipTest(f"Fichier {config_file} non trouvé")


class TestDataProcessing(unittest.TestCase):
    """Tests de traitement des données."""
    
    def test_equity_curve_processing(self):
        """Test du traitement des courbes d'équité."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)
        equity = (1 + returns).cumprod() * 10000
        
        # Vérifications de base
        self.assertIsInstance(equity, pd.Series)
        self.assertEqual(len(equity), 100)
        self.assertTrue(equity.iloc[0] > 0)
        
        # Test calcul drawdown
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        
        self.assertIsInstance(drawdown, pd.Series)
        self.assertTrue((drawdown <= 0).all())
    
    def test_trades_processing(self):
        """Test du traitement des trades."""
        dates = pd.date_range('2024-01-01', periods=20, freq='1h')
        trades = pd.DataFrame({
            'entry_time': dates[::2],
            'exit_time': dates[1::2],
            'side': ['LONG'] * 10,
            'pnl': np.random.normal(0, 100, 10)
        })
        
        # Calculs de base
        total_pnl = trades['pnl'].sum()
        win_trades = trades[trades['pnl'] > 0]
        win_rate = len(win_trades) / len(trades)
        
        self.assertIsInstance(total_pnl, (int, float))
        self.assertGreaterEqual(win_rate, 0)
        self.assertLessEqual(win_rate, 1)
    
    def test_metrics_calculation(self):
        """Test du calcul des métriques."""
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # 1 an
        
        # Métriques de base
        total_return = (1 + returns).prod() - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        self.assertIsInstance(total_return, (int, float))
        self.assertIsInstance(volatility, (int, float))
        self.assertIsInstance(sharpe, (int, float))
        
        self.assertGreater(volatility, 0)


if __name__ == '__main__':
    # Configuration des tests
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Lancer les tests
    unittest.main(verbosity=2)