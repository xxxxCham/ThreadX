"""
ThreadX Tkinter Application - Phase 8
=====================================

Application principale Tkinter pour ThreadX avec :
- Interface à onglets (Data, Indicators, Strategy, Backtest, Performance, Logs)
- Threading pour opérations non-bloquantes
- Drag & drop de paramètres JSON
- Intégration avec les phases 3/5/6

Features:
- Windows-optimized UI with DPI scaling
- Non-freezing operations via threading
- Parameter validation and error handling
- Real-time log display
- Export functionality

Author: ThreadX Framework
Version: Phase 8 - UI Components
"""

import json
import logging
import os
import re
import threading
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tkinter import ttk, filedialog, messagebox, scrolledtext
from typing import Any, Dict, List, Optional, Union
import traceback
import time

import pandas as pd
import numpy as np

# ThreadX imports
try:
    from threadx.config import get_settings
    from threadx.utils.log import get_logger, setup_logging_once
    from threadx.data.ingest import IngestionManager
    from threadx.indicators.bank import IndicatorBank
    from threadx.backtest.performance import PerformanceCalculator
    from threadx.backtest.engine import BacktestEngine, create_engine
    from threadx.optimization import create_optimization_ui
    from .downloads import create_downloads_page
    from .sweep import create_sweep_page
except ImportError:
    # Mock imports for development
    def get_settings():
        class MockSettings:
            def get(self, key: str, default=None):
                return default
        return MockSettings()
    
    def get_logger(name: str):
        return logging.getLogger(name)
    
    def setup_logging_once():
        logging.basicConfig(level=logging.INFO)
        
    class IngestionManager:
        def __init__(self, settings):
            self.settings = settings
        def download_ohlcv_1m(self, *args, **kwargs):
            return pd.DataFrame()
    
    # Mock BacktestEngine
    def create_engine():
        class MockEngine:
            def run(self, *args, **kwargs):
                from dataclasses import dataclass, field
                from typing import Dict, Any
                @dataclass
                class MockResult:
                    equity: pd.Series = field(default_factory=lambda: pd.Series([10000, 11000]))
                    returns: pd.Series = field(default_factory=lambda: pd.Series([0.0, 0.1]))
                    trades: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
                    meta: Dict[str, Any] = field(default_factory=dict)
                return MockResult()
        return MockEngine()

# Mock Phase 3/5/6 imports - replace with actual imports
try:
    # Imports des anciens modules - remplacés par les nouveaux
    pass
except ImportError:
    # Mock implementations for development
    # Fallback aux vrais moteurs ThreadX si disponibles
    try:
        from threadx.indicators.bank import IndicatorBank as Bank
    except ImportError:
        class Bank:
            def ensure(self, *args, **kwargs):
                time.sleep(1)  # Simulate work
                return {'result': np.random.randn(100)}
    
    # Utilisation du PerformanceCalculator existant ou fallback
    try:
        from threadx.backtest.performance import PerformanceCalculator
    except ImportError:
        class PerformanceCalculator:
            @staticmethod
            def summarize(returns, trades):
                return {
                    'final_equity': 11000,
                    'total_return': 0.10,
                    'cagr': 0.12,
                    'sharpe': 1.5,
                    'max_drawdown': -0.05,
                    'total_trades': len(trades) if trades is not None else 0,
                    'win_rate': 0.6
                }
    
    # Fallback pour BacktestEngine si indisponible
    if not hasattr(BacktestEngine, 'run'):
        class FallbackBacktestEngine:
            def run(self, *args, **kwargs):
                time.sleep(2)  # Simulate work
                dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
                returns = pd.Series(np.random.randn(1000) * 0.01, index=dates)
                equity = (1 + returns).cumprod() * 10000
                trades = pd.DataFrame({
                    'entry_time': dates[::50],
                    'exit_time': dates[50::50],
                    'pnl': np.random.randn(20) * 100,
                    'side': ['LONG'] * 20,
                    'entry_price': 50000 + np.random.randn(20) * 1000,
                    'exit_price': 50000 + np.random.randn(20) * 1000
                })
                from dataclasses import dataclass, field
                from typing import Dict, Any
                @dataclass
                class MockRunResult:
                    returns: pd.Series = field(default_factory=lambda: returns)
                    equity: pd.Series = field(default_factory=lambda: equity)
                    trades: pd.DataFrame = field(default_factory=lambda: trades)
                    meta: Dict[str, Any] = field(default_factory=dict)
                return MockRunResult()
        BacktestEngine = FallbackBacktestEngine

# Nord-inspired Dark Theme (inspired by Streamlit design)
THEME_COLORS = {
    'background': '#2E3440',      # Main background
    'panel': '#3B4252',           # Panels/cards
    'border': '#4C566A',          # Borders
    'text': '#ECEFF4',            # Main text
    'positive': '#A3BE8C',        # Green (positive)
    'info': '#88C0D0',            # Blue (info)
    'warning': '#EBCB8B',         # Yellow (warning)
    'danger': '#BF616A',          # Red (danger/negative)
    'grid': '#4C566A'             # Grid lines
}

# Default dates (inspired by Streamlit app)
DEFAULT_START_UTC = pd.Timestamp('2024-12-01 00:00:00', tz='UTC')
DEFAULT_END_UTC = pd.Timestamp('2025-01-31 23:59:59', tz='UTC')

# Pattern matching for candle files (from Streamlit app)
CANDLE_REGEXES = [
    r"(?P<symbol>[A-Za-z0-9]+)[\\-_](?P<tf>(?:[1-9][0-9]*[smhdwM]))",
    r"(?P<symbol>[A-Za-z0-9]+)[\\-_]tf(?P<tf>(?:[1-9][0-9]*[smhdwM]))",
    r"(?P<symbol>[A-Za-z0-9]+).*?(?P<tf>(?:[1-9][0-9]*[smhdwM]))",
]

def extract_sym_tf(filename: str) -> Optional[tuple[str, str]]:
    """Extract (symbol, timeframe) from filename (from Streamlit app)."""
    fname = filename.rsplit(".", 1)[0]  # Remove extension
    
    for pat in CANDLE_REGEXES:
        m = re.match(pat, fname, re.IGNORECASE)
        if m:
            sym = m.group("symbol").upper()
            tf = m.group("tf").lower()
            return (sym, tf)
    
    return None

def scan_dir_by_ext(dirpath: str, allowed_exts: set) -> Dict[str, str]:
    """Scan directory and select one file per SYM_TF pair (from Streamlit app)."""
    best_by_pair = {}
    
    if not os.path.isdir(dirpath):
        return best_by_pair
    
    try:
        files = os.listdir(dirpath)
    except Exception:
        return best_by_pair
    
    for fname in files:
        p = os.path.join(dirpath, fname)
        if not os.path.isfile(p):
            continue
            
        ext = os.path.splitext(fname)[1].lower()
        if ext not in allowed_exts:
            continue
            
        parsed = extract_sym_tf(fname)
        if parsed is None:
            continue
            
        sym, tf = parsed
        key = f"{sym}_{tf}"
        
        if key not in best_by_pair:
            best_by_pair[key] = p
    
    return best_by_pair

def _clean_series(df: pd.DataFrame) -> pd.DataFrame:
    """Clean series: UTC index, sorted, no missing OHLC (from Streamlit app)."""
    x = df.copy()
    original_size = len(x)
    
    # Handle datetime index
    if not isinstance(x.index, pd.DatetimeIndex):
        try:
            x.index = pd.to_datetime(x.index, utc=True, errors="coerce")
        except Exception:
            raise ValueError("Cannot convert index to datetime")
    
    if x.index.tz is None:
        x.index = x.index.tz_localize("UTC")
    
    # Remove duplicate indices
    duplicated_count = x.index.duplicated().sum()
    if duplicated_count > 0:
        x = x[~x.index.duplicated(keep="last")]
    
    # Sort index
    x = x.sort_index()
    
    # Remove rows with missing OHLC
    ohlc_cols = ["open", "high", "low", "close"]
    available_ohlc = [col for col in ohlc_cols if col in x.columns]
    if available_ohlc:
        na_count = x[available_ohlc].isnull().any(axis=1).sum()
        if na_count > 0:
            x = x.dropna(subset=available_ohlc)
    
    final_size = len(x)
    
    if final_size == 0:
        raise ValueError("No valid data after cleaning")
    
    return x

def read_series_simple(path: str) -> pd.DataFrame:
    """Simple series reader for JSON/Parquet files."""
    try:
        path_obj = Path(path)
        ext = path_obj.suffix.lower()
        
        if ext == '.parquet':
            df = pd.read_parquet(path)
        elif ext in ['.json', '.ndjson']:
            df = pd.read_json(path)
        elif ext == '.csv':
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
            
        # Basic cleaning
        if 'timestamp' in df.columns and df.index.name != 'timestamp':
            df = df.set_index('timestamp')
            
        return _clean_series(df)
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Error reading {path}: {e}")
        return pd.DataFrame()

class ThreadXApp(tk.Tk):
    """
    ThreadX TechinTerror Interface - Streamlit-inspired Tkinter App.
    
    Provides a complete UI for ThreadX backtesting framework with:
    - BTC homepage with automatic loading
    - Multi-tab interface (Home, Data, Indicators, Backtest, Performance, Logs, Downloads)
    - Non-blocking operations via threading
    - Nord-inspired Dark theme
    - Data scanning and cleaning (JSON/Parquet)
    - Manual downloads with 1m + 3h verification
    """
    
    def __init__(self):
        """Initialize the TechinTerror application."""
        super().__init__()
        
        # Setup logging
        setup_logging_once()
        self.logger = get_logger(__name__)
        self.logger.info("Initializing TechinTerror Application")
        
        # Initialize settings and components
        self.settings = get_settings()
        
        # Utilisation des vrais moteurs ThreadX
        try:
            # IndicatorBank principal pour tous les calculs d'indicateurs
            self.indicator_bank = IndicatorBank()
            self.logger.info("✅ IndicatorBank réel initialisé avec cache intelligent")
        except Exception as e:
            self.indicator_bank = Bank()
            self.logger.warning(f"⚠️ Fallback vers Bank mock: {e}")
        
        # BacktestEngine via create_engine (utilise le vrai moteur ThreadX)
        try:
            self.engine = create_engine()
            self.logger.info("✅ BacktestEngine réel initialisé avec support GPU/multi-GPU")
        except Exception as e:
            self.engine = BacktestEngine()
            self.logger.warning(f"⚠️ Fallback vers BacktestEngine basique: {e}")
            
        self.performance = PerformanceCalculator()
        
        # Gestionnaire d'ingestion de données
        try:
            self.ingestion_manager = IngestionManager(self.settings)
        except Exception as e:
            self.ingestion_manager = None
            self.logger.warning(f"⚠️ IngestionManager non disponible: {e}")
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.running_tasks = set()
        
        # Data storage
        self.current_params = {}
        self.last_results = None
        self.last_returns = None
        self.last_equity = None  # Add missing attribute for chart methods
        self.last_trades = None
        self.last_metrics = None
        self.last_result = None  # Add missing RunResult storage
        self.current_df = None
        self.current_data = None  # Add missing current_data attribute
        
        # File scanning results
        self.available_files = {}
        self.by_symbol = {}
        
        # Default paths (inspired by Streamlit app)
        self.json_dir = Path("data/crypto_data_json")
        self.parquet_dir = Path("data/crypto_data_parquet")
        
        # Setup UI
        self._setup_window()
        self._create_widgets()
        self._setup_drag_drop()
        
        # Auto-load BTC on startup
        self._auto_load_btc()
        
        self.logger.info("TechinTerror Application initialized successfully")
    
    def _setup_window(self):
        """Configure main window properties with dark theme."""
        self.title("ThreadX TechinTerror - Algorithmic Trading Interface")
        self.geometry("1400x900")
        self.minsize(1000, 700)
        
        # Configure for Windows DPI scaling
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except (ImportError, AttributeError):
            pass  # Not on Windows or DPI awareness not available
        
        # Apply dark theme
        self.configure(bg=THEME_COLORS['background'])
        
        # Icon (if available)
        try:
            self.iconbitmap(Path("assets/threadx.ico"))
        except tk.TclError:
            pass  # Icon not found, continue without
        
        # Bind close event
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        
    def _auto_load_btc(self):
        """Auto-load BTC data on startup (inspired by Streamlit homepage)."""
        def load_btc_async():
            try:
                self.logger.info("Auto-loading BTC data...")
                
                # Scan for available files
                self._scan_data_directories()
                
                # Look for BTC files
                btc_symbols = ['BTCUSDC', 'BTCUSDT', 'BTCUSD', 'BTC']
                for symbol in btc_symbols:
                    if symbol in self.by_symbol:
                        timeframes = list(self.by_symbol[symbol].keys())
                        if timeframes:
                            # Load first available timeframe
                            tf = timeframes[0]
                            if '1h' in timeframes:
                                tf = '1h'  # Prefer 1h
                            elif '15m' in timeframes:
                                tf = '15m'  # Then 15m
                                
                            self._load_and_display_data(symbol, tf)
                            self.logger.info(f"Auto-loaded {symbol} {tf} data")
                            return
                
                self.logger.info("No BTC data found for auto-loading")
                
            except Exception as e:
                self.logger.error(f"Error auto-loading BTC: {e}")
        
        # Load in background thread
        self.executor.submit(load_btc_async)
        
    def _scan_data_directories(self):
        """Scan data directories for available files."""
        try:
            # Scan JSON files
            json_files = scan_dir_by_ext(str(self.json_dir), {".json", ".ndjson", ".csv"})
            
            # Scan Parquet files  
            parquet_files = scan_dir_by_ext(str(self.parquet_dir), {".parquet"})
            
            # Combine all files
            self.available_files = {**json_files, **parquet_files}
            
            # Organize by symbol
            self.by_symbol = {}
            for key, path in self.available_files.items():
                fname = os.path.basename(path)
                parsed = extract_sym_tf(fname)
                if parsed:
                    symbol, tf = parsed
                    if symbol not in self.by_symbol:
                        self.by_symbol[symbol] = {}
                    self.by_symbol[symbol][tf] = path
                    
            self.logger.info(f"Scanned data: {len(self.available_files)} files, {len(self.by_symbol)} symbols")
            
        except Exception as e:
            self.logger.error(f"Error scanning data directories: {e}")
            
    def _load_and_display_data(self, symbol: str, timeframe: str):
        """Load and display data for given symbol/timeframe."""
        try:
            if symbol not in self.by_symbol or timeframe not in self.by_symbol[symbol]:
                self.logger.warning(f"Data not found for {symbol} {timeframe}")
                return
                
            path = self.by_symbol[symbol][timeframe]
            self.logger.info(f"Loading {symbol} {timeframe} from {path}")
            
            # Load data
            df = read_series_simple(path)
            if df.empty:
                self.logger.warning(f"Empty data loaded from {path}")
                return
                
            self.current_df = df
            
            # Update UI in main thread
            self.after(0, self._update_data_display, symbol, timeframe, df)
            
        except Exception as e:
            self.logger.error(f"Error loading data for {symbol} {timeframe}: {e}")
            
    def _update_data_display(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Update UI with loaded data (main thread)."""
        try:
            # Update status
            if hasattr(self, 'status_var'):
                self.status_var.set(f"Loaded {symbol} {timeframe}: {len(df)} bars")
            
            # Update symbol/timeframe selectors if they exist
            if hasattr(self, 'symbol_var'):
                self.symbol_var.set(symbol)
            if hasattr(self, 'timeframe_var'):
                self.timeframe_var.set(timeframe)
                
            # Update data info display
            self._update_data_info_display(df, symbol, timeframe)
            
            # Set as current data for other tabs
            self.current_data = df
            
        except Exception as e:
            self.logger.error(f"Error updating data display: {e}")
            
    def _update_data_info_display(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Update data information display."""
        try:
            if not hasattr(self, 'data_info_text'):
                return
                
            info_text = []
            info_text.append(f"Symbol: {symbol}")
            info_text.append(f"Timeframe: {timeframe}")
            info_text.append(f"Total bars: {len(df):,}")
            
            if not df.empty:
                info_text.append(f"Date range: {df.index[0]} to {df.index[-1]}")
                info_text.append(f"Columns: {', '.join(df.columns)}")
                
                # Basic stats if OHLC available
                if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                    info_text.append(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
                    info_text.append(f"Latest close: ${df['close'].iloc[-1]:.2f}")
            
            # Update text widget
            self.data_info_text.configure(state=tk.NORMAL)
            self.data_info_text.delete(1.0, tk.END)
            self.data_info_text.insert(1.0, "\n".join(info_text))
            self.data_info_text.configure(state=tk.DISABLED)
            
        except Exception as e:
            self.logger.error(f"Error updating data info display: {e}")
    
    def _create_widgets(self):
        """Create and layout all UI widgets with TechinTerror theme."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs (inspired by Streamlit layout)
        self._create_home_tab()           # New: BTC homepage
        self._create_data_tab()           # Enhanced data selection
        self._create_indicators_tab()     # Indicators regeneration
        self._create_optimization_tab()   # Parametric optimization & sweeps
        self._create_sweep_tab()          # NEW: Dedicated parametric sweeps
        self._create_backtest_tab()       # Backtest with Streamlit params
        self._create_performance_tab()    # Charts and metrics
        self._create_downloads_tab()      # Manual downloads 1m + 3h
        self._create_logs_tab()           # Logs display
        
        # Status bar with dark theme
        self.status_var = tk.StringVar(value="TechinTerror Ready - Auto-loading BTC...")
        self.status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _create_home_tab(self):
        """Create Home tab with BTC display (inspired by Streamlit homepage)."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🏠 Home")
        
        # Main container with padding
        main_frame = ttk.Frame(tab, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title section
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = ttk.Label(title_frame, text="ThreadX TechinTerror", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(side=tk.LEFT)
        
        subtitle_label = ttk.Label(title_frame, text="Crypto Backtesting Studio - BTC Dashboard",
                                  font=('Arial', 10))
        subtitle_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Quick Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Symbol and timeframe selection
        ttk.Label(control_frame, text="Symbol:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.home_symbol_var = tk.StringVar(value="BTCUSDC")
        symbol_combo = ttk.Combobox(control_frame, textvariable=self.home_symbol_var, 
                                   values=[], width=12, state="readonly")
        symbol_combo.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        ttk.Label(control_frame, text="Timeframe:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.home_timeframe_var = tk.StringVar(value="1h")
        tf_combo = ttk.Combobox(control_frame, textvariable=self.home_timeframe_var,
                               values=["1m", "5m", "15m", "1h", "4h", "1d"], width=8, state="readonly")
        tf_combo.grid(row=0, column=3, sticky=tk.W, padx=(0, 20))
        
        # Load button
        load_btn = ttk.Button(control_frame, text="Load Data", 
                             command=self._on_home_load_data)
        load_btn.grid(row=0, column=4, padx=(0, 20))
        
        # Auto-scan button
        scan_btn = ttk.Button(control_frame, text="Rescan Files", 
                             command=self._on_rescan_files)
        scan_btn.grid(row=0, column=5)
        
        # Data display area
        display_frame = ttk.LabelFrame(main_frame, text="Data Overview", padding="10")
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Quick stats on left, chart placeholder on right
        left_frame = ttk.Frame(display_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        
        # Quick stats
        stats_frame = ttk.LabelFrame(left_frame, text="Quick Stats", padding="10")
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.stats_text = scrolledtext.ScrolledText(stats_frame, height=8, width=30, 
                                                   state=tk.DISABLED, wrap=tk.WORD)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
        # Available files
        files_frame = ttk.LabelFrame(left_frame, text="Available Files", padding="10")
        files_frame.pack(fill=tk.BOTH, expand=True)
        
        self.files_text = scrolledtext.ScrolledText(files_frame, height=15, width=30,
                                                   state=tk.DISABLED, wrap=tk.WORD)
        self.files_text.pack(fill=tk.BOTH, expand=True)
        
        # Chart area (placeholder for now)
        chart_frame = ttk.LabelFrame(display_frame, text="Price Chart", padding="10")
        chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.chart_canvas = tk.Canvas(chart_frame, bg=THEME_COLORS['panel'], height=400)
        self.chart_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Initial message
        self.chart_canvas.create_text(400, 200, text="Chart will appear here after loading data",
                                     fill=THEME_COLORS['text'], font=('Arial', 12))
        
        # Store widgets for updates
        self.home_symbol_combo = symbol_combo
        self.home_tf_combo = tf_combo
        
        # Update available symbols
        self._update_home_symbols()

    def _create_data_tab(self):
        """Create Data configuration tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📁 Data")
        
        # Main frame with padding
        main_frame = ttk.Frame(tab, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Data source section
        source_frame = ttk.LabelFrame(main_frame, text="Data Source", padding="10")
        source_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(source_frame, text="Symbol:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.symbol_var = tk.StringVar(value="BTCUSDC")
        ttk.Entry(source_frame, textvariable=self.symbol_var, width=15).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(source_frame, text="Timeframe:").grid(row=0, column=2, sticky=tk.W, padx=(20, 10))
        self.timeframe_var = tk.StringVar(value="15m")
        timeframe_combo = ttk.Combobox(source_frame, textvariable=self.timeframe_var, 
                                     values=["1m", "5m", "15m", "1h", "4h", "1d"], width=10)
        timeframe_combo.grid(row=0, column=3, sticky=tk.W)
        timeframe_combo.state(['readonly'])
        
        # Date range
        ttk.Label(source_frame, text="Start Date:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.start_date_var = tk.StringVar(value="2024-01-01")
        ttk.Entry(source_frame, textvariable=self.start_date_var, width=12).grid(row=1, column=1, sticky=tk.W, pady=(10, 0))
        
        ttk.Label(source_frame, text="End Date:").grid(row=1, column=2, sticky=tk.W, padx=(20, 10), pady=(10, 0))
        self.end_date_var = tk.StringVar(value="2024-12-31")
        ttk.Entry(source_frame, textvariable=self.end_date_var, width=12).grid(row=1, column=3, sticky=tk.W, pady=(10, 0))
        
        # Data info display
        info_frame = ttk.LabelFrame(main_frame, text="Data Information", padding="10")
        info_frame.pack(fill=tk.BOTH, expand=True)
        
        self.data_info_text = scrolledtext.ScrolledText(info_frame, height=10, state=tk.DISABLED)
        self.data_info_text.pack(fill=tk.BOTH, expand=True)
        
        self._update_data_info("No data loaded")
    
    def _create_downloads_tab(self):
        """Create Downloads tab for manual 1m + 3h verification using the new DownloadsPage.""" 
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📥 Downloads")
        
        # Use the new DownloadsPage component
        try:
            from .downloads import create_downloads_page
            self.downloads_page = create_downloads_page(tab)
            self.downloads_page.pack(fill=tk.BOTH, expand=True)
            self.logger.info("✅ Onglet Downloads créé avec nouvelle interface")
        except Exception as e:
            self.logger.error(f"❌ Erreur création Downloads page: {e}")
            
            # Fallback simple interface
            main_frame = ttk.Frame(tab, padding="15")
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            error_label = ttk.Label(main_frame, 
                                  text=f"❌ Page de téléchargements non disponible\n\nErreur: {e}\n"
                                       f"Veuillez vérifier l'installation des composants.",
                                  justify=tk.CENTER)
            error_label.pack(expand=True)
    
    def _create_sweep_tab(self):
        """Create Sweep tab for parametric optimization using the new SweepPage."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🎯 Sweep")
        
        # Use the new SweepOptimizationPage component
        try:
            from .sweep import create_sweep_page
            self.sweep_page = create_sweep_page(tab, self.indicator_bank)
            self.sweep_page.pack(fill=tk.BOTH, expand=True)
            self.logger.info("✅ Onglet Sweep créé avec moteur unifié")
        except Exception as e:
            self.logger.error(f"❌ Erreur création Sweep page: {e}")
            
            # Fallback simple interface
            main_frame = ttk.Frame(tab, padding="15")
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            error_label = ttk.Label(main_frame, 
                                  text=f"❌ Page d'optimisation non disponible\n\nErreur: {e}\n"
                                       f"Veuillez vérifier l'installation des composants.",
                                  justify=tk.CENTER)
            error_label.pack(expand=True)
    
    def _create_indicators_tab(self):
        """Create Indicators configuration tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Indicators")
        
        main_frame = ttk.Frame(tab, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Indicators config
        config_frame = ttk.LabelFrame(main_frame, text="Indicator Configuration", padding="10")
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Bollinger Bands
        ttk.Label(config_frame, text="BB Period:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.bb_period_var = tk.StringVar(value="20")
        ttk.Entry(config_frame, textvariable=self.bb_period_var, width=10).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(config_frame, text="BB Std:").grid(row=0, column=2, sticky=tk.W, padx=(20, 10))
        self.bb_std_var = tk.StringVar(value="2.0")
        ttk.Entry(config_frame, textvariable=self.bb_std_var, width=10).grid(row=0, column=3, sticky=tk.W)
        
        # ATR
        ttk.Label(config_frame, text="ATR Period:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.atr_period_var = tk.StringVar(value="14")
        ttk.Entry(config_frame, textvariable=self.atr_period_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=(10, 0))
        
        # Regenerate button
        self.regenerate_btn = ttk.Button(config_frame, text="Regenerate Indicators", 
                                       command=self.trigger_regenerate)
        self.regenerate_btn.grid(row=2, column=0, columnspan=2, pady=(20, 0), sticky=tk.W)
        
        # Progress
        self.indicators_progress = ttk.Progressbar(config_frame, mode='indeterminate')
        self.indicators_progress.grid(row=2, column=2, columnspan=2, pady=(20, 0), sticky=tk.EW, padx=(20, 0))
        
        # Status display
        status_frame = ttk.LabelFrame(main_frame, text="Indicator Status", padding="10")
        status_frame.pack(fill=tk.BOTH, expand=True)
        
        self.indicators_text = scrolledtext.ScrolledText(status_frame, height=15, state=tk.DISABLED)
        self.indicators_text.pack(fill=tk.BOTH, expand=True)
        
        self._update_indicators_status("Ready to regenerate indicators")
    
    def _create_optimization_tab(self):
        """Create Parametric Optimization tab with unified engine."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🎯 Optimization")
        
        # Create optimization UI using the unified engine with shared IndicatorBank
        try:
            self.optimization_ui = create_optimization_ui(tab, self.indicator_bank)
            self.logger.info("✅ Onglet d'optimisation paramétrique créé avec moteur unifié")
        except Exception as e:
            # Fallback simple interface
            self.logger.error(f"❌ Erreur création UI optimisation: {e}")
            
            main_frame = ttk.Frame(tab, padding="10")
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            error_label = ttk.Label(main_frame, 
                                  text=f"❌ Module d'optimisation non disponible\n\nErreur: {e}\n"
                                       f"Veuillez vérifier l'installation des dépendances.",
                                  justify=tk.CENTER)
            error_label.pack(expand=True)
            
            # Basic placeholder controls
            controls_frame = ttk.Frame(main_frame)
            controls_frame.pack(pady=20)
            
            ttk.Button(controls_frame, text="🔄 Réessayer", 
                      command=self._retry_optimization_init).pack(side=tk.LEFT, padx=5)
            ttk.Button(controls_frame, text="📚 Documentation", 
                      command=self._open_optimization_docs).pack(side=tk.LEFT, padx=5)
    
    def _retry_optimization_init(self):
        """Retry initializing optimization module."""
        try:
            # Try to recreate the optimization UI
            tab_index = None
            for i, tab_id in enumerate(self.notebook.tabs()):
                if self.notebook.tab(tab_id, "text") == "🎯 Optimization":
                    tab_index = i
                    break
            
            if tab_index is not None:
                # Remove old tab
                self.notebook.forget(tab_index)
                # Recreate it
                self._create_optimization_tab()
                messagebox.showinfo("Succès", "Module d'optimisation rechargé avec succès!")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de recharger le module: {e}")
    
    def _open_optimization_docs(self):
        """Open optimization documentation."""
        import webbrowser
        try:
            # Try to open local docs or GitHub
            docs_path = Path("docs/optimization_guide.md")
            if docs_path.exists():
                webbrowser.open(f"file://{docs_path.absolute()}")
            else:
                webbrowser.open("https://github.com/ThreadX/docs/optimization")
        except Exception:
            messagebox.showinfo("Documentation", 
                               "Consultez le README.md pour la documentation d'optimisation")
    
    def _create_strategy_tab(self):
        """Create Strategy parameters tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Strategy")
        
        main_frame = ttk.Frame(tab, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Parameter loading section
        load_frame = ttk.LabelFrame(main_frame, text="Parameter Management", padding="10")
        load_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(load_frame, text="Load Parameters from JSON", 
                  command=self._load_params_dialog).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(load_frame, text="Save Parameters to JSON", 
                  command=self._save_params_dialog).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(load_frame, text="Reset to Defaults", 
                  command=self._reset_params).pack(side=tk.LEFT)
        
        # Strategy parameters
        params_frame = ttk.LabelFrame(main_frame, text="Strategy Parameters", padding="10")
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Entry parameters
        ttk.Label(params_frame, text="Entry Z-Score:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.entry_z_var = tk.StringVar(value="2.0")
        ttk.Entry(params_frame, textvariable=self.entry_z_var, width=10).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(params_frame, text="Stop Loss K:").grid(row=0, column=2, sticky=tk.W, padx=(20, 10))
        self.k_sl_var = tk.StringVar(value="1.5")
        ttk.Entry(params_frame, textvariable=self.k_sl_var, width=10).grid(row=0, column=3, sticky=tk.W)
        
        # Risk parameters
        ttk.Label(params_frame, text="Leverage:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.leverage_var = tk.StringVar(value="3")
        ttk.Entry(params_frame, textvariable=self.leverage_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=(10, 0))
        
        ttk.Label(params_frame, text="Risk %:").grid(row=1, column=2, sticky=tk.W, padx=(20, 10), pady=(10, 0))
        self.risk_var = tk.StringVar(value="2.0")
        ttk.Entry(params_frame, textvariable=self.risk_var, width=10).grid(row=1, column=3, sticky=tk.W, pady=(10, 0))
        
        # Trail parameters
        ttk.Label(params_frame, text="Trail K:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.trail_k_var = tk.StringVar(value="1.0")
        ttk.Entry(params_frame, textvariable=self.trail_k_var, width=10).grid(row=2, column=1, sticky=tk.W, pady=(10, 0))
        
        # Validation button
        ttk.Button(params_frame, text="Validate Parameters", 
                  command=self._validate_params).grid(row=3, column=0, columnspan=2, pady=(20, 0), sticky=tk.W)
        
        # Current parameters display
        current_frame = ttk.LabelFrame(main_frame, text="Current Parameters", padding="10")
        current_frame.pack(fill=tk.BOTH, expand=True)
        
        self.params_text = scrolledtext.ScrolledText(current_frame, height=10, state=tk.DISABLED)
        self.params_text.pack(fill=tk.BOTH, expand=True)
        
        self._update_params_display()
    
    def _create_backtest_tab(self):
        """Create Backtest execution tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Backtest")
        
        main_frame = ttk.Frame(tab, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Execution section
        exec_frame = ttk.LabelFrame(main_frame, text="Backtest Execution", padding="10")
        exec_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Première ligne: boutons principaux
        buttons_frame = ttk.Frame(exec_frame)
        buttons_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.backtest_btn = ttk.Button(buttons_frame, text="🚀 Run Backtest", 
                                     command=self.trigger_backtest, 
                                     style="Accent.TButton")
        self.backtest_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.pause_btn = ttk.Button(buttons_frame, text="⏸️ Pause", 
                                  command=self._pause_backtest,
                                  state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.resume_btn = ttk.Button(buttons_frame, text="▶️ Resume", 
                                   command=self._resume_backtest,
                                   state=tk.DISABLED)
        self.resume_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_btn = ttk.Button(buttons_frame, text="⏹️ Stop", 
                                 command=self._stop_backtest,
                                 state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 20))
        
        # Status du backtest
        self.backtest_status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(buttons_frame, textvariable=self.backtest_status_var, 
                               foreground="blue")
        status_label.pack(side=tk.RIGHT, padx=(20, 0))
        
        # Deuxième ligne: barre de progression
        progress_frame = ttk.Frame(exec_frame)
        progress_frame.pack(fill=tk.X)
        
        self.backtest_progress = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.backtest_progress.pack(fill=tk.X, pady=(0, 5))
        
        # Label de progression détaillé
        self.progress_detail_var = tk.StringVar(value="")
        progress_detail_label = ttk.Label(progress_frame, textvariable=self.progress_detail_var,
                                        foreground="gray")
        progress_detail_label.pack()
        
        # Options
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        options_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.use_gpu_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Use GPU Acceleration", 
                       variable=self.use_gpu_var).pack(side=tk.LEFT, padx=(0, 20))
        
        self.cache_indicators_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Cache Indicators", 
                       variable=self.cache_indicators_var).pack(side=tk.LEFT)
        
        # Results summary
        results_frame = ttk.LabelFrame(main_frame, text="Backtest Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, state=tk.DISABLED)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        self._update_results_display("No backtest results available")
    
    def _create_performance_tab(self):
        """Create Performance analysis tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Performance")
        
        main_frame = ttk.Frame(tab, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Charts section
        charts_frame = ttk.LabelFrame(main_frame, text="Charts", padding="10")
        charts_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(charts_frame, text="Show Equity Curve", 
                  command=self._show_equity_chart).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(charts_frame, text="Show Drawdown", 
                  command=self._show_drawdown_chart).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(charts_frame, text="Export Charts", 
                  command=self._export_charts).pack(side=tk.LEFT)
        
        # Tables section
        tables_frame = ttk.LabelFrame(main_frame, text="Tables", padding="10")
        tables_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(tables_frame, text="Show Trades", 
                  command=self._show_trades_table).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(tables_frame, text="Show Metrics", 
                  command=self._show_metrics_table).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(tables_frame, text="Export Data", 
                  command=self._export_data).pack(side=tk.LEFT)
        
        # Performance display
        perf_frame = ttk.LabelFrame(main_frame, text="Performance Summary", padding="10")
        perf_frame.pack(fill=tk.BOTH, expand=True)
        
        self.performance_text = scrolledtext.ScrolledText(perf_frame, height=12, state=tk.DISABLED)
        self.performance_text.pack(fill=tk.BOTH, expand=True)
        
        self._update_performance_display("No performance data available")
    
    def _create_logs_tab(self):
        """Create Logs display tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Logs")
        
        main_frame = ttk.Frame(tab, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Log controls
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(controls_frame, text="Clear Logs", 
                  command=self._clear_logs).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(controls_frame, text="Export Logs", 
                  command=self._export_logs).pack(side=tk.LEFT, padx=(0, 10))
        
        # Log level filter
        ttk.Label(controls_frame, text="Level:").pack(side=tk.LEFT, padx=(20, 5))
        self.log_level_var = tk.StringVar(value="INFO")
        log_level_combo = ttk.Combobox(controls_frame, textvariable=self.log_level_var,
                                     values=["DEBUG", "INFO", "WARNING", "ERROR"], width=10)
        log_level_combo.pack(side=tk.LEFT)
        log_level_combo.state(['readonly'])
        
        # Log display
        self.logs_text = scrolledtext.ScrolledText(main_frame, height=25, state=tk.DISABLED)
        self.logs_text.pack(fill=tk.BOTH, expand=True)
        
        # Setup log handler to display in UI
        self._setup_log_handler()
    
    def _setup_drag_drop(self):
        """Setup drag and drop functionality for JSON files (Windows-compatible)."""
        try:
            # Try to use tkdnd if available (optional dependency)
            import tkdnd
            from tkdnd import TkDND, DND_FILES
            
            def drop_handler(event):
                files = event.data.split()
                for file_path in files:
                    file_path = file_path.strip('{}')  # Remove braces if present
                    if file_path.endswith('.json'):
                        self._load_params_from_file(Path(file_path))
                        break
            
            dnd = TkDND(self)
            dnd.bindtarget(self, drop_handler, DND_FILES)
            self.logger.info("Drag & drop enabled")
            
        except ImportError:
            self.logger.info("tkdnd not available, drag & drop disabled")
    
    def _setup_log_handler(self):
        """Setup log handler to display logs in UI."""
        class UILogHandler(logging.Handler):
            def __init__(self, text_widget):
                super().__init__()
                self.text_widget = text_widget
                
            def emit(self, record):
                msg = self.format(record)
                # Thread-safe UI update
                self.text_widget.after(0, lambda: self._append_log(msg))
                
            def _append_log(self, msg):
                self.text_widget.config(state=tk.NORMAL)
                self.text_widget.insert(tk.END, msg + '\n')
                self.text_widget.see(tk.END)
                self.text_widget.config(state=tk.DISABLED)
        
        # Add handler to root logger
        handler = UILogHandler(self.logs_text)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(handler)
    
    # Public API methods
    
    def load_params_from_json(self, path: Union[Path, str]) -> dict:
        """
        Load parameters from JSON file.
        
        Parameters
        ----------
        path : Path or str
            Path to JSON parameter file
            
        Returns
        -------
        dict
            Loaded parameters
            
        Raises
        ------
        FileNotFoundError
            If file doesn't exist
        ValueError
            If JSON is invalid or parameters are malformed
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Parameter file not found: {path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                params = json.load(f)
            
            # Validate and apply parameters
            self._validate_loaded_params(params)
            self._apply_params(params)
            
            self.logger.info(f"Loaded parameters from {path}")
            return params
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in parameter file: {e}")
        except Exception as e:
            self.logger.error(f"Failed to load parameters: {e}")
            raise
    
    def trigger_regenerate(self) -> None:
        """
        Trigger indicator regeneration in background thread.
        
        Calls bank.ensure() from Phase 3 to regenerate all indicators
        for current symbol/timeframe configuration.
        """
        if self._is_task_running('regenerate'):
            messagebox.showwarning("Warning", "Indicator regeneration already in progress")
            return
        
        def regenerate_task():
            task_id = 'regenerate'
            self.running_tasks.add(task_id)
            
            try:
                self._set_status("Regenerating indicators...")
                self._toggle_regenerate_ui(False)
                
                # Get current parameters
                symbol = self.symbol_var.get()
                timeframe = self.timeframe_var.get()
                bb_period = int(self.bb_period_var.get())
                bb_std = float(self.bb_std_var.get())
                atr_period = int(self.atr_period_var.get())
                
                self.logger.info(f"Starting indicator regeneration for {symbol} {timeframe}")
                start_time = time.time()
                
                # Call Phase 3 IndicatorBank.ensure() avec les bons paramètres
                if self.current_df is not None and not self.current_df.empty:
                    try:
                        # Calcul Bollinger Bands
                        bb_result = self.indicator_bank.ensure(
                            indicator_type='bollinger',
                            params={'period': bb_period, 'std': bb_std},
                            data=self.current_df,
                            symbol=symbol,
                            timeframe=timeframe
                        )
                        
                        # Calcul ATR
                        atr_result = self.indicator_bank.ensure(
                            indicator_type='atr',
                            params={'period': atr_period, 'method': 'ema'},
                            data=self.current_df,
                            symbol=symbol,
                            timeframe=timeframe
                        )
                        
                        result = bb_result is not None and atr_result is not None
                        
                    except Exception as e:
                        self.logger.error(f"Erreur calcul indicateurs: {e}")
                        result = False
                else:
                    self.logger.warning("Aucune donnée disponible pour calculer les indicateurs")
                    result = False
                
                elapsed = time.time() - start_time
                
                if result:
                    message = f"Indicators regenerated successfully in {elapsed:.2f}s"
                    self.logger.info(message)
                    self._update_indicators_status(message)
                    self._set_status("Ready")
                else:
                    message = "Indicator regeneration failed"
                    self.logger.error(message)
                    self._update_indicators_status(message)
                    self._set_status("Error")
                    
            except Exception as e:
                error_msg = f"Indicator regeneration error: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                self._update_indicators_status(error_msg)
                self._set_status("Error")
                messagebox.showerror("Error", error_msg)
                
            finally:
                self.running_tasks.discard(task_id)
                self._toggle_regenerate_ui(True)
        
        # Run in thread
        self.executor.submit(regenerate_task)
    
    def trigger_backtest(self) -> None:
        """
        Trigger backtest execution avec nouveau pipeline ThreadX.
        
        Pipeline complet :
        1. bank.ensure(...) → indicators  
        2. engine.run(df, indicators, params) → RunResult
        3. performance.summarize(result.returns, result.trades) → metrics
        4. UI update avec résultats
        """
        if self._is_task_running('backtest'):
            messagebox.showwarning("Warning", "Backtest already in progress")
            return
        
        # Validate parameters first
        if not self._validate_params():
            return
        
        def backtest_task():
            task_id = 'backtest'
            self.running_tasks.add(task_id)
            
            try:
                self._set_status("Running backtest...")
                self._toggle_backtest_ui(False)
                
                # Collect parameters
                params = self._get_current_params()
                
                self.logger.info(f"🎯 Démarrage backtest ThreadX: {params['symbol']} {params['timeframe']}")
                start_time = time.time()
                
                # === ÉTAPE 1: Préparation des données ===
                if not hasattr(self, 'current_data') or self.current_data.empty:
                    raise ValueError("Aucune donnée chargée. Veuillez charger des données d'abord.")
                
                df_1m = self.current_data.copy()
                
                # === ÉTAPE 2: Calcul des indicateurs via IndicatorBank ===
                self._set_status("Calculating indicators...")
                
                # Bollinger Bands
                bollinger_params = {
                    "period": params.get('bb_period', 20),
                    "std": params.get('bb_std', 2.0)
                }
                bollinger_result = self.indicator_bank.ensure(
                    "bollinger", bollinger_params, df_1m,
                    symbol=params['symbol'], timeframe=params['timeframe']
                )
                
                # ATR (Average True Range)
                atr_params = {"period": 14}
                atr_result = self.indicator_bank.ensure(
                    "atr", atr_params, df_1m,
                    symbol=params['symbol'], timeframe=params['timeframe']
                )
                
                indicators = {
                    "bollinger": bollinger_result,
                    "atr": atr_result
                }
                
                self.logger.info(f"✅ Indicateurs calculés: {list(indicators.keys())}")
                
                # === ÉTAPE 3: Paramètres de stratégie ===
                strategy_params = {
                    "entry_z": params.get('entry_z', 2.0),
                    "k_sl": params.get('k_sl', 1.5),
                    "leverage": params.get('leverage', 3.0),
                    "initial_capital": 10000.0,
                    "fees_bps": 10.0,  # 10 bps = 0.1%
                    "slip_bps": 5.0    # 5 bps slippage
                }
                
                # === ÉTAPE 4: Exécution backtest via BacktestEngine ===
                self._set_status("Running backtest engine...")
                
                result = self.engine.run(
                    df_1m=df_1m,
                    indicators=indicators,
                    params=strategy_params,
                    symbol=params['symbol'],
                    timeframe=params['timeframe'],
                    seed=42,
                    use_gpu=self.use_gpu_var.get()
                )
                
                # === ÉTAPE 5: Calcul métriques via PerformanceCalculator ===
                self._set_status("Calculating performance metrics...")
                
                metrics = self.performance.summarize(
                    trades=result.trades,
                    returns=result.returns
                )
                elapsed = time.time() - start_time
                
                # === ÉTAPE 6: Stockage résultats ===
                self.last_returns = result.returns
                self.last_trades = result.trades
                self.last_metrics = metrics
                self.last_equity = result.equity
                self.last_result = result  # Stockage complet du RunResult
                
                # === ÉTAPE 7: Logs et UI update ===
                self.logger.info(f"✅ Backtest terminé en {elapsed:.2f}s")
                self.logger.info(f"   Trades: {len(result.trades)}")
                self.logger.info(f"   Equity finale: ${result.equity.iloc[-1]:,.2f}")
                self.logger.info(f"   Return total: {((result.equity.iloc[-1] / result.equity.iloc[0]) - 1) * 100:.2f}%")
                self.logger.info(f"   Sharpe: {metrics.get('sharpe_ratio', 0):.3f}")
                self.logger.info(f"   Max DD: {metrics.get('max_drawdown', 0):.2%}")
                
                # Format résultats pour affichage
                results_summary = f"""🎯 BACKTEST THREADX TERMINÉ
════════════════════════════════════════════════

⏱️  Durée d'exécution: {elapsed:.2f}s
🎲  Seed: 42 (déterministe)
🖥️  Backend: {result.meta.get('backend', 'unknown')}
📊  Points de données: {result.meta.get('data_points', 0):,}

📈 RÉSULTATS DE TRADING
────────────────────────────────────────────────
💰 Capital initial: ${strategy_params['initial_capital']:,.2f}
💎 Equity finale: ${result.equity.iloc[-1]:,.2f}
📊 Return total: {((result.equity.iloc[-1] / result.equity.iloc[0]) - 1) * 100:.2f}%
🎯 Trades exécutés: {len(result.trades)}
📅 Période: {df_1m.index[0].strftime('%Y-%m-%d')} → {df_1m.index[-1].strftime('%Y-%m-%d')}

🔗 Métadonnées device: {result.meta.get('devices', [])}
⚡ Performance flags: {result.meta.get('performance_flags', [])}
"""
                
                self._update_results_display(results_summary)
                self._update_performance_display(self._format_metrics(metrics))
                self._set_status("Ready")
                
                # Switch to performance tab pour voir résultats
                self.notebook.select(4)  # Performance tab index
                
            except Exception as e:
                error_msg = f"❌ Erreur backtest ThreadX: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                self._update_results_display(error_msg)
                self._set_status("Error")
                messagebox.showerror("Backtest Error", error_msg)
                
            finally:
                self.running_tasks.discard(task_id)
                self._toggle_backtest_ui(True)
        
        # Update UI pour démarrage
        self._update_backtest_ui_state("running")
        
        # Run in thread
        self.executor.submit(backtest_task)
    
    def export_results(self, dir_path: Union[Path, str]) -> List[Path]:
        """
        Export all results to specified directory.
        
        Parameters
        ----------
        dir_path : Path or str
            Directory to export results to
            
        Returns
        -------
        List[Path]
            List of exported file paths
            
        Raises
        ------
        ValueError
            If no results available to export
        """
        if not self._has_results():
            raise ValueError("No results available to export")
        
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = []
        
        try:
            # Export trades
            if self.last_trades is not None:
                trades_path = dir_path / "trades.csv"
                self.last_trades.to_csv(trades_path, index=False)
                exported_files.append(trades_path)
            
            # Export metrics
            if self.last_metrics is not None:
                metrics_path = dir_path / "metrics.json"
                with open(metrics_path, 'w') as f:
                    json.dump(self.last_metrics, f, indent=2, default=str)
                exported_files.append(metrics_path)
            
            # Export returns
            if self.last_returns is not None:
                returns_path = dir_path / "returns.csv"
                self.last_returns.to_csv(returns_path)
                exported_files.append(returns_path)
            
            # Export parameters
            params = self._get_current_params()
            params_path = dir_path / "parameters.json"
            with open(params_path, 'w') as f:
                json.dump(params, f, indent=2)
            exported_files.append(params_path)
            
            self.logger.info(f"Exported {len(exported_files)} files to {dir_path}")
            return exported_files
            
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            raise
    
    # Private helper methods
    
    def _is_task_running(self, task_name: str) -> bool:
        """Check if specific task is running."""
        return task_name in self.running_tasks
    
    def _set_status(self, message: str):
        """Update status bar message (thread-safe)."""
        self.after(0, lambda: self.status_var.set(message))
    
    def _toggle_regenerate_ui(self, enabled: bool):
        """Toggle regenerate UI elements (thread-safe)."""
        def toggle():
            state = tk.NORMAL if enabled else tk.DISABLED
            self.regenerate_btn.config(state=state)
            if enabled:
                self.indicators_progress.stop()
            else:
                self.indicators_progress.start()
        
        self.after(0, toggle)
    
    def _toggle_backtest_ui(self, enabled: bool):
        """Toggle backtest UI elements (thread-safe)."""
        def toggle():
            state = tk.NORMAL if enabled else tk.DISABLED
            self.backtest_btn.config(state=state)
            if enabled:
                self.backtest_progress.stop()
            else:
                self.backtest_progress.start()
        
        self.after(0, toggle)
    
    def _get_current_params(self) -> dict:
        """Get current parameter values as dictionary."""
        return {
            'symbol': self.symbol_var.get(),
            'timeframe': self.timeframe_var.get(),
            'start_date': self.start_date_var.get(),
            'end_date': self.end_date_var.get(),
            'bb_period': int(self.bb_period_var.get()),
            'bb_std': float(self.bb_std_var.get()),
            'atr_period': int(self.atr_period_var.get()),
            'entry_z': float(self.entry_z_var.get()),
            'k_sl': float(self.k_sl_var.get()),
            'leverage': int(self.leverage_var.get()),
            'risk': float(self.risk_var.get()),
            'trail_k': float(self.trail_k_var.get())
        }
    
    def _validate_params(self) -> bool:
        """Validate current parameters."""
        try:
            params = self._get_current_params()
            
            # Basic validation
            if params['bb_period'] <= 0 or params['bb_period'] > 200:
                raise ValueError("BB Period must be between 1 and 200")
            if params['bb_std'] <= 0 or params['bb_std'] > 5:
                raise ValueError("BB Std must be between 0 and 5")
            if params['entry_z'] <= 0 or params['entry_z'] > 5:
                raise ValueError("Entry Z must be between 0 and 5")
            if params['leverage'] <= 0 or params['leverage'] > 20:
                raise ValueError("Leverage must be between 1 and 20")
            if params['risk'] <= 0 or params['risk'] > 0.5:
                raise ValueError("Risk must be between 0 and 50%")
            
            return True
            
        except ValueError as e:
            messagebox.showerror("Parameter Error", str(e))
            return False
        except Exception as e:
            messagebox.showerror("Validation Error", f"Parameter validation failed: {e}")
            return False
    
    def _has_results(self) -> bool:
        """Check if results are available."""
        return (self.last_returns is not None and 
                self.last_trades is not None and 
                self.last_metrics is not None)
    
    def _format_metrics(self, metrics: dict) -> str:
        """Format metrics dictionary for display."""
        lines = ["Performance Metrics", "=" * 20, ""]
        
        key_metrics = [
            ('Final Equity', 'final_equity', '${:,.2f}'),
            ('Total Return', 'total_return', '{:.2%}'),
            ('CAGR', 'cagr', '{:.2%}'),
            ('Sharpe Ratio', 'sharpe', '{:.3f}'),
            ('Max Drawdown', 'max_drawdown', '{:.2%}'),
            ('Total Trades', 'total_trades', '{:,}'),
            ('Win Rate', 'win_rate', '{:.2%}')
        ]
        
        for label, key, fmt in key_metrics:
            value = metrics.get(key, 0)
            try:
                formatted = fmt.format(value)
            except (ValueError, TypeError):
                formatted = str(value)
            lines.append(f"{label:.<20} {formatted}")
        
        return '\n'.join(lines)
    
    # UI event handlers
    
    def _load_params_dialog(self):
        """Open file dialog to load parameters."""
        file_path = filedialog.askopenfilename(
            title="Load Parameters",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            self._load_params_from_file(Path(file_path))
    
    def _load_params_from_file(self, path: Path):
        """Load parameters from file with error handling."""
        try:
            params = self.load_params_from_json(path)
            messagebox.showinfo("Success", f"Parameters loaded from {path.name}")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load parameters: {e}")
    
    def _save_params_dialog(self):
        """Open file dialog to save parameters."""
        file_path = filedialog.asksaveasfilename(
            title="Save Parameters",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                params = self._get_current_params()
                with open(file_path, 'w') as f:
                    json.dump(params, f, indent=2)
                messagebox.showinfo("Success", f"Parameters saved to {Path(file_path).name}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save parameters: {e}")
    
    def _reset_params(self):
        """Reset parameters to defaults."""
        self.symbol_var.set("BTCUSDC")
        self.timeframe_var.set("15m")
        self.start_date_var.set("2024-01-01")
        self.end_date_var.set("2024-12-31")
        self.bb_period_var.set("20")
        self.bb_std_var.set("2.0")
        self.atr_period_var.set("14")
        self.entry_z_var.set("2.0")
        self.k_sl_var.set("1.5")
        self.leverage_var.set("3")
        self.risk_var.set("2.0")
        self.trail_k_var.set("1.0")
        
        self._update_params_display()
        messagebox.showinfo("Reset", "Parameters reset to defaults")
    
    def _validate_loaded_params(self, params: dict):
        """Validate loaded parameters structure."""
        required_keys = ['symbol', 'timeframe', 'bb_period', 'bb_std', 'entry_z', 'leverage', 'risk']
        missing_keys = [key for key in required_keys if key not in params]
        
        if missing_keys:
            raise ValueError(f"Missing required parameters: {missing_keys}")
    
    def _apply_params(self, params: dict):
        """Apply loaded parameters to UI."""
        self.symbol_var.set(params.get('symbol', 'BTCUSDC'))
        self.timeframe_var.set(params.get('timeframe', '15m'))
        self.start_date_var.set(params.get('start_date', '2024-01-01'))
        self.end_date_var.set(params.get('end_date', '2024-12-31'))
        self.bb_period_var.set(str(params.get('bb_period', 20)))
        self.bb_std_var.set(str(params.get('bb_std', 2.0)))
        self.atr_period_var.set(str(params.get('atr_period', 14)))
        self.entry_z_var.set(str(params.get('entry_z', 2.0)))
        self.k_sl_var.set(str(params.get('k_sl', 1.5)))
        self.leverage_var.set(str(params.get('leverage', 3)))
        self.risk_var.set(str(params.get('risk', 2.0)))
        self.trail_k_var.set(str(params.get('trail_k', 1.0)))
        
        self._update_params_display()
    
    def _update_data_info(self, message: str):
        """Update data info display."""
        self.data_info_text.config(state=tk.NORMAL)
        self.data_info_text.delete(1.0, tk.END)
        self.data_info_text.insert(tk.END, message)
        self.data_info_text.config(state=tk.DISABLED)
    
    def _update_indicators_status(self, message: str):
        """Update indicators status display."""
        self.indicators_text.config(state=tk.NORMAL)
        self.indicators_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.indicators_text.see(tk.END)
        self.indicators_text.config(state=tk.DISABLED)
    
    def _update_params_display(self):
        """Update parameters display."""
        params = self._get_current_params()
        params_str = json.dumps(params, indent=2)
        
        self.params_text.config(state=tk.NORMAL)
        self.params_text.delete(1.0, tk.END)
        self.params_text.insert(tk.END, params_str)
        self.params_text.config(state=tk.DISABLED)
    
    def _update_results_display(self, message: str):
        """Update results display."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, message)
        self.results_text.config(state=tk.DISABLED)
    
    def _update_performance_display(self, message: str):
        """Update performance display."""
        self.performance_text.config(state=tk.NORMAL)
        self.performance_text.delete(1.0, tk.END)
        self.performance_text.insert(tk.END, message)
        self.performance_text.config(state=tk.DISABLED)
    
    def _clear_logs(self):
        """Clear log display."""
        self.logs_text.config(state=tk.NORMAL)
        self.logs_text.delete(1.0, tk.END)
        self.logs_text.config(state=tk.DISABLED)
    
    def _export_logs(self):
        """Export logs to file."""
        file_path = filedialog.asksaveasfilename(
            title="Export Logs",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                content = self.logs_text.get(1.0, tk.END)
                with open(file_path, 'w') as f:
                    f.write(content)
                messagebox.showinfo("Success", f"Logs exported to {Path(file_path).name}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export logs: {e}")
    
    def _pause_backtest(self):
        """Pause running backtest."""
        if self._is_task_running('backtest') and self.backtest_controller:
            self.logger.info("🎯 Backtest pause requested")
            self.backtest_controller.pause()
            self._update_backtest_ui_state("paused")
            self.backtest_status_var.set("Paused")
            self.progress_detail_var.set("Backtest paused by user")
    
    def _resume_backtest(self):
        """Resume paused backtest."""
        if self._is_task_running('backtest') and self.backtest_controller:
            self.logger.info("🎯 Backtest resume requested")
            self.backtest_controller.resume()
            self._update_backtest_ui_state("running")
            self.backtest_status_var.set("Running")
            self.progress_detail_var.set("Backtest resumed")
    
    def _stop_backtest(self):
        """Stop running backtest completely."""
        if self._is_task_running('backtest') and self.backtest_controller:
            self.logger.info("🎯 Backtest stop requested")
            self.backtest_controller.stop()
            self._update_backtest_ui_state("stopped")
            self.backtest_status_var.set("Stopped")
            self.progress_detail_var.set("Backtest stopped by user")
    
    def _update_backtest_ui_state(self, state: str):
        """Update UI buttons based on backtest state."""
        def update():
            if state == "ready":
                self.backtest_btn.config(state=tk.NORMAL)
                self.pause_btn.config(state=tk.DISABLED)
                self.resume_btn.config(state=tk.DISABLED)
                self.stop_btn.config(state=tk.DISABLED)
                self.backtest_progress.stop()
                self.backtest_status_var.set("Ready")
                self.progress_detail_var.set("")
            elif state == "running":
                self.backtest_btn.config(state=tk.DISABLED)
                self.pause_btn.config(state=tk.NORMAL)
                self.resume_btn.config(state=tk.DISABLED)
                self.stop_btn.config(state=tk.NORMAL)
                self.backtest_progress.start()
                self.backtest_status_var.set("Running")
            elif state == "paused":
                self.backtest_btn.config(state=tk.DISABLED)
                self.pause_btn.config(state=tk.DISABLED)
                self.resume_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.NORMAL)
                self.backtest_progress.stop()
                self.backtest_status_var.set("Paused")
            elif state == "stopped":
                self.backtest_btn.config(state=tk.NORMAL)
                self.pause_btn.config(state=tk.DISABLED)
                self.resume_btn.config(state=tk.DISABLED)
                self.stop_btn.config(state=tk.DISABLED)
                self.backtest_progress.stop()
                self.backtest_status_var.set("Stopped")
                
        self.after(0, update)
    
    def _cancel_backtest(self):
        """Cancel running backtest (deprecated - use _stop_backtest instead)."""
        self._stop_backtest()
    
    # Chart and table methods (delegated to other modules)
    
    def _show_equity_chart(self):
        """Show equity curve chart."""
        if not self._has_results():
            messagebox.showwarning("No Data", "No backtest results available")
            return
        
        try:
            from threadx.ui.charts import plot_equity
            
            # Utiliser l'equity calculée par le BacktestEngine
            equity = self.last_equity if hasattr(self, 'last_equity') and self.last_equity is not None else (1 + self.last_returns).cumprod() * 10000
            
            chart_path = plot_equity(equity, save_path=Path("equity_chart.png"))
            if chart_path:
                messagebox.showinfo("Chart", f"Equity chart saved to {chart_path.name}")
            
        except ImportError:
            messagebox.showerror("Error", "Chart module not available")
        except Exception as e:
            messagebox.showerror("Chart Error", f"Failed to create equity chart: {e}")
    
    def _show_drawdown_chart(self):
        """Show drawdown chart."""
        if not self._has_results():
            messagebox.showwarning("No Data", "No backtest results available")
            return
        
        try:
            from threadx.ui.charts import plot_drawdown
            
            # Utiliser l'equity calculée par le BacktestEngine
            equity = self.last_equity if hasattr(self, 'last_equity') and self.last_equity is not None else (1 + self.last_returns).cumprod() * 10000
            
            chart_path = plot_drawdown(equity, save_path=Path("drawdown_chart.png"))
            if chart_path:
                messagebox.showinfo("Chart", f"Drawdown chart saved to {chart_path.name}")
            
        except ImportError:
            messagebox.showerror("Error", "Chart module not available")
        except Exception as e:
            messagebox.showerror("Chart Error", f"Failed to create drawdown chart: {e}")
    
    def _export_charts(self):
        """Export all charts."""
        if not self._has_results():
            messagebox.showwarning("No Data", "No backtest results available")
            return
        
        dir_path = filedialog.askdirectory(title="Select Export Directory")
        if not dir_path:
            return
        
        try:
            from threadx.ui.charts import plot_equity, plot_drawdown
            
            # Calculate equity curve
            if self.last_returns is not None:
                equity = (1 + self.last_returns).cumprod() * 10000
            else:
                equity = pd.Series([10000, 11000])  # Fallback
            
            # Export charts
            exported_files = []
            
            equity_path = plot_equity(equity, save_path=Path(dir_path) / "equity_curve.png")
            if equity_path:
                exported_files.append(equity_path)
            
            drawdown_path = plot_drawdown(equity, save_path=Path(dir_path) / "drawdown.png")
            if drawdown_path:
                exported_files.append(drawdown_path)
            
            messagebox.showinfo("Export", f"Exported {len(exported_files)} charts to {Path(dir_path).name}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export charts: {e}")
    
    def _show_trades_table(self):
        """Show trades table in new window."""
        if self.last_trades is None:
            messagebox.showwarning("No Data", "No trades data available")
            return
        
        try:
            from threadx.ui.tables import render_trades_table
            
            table_window = tk.Toplevel(self)
            table_window.title("Trades Table")
            table_window.geometry("800x600")
            
            table_widget = render_trades_table(self.last_trades)
            # Note: Actual table rendering would be implemented in tables.py
            
            messagebox.showinfo("Table", "Trades table opened in new window")
            
        except ImportError:
            messagebox.showerror("Error", "Table module not available")
        except Exception as e:
            messagebox.showerror("Table Error", f"Failed to show trades table: {e}")
    
    def _show_metrics_table(self):
        """Show metrics table in new window."""
        if self.last_metrics is None:
            messagebox.showwarning("No Data", "No metrics data available")
            return
        
        try:
            from threadx.ui.tables import render_metrics_table
            
            table_window = tk.Toplevel(self)
            table_window.title("Performance Metrics")
            table_window.geometry("600x400")
            
            table_widget = render_metrics_table(self.last_metrics)
            # Note: Actual table rendering would be implemented in tables.py
            
            messagebox.showinfo("Table", "Metrics table opened in new window")
            
        except ImportError:
            messagebox.showerror("Error", "Table module not available")
        except Exception as e:
            messagebox.showerror("Table Error", f"Failed to show metrics table: {e}")
    
    def _export_data(self):
        """Export trades and metrics data."""
        if not self._has_results():
            messagebox.showwarning("No Data", "No data available to export")
            return
        
        dir_path = filedialog.askdirectory(title="Select Export Directory")
        if not dir_path:
            return
        
        try:
            exported_files = self.export_results(dir_path)
            messagebox.showinfo("Export", f"Exported {len(exported_files)} files to {Path(dir_path).name}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data: {e}")
    
    def _on_home_load_data(self):
        """Load data for selected symbol/timeframe on home tab."""
        try:
            symbol = self.home_symbol_var.get()
            timeframe = self.home_timeframe_var.get()
            
            if not symbol or not timeframe:
                messagebox.showwarning("Missing Selection", "Please select both symbol and timeframe")
                return
                
            self.status_var.set(f"Loading {symbol} {timeframe}...")
            
            # Load in background
            def load_async():
                self._load_and_display_data(symbol, timeframe)
                
            self.executor.submit(load_async)
            
        except Exception as e:
            self.logger.error(f"Error in home load data: {e}")
            messagebox.showerror("Load Error", f"Error loading data: {e}")
    
    def _on_rescan_files(self):
        """Rescan data directories for available files."""
        try:
            self.status_var.set("Rescanning data directories...")
            
            def rescan_async():
                self._scan_data_directories()
                self.after(0, self._update_home_symbols)
                self.after(0, lambda: self.status_var.set("Rescan completed"))
                
            self.executor.submit(rescan_async)
            
        except Exception as e:
            self.logger.error(f"Error rescanning files: {e}")
            messagebox.showerror("Rescan Error", f"Error rescanning: {e}")
    
    def _update_home_symbols(self):
        """Update available symbols in home tab combo box."""
        try:
            if hasattr(self, 'home_symbol_combo') and self.by_symbol:
                symbols = sorted(self.by_symbol.keys())
                self.home_symbol_combo['values'] = symbols
                
                # Set default to BTC variant if available
                for btc_variant in ['BTCUSDC', 'BTCUSDT', 'BTCUSD', 'BTC']:
                    if btc_variant in symbols:
                        self.home_symbol_var.set(btc_variant)
                        break
                else:
                    if symbols:
                        self.home_symbol_var.set(symbols[0])
                        
            # Update files display
            self._update_files_display()
            
        except Exception as e:
            self.logger.error(f"Error updating home symbols: {e}")
    
    def _update_files_display(self):
        """Update the files display in home tab."""
        try:
            if not hasattr(self, 'files_text'):
                return
                
            files_info = []
            files_info.append(f"Available Files: {len(self.available_files)}")
            files_info.append(f"Symbols: {len(self.by_symbol)}")
            files_info.append("")
            
            for symbol in sorted(self.by_symbol.keys())[:10]:  # Show first 10
                timeframes = sorted(self.by_symbol[symbol].keys())
                files_info.append(f"{symbol}: {', '.join(timeframes)}")
                
            if len(self.by_symbol) > 10:
                files_info.append(f"... and {len(self.by_symbol) - 10} more")
            
            # Update display
            self.files_text.configure(state=tk.NORMAL)
            self.files_text.delete(1.0, tk.END)
            self.files_text.insert(1.0, "\n".join(files_info))
            self.files_text.configure(state=tk.DISABLED)
            
        except Exception as e:
            self.logger.error(f"Error updating files display: {e}")
    
    def _on_start_download(self):
        """Start manual download process."""
        try:
            # Get parameters
            tokens_str = self.download_tokens_var.get().strip()
            start_date = self.download_start_var.get().strip()
            end_date = self.download_end_var.get().strip()
            verify_3h = self.verify_3h_var.get()
            dry_run = self.dry_run_var.get()
            
            if not tokens_str or not start_date or not end_date:
                messagebox.showwarning("Missing Parameters", "Please fill in all required fields")
                return
                
            # Parse tokens
            tokens = [token.strip().upper() for token in tokens_str.split(',') if token.strip()]
            
            if not tokens:
                messagebox.showwarning("No Tokens", "Please specify at least one token")
                return
                
            # Validate dates
            try:
                pd.to_datetime(start_date)
                pd.to_datetime(end_date)
            except Exception:
                messagebox.showerror("Invalid Dates", "Please use YYYY-MM-DD format for dates")
                return
            
            # Start download in background
            self._start_download_async(tokens, start_date, end_date, verify_3h, dry_run)
            
        except Exception as e:
            self.logger.error(f"Error starting download: {e}")
            messagebox.showerror("Download Error", f"Error starting download: {e}")
    
    def _start_download_async(self, tokens: List[str], start_date: str, end_date: str, 
                             verify_3h: bool, dry_run: bool):
        """Execute download in background thread."""
        try:
            # Update UI
            self.download_progress.start()
            self.status_var.set("Downloading data...")
            
            # Clear results
            self.download_results_text.configure(state=tk.NORMAL)
            self.download_results_text.delete(1.0, tk.END)
            self.download_results_text.configure(state=tk.DISABLED)
            
            def download_worker():
                results = []
                
                try:
                    if dry_run:
                        results.append("=== DRY RUN MODE - No actual downloads ===\n")
                    
                    results.append(f"Download configuration:")
                    results.append(f"- Tokens: {', '.join(tokens)}")
                    results.append(f"- Date range: {start_date} to {end_date}")
                    results.append(f"- 3h verification: {'Yes' if verify_3h else 'No'}")
                    results.append(f"- Mode: {'Dry run' if dry_run else 'Live'}")
                    results.append("")
                    
                    # Process each token
                    for i, token in enumerate(tokens):
                        results.append(f"Processing {token} ({i+1}/{len(tokens)})...")
                        
                        if not dry_run:
                            # Actual download using ingestion manager
                            try:
                                df_1m = self.ingestion_manager.download_ohlcv_1m(
                                    token, start_date, end_date
                                )
                                results.append(f"✓ {token} 1m: {len(df_1m)} bars downloaded")
                                
                                if verify_3h:
                                    df_3h = self.ingestion_manager.resample_from_1m_api(
                                        token, "3h", start_date, end_date
                                    )
                                    results.append(f"✓ {token} 3h: {len(df_3h)} bars resampled")
                                    
                                    # Basic verification
                                    if len(df_3h) > 0:
                                        results.append(f"  → 3h verification: PASSED")
                                    else:
                                        results.append(f"  → 3h verification: FAILED (empty)")
                                        
                            except Exception as e:
                                results.append(f"✗ {token}: Error - {e}")
                        else:
                            # Dry run - just simulate
                            results.append(f"✓ {token}: Would download 1m data for {start_date} to {end_date}")
                            if verify_3h:
                                results.append(f"✓ {token}: Would verify with 3h resampling")
                        
                        results.append("")
                        
                        # Update UI
                        self.after(0, lambda text="\n".join(results): self._update_download_results(text))
                    
                    results.append("=== Download completed ===")
                    
                except Exception as e:
                    results.append(f"ERROR: {e}")
                    self.logger.error(f"Download worker error: {e}")
                
                # Final UI update
                self.after(0, lambda: self.download_progress.stop())
                self.after(0, lambda: self.status_var.set("Download completed"))
                self.after(0, lambda text="\n".join(results): self._update_download_results(text))
            
            self.executor.submit(download_worker)
            
        except Exception as e:
            self.logger.error(f"Error in download async: {e}")
            self.download_progress.stop()
            self.status_var.set("Download failed")
    
    def _update_download_results(self, text: str):
        """Update download results display."""
        try:
            self.download_results_text.configure(state=tk.NORMAL)
            self.download_results_text.delete(1.0, tk.END)
            self.download_results_text.insert(1.0, text)
            self.download_results_text.see(tk.END)
            self.download_results_text.configure(state=tk.DISABLED)
        except Exception as e:
            self.logger.error(f"Error updating download results: {e}")

    def _on_closing(self):
        """Handle application closing."""
        self.logger.info("Closing ThreadX Application")
        
        # Cancel running tasks
        for task_id in list(self.running_tasks):
            self.logger.info(f"Cancelling task: {task_id}")
        
        # Shutdown executor
        self.executor.shutdown(wait=False)
        
        # Close application
        self.destroy()


def run_app() -> None:
    """
    Run the ThreadX Tkinter application.
    
    Main entry point for the ThreadX UI. Creates and runs the Tkinter
    application with proper error handling and logging.
    
    Examples
    --------
    >>> from threadx.ui import run_app
    >>> run_app()
    """
    try:
        # Setup logging
        setup_logging_once()
        logger = get_logger(__name__)
        logger.info("Starting ThreadX Application")
        
        # Create and run app
        app = ThreadXApp()
        app.mainloop()
        
        logger.info("ThreadX Application closed")
        
    except Exception as e:
        # Fallback error handling
        import traceback
        error_msg = f"Failed to start ThreadX Application: {e}\n{traceback.format_exc()}"
        
        try:
            messagebox.showerror("Startup Error", error_msg)
        except:
            print(error_msg)


if __name__ == '__main__':
    run_app()