"""
ThreadX Unified Interface v1.0
Interface unifiée inspirée de TradXPro, adaptée à ThreadX
Gestion complète des données, indicateurs et backtests ThreadX
"""

from __future__ import annotations

import os
import sys
import json
import time
import queue
import logging
import threading
import platform
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, List, Tuple, Callable, Any
from functools import lru_cache

import numpy as np
import pandas as pd

# Import ThreadX core components
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from threadx.config.loaders import load_settings, load_strategies
    from threadx.data.token_diversity_data_source import TokenDiversityDataSource
    from threadx.backtest.engine import BacktestEngine
    from threadx.indicators.bank import IndicatorBank

    THREADX_AVAILABLE = True
    print("✅ ThreadX modules loaded successfully")
except ImportError as e:
    print(f"⚠️ ThreadX modules not available: {e}")
    THREADX_AVAILABLE = False

# Tkinter GUI (optional)
TK_AVAILABLE = False
try:
    import tkinter as tk
    from tkinter.scrolledtext import ScrolledText
    from tkinter import ttk, messagebox, filedialog
    from tkinter import font as tkFont

    TK_AVAILABLE = True
except ImportError:
    tk = None
    ScrolledText = None
    ttk = None
    messagebox = None
    filedialog = None
    tkFont = None

# Progress bar
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(it, **_):
        return it


# Environment variables
IS_WINDOWS = platform.system() == "Windows"

# ThreadX paths configuration
THREADX_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = THREADX_ROOT / "data"
CACHE_ROOT = DATA_ROOT / "cache"
INDICATORS_ROOT = DATA_ROOT / "indicators"
BACKTEST_ROOT = DATA_ROOT / "backtests"
EXPORTS_ROOT = DATA_ROOT / "exports"

# Create directories
for path in [DATA_ROOT, CACHE_ROOT, INDICATORS_ROOT, BACKTEST_ROOT, EXPORTS_ROOT]:
    path.mkdir(parents=True, exist_ok=True)

# Logging setup
LOG_FILE = THREADX_ROOT / "logs" / "threadx_unified.log"
LOG_FILE.parent.mkdir(exist_ok=True)

logger = logging.getLogger("ThreadXUnified")
logger.setLevel(logging.INFO)

# File handler
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Queue for GUI logging
log_queue: queue.Queue[logging.LogRecord] = queue.Queue()


class QueueHandler(logging.Handler):
    def __init__(self, log_queue: queue.Queue[logging.LogRecord]):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        self.log_queue.put(record)


queue_handler = QueueHandler(log_queue)
queue_handler.setFormatter(formatter)
logger.addHandler(queue_handler)

# Console handler for CLI mode
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# =========================================================
#  ThreadX Core Integration
# =========================================================


class ThreadXCore:
    """Core ThreadX operations wrapper"""

    def __init__(self):
        self.config = None
        self.strategies = None
        self.data_source = None
        self.backtest_engine = None
        self.indicator_bank = None
        self.initialized = False

    def initialize(self) -> bool:
        """Initialize ThreadX components"""
        if not THREADX_AVAILABLE:
            logger.error("ThreadX modules not available")
            return False

        try:
            # Load configuration
            self.config = load_settings()
            self.strategies = load_strategies()
            logger.info("✅ ThreadX configuration loaded")

            # Initialize data source
            self.data_source = TokenDiversityDataSource(self.config)
            logger.info("✅ TokenDiversityDataSource initialized")

            # Initialize backtest engine
            self.backtest_engine = BacktestEngine(self.config)
            logger.info("✅ BacktestEngine initialized")

            # Initialize indicator bank
            self.indicator_bank = IndicatorBank()
            logger.info("✅ IndicatorBank initialized")

            self.initialized = True
            logger.info("🚀 ThreadX Core fully initialized")
            return True

        except Exception as e:
            logger.error(f"❌ ThreadX initialization failed: {e}")
            return False

    def get_available_symbols(self) -> List[str]:
        """Get available trading symbols"""
        try:
            if self.data_source and self.initialized:
                # Use ThreadX data source
                return self.data_source.get_available_symbols()
            else:
                # Fallback to manual list
                return [
                    "BTCUSDC",
                    "ETHUSDC",
                    "ADAUSDC",
                    "SOLUSDC",
                    "DOTUSDC",
                    "AVAXUSDC",
                    "MATICUSDC",
                    "LINKUSDC",
                    "UNIUSDC",
                    "ATOMUSDC",
                ]
        except Exception as e:
            logger.error(f"Error getting symbols: {e}")
            return []

    def get_available_strategies(self) -> List[str]:
        """Get available trading strategies"""
        try:
            if self.strategies:
                return list(self.strategies.keys())
            return ["bb_atr", "rsi_crossover", "macd_strategy"]
        except Exception as e:
            logger.error(f"Error getting strategies: {e}")
            return []

    def run_backtest(
        self, strategy: str, symbols: List[str], timeframe: str = "1h", **kwargs
    ) -> Dict[str, Any]:
        """Run backtest with given parameters"""
        try:
            if not self.initialized:
                raise ValueError("ThreadX Core not initialized")

            logger.info(f"🎯 Running backtest: {strategy} on {len(symbols)} symbols")

            # Prepare backtest parameters
            params = {
                "strategy": strategy,
                "symbols": symbols,
                "timeframe": timeframe,
                **kwargs,
            }

            # Run backtest
            results = self.backtest_engine.run(params)

            logger.info("✅ Backtest completed successfully")
            return results

        except Exception as e:
            logger.error(f"❌ Backtest failed: {e}")
            return {"error": str(e), "success": False}


# Global ThreadX core instance
threadx_core = ThreadXCore()

# =========================================================
#  ThreadX Unified GUI Application
# =========================================================


class ThreadXUnifiedApp:
    """Main ThreadX Unified Application"""

    def __init__(self):
        if not TK_AVAILABLE:
            raise ImportError("Tkinter not available - GUI mode unavailable")

        self.root = tk.Tk()
        self.root.title("ThreadX Unified Interface v1.0")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)

        # Variables
        self.log_text = None
        self.progress_var = None
        self.status_var = None
        self.notebook = None

        # ThreadX integration
        self.threadx_ready = False

        # Setup
        self.setup_styles()
        self.setup_ui()
        self.setup_logging_display()
        self.init_threadx()

    def setup_styles(self):
        """Configure visual styles"""
        style = ttk.Style()

        # Try modern theme
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")

        # Custom styles
        style.configure("Title.TLabel", font=("Arial", 12, "bold"))
        style.configure("Header.TLabel", font=("Arial", 10, "bold"))
        style.configure("Success.TLabel", foreground="green")
        style.configure("Error.TLabel", foreground="red")
        style.configure("Warning.TLabel", foreground="orange")

    def setup_ui(self):
        """Setup main user interface"""
        # Main menu
        self.create_menu()

        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Header
        self.create_header(main_frame)

        # Main content (notebook)
        self.create_notebook(main_frame)

        # Status bar
        self.create_status_bar()

    def create_menu(self):
        """Create application menu"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(
            label="📁 Open Data Folder", command=self.open_data_folder
        )
        file_menu.add_command(label="💾 Export Results", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="❌ Exit", command=self.root.quit)

        # ThreadX menu
        threadx_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="ThreadX", menu=threadx_menu)
        threadx_menu.add_command(
            label="🔄 Reinitialize Core", command=self.init_threadx
        )
        threadx_menu.add_command(label="⚙️ Configuration", command=self.show_config)
        threadx_menu.add_command(label="🧪 Run Tests", command=self.run_tests)

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="📊 Data Manager", command=self.open_data_manager)
        tools_menu.add_command(
            label="📈 Indicator Calculator", command=self.open_indicator_calc
        )
        tools_menu.add_command(label="🧹 Clean Cache", command=self.clean_cache)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="❓ About", command=self.show_about)
        help_menu.add_command(label="📚 Documentation", command=self.show_docs)

    def create_header(self, parent):
        """Create application header"""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        # Title
        title_label = ttk.Label(
            header_frame, text="ThreadX Unified Interface", style="Title.TLabel"
        )
        title_label.pack(side=tk.LEFT)

        # Status indicator
        self.status_indicator = ttk.Label(
            header_frame, text="⚪ Starting...", style="Header.TLabel"
        )
        self.status_indicator.pack(side=tk.RIGHT)

    def create_notebook(self, parent):
        """Create main notebook with tabs"""
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Tab 1: Dashboard
        self.create_dashboard_tab()

        # Tab 2: Data Management
        self.create_data_tab()

        # Tab 3: Backtesting
        self.create_backtest_tab()

        # Tab 4: Indicators
        self.create_indicators_tab()

        # Tab 5: Optimization
        self.create_optimization_tab()

        # Tab 6: Logs
        self.create_logs_tab()

    def create_dashboard_tab(self):
        """Create dashboard tab"""
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="🏠 Dashboard")

        # Main dashboard content
        dashboard_container = ttk.Frame(tab_frame)
        dashboard_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Welcome section
        welcome_frame = ttk.LabelFrame(
            dashboard_container, text="Welcome to ThreadX", padding=20
        )
        welcome_frame.pack(fill=tk.X, pady=(0, 20))

        welcome_text = """
🚀 ThreadX Unified Interface v1.0

This interface provides comprehensive access to all ThreadX functionality:
• Data management and validation
• Strategy backtesting and optimization  
• Technical indicator calculation
• Performance monitoring and reporting

Status: Ready for trading analysis
        """

        ttk.Label(welcome_frame, text=welcome_text.strip(), justify=tk.LEFT).pack(
            anchor=tk.W
        )

        # Quick actions
        actions_frame = ttk.LabelFrame(
            dashboard_container, text="Quick Actions", padding=20
        )
        actions_frame.pack(fill=tk.X, pady=(0, 20))

        buttons_frame = ttk.Frame(actions_frame)
        buttons_frame.pack(fill=tk.X)

        ttk.Button(
            buttons_frame, text="📊 Load Data", command=lambda: self.notebook.select(1)
        ).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(
            buttons_frame,
            text="🎯 Run Backtest",
            command=lambda: self.notebook.select(2),
        ).pack(side=tk.LEFT, padx=10)
        ttk.Button(
            buttons_frame,
            text="📈 Calculate Indicators",
            command=lambda: self.notebook.select(3),
        ).pack(side=tk.LEFT, padx=10)

        # System status
        status_frame = ttk.LabelFrame(
            dashboard_container, text="System Status", padding=20
        )
        status_frame.pack(fill=tk.BOTH, expand=True)

        self.status_text = tk.Text(
            status_frame, height=10, wrap=tk.WORD, font=("Consolas", 9)
        )
        status_scroll = ttk.Scrollbar(
            status_frame, orient=tk.VERTICAL, command=self.status_text.yview
        )
        self.status_text.configure(yscrollcommand=status_scroll.set)

        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        status_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def create_data_tab(self):
        """Create data management tab"""
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="📊 Data")

        # Data management interface
        ttk.Label(tab_frame, text="Data Management", style="Title.TLabel").pack(pady=10)

        # Data sources
        sources_frame = ttk.LabelFrame(tab_frame, text="Data Sources", padding=10)
        sources_frame.pack(fill=tk.X, padx=20, pady=10)

        ttk.Button(
            sources_frame, text="🔄 Refresh Symbols", command=self.refresh_symbols
        ).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(sources_frame, text="📥 Import Data", command=self.import_data).pack(
            side=tk.LEFT, padx=10
        )
        ttk.Button(
            sources_frame, text="✅ Validate Data", command=self.validate_data
        ).pack(side=tk.LEFT, padx=10)

        # Symbols list
        symbols_frame = ttk.LabelFrame(tab_frame, text="Available Symbols", padding=10)
        symbols_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Symbols treeview
        columns = ("Symbol", "Status", "Records", "Last Update")
        self.symbols_tree = ttk.Treeview(
            symbols_frame, columns=columns, show="headings", height=15
        )

        for col in columns:
            self.symbols_tree.heading(col, text=col)
            self.symbols_tree.column(col, width=150)

        symbols_scroll = ttk.Scrollbar(
            symbols_frame, orient=tk.VERTICAL, command=self.symbols_tree.yview
        )
        self.symbols_tree.configure(yscrollcommand=symbols_scroll.set)

        self.symbols_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        symbols_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def create_backtest_tab(self):
        """Create backtesting tab"""
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="🎯 Backtest")

        ttk.Label(tab_frame, text="Strategy Backtesting", style="Title.TLabel").pack(
            pady=10
        )

        # Parameters frame
        params_frame = ttk.LabelFrame(tab_frame, text="Backtest Parameters", padding=15)
        params_frame.pack(fill=tk.X, padx=20, pady=10)

        # Strategy selection
        strategy_frame = ttk.Frame(params_frame)
        strategy_frame.pack(fill=tk.X, pady=5)
        ttk.Label(strategy_frame, text="Strategy:").pack(side=tk.LEFT)
        self.strategy_var = tk.StringVar()
        self.strategy_combo = ttk.Combobox(
            strategy_frame, textvariable=self.strategy_var, width=20, state="readonly"
        )
        self.strategy_combo.pack(side=tk.LEFT, padx=(10, 0))

        # Symbol selection
        symbols_frame = ttk.Frame(params_frame)
        symbols_frame.pack(fill=tk.X, pady=5)
        ttk.Label(symbols_frame, text="Symbols:").pack(side=tk.LEFT)
        self.symbols_var = tk.StringVar()
        symbols_entry = ttk.Entry(
            symbols_frame, textvariable=self.symbols_var, width=40
        )
        symbols_entry.pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(symbols_frame, text="Select", command=self.select_symbols).pack(
            side=tk.LEFT, padx=(5, 0)
        )

        # Timeframe
        tf_frame = ttk.Frame(params_frame)
        tf_frame.pack(fill=tk.X, pady=5)
        ttk.Label(tf_frame, text="Timeframe:").pack(side=tk.LEFT)
        self.timeframe_var = tk.StringVar(value="1h")
        tf_combo = ttk.Combobox(
            tf_frame,
            textvariable=self.timeframe_var,
            values=["5m", "15m", "30m", "1h", "4h", "1d"],
            width=10,
            state="readonly",
        )
        tf_combo.pack(side=tk.LEFT, padx=(10, 0))

        # Run button
        ttk.Button(
            params_frame, text="🚀 Run Backtest", command=self.run_backtest_action
        ).pack(pady=10)

        # Results frame
        results_frame = ttk.LabelFrame(tab_frame, text="Backtest Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.results_text = ScrolledText(results_frame, height=20, font=("Consolas", 9))
        self.results_text.pack(fill=tk.BOTH, expand=True)

    def create_indicators_tab(self):
        """Create indicators tab"""
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="📈 Indicators")

        ttk.Label(tab_frame, text="Technical Indicators", style="Title.TLabel").pack(
            pady=10
        )

        # Indicator selection
        indicator_frame = ttk.LabelFrame(
            tab_frame, text="Calculate Indicators", padding=15
        )
        indicator_frame.pack(fill=tk.X, padx=20, pady=10)

        # Available indicators
        indicators = [
            "RSI",
            "MACD",
            "Bollinger Bands",
            "ATR",
            "EMA",
            "SMA",
            "Stochastic",
        ]

        self.selected_indicators = {}
        for i, indicator in enumerate(indicators):
            var = tk.BooleanVar()
            self.selected_indicators[indicator] = var
            ttk.Checkbutton(indicator_frame, text=indicator, variable=var).grid(
                row=i // 3, column=i % 3, sticky=tk.W, padx=10, pady=2
            )

        ttk.Button(
            indicator_frame,
            text="📊 Calculate Selected",
            command=self.calculate_indicators,
        ).pack(pady=15)

        # Results display
        results_frame = ttk.LabelFrame(
            tab_frame, text="Calculation Results", padding=10
        )
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.indicators_text = ScrolledText(
            results_frame, height=15, font=("Consolas", 9)
        )
        self.indicators_text.pack(fill=tk.BOTH, expand=True)

    def create_optimization_tab(self):
        """Create optimization tab"""
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="⚡ Optimization")

        ttk.Label(tab_frame, text="Strategy Optimization", style="Title.TLabel").pack(
            pady=10
        )

        # Optimization controls
        opt_frame = ttk.LabelFrame(tab_frame, text="Optimization Settings", padding=15)
        opt_frame.pack(fill=tk.X, padx=20, pady=10)

        ttk.Label(opt_frame, text="Optimization coming soon...").pack(pady=20)

    def create_logs_tab(self):
        """Create logs tab"""
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="📋 Logs")

        # Log display
        self.log_text = ScrolledText(tab_frame, height=30, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Log controls
        controls_frame = ttk.Frame(tab_frame)
        controls_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        ttk.Button(controls_frame, text="🧹 Clear Logs", command=self.clear_logs).pack(
            side=tk.LEFT
        )
        ttk.Button(controls_frame, text="💾 Save Logs", command=self.save_logs).pack(
            side=tk.LEFT, padx=(10, 0)
        )

    def create_status_bar(self):
        """Create status bar"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_var = tk.StringVar()
        self.status_var.set("Ready")

        status_label = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=5,
        )
        status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            status_frame, variable=self.progress_var, length=200
        )
        self.progress_bar.pack(side=tk.RIGHT, padx=(0, 5))

    def setup_logging_display(self):
        """Setup logging display in GUI"""

        def check_log_queue():
            try:
                while True:
                    record = log_queue.get_nowait()
                    if self.log_text:
                        formatted = queue_handler.format(record)
                        self.log_text.insert(tk.END, formatted + "\n")
                        self.log_text.see(tk.END)

                        # Also show in status text
                        if hasattr(self, "status_text"):
                            self.status_text.insert(tk.END, formatted + "\n")
                            self.status_text.see(tk.END)
            except queue.Empty:
                pass

            # Schedule next check
            self.root.after(100, check_log_queue)

        self.root.after(100, check_log_queue)

    def init_threadx(self):
        """Initialize ThreadX core in background"""

        def init_worker():
            try:
                self.update_status("🔄 Initializing ThreadX Core...")
                success = threadx_core.initialize()

                if success:
                    self.root.after(0, lambda: self.threadx_init_success())
                else:
                    self.root.after(0, lambda: self.threadx_init_failed())

            except Exception as e:
                self.root.after(0, lambda: self.threadx_init_error(str(e)))

        threading.Thread(target=init_worker, daemon=True).start()

    def threadx_init_success(self):
        """Handle successful ThreadX initialization"""
        self.threadx_ready = True
        self.status_indicator.config(text="🟢 ThreadX Ready", style="Success.TLabel")
        self.update_status("✅ ThreadX Core initialized successfully")

        # Populate combo boxes
        self.populate_strategies()
        self.refresh_symbols()

    def threadx_init_failed(self):
        """Handle failed ThreadX initialization"""
        self.status_indicator.config(text="🔴 ThreadX Failed", style="Error.TLabel")
        self.update_status("❌ ThreadX initialization failed")

    def threadx_init_error(self, error: str):
        """Handle ThreadX initialization error"""
        self.status_indicator.config(text="🟡 ThreadX Error", style="Warning.TLabel")
        self.update_status(f"⚠️ ThreadX error: {error}")

    def populate_strategies(self):
        """Populate strategy combo box"""
        try:
            strategies = threadx_core.get_available_strategies()
            self.strategy_combo["values"] = strategies
            if strategies:
                self.strategy_var.set(strategies[0])
        except Exception as e:
            logger.error(f"Error populating strategies: {e}")

    def refresh_symbols(self):
        """Refresh symbols list"""

        def refresh_worker():
            try:
                self.update_status("🔄 Refreshing symbols...")
                symbols = threadx_core.get_available_symbols()

                self.root.after(0, lambda: self.update_symbols_tree(symbols))

            except Exception as e:
                logger.error(f"Error refreshing symbols: {e}")

        threading.Thread(target=refresh_worker, daemon=True).start()

    def update_symbols_tree(self, symbols: List[str]):
        """Update symbols treeview"""
        # Clear existing items
        for item in self.symbols_tree.get_children():
            self.symbols_tree.delete(item)

        # Add symbols
        for symbol in symbols:
            self.symbols_tree.insert(
                "", tk.END, values=(symbol, "Available", "N/A", "N/A")
            )

        self.update_status(f"✅ Loaded {len(symbols)} symbols")

    def run_backtest_action(self):
        """Run backtest action"""
        if not self.threadx_ready:
            messagebox.showerror(
                "Error", "ThreadX not ready. Please wait for initialization."
            )
            return

        strategy = self.strategy_var.get()
        symbols_text = self.symbols_var.get()
        timeframe = self.timeframe_var.get()

        if not strategy or not symbols_text:
            messagebox.showerror("Error", "Please select strategy and symbols.")
            return

        symbols = [s.strip().upper() for s in symbols_text.split(",")]

        def backtest_worker():
            try:
                self.update_status(f"🎯 Running backtest: {strategy}")
                self.update_progress(0)

                results = threadx_core.run_backtest(
                    strategy=strategy, symbols=symbols, timeframe=timeframe
                )

                self.root.after(0, lambda: self.display_backtest_results(results))

            except Exception as e:
                self.root.after(0, lambda: self.backtest_error(str(e)))

        threading.Thread(target=backtest_worker, daemon=True).start()

    def display_backtest_results(self, results: Dict[str, Any]):
        """Display backtest results"""
        self.results_text.delete(1.0, tk.END)

        if results.get("success", True):
            self.results_text.insert(tk.END, "🎉 Backtest completed successfully!\n\n")
            self.results_text.insert(tk.END, json.dumps(results, indent=2))
        else:
            self.results_text.insert(
                tk.END, f"❌ Backtest failed: {results.get('error', 'Unknown error')}"
            )

        self.update_status("✅ Backtest completed")
        self.update_progress(100)

    def backtest_error(self, error: str):
        """Handle backtest error"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"❌ Backtest error: {error}")
        self.update_status("❌ Backtest failed")

    def calculate_indicators(self):
        """Calculate selected indicators"""
        selected = [name for name, var in self.selected_indicators.items() if var.get()]

        if not selected:
            messagebox.showwarning("Warning", "Please select at least one indicator.")
            return

        self.indicators_text.delete(1.0, tk.END)
        self.indicators_text.insert(
            tk.END, f"📊 Calculating indicators: {', '.join(selected)}\n\n"
        )
        self.indicators_text.insert(
            tk.END, "Indicator calculation functionality coming soon...\n"
        )

    def update_status(self, message: str):
        """Update status bar"""
        if self.status_var:
            self.status_var.set(message)

    def update_progress(self, value: float):
        """Update progress bar"""
        if self.progress_var:
            self.progress_var.set(value)

    def select_symbols(self):
        """Open symbol selection dialog"""
        # Simple implementation - could be enhanced with multi-select dialog
        result = tk.simpledialog.askstring(
            "Symbols", "Enter symbols (comma-separated):"
        )
        if result:
            self.symbols_var.set(result)

    def import_data(self):
        """Import data action"""
        messagebox.showinfo("Info", "Data import functionality coming soon...")

    def validate_data(self):
        """Validate data action"""
        messagebox.showinfo("Info", "Data validation functionality coming soon...")

    def open_data_folder(self):
        """Open data folder"""
        try:
            if IS_WINDOWS:
                os.startfile(str(DATA_ROOT))
            else:
                os.system(f"xdg-open {DATA_ROOT}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open data folder: {e}")

    def export_results(self):
        """Export results"""
        messagebox.showinfo("Info", "Export functionality coming soon...")

    def open_data_manager(self):
        """Open data manager"""
        messagebox.showinfo("Info", "Opening data manager...")

    def open_indicator_calc(self):
        """Open indicator calculator"""
        messagebox.showinfo("Info", "Opening indicator calculator...")

    def clean_cache(self):
        """Clean cache"""
        result = messagebox.askyesno("Confirm", "Clean all cache files?")
        if result:
            try:
                import shutil

                if CACHE_ROOT.exists():
                    shutil.rmtree(CACHE_ROOT)
                    CACHE_ROOT.mkdir()
                messagebox.showinfo("Success", "Cache cleaned successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Cache cleaning failed: {e}")

    def show_config(self):
        """Show configuration"""
        config_text = "ThreadX Configuration:\n\n"
        config_text += f"ThreadX Root: {THREADX_ROOT}\n"
        config_text += f"Data Root: {DATA_ROOT}\n"
        config_text += f"Cache Root: {CACHE_ROOT}\n"
        config_text += f"ThreadX Ready: {self.threadx_ready}\n"

        messagebox.showinfo("Configuration", config_text)

    def run_tests(self):
        """Run ThreadX tests"""
        messagebox.showinfo("Info", "Running ThreadX tests...")

    def show_about(self):
        """Show about dialog"""
        about_text = """ThreadX Unified Interface v1.0

A comprehensive trading analysis platform inspired by TradXPro
and integrated with ThreadX core functionality.

Features:
• Data management and validation
• Strategy backtesting
• Technical indicator calculation
• Performance optimization
• Real-time monitoring

© 2024 ThreadX Project"""

        messagebox.showinfo("About", about_text)

    def show_docs(self):
        """Show documentation"""
        messagebox.showinfo("Documentation", "Documentation will be available soon...")

    def clear_logs(self):
        """Clear log display"""
        if self.log_text:
            self.log_text.delete(1.0, tk.END)

    def save_logs(self):
        """Save logs to file"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".log",
                filetypes=[("Log files", "*.log"), ("Text files", "*.txt")],
            )
            if filename:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(self.log_text.get(1.0, tk.END))
                messagebox.showinfo("Success", f"Logs saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save logs: {e}")

    def run(self):
        """Run the application"""
        logger.info("🚀 Starting ThreadX Unified Interface")
        self.root.mainloop()


# =========================================================
#  CLI Mode Functions
# =========================================================


def run_cli_mode():
    """Run in CLI mode when GUI not available"""
    print("=" * 60)
    print("ThreadX Unified Interface - CLI Mode")
    print("=" * 60)

    # Initialize ThreadX
    print("🔄 Initializing ThreadX Core...")
    success = threadx_core.initialize()

    if success:
        print("✅ ThreadX Core initialized successfully")

        # Show available options
        print("\nAvailable actions:")
        print("1. List available symbols")
        print("2. List available strategies")
        print("3. Run sample backtest")
        print("4. Exit")

        while True:
            try:
                choice = input("\nSelect action (1-4): ").strip()

                if choice == "1":
                    symbols = threadx_core.get_available_symbols()
                    print(f"\nAvailable symbols ({len(symbols)}):")
                    for symbol in symbols[:10]:  # Show first 10
                        print(f"  • {symbol}")
                    if len(symbols) > 10:
                        print(f"  ... and {len(symbols) - 10} more")

                elif choice == "2":
                    strategies = threadx_core.get_available_strategies()
                    print(f"\nAvailable strategies ({len(strategies)}):")
                    for strategy in strategies:
                        print(f"  • {strategy}")

                elif choice == "3":
                    print("\n🎯 Running sample backtest...")
                    results = threadx_core.run_backtest(
                        strategy="bb_atr",
                        symbols=["BTCUSDC", "ETHUSDC"],
                        timeframe="1h",
                    )
                    print(f"Results: {results}")

                elif choice == "4":
                    print("👋 Goodbye!")
                    break

                else:
                    print("Invalid choice. Please select 1-4.")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    else:
        print("❌ ThreadX initialization failed")


# =========================================================
#  Main Entry Point
# =========================================================


def main():
    """Main application entry point"""
    print("🚀 ThreadX Unified Interface v1.0")
    print(f"Platform: {platform.system()}")
    print(f"ThreadX Available: {'✅' if THREADX_AVAILABLE else '❌'}")
    print(f"GUI Available: {'✅' if TK_AVAILABLE else '❌'}")

    if TK_AVAILABLE:
        try:
            app = ThreadXUnifiedApp()
            app.run()
        except Exception as e:
            print(f"❌ GUI failed to start: {e}")
            print("🔄 Falling back to CLI mode...")
            run_cli_mode()
    else:
        print("🖥️ Running in CLI mode (Tkinter not available)")
        run_cli_mode()


if __name__ == "__main__":
    main()
