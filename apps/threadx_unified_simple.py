"""
ThreadX Unified Interface v1.0 - Simplified Version
Interface unifiée inspirée de TradXPro, adaptée à ThreadX
"""

import os
import sys
import json
import queue
import logging
import threading
import platform
from pathlib import Path
from typing import Dict, List, Any

# ThreadX integration - graceful fallback
THREADX_AVAILABLE = False
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from threadx.config.loaders import load_settings
    from threadx.data.token_diversity_data_source import TokenDiversityDataSource
    from threadx.backtest.engine import BacktestEngine
    from threadx.indicators.bank import IndicatorBank

    THREADX_AVAILABLE = True
    print("✅ ThreadX modules loaded successfully")
except ImportError as e:
    print(f"⚠️ ThreadX modules not available: {e}")
    print("🔄 Running in demo mode")

# GUI components
TK_AVAILABLE = False
try:
    import tkinter as tk
    from tkinter.scrolledtext import ScrolledText
    from tkinter import ttk, messagebox, filedialog

    TK_AVAILABLE = True
except ImportError:
    print("❌ Tkinter not available")

# Configuration
IS_WINDOWS = platform.system() == "Windows"
THREADX_ROOT = Path(__file__).parent.parent
DATA_ROOT = THREADX_ROOT / "data"

# Create directories
DATA_ROOT.mkdir(exist_ok=True)
(DATA_ROOT / "cache").mkdir(exist_ok=True)
(DATA_ROOT / "exports").mkdir(exist_ok=True)

# Logging setup
LOG_FILE = THREADX_ROOT / "logs" / "threadx_unified.log"
LOG_FILE.parent.mkdir(exist_ok=True)

logger = logging.getLogger("ThreadXUnified")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# File handler
file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

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


class ThreadXCore:
    """Core ThreadX operations wrapper"""

    def __init__(self):
        self.config = None
        self.data_source = None
        self.backtest_engine = None
        self.indicator_bank = None
        self.initialized = False

    def initialize(self) -> bool:
        """Initialize ThreadX components"""
        if not THREADX_AVAILABLE:
            logger.warning("ThreadX modules not available - running in demo mode")
            self.initialized = True  # Demo mode
            return True

        try:
            # Load configuration
            self.config = load_settings()
            logger.info("✅ ThreadX configuration loaded")

            # Initialize components
            self.data_source = TokenDiversityDataSource(self.config)
            self.backtest_engine = BacktestEngine(self.config)
            self.indicator_bank = IndicatorBank()

            self.initialized = True
            logger.info("🚀 ThreadX Core fully initialized")
            return True

        except Exception as e:
            logger.error(f"❌ ThreadX initialization failed: {e}")
            # Fallback to demo mode
            self.initialized = True
            return True

    def get_available_symbols(self) -> List[str]:
        """Get available trading symbols"""
        if THREADX_AVAILABLE and self.data_source:
            try:
                return self.data_source.get_available_symbols()
            except Exception as e:
                logger.error(f"Error getting symbols: {e}")

        # Demo data
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

    def get_available_strategies(self) -> List[str]:
        """Get available trading strategies"""
        return ["bb_atr", "rsi_crossover", "macd_strategy", "demo_strategy"]

    def run_backtest(
        self, strategy: str, symbols: List[str], timeframe: str = "1h", **kwargs
    ) -> Dict[str, Any]:
        """Run backtest with given parameters"""
        try:
            logger.info(f"🎯 Running backtest: {strategy} on {len(symbols)} symbols")

            if THREADX_AVAILABLE and self.backtest_engine:
                # Real backtest
                params = {
                    "strategy": strategy,
                    "symbols": symbols,
                    "timeframe": timeframe,
                    **kwargs,
                }
                results = self.backtest_engine.run(params)
                logger.info("✅ Backtest completed successfully")
                return results
            else:
                # Demo results
                import random

                return {
                    "success": True,
                    "strategy": strategy,
                    "symbols": symbols,
                    "timeframe": timeframe,
                    "total_return": round(random.uniform(-0.2, 0.5), 4),
                    "sharpe_ratio": round(random.uniform(0.5, 2.0), 2),
                    "max_drawdown": round(random.uniform(0.1, 0.3), 4),
                    "trades": random.randint(50, 200),
                    "win_rate": round(random.uniform(0.4, 0.7), 3),
                    "note": "Demo results - not real backtest data",
                }

        except Exception as e:
            logger.error(f"❌ Backtest failed: {e}")
            return {"error": str(e), "success": False}


# Global ThreadX core instance
threadx_core = ThreadXCore()


class ThreadXUnifiedApp:
    """Main ThreadX Unified Application"""

    def __init__(self):
        if not TK_AVAILABLE:
            raise ImportError("Tkinter not available - GUI mode unavailable")

        self.root = tk.Tk()
        self.root.title("ThreadX Unified Interface v1.0")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 600)

        # Variables
        self.log_text = None
        self.progress_var = None
        self.status_var = None
        self.notebook = None
        self.threadx_ready = False

        # Setup
        self.setup_styles()
        self.setup_ui()
        self.setup_logging_display()
        self.init_threadx()

    def setup_styles(self):
        """Configure visual styles"""
        style = ttk.Style()

        # Use modern theme if available
        available_themes = style.theme_names()
        if "vista" in available_themes:
            style.theme_use("vista")
        elif "clam" in available_themes:
            style.theme_use("clam")

        # Custom styles
        style.configure("Title.TLabel", font=("Arial", 14, "bold"))
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
        threadx_menu.add_command(label="🔄 Reinitialize", command=self.init_threadx)
        threadx_menu.add_command(label="⚙️ Configuration", command=self.show_config)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="❓ About", command=self.show_about)

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

        # Tabs
        self.create_dashboard_tab()
        self.create_backtest_tab()
        self.create_data_tab()
        self.create_logs_tab()

    def create_dashboard_tab(self):
        """Create dashboard tab"""
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="🏠 Dashboard")

        # Main container
        container = ttk.Frame(tab_frame)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Welcome section
        welcome_frame = ttk.LabelFrame(container, text="Welcome to ThreadX", padding=20)
        welcome_frame.pack(fill=tk.X, pady=(0, 20))

        welcome_text = """🚀 ThreadX Unified Interface v1.0

This interface provides comprehensive access to ThreadX functionality:
• Strategy backtesting and analysis
• Data management and validation  
• Technical indicator calculation
• Performance monitoring and reporting

Inspired by TradXPro architecture, adapted for ThreadX."""

        ttk.Label(welcome_frame, text=welcome_text.strip(), justify=tk.LEFT).pack(
            anchor=tk.W
        )

        # Quick actions
        actions_frame = ttk.LabelFrame(container, text="Quick Actions", padding=20)
        actions_frame.pack(fill=tk.X, pady=(0, 20))

        buttons_frame = ttk.Frame(actions_frame)
        buttons_frame.pack()

        ttk.Button(
            buttons_frame,
            text="🎯 Run Backtest",
            command=lambda: self.notebook.select(1),
        ).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(
            buttons_frame, text="📊 View Data", command=lambda: self.notebook.select(2)
        ).pack(side=tk.LEFT, padx=10)
        ttk.Button(
            buttons_frame, text="📋 Check Logs", command=lambda: self.notebook.select(3)
        ).pack(side=tk.LEFT, padx=10)

        # System status
        status_frame = ttk.LabelFrame(container, text="System Status", padding=10)
        status_frame.pack(fill=tk.BOTH, expand=True)

        self.status_text = ScrolledText(status_frame, height=12, font=("Consolas", 9))
        self.status_text.pack(fill=tk.BOTH, expand=True)

    def create_backtest_tab(self):
        """Create backtesting tab"""
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="🎯 Backtest")

        # Container
        container = ttk.Frame(tab_frame)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        ttk.Label(container, text="Strategy Backtesting", style="Title.TLabel").pack(
            pady=(0, 20)
        )

        # Parameters frame
        params_frame = ttk.LabelFrame(container, text="Backtest Parameters", padding=15)
        params_frame.pack(fill=tk.X, pady=(0, 20))

        # Strategy selection
        ttk.Label(params_frame, text="Strategy:").pack(anchor=tk.W)
        self.strategy_var = tk.StringVar(value="bb_atr")
        strategy_combo = ttk.Combobox(
            params_frame, textvariable=self.strategy_var, width=30, state="readonly"
        )
        strategy_combo.pack(fill=tk.X, pady=(5, 10))

        # Symbols
        ttk.Label(params_frame, text="Symbols (comma-separated):").pack(anchor=tk.W)
        self.symbols_var = tk.StringVar(value="BTCUSDC,ETHUSDC")
        symbols_entry = ttk.Entry(params_frame, textvariable=self.symbols_var, width=50)
        symbols_entry.pack(fill=tk.X, pady=(5, 10))

        # Timeframe
        ttk.Label(params_frame, text="Timeframe:").pack(anchor=tk.W)
        self.timeframe_var = tk.StringVar(value="1h")
        tf_combo = ttk.Combobox(
            params_frame,
            textvariable=self.timeframe_var,
            values=["5m", "15m", "30m", "1h", "4h", "1d"],
            width=15,
            state="readonly",
        )
        tf_combo.pack(anchor=tk.W, pady=(5, 10))

        # Run button
        ttk.Button(
            params_frame, text="🚀 Run Backtest", command=self.run_backtest_action
        ).pack(pady=10)

        # Results frame
        results_frame = ttk.LabelFrame(container, text="Backtest Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True)

        self.results_text = ScrolledText(results_frame, height=15, font=("Consolas", 9))
        self.results_text.pack(fill=tk.BOTH, expand=True)

    def create_data_tab(self):
        """Create data management tab"""
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="📊 Data")

        # Container
        container = ttk.Frame(tab_frame)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        ttk.Label(container, text="Data Management", style="Title.TLabel").pack(
            pady=(0, 20)
        )

        # Symbols display
        symbols_frame = ttk.LabelFrame(container, text="Available Symbols", padding=10)
        symbols_frame.pack(fill=tk.BOTH, expand=True)

        # Symbols listbox
        symbols_listbox = tk.Listbox(symbols_frame, height=20)
        symbols_scroll = ttk.Scrollbar(
            symbols_frame, orient=tk.VERTICAL, command=symbols_listbox.yview
        )
        symbols_listbox.configure(yscrollcommand=symbols_scroll.set)

        symbols_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        symbols_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Populate symbols
        self.symbols_listbox = symbols_listbox
        self.refresh_symbols_display()

        # Refresh button
        ttk.Button(
            container, text="🔄 Refresh Symbols", command=self.refresh_symbols_display
        ).pack(pady=10)

    def create_logs_tab(self):
        """Create logs tab"""
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="📋 Logs")

        # Container
        container = ttk.Frame(tab_frame)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Log display
        self.log_text = ScrolledText(container, height=30, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Controls
        controls_frame = ttk.Frame(container)
        controls_frame.pack(fill=tk.X)

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
                    self.root.after(0, self.threadx_init_success)
                else:
                    self.root.after(0, self.threadx_init_failed)

            except Exception as e:
                self.root.after(0, lambda: self.threadx_init_error(str(e)))

        threading.Thread(target=init_worker, daemon=True).start()

    def threadx_init_success(self):
        """Handle successful ThreadX initialization"""
        self.threadx_ready = True
        mode = "ThreadX Ready" if THREADX_AVAILABLE else "Demo Mode"
        self.status_indicator.config(text=f"🟢 {mode}", style="Success.TLabel")
        self.update_status(f"✅ {mode} - System initialized")

        # Populate strategy combo box
        strategies = threadx_core.get_available_strategies()
        if hasattr(self, "strategy_var"):
            try:
                strategy_combo = None
                for widget in self.root.winfo_children():
                    if hasattr(widget, "winfo_children"):
                        for child in widget.winfo_children():
                            if isinstance(child, ttk.Combobox):
                                strategy_combo = child
                                break
                if strategy_combo:
                    strategy_combo["values"] = strategies
            except Exception as e:
                logger.debug(f"Could not update strategy combo: {e}")

    def threadx_init_failed(self):
        """Handle failed ThreadX initialization"""
        self.status_indicator.config(text="🔴 ThreadX Failed", style="Error.TLabel")
        self.update_status("❌ ThreadX initialization failed")

    def threadx_init_error(self, error: str):
        """Handle ThreadX initialization error"""
        self.status_indicator.config(text="🟡 ThreadX Error", style="Warning.TLabel")
        self.update_status(f"⚠️ ThreadX error: {error}")

    def refresh_symbols_display(self):
        """Refresh symbols display"""
        if hasattr(self, "symbols_listbox"):
            self.symbols_listbox.delete(0, tk.END)
            symbols = threadx_core.get_available_symbols()
            for symbol in symbols:
                self.symbols_listbox.insert(tk.END, symbol)

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
                self.update_progress(25)

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

            # Format results nicely
            if "note" in results:
                self.results_text.insert(tk.END, f"📝 Note: {results['note']}\n\n")

            self.results_text.insert(tk.END, "📊 Results Summary:\n")
            self.results_text.insert(tk.END, f"{'-'*40}\n")

            for key, value in results.items():
                if key not in ["success", "note"]:
                    if isinstance(value, list):
                        self.results_text.insert(
                            tk.END, f"{key}: {', '.join(map(str, value))}\n"
                        )
                    else:
                        self.results_text.insert(tk.END, f"{key}: {value}\n")

            self.results_text.insert(tk.END, f"{'-'*40}\n\n")
            self.results_text.insert(tk.END, "Raw JSON:\n")
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

    def update_status(self, message: str):
        """Update status bar"""
        if self.status_var:
            self.status_var.set(message)

    def update_progress(self, value: float):
        """Update progress bar"""
        if self.progress_var:
            self.progress_var.set(value)

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
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("Text files", "*.txt")],
            )
            if filename:
                content = self.results_text.get(1.0, tk.END)
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(content)
                messagebox.showinfo("Success", f"Results exported to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not export results: {e}")

    def show_config(self):
        """Show configuration"""
        config_text = f"""ThreadX Configuration:

ThreadX Root: {THREADX_ROOT}
Data Root: {DATA_ROOT}
ThreadX Available: {'✅' if THREADX_AVAILABLE else '❌'}
GUI Available: {'✅' if TK_AVAILABLE else '❌'}
Platform: {platform.system()}
ThreadX Ready: {'✅' if self.threadx_ready else '❌'}"""

        messagebox.showinfo("Configuration", config_text)

    def show_about(self):
        """Show about dialog"""
        about_text = """ThreadX Unified Interface v1.0

A comprehensive trading analysis platform inspired by TradXPro
and integrated with ThreadX core functionality.

Features:
• Strategy backtesting and analysis
• Data management and validation
• Technical indicator calculation
• Performance monitoring
• Real-time logging and status

© 2024 ThreadX Project"""

        messagebox.showinfo("About", about_text)

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


def run_cli_mode():
    """Run in CLI mode when GUI not available"""
    print("=" * 60)
    print("ThreadX Unified Interface - CLI Mode")
    print("=" * 60)

    # Initialize ThreadX
    print("🔄 Initializing ThreadX Core...")
    success = threadx_core.initialize()

    if success:
        mode = "ThreadX Ready" if THREADX_AVAILABLE else "Demo Mode"
        print(f"✅ {mode}")

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
                    for symbol in symbols:
                        print(f"  • {symbol}")

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
                    print(f"\nResults: {json.dumps(results, indent=2)}")

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
