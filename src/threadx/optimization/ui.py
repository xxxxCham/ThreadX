"""
ThreadX Parametric Optimization UI Module
=========================================

Interface utilisateur pour l'optimisation paramétrique unifiée.
Utilise le moteur UnifiedOptimizationEngine qui centralise tous les calculs via IndicatorBank.

Author: ThreadX Framework
Version: Phase 10 - Unified Compute Engine
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np

from threadx.optimization.engine import UnifiedOptimizationEngine, DEFAULT_SWEEP_CONFIG
from threadx.indicators.bank import IndicatorBank
from threadx.utils.log import get_logger

logger = get_logger(__name__)


class ParametricOptimizationUI:
    """Interface d'optimisation paramétrique utilisant le moteur unifié."""

    def __init__(
        self, parent: tk.Widget, indicator_bank: Optional[IndicatorBank] = None
    ):
        """
        Initialise l'interface d'optimisation.

        Args:
            parent: Widget parent pour l'interface
            indicator_bank: Instance IndicatorBank partagée (recommandé)
        """
        self.parent = parent
        self.indicator_bank = indicator_bank or IndicatorBank()

        # Moteur d'optimisation unifié
        self.optimization_engine = UnifiedOptimizationEngine(
            indicator_bank=self.indicator_bank, max_workers=4
        )

        # État de l'interface
        self.current_data: Optional[pd.DataFrame] = None
        self.current_results: Optional[pd.DataFrame] = None
        self.optimization_thread: Optional[threading.Thread] = None
        self.is_running = False

        # Configuration par défaut
        self.config = DEFAULT_SWEEP_CONFIG.copy()

        # Interface utilisateur
        self.setup_ui()

        logger.info("ParametricOptimizationUI initialisé avec moteur unifié")

    def setup_ui(self):
        """Construit l'interface utilisateur."""
        # Frame principal
        self.main_frame = ttk.Frame(self.parent)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Section configuration
        self.create_config_section()

        # Section contrôles
        self.create_controls_section()

        # Section progress
        self.create_progress_section()

        # Section résultats
        self.create_results_section()

    def create_config_section(self):
        """Crée la section de configuration."""
        config_frame = ttk.LabelFrame(
            self.main_frame, text="📊 Configuration du Sweep", padding=10
        )
        config_frame.pack(fill=tk.X, padx=5, pady=5)

        # Dataset configuration
        dataset_frame = ttk.Frame(config_frame)
        dataset_frame.pack(fill=tk.X, pady=5)

        ttk.Label(dataset_frame, text="Dataset:").pack(side=tk.LEFT)

        self.symbol_var = tk.StringVar(value=self.config["dataset"]["symbol"])
        symbol_combo = ttk.Combobox(
            dataset_frame,
            textvariable=self.symbol_var,
            values=["BTCUSDC", "ETHUSDC", "ADAUSDC"],
            width=12,
        )
        symbol_combo.pack(side=tk.LEFT, padx=5)

        self.timeframe_var = tk.StringVar(value=self.config["dataset"]["timeframe"])
        tf_combo = ttk.Combobox(
            dataset_frame,
            textvariable=self.timeframe_var,
            values=["15m", "1h", "4h", "1d"],
            width=8,
        )
        tf_combo.pack(side=tk.LEFT, padx=5)

        ttk.Button(
            dataset_frame, text="📁 Charger Données", command=self.load_data
        ).pack(side=tk.LEFT, padx=10)

        self.data_status = ttk.Label(dataset_frame, text="❌ Aucune donnée")
        self.data_status.pack(side=tk.LEFT, padx=10)

        # Parameter grid configuration
        self.create_parameter_grid_config(config_frame)

    def create_parameter_grid_config(self, parent):
        """Crée la configuration de grille de paramètres."""
        grid_frame = ttk.LabelFrame(parent, text="🎛️ Grille de Paramètres", padding=5)
        grid_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Notebook pour les différents indicateurs
        self.param_notebook = ttk.Notebook(grid_frame)
        self.param_notebook.pack(fill=tk.BOTH, expand=True)

        # Onglet Bollinger Bands
        self.create_bollinger_tab()

        # Onglet ATR
        self.create_atr_tab()

        # Contrôles de configuration
        config_controls = ttk.Frame(grid_frame)
        config_controls.pack(fill=tk.X, pady=5)

        ttk.Button(
            config_controls, text="💾 Sauver Config", command=self.save_config
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            config_controls, text="📂 Charger Config", command=self.load_config
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(config_controls, text="🔄 Reset", command=self.reset_config).pack(
            side=tk.LEFT, padx=5
        )

    def create_bollinger_tab(self):
        """Crée l'onglet de configuration Bollinger Bands."""
        bb_frame = ttk.Frame(self.param_notebook)
        self.param_notebook.add(bb_frame, text="Bollinger Bands")

        # Enable/Disable
        self.bb_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            bb_frame, text="Activer Bollinger Bands", variable=self.bb_enabled
        ).pack(anchor=tk.W, pady=5)

        # Period configuration
        period_frame = ttk.Frame(bb_frame)
        period_frame.pack(fill=tk.X, pady=5)

        ttk.Label(period_frame, text="Période:").pack(side=tk.LEFT)
        self.bb_period_values = tk.StringVar(value="10,15,20,25,30")
        ttk.Entry(period_frame, textvariable=self.bb_period_values, width=30).pack(
            side=tk.LEFT, padx=5
        )

        # Standard deviation configuration
        std_frame = ttk.Frame(bb_frame)
        std_frame.pack(fill=tk.X, pady=5)

        ttk.Label(std_frame, text="Écart-type - Start:").pack(side=tk.LEFT)
        self.bb_std_start = tk.StringVar(value="1.5")
        ttk.Entry(std_frame, textvariable=self.bb_std_start, width=8).pack(
            side=tk.LEFT, padx=2
        )

        ttk.Label(std_frame, text="Stop:").pack(side=tk.LEFT, padx=(10, 0))
        self.bb_std_stop = tk.StringVar(value="3.0")
        ttk.Entry(std_frame, textvariable=self.bb_std_stop, width=8).pack(
            side=tk.LEFT, padx=2
        )

        ttk.Label(std_frame, text="Step:").pack(side=tk.LEFT, padx=(10, 0))
        self.bb_std_step = tk.StringVar(value="0.1")
        ttk.Entry(std_frame, textvariable=self.bb_std_step, width=8).pack(
            side=tk.LEFT, padx=2
        )

    def create_atr_tab(self):
        """Crée l'onglet de configuration ATR."""
        atr_frame = ttk.Frame(self.param_notebook)
        self.param_notebook.add(atr_frame, text="ATR")

        # Enable/Disable
        self.atr_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(atr_frame, text="Activer ATR", variable=self.atr_enabled).pack(
            anchor=tk.W, pady=5
        )

        # Period configuration
        period_frame = ttk.Frame(atr_frame)
        period_frame.pack(fill=tk.X, pady=5)

        ttk.Label(period_frame, text="Période - Start:").pack(side=tk.LEFT)
        self.atr_period_start = tk.StringVar(value="10")
        ttk.Entry(period_frame, textvariable=self.atr_period_start, width=8).pack(
            side=tk.LEFT, padx=2
        )

        ttk.Label(period_frame, text="Stop:").pack(side=tk.LEFT, padx=(10, 0))
        self.atr_period_stop = tk.StringVar(value="30")
        ttk.Entry(period_frame, textvariable=self.atr_period_stop, width=8).pack(
            side=tk.LEFT, padx=2
        )

        ttk.Label(period_frame, text="Step:").pack(side=tk.LEFT, padx=(10, 0))
        self.atr_period_step = tk.StringVar(value="2")
        ttk.Entry(period_frame, textvariable=self.atr_period_step, width=8).pack(
            side=tk.LEFT, padx=2
        )

        # Method configuration
        method_frame = ttk.Frame(atr_frame)
        method_frame.pack(fill=tk.X, pady=5)

        ttk.Label(method_frame, text="Méthodes:").pack(side=tk.LEFT)
        self.atr_methods = tk.StringVar(value="ema,sma")
        ttk.Entry(method_frame, textvariable=self.atr_methods, width=20).pack(
            side=tk.LEFT, padx=5
        )

    def create_controls_section(self):
        """Crée la section de contrôles."""
        controls_frame = ttk.LabelFrame(
            self.main_frame, text="🎮 Contrôles", padding=10
        )
        controls_frame.pack(fill=tk.X, padx=5, pady=5)

        # Boutons principaux
        self.start_button = ttk.Button(
            controls_frame,
            text="🚀 Démarrer Sweep",
            command=self.start_optimization,
            style="Accent.TButton",
        )
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.pause_button = ttk.Button(
            controls_frame,
            text="⏸️ Pause",
            command=self.pause_optimization,
            state=tk.DISABLED,
        )
        self.pause_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(
            controls_frame,
            text="⏹️ Stop",
            command=self.stop_optimization,
            state=tk.DISABLED,
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Configuration des workers
        ttk.Label(controls_frame, text="Workers:").pack(side=tk.LEFT, padx=(20, 5))
        self.workers_var = tk.IntVar(value=4)
        workers_spin = tk.Spinbox(
            controls_frame, from_=1, to=16, width=5, textvariable=self.workers_var
        )
        workers_spin.pack(side=tk.LEFT, padx=5)

        # Statut du moteur
        self.engine_status = ttk.Label(controls_frame, text="💾 Cache: 0 entries")
        self.engine_status.pack(side=tk.RIGHT, padx=10)

    def create_progress_section(self):
        """Crée la section de progression."""
        progress_frame = ttk.LabelFrame(
            self.main_frame, text="📈 Progression", padding=10
        )
        progress_frame.pack(fill=tk.X, padx=5, pady=5)

        # Barre de progression
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame, variable=self.progress_var, maximum=100, length=400
        )
        self.progress_bar.pack(fill=tk.X, pady=5)

        # Labels de statut
        status_frame = ttk.Frame(progress_frame)
        status_frame.pack(fill=tk.X)

        self.status_label = ttk.Label(status_frame, text="Prêt à démarrer")
        self.status_label.pack(side=tk.LEFT)

        self.eta_label = ttk.Label(status_frame, text="")
        self.eta_label.pack(side=tk.RIGHT)

    def create_results_section(self):
        """Crée la section des résultats."""
        results_frame = ttk.LabelFrame(self.main_frame, text="🏆 Résultats", padding=5)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Contrôles des résultats
        results_controls = ttk.Frame(results_frame)
        results_controls.pack(fill=tk.X, pady=5)

        ttk.Button(
            results_controls, text="💾 Exporter CSV", command=self.export_results
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            results_controls, text="📊 Visualiser", command=self.visualize_results
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            results_controls, text="🔍 Analyser Top", command=self.analyze_top_results
        ).pack(side=tk.LEFT, padx=5)

        # Tableau des résultats
        self.create_results_tree(results_frame)

    def create_results_tree(self, parent):
        """Crée le tableau des résultats."""
        # Frame avec scrollbars
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        # Treeview
        columns = (
            "rank",
            "indicator",
            "params",
            "pnl",
            "sharpe",
            "drawdown",
            "trades",
            "winrate",
        )
        self.results_tree = ttk.Treeview(
            tree_frame, columns=columns, show="headings", height=15
        )

        # Headers
        self.results_tree.heading("rank", text="Rang")
        self.results_tree.heading("indicator", text="Indicateur")
        self.results_tree.heading("params", text="Paramètres")
        self.results_tree.heading("pnl", text="PnL")
        self.results_tree.heading("sharpe", text="Sharpe")
        self.results_tree.heading("drawdown", text="MaxDD")
        self.results_tree.heading("trades", text="Trades")
        self.results_tree.heading("winrate", text="Win%")

        # Column widths
        self.results_tree.column("rank", width=50, anchor=tk.CENTER)
        self.results_tree.column("indicator", width=100, anchor=tk.CENTER)
        self.results_tree.column("params", width=200, anchor=tk.W)
        self.results_tree.column("pnl", width=80, anchor=tk.E)
        self.results_tree.column("sharpe", width=70, anchor=tk.E)
        self.results_tree.column("drawdown", width=80, anchor=tk.E)
        self.results_tree.column("trades", width=70, anchor=tk.E)
        self.results_tree.column("winrate", width=70, anchor=tk.E)

        # Scrollbars
        v_scrollbar = ttk.Scrollbar(
            tree_frame, orient=tk.VERTICAL, command=self.results_tree.yview
        )
        h_scrollbar = ttk.Scrollbar(
            tree_frame, orient=tk.HORIZONTAL, command=self.results_tree.xview
        )
        self.results_tree.configure(
            yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set
        )

        # Grid layout
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

    def load_data(self):
        """Charge les données pour l'optimisation."""
        try:
            # Pour l'instant, on utilise des données de test
            # En production, ceci chargerait depuis les données réelles
            symbol = self.symbol_var.get()
            timeframe = self.timeframe_var.get()

            # Simulation de données OHLCV
            dates = pd.date_range("2024-01-01", "2024-12-31", freq="1H")[:1000]
            np.random.seed(42)

            data = pd.DataFrame(
                {
                    "timestamp": dates,
                    "open": 50000 + np.random.randn(len(dates)).cumsum() * 100,
                    "high": 0,
                    "low": 0,
                    "close": 0,
                    "volume": np.random.randint(100, 1000, len(dates)),
                }
            )

            # Calcul OHLC basique
            data["close"] = data["open"] + np.random.randn(len(dates)) * 50
            data["high"] = (
                np.maximum(data["open"], data["close"])
                + np.random.rand(len(dates)) * 100
            )
            data["low"] = (
                np.minimum(data["open"], data["close"])
                - np.random.rand(len(dates)) * 100
            )

            data = data.set_index("timestamp")

            self.current_data = data
            self.data_status.config(text=f"✅ {len(data)} barres chargées")

            logger.info(f"Données chargées: {symbol} {timeframe} - {len(data)} barres")

        except Exception as e:
            logger.error(f"Erreur chargement données: {e}")
            messagebox.showerror("Erreur", f"Impossible de charger les données: {e}")

    def build_config_from_ui(self) -> Dict:
        """Construit la configuration à partir de l'interface."""
        config = {
            "dataset": {
                "symbol": self.symbol_var.get(),
                "timeframe": self.timeframe_var.get(),
                "start": "2024-01-01",
                "end": "2024-12-31",
            },
            "grid": {},
            "scoring": {
                "primary": "pnl",
                "secondary": ["sharpe", "-max_drawdown"],
                "top_k": 50,
            },
        }

        # Bollinger Bands
        if self.bb_enabled.get():
            try:
                periods = [
                    int(x.strip()) for x in self.bb_period_values.get().split(",")
                ]
                config["grid"]["bollinger"] = {
                    "period": periods,
                    "std": {
                        "start": float(self.bb_std_start.get()),
                        "stop": float(self.bb_std_stop.get()),
                        "step": float(self.bb_std_step.get()),
                    },
                }
            except ValueError as e:
                logger.error(f"Erreur config Bollinger: {e}")

        # ATR
        if self.atr_enabled.get():
            try:
                methods = [x.strip() for x in self.atr_methods.get().split(",")]
                config["grid"]["atr"] = {
                    "period": {
                        "start": int(self.atr_period_start.get()),
                        "stop": int(self.atr_period_stop.get()),
                        "step": int(self.atr_period_step.get()),
                    },
                    "method": methods,
                }
            except ValueError as e:
                logger.error(f"Erreur config ATR: {e}")

        return config

    def start_optimization(self):
        """Démarre l'optimisation paramétrique."""
        if not self.current_data is not None:
            messagebox.showwarning("Attention", "Veuillez d'abord charger des données")
            return

        if self.is_running:
            messagebox.showwarning("Attention", "Une optimisation est déjà en cours")
            return

        try:
            # Configuration
            self.config = self.build_config_from_ui()

            # Mise à jour du moteur
            self.optimization_engine.max_workers = self.workers_var.get()
            self.optimization_engine.progress_callback = self.update_progress

            # Thread d'optimisation
            self.optimization_thread = threading.Thread(
                target=self._run_optimization_thread, daemon=True
            )

            # Interface
            self.is_running = True
            self.start_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.NORMAL)

            # Démarrage
            self.optimization_thread.start()

            logger.info("Optimisation paramétrique démarrée")

        except Exception as e:
            logger.error(f"Erreur démarrage optimisation: {e}")
            messagebox.showerror(
                "Erreur", f"Impossible de démarrer l'optimisation: {e}"
            )
            self._reset_ui_state()

    def _run_optimization_thread(self):
        """Thread d'exécution de l'optimisation."""
        try:
            if self.current_data is None:
                raise ValueError("Aucune donnée chargée")

            results = self.optimization_engine.run_parameter_sweep(
                self.config, self.current_data
            )

            # Mise à jour des résultats dans le thread principal
            self.parent.after(0, lambda: self._optimization_completed(results))

        except Exception as e:
            logger.error(f"Erreur dans le thread d'optimisation: {e}")
            self.parent.after(0, lambda: self._optimization_failed(str(e)))

    def _optimization_completed(self, results: pd.DataFrame):
        """Callback quand l'optimisation est terminée."""
        self.current_results = results
        self._populate_results_tree(results)
        self._reset_ui_state()

        # Statistiques
        stats = self.optimization_engine.get_indicator_bank_stats()
        self.engine_status.config(
            text=f"💾 Cache: {stats.get('cache_size', 0)} entries"
        )

        messagebox.showinfo(
            "Succès", f"Optimisation terminée!\n{len(results)} résultats obtenus"
        )
        logger.info(f"Optimisation terminée: {len(results)} résultats")

    def _optimization_failed(self, error: str):
        """Callback quand l'optimisation échoue."""
        self._reset_ui_state()
        messagebox.showerror("Erreur", f"L'optimisation a échoué: {error}")
        logger.error(f"Optimisation échouée: {error}")

    def _reset_ui_state(self):
        """Remet l'interface à l'état initial."""
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.status_label.config(text="Prêt à démarrer")
        self.eta_label.config(text="")

    def _populate_results_tree(self, results: pd.DataFrame):
        """Peuple le tableau des résultats."""
        # Vider le tableau
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        if results.empty:
            return

        # Ajouter les résultats
        for _, row in results.head(100).iterrows():  # Top 100
            # Format des paramètres
            params_str = ""
            for col in row.index:
                if col not in [
                    "rank",
                    "indicator_type",
                    "pnl",
                    "sharpe",
                    "max_drawdown",
                    "profit_factor",
                    "total_trades",
                    "win_rate",
                    "duration_sec",
                ]:
                    params_str += f"{col}={row[col]:.2f if isinstance(row[col], float) else row[col]} "

            values = (
                int(row.get("rank", 0)),
                row.get("indicator_type", ""),
                params_str.strip(),
                f"{row.get('pnl', 0):.2f}",
                f"{row.get('sharpe', 0):.3f}",
                f"{row.get('max_drawdown', 0):.3f}",
                int(row.get("total_trades", 0)),
                f"{row.get('win_rate', 0)*100:.1f}%",
            )

            self.results_tree.insert("", tk.END, values=values)

    def update_progress(
        self, progress: float, completed: int, total: int, eta: Optional[float]
    ):
        """Callback de mise à jour de la progression."""
        self.progress_var.set(progress * 100)
        self.status_label.config(text=f"Progression: {completed}/{total}")

        if eta:
            eta_str = f"ETA: {eta/60:.1f}min" if eta > 60 else f"ETA: {eta:.0f}s"
            self.eta_label.config(text=eta_str)

    def pause_optimization(self):
        """Met en pause/reprend l'optimisation."""
        if self.optimization_engine.should_pause:
            self.optimization_engine.resume()
            self.pause_button.config(text="⏸️ Pause")
        else:
            self.optimization_engine.pause()
            self.pause_button.config(text="▶️ Reprendre")

    def stop_optimization(self):
        """Arrête l'optimisation."""
        self.optimization_engine.stop()
        if self.optimization_thread and self.optimization_thread.is_alive():
            self.optimization_thread.join(timeout=1.0)
        self._reset_ui_state()

    def save_config(self):
        """Sauvegarde la configuration."""
        try:
            config = self.build_config_from_ui()
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            )
            if filename:
                with open(filename, "w") as f:
                    json.dump(config, f, indent=2)
                messagebox.showinfo("Succès", "Configuration sauvegardée")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de sauvegarder: {e}")

    def load_config(self):
        """Charge une configuration."""
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if filename:
                with open(filename, "r") as f:
                    config = json.load(f)
                self._apply_config_to_ui(config)
                messagebox.showinfo("Succès", "Configuration chargée")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger: {e}")

    def _apply_config_to_ui(self, config: Dict):
        """Applique une configuration à l'interface."""
        # Dataset
        if "dataset" in config:
            self.symbol_var.set(config["dataset"].get("symbol", "BTCUSDC"))
            self.timeframe_var.set(config["dataset"].get("timeframe", "1h"))

        # Grid
        if "grid" in config:
            # Bollinger
            if "bollinger" in config["grid"]:
                bb_config = config["grid"]["bollinger"]
                self.bb_enabled.set(True)
                if "period" in bb_config:
                    self.bb_period_values.set(",".join(map(str, bb_config["period"])))
                if "std" in bb_config:
                    std_config = bb_config["std"]
                    self.bb_std_start.set(str(std_config.get("start", 1.5)))
                    self.bb_std_stop.set(str(std_config.get("stop", 3.0)))
                    self.bb_std_step.set(str(std_config.get("step", 0.1)))

            # ATR
            if "atr" in config["grid"]:
                atr_config = config["grid"]["atr"]
                self.atr_enabled.set(True)
                if "period" in atr_config:
                    period_config = atr_config["period"]
                    self.atr_period_start.set(str(period_config.get("start", 10)))
                    self.atr_period_stop.set(str(period_config.get("stop", 30)))
                    self.atr_period_step.set(str(period_config.get("step", 2)))
                if "method" in atr_config:
                    self.atr_methods.set(",".join(atr_config["method"]))

    def reset_config(self):
        """Remet la configuration par défaut."""
        self._apply_config_to_ui(DEFAULT_SWEEP_CONFIG)

    def export_results(self):
        """Exporte les résultats en CSV."""
        if self.current_results is None or self.current_results.empty:
            messagebox.showwarning("Attention", "Aucun résultat à exporter")
            return

        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            )
            if filename:
                self.current_results.to_csv(filename, index=False)
                messagebox.showinfo("Succès", f"Résultats exportés: {filename}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'exporter: {e}")

    def visualize_results(self):
        """Lance la visualisation des résultats."""
        if self.current_results is None or self.current_results.empty:
            messagebox.showwarning("Attention", "Aucun résultat à visualiser")
            return

        # TODO: Implémenter la visualisation
        messagebox.showinfo("Info", "Visualisation à implémenter")

    def analyze_top_results(self):
        """Analyse les meilleurs résultats."""
        if self.current_results is None or self.current_results.empty:
            messagebox.showwarning("Attention", "Aucun résultat à analyser")
            return

        # TODO: Implémenter l'analyse
        messagebox.showinfo("Info", "Analyse détaillée à implémenter")


def create_optimization_ui(
    parent: tk.Widget, indicator_bank: Optional[IndicatorBank] = None
) -> ParametricOptimizationUI:
    """
    Factory function pour créer l'interface d'optimisation.

    Args:
        parent: Widget parent
        indicator_bank: Instance IndicatorBank partagée

    Returns:
        ParametricOptimizationUI configurée
    """
    return ParametricOptimizationUI(parent, indicator_bank)
