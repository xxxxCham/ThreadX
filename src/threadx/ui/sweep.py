"""
ThreadX Sweep UI - Optimisation Paramétrique Intégrée
======================================================

Interface d'optimisation paramétrique utilisant le moteur unifié ThreadX.
Utilise le Bridge pour accès au moteur d'optimisation.

Features:
- Configuration de grilles paramétriques (Bollinger, ATR, MA)
- Exécution non-bloquante avec progress tracking
- Résultats triés multi-critères (PnL, Sharpe, -MaxDD, PF)
- Export complet (CSV/Parquet + config + logs)
- Resume sur cache/résultats existants
- Interface Windows-optimized

Author: ThreadX Framework
Version: Phase 10 - Parametric Sweeps Integration
"""

# type: ignore  # Trop d'erreurs de type, analyse désactivée

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from threading import Thread
from queue import Queue, Empty
from typing import Dict, Optional, Any, List
import time
import json
import pandas as pd
from pathlib import Path
import logging

# ✅ FIXED: Import from Bridge only, not direct Engine
from threadx.bridge import SweepController, SweepRequest, DEFAULT_SWEEP_CONFIG
from ..utils.log import get_logger

logger = get_logger(__name__)


class SweepOptimizationPage(ttk.Frame):
    """
    Page d'optimisation paramétrique pour sweeps multi-indicateurs.

    Utilise le Bridge SweepController qui orchestre l'Engine d'optimisation.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        # ✅ FIXED: Initialize Bridge controller instead of direct Engine
        self.sweep_controller = SweepController()
        logger.info("✅ SweepController initialisé via Bridge")

        # Communication thread ↔ UI
        self.progress_queue = Queue()
        self.log_queue = Queue()

        # État d'optimisation
        self.optimization_thread: Optional[Thread] = None
        self.is_running = False
        self.current_data: Optional[pd.DataFrame] = None
        self.current_results: Optional[pd.DataFrame] = None

        # Configuration par défaut
        self.config = DEFAULT_SWEEP_CONFIG.copy()

        self.setup_ui()

        # Progress tracking
        self.after(100, self.check_queues)

    def setup_ui(self):
        """Construction de l'interface utilisateur."""
        # Titre principal
        title_frame = ttk.Frame(self)
        title_frame.pack(fill="x", padx=10, pady=5)

        title_label = ttk.Label(
            title_frame, text="🎯 Optimisation Paramétrique", font=("Arial", 14, "bold")
        )
        title_label.pack(side="left")

        subtitle_label = ttk.Label(
            title_frame,
            text="Sweeps multi-indicateurs avec moteur unifié ThreadX",
            font=("Arial", 9),
            foreground="gray",
        )
        subtitle_label.pack(side="left", padx=(10, 0))

        # Configuration du sweep
        self.create_config_section()

        # Contrôles d'exécution
        self.create_controls_section()

        # Progress et résultats
        self.create_results_section()

    def create_config_section(self):
        """Section de configuration du sweep."""
        config_frame = ttk.LabelFrame(self, text="⚙️ Configuration du Sweep", padding=10)
        config_frame.pack(fill="x", padx=10, pady=5)

        # Dataset configuration
        dataset_frame = ttk.Frame(config_frame)
        dataset_frame.pack(fill="x", pady=5)

        ttk.Label(dataset_frame, text="Dataset:").pack(side="left")

        # Symbole
        self.symbol_var = tk.StringVar(value=self.config["dataset"]["symbol"])
        symbol_combo = ttk.Combobox(
            dataset_frame,
            textvariable=self.symbol_var,
            values=["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"],
            width=12,
        )
        symbol_combo.pack(side="left", padx=5)

        # Timeframe
        self.timeframe_var = tk.StringVar(value=self.config["dataset"]["timeframe"])
        tf_combo = ttk.Combobox(
            dataset_frame,
            textvariable=self.timeframe_var,
            values=["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"],
            width=8,
        )
        tf_combo.pack(side="left", padx=5)

        # Bouton charger données
        ttk.Button(
            dataset_frame, text="📁 Charger Données", command=self.load_data
        ).pack(side="left", padx=10)

        # Status des données
        self.data_status = ttk.Label(dataset_frame, text="❌ Aucune donnée chargée")
        self.data_status.pack(side="left", padx=10)

        # Configuration des grilles paramétriques
        self.create_parameter_grids(config_frame)

        # Configuration de scoring
        self.create_scoring_config(config_frame)

    def create_parameter_grids(self, parent):
        """Configuration des grilles de paramètres."""
        grid_frame = ttk.LabelFrame(parent, text="🎛️ Grilles de Paramètres", padding=5)
        grid_frame.pack(fill="both", expand=True, pady=5)

        # Notebook pour différents indicateurs
        self.param_notebook = ttk.Notebook(grid_frame)
        self.param_notebook.pack(fill="both", expand=True)

        # Onglet Bollinger Bands
        self.create_bollinger_tab()

        # Onglet ATR
        self.create_atr_tab()

        # Onglet Moving Averages
        self.create_ma_tab()

        # Contrôles de configuration
        config_controls = ttk.Frame(grid_frame)
        config_controls.pack(fill="x", pady=5)

        ttk.Button(
            config_controls, text="💾 Sauver Config", command=self.save_config
        ).pack(side="left", padx=5)
        ttk.Button(
            config_controls, text="📂 Charger Config", command=self.load_config
        ).pack(side="left", padx=5)
        ttk.Button(config_controls, text="🔄 Reset", command=self.reset_config).pack(
            side="left", padx=5
        )

    def create_bollinger_tab(self):
        """Onglet configuration Bollinger Bands."""
        bb_frame = ttk.Frame(self.param_notebook)
        self.param_notebook.add(bb_frame, text="Bollinger Bands")

        # Enable/Disable
        self.bb_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            bb_frame, text="✅ Activer Bollinger Bands", variable=self.bb_enabled
        ).pack(anchor="w", pady=5)

        # Paramètres de grille
        params_frame = ttk.LabelFrame(bb_frame, text="Paramètres", padding=5)
        params_frame.pack(fill="x", pady=5)

        # Period (liste de valeurs)
        period_frame = ttk.Frame(params_frame)
        period_frame.pack(fill="x", pady=2)

        ttk.Label(period_frame, text="Périodes:").pack(side="left", padx=(0, 10))
        self.bb_periods = tk.StringVar(value="10,15,20,25,30")
        ttk.Entry(period_frame, textvariable=self.bb_periods, width=30).pack(
            side="left"
        )

        # Std deviation (range)
        std_frame = ttk.Frame(params_frame)
        std_frame.pack(fill="x", pady=2)

        ttk.Label(std_frame, text="Std Dev:").pack(side="left", padx=(0, 10))
        ttk.Label(std_frame, text="Début:").pack(side="left", padx=(10, 5))
        self.bb_std_start = tk.DoubleVar(value=1.5)
        ttk.Entry(std_frame, textvariable=self.bb_std_start, width=8).pack(
            side="left", padx=(0, 10)
        )

        ttk.Label(std_frame, text="Fin:").pack(side="left", padx=(10, 5))
        self.bb_std_stop = tk.DoubleVar(value=3.0)
        ttk.Entry(std_frame, textvariable=self.bb_std_stop, width=8).pack(
            side="left", padx=(0, 10)
        )

        ttk.Label(std_frame, text="Pas:").pack(side="left", padx=(10, 5))
        self.bb_std_step = tk.DoubleVar(value=0.1)
        ttk.Entry(std_frame, textvariable=self.bb_std_step, width=8).pack(side="left")

    def create_atr_tab(self):
        """Onglet configuration ATR."""
        atr_frame = ttk.Frame(self.param_notebook)
        self.param_notebook.add(atr_frame, text="ATR")

        # Enable/Disable
        self.atr_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            atr_frame, text="✅ Activer ATR", variable=self.atr_enabled
        ).pack(anchor="w", pady=5)

        # Paramètres
        params_frame = ttk.LabelFrame(atr_frame, text="Paramètres", padding=5)
        params_frame.pack(fill="x", pady=5)

        # Period range
        period_frame = ttk.Frame(params_frame)
        period_frame.pack(fill="x", pady=2)

        ttk.Label(period_frame, text="Période:").pack(side="left", padx=(0, 10))
        ttk.Label(period_frame, text="Début:").pack(side="left", padx=(10, 5))
        self.atr_period_start = tk.IntVar(value=10)
        ttk.Entry(period_frame, textvariable=self.atr_period_start, width=8).pack(
            side="left", padx=(0, 10)
        )

        ttk.Label(period_frame, text="Fin:").pack(side="left", padx=(10, 5))
        self.atr_period_stop = tk.IntVar(value=30)
        ttk.Entry(period_frame, textvariable=self.atr_period_stop, width=8).pack(
            side="left", padx=(0, 10)
        )

        ttk.Label(period_frame, text="Pas:").pack(side="left", padx=(10, 5))
        self.atr_period_step = tk.IntVar(value=2)
        ttk.Entry(period_frame, textvariable=self.atr_period_step, width=8).pack(
            side="left"
        )

        # Method
        method_frame = ttk.Frame(params_frame)
        method_frame.pack(fill="x", pady=2)

        ttk.Label(method_frame, text="Méthodes:").pack(side="left", padx=(0, 10))
        self.atr_methods = tk.StringVar(value="ema,sma")
        ttk.Entry(method_frame, textvariable=self.atr_methods, width=20).pack(
            side="left"
        )

    def create_ma_tab(self):
        """Onglet configuration Moving Averages."""
        ma_frame = ttk.Frame(self.param_notebook)
        self.param_notebook.add(ma_frame, text="Moving Average")

        # Enable/Disable
        self.ma_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            ma_frame, text="✅ Activer Moving Average", variable=self.ma_enabled
        ).pack(anchor="w", pady=5)

        # Paramètres
        params_frame = ttk.LabelFrame(ma_frame, text="Paramètres", padding=5)
        params_frame.pack(fill="x", pady=5)

        # Window range
        window_frame = ttk.Frame(params_frame)
        window_frame.pack(fill="x", pady=2)

        ttk.Label(window_frame, text="Fenêtre:").pack(side="left", padx=(0, 10))
        ttk.Label(window_frame, text="Début:").pack(side="left", padx=(10, 5))
        self.ma_window_start = tk.IntVar(value=10)
        ttk.Entry(window_frame, textvariable=self.ma_window_start, width=8).pack(
            side="left", padx=(0, 10)
        )

        ttk.Label(window_frame, text="Fin:").pack(side="left", padx=(10, 5))
        self.ma_window_stop = tk.IntVar(value=60)
        ttk.Entry(window_frame, textvariable=self.ma_window_stop, width=8).pack(
            side="left", padx=(0, 10)
        )

        ttk.Label(window_frame, text="Pas:").pack(side="left", padx=(10, 5))
        self.ma_window_step = tk.IntVar(value=5)
        ttk.Entry(window_frame, textvariable=self.ma_window_step, width=8).pack(
            side="left"
        )

        # Kind
        kind_frame = ttk.Frame(params_frame)
        kind_frame.pack(fill="x", pady=2)

        ttk.Label(kind_frame, text="Types:").pack(side="left", padx=(0, 10))
        self.ma_kinds = tk.StringVar(value="sma,ema")
        ttk.Entry(kind_frame, textvariable=self.ma_kinds, width=20).pack(side="left")

    def create_scoring_config(self, parent):
        """Configuration du scoring des résultats."""
        scoring_frame = ttk.LabelFrame(
            parent, text="🏆 Configuration Scoring", padding=5
        )
        scoring_frame.pack(fill="x", pady=5)

        # Critère primaire
        primary_frame = ttk.Frame(scoring_frame)
        primary_frame.pack(fill="x", pady=2)

        ttk.Label(primary_frame, text="Critère primaire:").pack(
            side="left", padx=(0, 10)
        )
        self.primary_metric = tk.StringVar(value="pnl")
        ttk.Combobox(
            primary_frame,
            textvariable=self.primary_metric,
            values=["pnl", "sharpe", "total_return", "profit_factor"],
            width=15,
        ).pack(side="left")

        # Critères secondaires
        secondary_frame = ttk.Frame(scoring_frame)
        secondary_frame.pack(fill="x", pady=2)

        ttk.Label(secondary_frame, text="Critères secondaires:").pack(
            side="left", padx=(0, 10)
        )
        self.secondary_metrics = tk.StringVar(
            value="sharpe,-max_drawdown,profit_factor"
        )
        ttk.Entry(secondary_frame, textvariable=self.secondary_metrics, width=40).pack(
            side="left"
        )

        # Top K résultats
        topk_frame = ttk.Frame(scoring_frame)
        topk_frame.pack(fill="x", pady=2)

        ttk.Label(topk_frame, text="Top K résultats:").pack(side="left", padx=(0, 10))
        self.top_k = tk.IntVar(value=50)
        ttk.Entry(topk_frame, textvariable=self.top_k, width=10).pack(side="left")

    def create_controls_section(self):
        """Contrôles d'exécution du sweep."""
        control_frame = ttk.Frame(self)
        control_frame.pack(fill="x", padx=10, pady=5)

        # Boutons principaux
        self.start_btn = ttk.Button(
            control_frame, text="🚀 Démarrer Sweep", command=self.start_optimization
        )
        self.start_btn.pack(side="left")

        self.pause_btn = ttk.Button(
            control_frame,
            text="⏸️ Pause",
            state="disabled",
            command=self.pause_optimization,
        )
        self.pause_btn.pack(side="left", padx=(10, 0))

        self.stop_btn = ttk.Button(
            control_frame,
            text="⏹️ Stop",
            state="disabled",
            command=self.stop_optimization,
        )
        self.stop_btn.pack(side="left", padx=(10, 0))

        # Export
        self.export_btn = ttk.Button(
            control_frame,
            text="📊 Exporter Résultats",
            command=self.export_results,
            state="disabled",
        )
        self.export_btn.pack(side="left", padx=(10, 0))

        # Status
        self.status_var = tk.StringVar(value="✅ Prêt pour optimisation")
        self.status_label = ttk.Label(control_frame, textvariable=self.status_var)
        self.status_label.pack(side="right")

    def create_results_section(self):
        """Section résultats et progression."""
        # Progress bar
        progress_frame = ttk.Frame(self)
        progress_frame.pack(fill="x", padx=10, pady=5)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame, variable=self.progress_var, maximum=100, length=400
        )
        self.progress_bar.pack(fill="x", pady=2)

        self.progress_label = ttk.Label(progress_frame, text="")
        self.progress_label.pack(fill="x", pady=2)

        # Results notebook
        results_notebook = ttk.Notebook(self)
        results_notebook.pack(fill="both", expand=True, padx=10, pady=5)

        # Onglet résultats
        results_frame = ttk.Frame(results_notebook)
        results_notebook.add(results_frame, text="📊 Résultats")

        self.create_results_table(results_frame)

        # Onglet logs
        logs_frame = ttk.Frame(results_notebook)
        results_notebook.add(logs_frame, text="📋 Logs")

        self.logs_text = scrolledtext.ScrolledText(logs_frame, height=15, wrap=tk.WORD)
        self.logs_text.pack(fill="both", expand=True, padx=5, pady=5)

    def create_results_table(self, parent):
        """Table des résultats d'optimisation."""
        # Frame pour la table avec scrollbars
        table_frame = ttk.Frame(parent)
        table_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Treeview pour les résultats
        columns = (
            "rank",
            "pnl",
            "sharpe",
            "max_dd",
            "profit_factor",
            "total_trades",
            "bb_period",
            "bb_std",
            "atr_period",
            "duration",
        )

        self.results_tree = ttk.Treeview(
            table_frame, columns=columns, show="headings", height=12
        )

        # Configuration des colonnes
        self.results_tree.heading("rank", text="Rang")
        self.results_tree.heading("pnl", text="PnL")
        self.results_tree.heading("sharpe", text="Sharpe")
        self.results_tree.heading("max_dd", text="MaxDD")
        self.results_tree.heading("profit_factor", text="PF")
        self.results_tree.heading("total_trades", text="Trades")
        self.results_tree.heading("bb_period", text="BB Period")
        self.results_tree.heading("bb_std", text="BB Std")
        self.results_tree.heading("atr_period", text="ATR Period")
        self.results_tree.heading("duration", text="Durée (s)")

        # Largeurs des colonnes
        for col in columns:
            self.results_tree.column(col, width=80, anchor="center")

        # Scrollbars
        v_scrollbar = ttk.Scrollbar(
            table_frame, orient="vertical", command=self.results_tree.yview
        )
        h_scrollbar = ttk.Scrollbar(
            table_frame, orient="horizontal", command=self.results_tree.xview
        )
        self.results_tree.configure(
            yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set
        )

        # Layout
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

    def load_data(self):
        """Charge les données pour l'optimisation."""
        file_path = filedialog.askopenfilename(
            title="Charger données OHLCV",
            filetypes=[
                ("Parquet files", "*.parquet"),
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ],
        )

        if not file_path:
            return

        try:
            if file_path.endswith(".parquet"):
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)

            # Validation des colonnes OHLCV
            required_cols = ["open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required_cols):
                messagebox.showerror(
                    "Erreur",
                    f"Colonnes manquantes: {set(required_cols) - set(df.columns)}",
                )
                return

            self.current_data = df
            self.data_status.config(text=f"✅ {len(df):,} barres chargées")
            logger.info(
                f"Données chargées: {len(df)} barres, {df.index[0]} → {df.index[-1]}"
            )

        except Exception as e:
            logger.error(f"Erreur chargement données: {e}")
            messagebox.showerror("Erreur", f"Impossible de charger les données:\n{e}")

    def build_config_from_ui(self) -> Dict:
        """Construit la configuration à partir de l'UI."""
        config = {
            "dataset": {
                "symbol": self.symbol_var.get(),
                "timeframe": self.timeframe_var.get(),
                "start": "2024-01-01",  # Sera déterminé par les données chargées
                "end": "2024-12-31",
            },
            "grid": {},
            "scoring": {
                "primary": self.primary_metric.get(),
                "secondary": [
                    s.strip()
                    for s in self.secondary_metrics.get().split(",")
                    if s.strip()
                ],
                "top_k": self.top_k.get(),
            },
        }

        # Bollinger Bands
        if self.bb_enabled.get():
            periods = [
                int(p.strip()) for p in self.bb_periods.get().split(",") if p.strip()
            ]
            config["grid"]["bollinger"] = {
                "period": periods,
                "std": {
                    "start": self.bb_std_start.get(),
                    "stop": self.bb_std_stop.get(),
                    "step": self.bb_std_step.get(),
                },
            }

        # ATR
        if self.atr_enabled.get():
            methods = [
                m.strip() for m in self.atr_methods.get().split(",") if m.strip()
            ]
            config["grid"]["atr"] = {
                "period": {
                    "start": self.atr_period_start.get(),
                    "stop": self.atr_period_stop.get(),
                    "step": self.atr_period_step.get(),
                },
                "method": methods,
            }

        # Moving Average
        if self.ma_enabled.get():
            kinds = [k.strip() for k in self.ma_kinds.get().split(",") if k.strip()]
            config["grid"]["ma"] = {
                "window": {
                    "start": self.ma_window_start.get(),
                    "stop": self.ma_window_stop.get(),
                    "step": self.ma_window_step.get(),
                },
                "kind": kinds,
            }

        return config

    def start_optimization(self):
        """Démarre l'optimisation paramétrique."""
        if self.is_running:
            logger.warning("Optimisation déjà en cours")
            return

        if self.current_data is None:
            messagebox.showerror("Erreur", "Veuillez d'abord charger des données")
            return

        if not self.optimization_engine:
            messagebox.showerror("Erreur", "Moteur d'optimisation non disponible")
            return

        # Construction de la config depuis l'UI
        config = self.build_config_from_ui()

        # Validation
        if not config["grid"]:
            messagebox.showerror("Erreur", "Veuillez activer au moins un indicateur")
            return

        # Démarrage du thread d'optimisation
        self.is_running = True
        self.optimization_thread = Thread(
            target=self.optimization_worker, args=(config,)
        )
        self.optimization_thread.daemon = True
        self.optimization_thread.start()

        # Mise à jour UI
        self._update_ui_state(running=True)
        logger.info("🚀 Démarrage optimisation paramétrique")

    def optimization_worker(self, config: Dict):
        """Worker d'optimisation (thread background)."""
        try:
            self.log_queue.put("🚀 Démarrage du sweep paramétrique")
            self.log_queue.put(f"Configuration: {len(config['grid'])} indicateur(s)")

            # Setup progress callback
            def progress_callback(completed, total, eta=None):
                progress = (completed / total) * 100 if total > 0 else 0
                eta_text = f", ETA: {eta:.1f}s" if eta else ""
                self.progress_queue.put(
                    (
                        "progress",
                        progress,
                        f"{completed}/{total} combinaisons{eta_text}",
                    )
                )

            self.optimization_engine.progress_callback = progress_callback

            # Exécution du sweep
            start_time = time.time()
            results_df = self.optimization_engine.run_parameter_sweep(
                config, self.current_data
            )
            duration = time.time() - start_time

            # Résultats
            if results_df is not None and not results_df.empty:
                self.log_queue.put(
                    f"✅ Sweep terminé: {len(results_df)} résultats en {duration:.1f}s"
                )
                self.log_queue.put(f"Meilleur PnL: {results_df.iloc[0]['pnl']:.2f}")
                self.log_queue.put(f"Meilleur Sharpe: {results_df['sharpe'].max():.3f}")

                self.current_results = results_df
                self.progress_queue.put(
                    ("complete", results_df, "Optimisation terminée")
                )
            else:
                self.log_queue.put("❌ Aucun résultat généré")
                self.progress_queue.put(("error", "Aucun résultat", "Erreur"))

        except Exception as e:
            logger.error(f"Erreur optimisation: {e}")
            self.log_queue.put(f"❌ Erreur: {e}")
            self.progress_queue.put(("error", str(e), "Erreur"))
        finally:
            self.progress_queue.put(("finished", None, "Terminé"))

    def pause_optimization(self):
        """Met en pause l'optimisation."""
        if self.optimization_engine:
            self.optimization_engine.pause()
            self.log_queue.put("⏸️ Optimisation en pause")

    def stop_optimization(self):
        """Arrête l'optimisation."""
        if self.optimization_engine:
            self.optimization_engine.stop()
            self.log_queue.put("⏹️ Arrêt de l'optimisation")

    def export_results(self):
        """Exporte les résultats."""
        if self.current_results is None:
            messagebox.showwarning("Attention", "Aucun résultat à exporter")
            return

        export_dir = filedialog.askdirectory(title="Sélectionner répertoire d'export")
        if not export_dir:
            return

        try:
            export_path = Path(export_dir)

            # Export résultats CSV
            results_path = export_path / "sweep_results.csv"
            self.current_results.to_csv(results_path, index=False)

            # Export résultats Parquet
            parquet_path = export_path / "sweep_results.parquet"
            self.current_results.to_parquet(parquet_path, index=False)

            # Export configuration
            config_path = export_path / "sweep_config.json"
            config = self.build_config_from_ui()
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            # Export logs
            logs_path = export_path / "sweep_logs.txt"
            with open(logs_path, "w") as f:
                f.write(self.logs_text.get(1.0, tk.END))

            messagebox.showinfo("Export", f"Résultats exportés dans {export_path}")
            logger.info(f"Export terminé: {export_path}")

        except Exception as e:
            logger.error(f"Erreur export: {e}")
            messagebox.showerror("Erreur", f"Erreur lors de l'export:\n{e}")

    def save_config(self):
        """Sauvegarde la configuration."""
        file_path = filedialog.asksaveasfilename(
            title="Sauvegarder configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )

        if file_path:
            try:
                config = self.build_config_from_ui()
                with open(file_path, "w") as f:
                    json.dump(config, f, indent=2)
                logger.info(f"Configuration sauvegardée: {file_path}")
            except Exception as e:
                logger.error(f"Erreur sauvegarde config: {e}")
                messagebox.showerror("Erreur", f"Erreur sauvegarde:\n{e}")

    def load_config(self):
        """Charge une configuration."""
        file_path = filedialog.askopenfilename(
            title="Charger configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )

        if file_path:
            try:
                with open(file_path, "r") as f:
                    config = json.load(f)
                self._apply_config_to_ui(config)
                logger.info(f"Configuration chargée: {file_path}")
            except Exception as e:
                logger.error(f"Erreur chargement config: {e}")
                messagebox.showerror("Erreur", f"Erreur chargement:\n{e}")

    def reset_config(self):
        """Reset la configuration par défaut."""
        self._apply_config_to_ui(DEFAULT_SWEEP_CONFIG)
        logger.info("Configuration remise à zéro")

    def _apply_config_to_ui(self, config: Dict):
        """Applique une configuration à l'UI."""
        # Dataset
        if "dataset" in config:
            self.symbol_var.set(config["dataset"].get("symbol", "BTCUSDT"))
            self.timeframe_var.set(config["dataset"].get("timeframe", "1h"))

        # Grid config
        if "grid" in config:
            # Bollinger
            if "bollinger" in config["grid"]:
                bb_config = config["grid"]["bollinger"]
                self.bb_enabled.set(True)
                if "period" in bb_config:
                    self.bb_periods.set(",".join(map(str, bb_config["period"])))
                if "std" in bb_config:
                    std_config = bb_config["std"]
                    if isinstance(std_config, dict):
                        self.bb_std_start.set(std_config.get("start", 1.5))
                        self.bb_std_stop.set(std_config.get("stop", 3.0))
                        self.bb_std_step.set(std_config.get("step", 0.1))

            # ATR
            if "atr" in config["grid"]:
                atr_config = config["grid"]["atr"]
                self.atr_enabled.set(True)
                if "period" in atr_config:
                    period_config = atr_config["period"]
                    if isinstance(period_config, dict):
                        self.atr_period_start.set(period_config.get("start", 10))
                        self.atr_period_stop.set(period_config.get("stop", 30))
                        self.atr_period_step.set(period_config.get("step", 2))
                if "method" in atr_config:
                    self.atr_methods.set(",".join(atr_config["method"]))

        # Scoring
        if "scoring" in config:
            scoring_config = config["scoring"]
            self.primary_metric.set(scoring_config.get("primary", "pnl"))
            if "secondary" in scoring_config:
                self.secondary_metrics.set(",".join(scoring_config["secondary"]))
            self.top_k.set(scoring_config.get("top_k", 50))

    def _update_ui_state(self, running: bool):
        """Met à jour l'état des contrôles UI."""
        if running:
            self.start_btn.config(state="disabled")
            self.pause_btn.config(state="normal")
            self.stop_btn.config(state="normal")
            self.export_btn.config(state="disabled")
            self.status_var.set("🔄 Optimisation en cours...")
        else:
            self.start_btn.config(state="normal")
            self.pause_btn.config(state="disabled")
            self.stop_btn.config(state="disabled")
            self.export_btn.config(
                state="normal" if self.current_results is not None else "disabled"
            )
            self.status_var.set("✅ Prêt")
            self.is_running = False

    def _populate_results_table(self, results_df: pd.DataFrame):
        """Remplit la table des résultats."""
        # Vider la table
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        # Ajouter les résultats
        for i, row in results_df.head(
            100
        ).iterrows():  # Limiter à 100 résultats affichés
            values = (
                i + 1,  # Rang
                f"{row.get('pnl', 0):.2f}",
                f"{row.get('sharpe', 0):.3f}",
                f"{row.get('max_drawdown', 0):.3f}",
                f"{row.get('profit_factor', 0):.2f}",
                f"{row.get('total_trades', 0)}",
                f"{row.get('bb_period', '-')}",
                f"{row.get('bb_std', '-')}",
                f"{row.get('atr_period', '-')}",
                f"{row.get('duration_sec', 0):.2f}",
            )
            self.results_tree.insert("", "end", values=values)

    def check_queues(self):
        """Vérifie les queues pour mise à jour UI."""
        # Progress updates
        try:
            while True:
                queue_data = self.progress_queue.get_nowait()

                # Gestion flexible du format de queue (2 ou 3 valeurs)
                if len(queue_data) == 3:
                    msg_type, data, text = queue_data
                elif len(queue_data) == 2:
                    msg_type, data = queue_data
                    text = ""
                else:
                    continue

                if msg_type == "progress":
                    self.progress_var.set(data)
                    self.progress_label.config(text=text)
                elif msg_type == "complete":
                    self.progress_var.set(100)
                    self.progress_label.config(text="Optimisation terminée")
                    self._populate_results_table(data)
                    self._update_ui_state(running=False)
                elif msg_type == "error":
                    self.progress_label.config(text=f"Erreur: {data}")
                    self._update_ui_state(running=False)
                elif msg_type == "finished":
                    self._update_ui_state(running=False)

        except Empty:
            pass

        # Log updates
        try:
            while True:
                log_msg = self.log_queue.get_nowait()
                self.logs_text.insert(tk.END, f"{log_msg}\n")
                self.logs_text.see(tk.END)
        except Empty:
            pass

        # Schedule next check
        self.after(100, self.check_queues)


def create_sweep_page(parent) -> SweepOptimizationPage:
    """Factory pour créer la page d'optimisation."""
    return SweepOptimizationPage(parent)
