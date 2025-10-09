#!/usr/bin/env python3
"""
ThreadX - Interface Graphique Diversity Data Manager
==================================================

Interface complète pour la gestion et mise à jour des tokens crypto.
Fonctionnalités :
- Sélection par groupes (L1, L2, DeFi, AI, Gaming, Meme)
- Configuration timeframes et indicateurs
- Mise à jour automatique avec barre de progression
- Export multi-formats (CSV, Parquet, Excel)
- Logs en temps réel avec coloration
- Sauvegarde organisée par groupe/symbole/timeframe
"""

import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import threading
import queue
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import time

# Ajouter le chemin ThreadX au sys.path
THREADX_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(THREADX_ROOT / "src"))

# Tentative d'import ThreadX
try:
    from threadx.data.providers.token_diversity import (
        TokenDiversityManager,
        IndicatorSpec,
        PriceSourceSpec,
    )

    THREADX_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  ThreadX modules non disponibles: {e}")
    THREADX_AVAILABLE = False

# Configuration par défaut des groupes de tokens
DEFAULT_GROUPS = {
    "L1": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT", "DOTUSDT"],
    "L2": ["ARBUSDT", "OPUSDT", "MATICUSDT", "ATOMUSDT", "AVAXUSDT", "NEARUSDT"],
    "DeFi": ["UNIUSDT", "AAVEUSDT", "COMPUSDT", "MKRUSDT", "CRVUSDT", "SUSHIUSDT"],
    "AI": ["FETUSD", "RENDERUSDT", "AGIXUSDT", "OCEANUSDT", "THETAUSDT", "GRTUSDT"],
    "Gaming": ["AXSUSDT", "SANDUSDT", "MANAUSDT", "ENJUSDT", "CHZUSDT", "GALAUSDT"],
    "Meme": ["DOGEUSDT", "SHIBUSD", "PEPEUSDT", "FLOKIUSDT", "BONKUSDT", "WIFUSDT"],
    "Infrastructure": ["LINKUSDT", "FILUSDT", "ICPUSDT", "HBARUSDT"],
    "Privacy": ["XMRUSDT", "ZECUSDT", "DASHUSDT"],
}

DEFAULT_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]
DEFAULT_INDICATORS = [
    "RSI",
    "MACD",
    "Bollinger Bands",
    "SMA 20",
    "EMA 50",
    "ATR",
    "VWAP",
]


class QueueHandler(logging.Handler):
    """Handler pour rediriger les logs vers une queue."""

    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(record)


class ThreadXDataManagerGUI:
    """Interface graphique principale ThreadX."""

    def __init__(self):
        self.root = tk.Tk()
        self.setup_window()
        self.setup_paths()
        self.setup_logging()

        # État de l'application
        self.is_running = False
        self.current_thread = None
        self.manager = None

        # Queues pour communication inter-threads
        self.progress_queue = queue.Queue()
        self.log_queue = queue.Queue()

        # Variables d'interface
        self.group_vars = {}
        self.timeframe_vars = {}
        self.indicator_vars = {}

        # Création de l'interface
        self.create_interface()
        self.init_threadx_manager()
        self.start_queue_monitoring()

    def setup_window(self):
        """Configuration de la fenêtre principale."""
        self.root.title("ThreadX - Diversity Data Manager")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)

        # Centrer la fenêtre
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (1400 // 2)
        y = (self.root.winfo_screenheight() // 2) - (900 // 2)
        self.root.geometry(f"1400x900+{x}+{y}")

        # Style moderne
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")

        # Couleurs personnalisées
        style.configure("Title.TLabel", font=("Arial", 12, "bold"))
        style.configure("Header.TLabel", font=("Arial", 10, "bold"))
        style.configure("Success.TLabel", foreground="green")
        style.configure("Error.TLabel", foreground="red")

    def setup_paths(self):
        """Configuration des chemins de sauvegarde."""
        self.base_path = THREADX_ROOT

        # Structure de dossiers
        self.paths = {
            "data": self.base_path / "data",
            "cache": self.base_path / "data" / "cache",
            "processed": self.base_path / "data" / "processed",
            "exports": self.base_path / "data" / "exports",
            "logs": self.base_path / "logs",
            "configs": self.base_path / "configs",
        }

        # Créer tous les dossiers
        for path_name, path in self.paths.items():
            path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Dossier {path_name}: {path}")

        # Sous-dossiers par groupe
        for group_name in DEFAULT_GROUPS.keys():
            group_dir = self.paths["processed"] / group_name.lower()
            group_dir.mkdir(exist_ok=True)
            print(f"✅ Groupe {group_name}: {group_dir}")

    def setup_logging(self):
        """Configuration du système de logs."""
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = self.paths["logs"] / f"threadx_gui_{timestamp}.log"

        # Logger principal
        self.logger = logging.getLogger("ThreadXGUI")
        self.logger.setLevel(logging.INFO)

        # Handler vers fichier
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Handler vers queue pour GUI
        queue_handler = QueueHandler(self.log_queue)
        queue_handler.setFormatter(file_formatter)
        self.logger.addHandler(queue_handler)

        self.logger.info("🚀 ThreadX Data Manager GUI démarré")

    def init_threadx_manager(self):
        """Initialise le TokenDiversityManager."""
        if not THREADX_AVAILABLE:
            self.logger.warning("⚠️  ThreadX non disponible - Mode simulation")
            return

        try:
            self.manager = TokenDiversityManager()
            self.logger.info("✅ TokenDiversityManager initialisé")
        except Exception as e:
            self.logger.error(f"❌ Erreur initialisation manager: {e}")
            self.manager = None

    def create_interface(self):
        """Crée l'interface utilisateur complète."""
        # Frame principal avec onglets
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Onglets
        self.create_main_tab(notebook)
        self.create_config_tab(notebook)
        self.create_export_tab(notebook)
        self.create_logs_tab(notebook)

    def create_main_tab(self, parent):
        """Onglet principal de gestion des données."""
        main_frame = ttk.Frame(parent)
        parent.add(main_frame, text="📊 Mise à Jour Données")

        # ==> Section Sélection des Groupes
        groups_frame = ttk.LabelFrame(
            main_frame, text="🎯 Sélection des Groupes de Tokens"
        )
        groups_frame.pack(fill=tk.X, padx=5, pady=5)

        # Checkboxes groupes en grille
        groups_grid = ttk.Frame(groups_frame)
        groups_grid.pack(padx=10, pady=10)

        row, col = 0, 0
        for group_name, symbols in DEFAULT_GROUPS.items():
            var = tk.BooleanVar(value=True)
            self.group_vars[group_name] = var

            cb = ttk.Checkbutton(
                groups_grid, text=f"{group_name} ({len(symbols)})", variable=var
            )
            cb.grid(row=row, column=col, sticky=tk.W, padx=10, pady=3)

            col += 1
            if col > 3:  # 4 colonnes
                col = 0
                row += 1

        # Boutons sélection rapide
        quick_buttons = ttk.Frame(groups_frame)
        quick_buttons.pack(pady=5)

        ttk.Button(
            quick_buttons, text="✅ Tout sélectionner", command=self.select_all_groups
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            quick_buttons,
            text="❌ Tout désélectionner",
            command=self.deselect_all_groups,
        ).pack(side=tk.LEFT, padx=5)

        # ==> Section Timeframes
        tf_frame = ttk.LabelFrame(main_frame, text="⏰ Timeframes")
        tf_frame.pack(fill=tk.X, padx=5, pady=5)

        tf_grid = ttk.Frame(tf_frame)
        tf_grid.pack(padx=10, pady=10)

        col = 0
        for tf in DEFAULT_TIMEFRAMES:
            var = tk.BooleanVar(value=tf in ["1h", "4h", "1d"])  # Défaut
            self.timeframe_vars[tf] = var

            ttk.Checkbutton(tf_grid, text=tf, variable=var).grid(
                row=0, column=col, sticky=tk.W, padx=15, pady=3
            )
            col += 1

        # ==> Section Indicateurs
        ind_frame = ttk.LabelFrame(main_frame, text="📈 Indicateurs Techniques")
        ind_frame.pack(fill=tk.X, padx=5, pady=5)

        ind_grid = ttk.Frame(ind_frame)
        ind_grid.pack(padx=10, pady=10)

        row, col = 0, 0
        for indicator in DEFAULT_INDICATORS:
            var = tk.BooleanVar(value=indicator in ["RSI", "MACD", "SMA 20"])  # Défaut
            self.indicator_vars[indicator] = var

            ttk.Checkbutton(ind_grid, text=indicator, variable=var).grid(
                row=row, column=col, sticky=tk.W, padx=15, pady=3
            )

            col += 1
            if col > 3:
                col = 0
                row += 1

        # ==> Section Configuration Temporelle
        time_frame = ttk.LabelFrame(main_frame, text="📅 Période de Données")
        time_frame.pack(fill=tk.X, padx=5, pady=5)

        time_grid = ttk.Frame(time_frame)
        time_grid.pack(padx=10, pady=10)

        ttk.Label(time_grid, text="Période:").grid(row=0, column=0, sticky=tk.W, padx=5)

        self.period_var = tk.StringVar(value="30_days")
        periods = [
            ("7 jours", "7_days"),
            ("30 jours", "30_days"),
            ("90 jours", "90_days"),
            ("6 mois", "180_days"),
            ("1 an", "365_days"),
        ]

        col = 1
        for text, value in periods:
            ttk.Radiobutton(
                time_grid, text=text, variable=self.period_var, value=value
            ).grid(row=0, column=col, padx=10)
            col += 1

        # ==> Section Contrôles
        control_frame = ttk.LabelFrame(main_frame, text="🎮 Contrôles")
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Bouton principal
        main_btn_frame = ttk.Frame(control_frame)
        main_btn_frame.pack(pady=15)

        self.start_button = ttk.Button(
            main_btn_frame,
            text="🚀 METTRE À JOUR TOUS LES TOKENS",
            command=self.start_data_update,
            style="Title.TLabel",
        )
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.stop_button = ttk.Button(
            main_btn_frame,
            text="⏹️ ARRÊTER",
            command=self.stop_data_update,
            state=tk.DISABLED,
        )
        self.stop_button.pack(side=tk.LEFT, padx=10)

        # Barre de progression
        progress_frame = ttk.Frame(control_frame)
        progress_frame.pack(fill=tk.X, padx=10, pady=10)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame, variable=self.progress_var, maximum=100, length=400
        )
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.progress_label = ttk.Label(progress_frame, text="Prêt")
        self.progress_label.pack(side=tk.RIGHT, padx=10)

        # Statistiques temps réel
        stats_frame = ttk.Frame(control_frame)
        stats_frame.pack(fill=tk.X, padx=10, pady=5)

        self.stats_label = ttk.Label(stats_frame, text="📊 Aucune opération en cours")
        self.stats_label.pack()

    def create_config_tab(self, parent):
        """Onglet de configuration."""
        config_frame = ttk.Frame(parent)
        parent.add(config_frame, text="⚙️ Configuration")

        # Configuration des dossiers
        paths_frame = ttk.LabelFrame(config_frame, text="📁 Dossiers de Sauvegarde")
        paths_frame.pack(fill=tk.X, padx=5, pady=5)

        paths_info = ttk.Frame(paths_frame)
        paths_info.pack(padx=10, pady=10)

        ttk.Label(paths_info, text="Dossier de base:", style="Header.TLabel").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        ttk.Label(paths_info, text=str(self.base_path)).grid(
            row=0, column=1, sticky=tk.W, padx=20
        )

        ttk.Label(paths_info, text="Données traitées:", style="Header.TLabel").grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        ttk.Label(paths_info, text=str(self.paths["processed"])).grid(
            row=1, column=1, sticky=tk.W, padx=20
        )

        ttk.Label(paths_info, text="Exports:", style="Header.TLabel").grid(
            row=2, column=0, sticky=tk.W, pady=5
        )
        ttk.Label(paths_info, text=str(self.paths["exports"])).grid(
            row=2, column=1, sticky=tk.W, padx=20
        )

        # Configuration performance
        perf_frame = ttk.LabelFrame(config_frame, text="⚡ Performance")
        perf_frame.pack(fill=tk.X, padx=5, pady=5)

        perf_grid = ttk.Frame(perf_frame)
        perf_grid.pack(padx=10, pady=10)

        ttk.Label(perf_grid, text="Threads parallèles:").grid(
            row=0, column=0, sticky=tk.W, padx=5
        )
        self.threads_var = tk.IntVar(value=4)
        ttk.Spinbox(
            perf_grid, from_=1, to=16, textvariable=self.threads_var, width=5
        ).grid(row=0, column=1, padx=10)

        ttk.Label(perf_grid, text="Cache TTL (min):").grid(
            row=1, column=0, sticky=tk.W, padx=5
        )
        self.cache_ttl_var = tk.IntVar(value=15)
        ttk.Spinbox(
            perf_grid, from_=5, to=120, textvariable=self.cache_ttl_var, width=5
        ).grid(row=1, column=1, padx=10)

        # Boutons configuration
        config_buttons = ttk.Frame(config_frame)
        config_buttons.pack(pady=20)

        ttk.Button(
            config_buttons, text="💾 Sauvegarder Config", command=self.save_config
        ).pack(side=tk.LEFT, padx=10)
        ttk.Button(
            config_buttons, text="🔄 Recharger Config", command=self.load_config
        ).pack(side=tk.LEFT, padx=10)
        ttk.Button(
            config_buttons,
            text="📂 Ouvrir Dossier Données",
            command=self.open_data_folder,
        ).pack(side=tk.LEFT, padx=10)

    def create_export_tab(self, parent):
        """Onglet d'export et visualisation."""
        export_frame = ttk.Frame(parent)
        parent.add(export_frame, text="📤 Export & Visualisation")

        # Liste des données disponibles
        list_frame = ttk.LabelFrame(export_frame, text="📋 Données Disponibles")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Treeview avec colonnes
        columns = ("Groupe", "Symbole", "Timeframe", "Lignes", "Dernière MAJ", "Taille")
        self.data_tree = ttk.Treeview(
            list_frame, columns=columns, show="headings", height=15
        )

        # Configuration des colonnes
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=120)

        # Scrollbars
        tree_scroll_v = ttk.Scrollbar(
            list_frame, orient=tk.VERTICAL, command=self.data_tree.yview
        )
        tree_scroll_h = ttk.Scrollbar(
            list_frame, orient=tk.HORIZONTAL, command=self.data_tree.xview
        )
        self.data_tree.configure(
            yscrollcommand=tree_scroll_v.set, xscrollcommand=tree_scroll_h.set
        )

        self.data_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll_v.pack(side=tk.RIGHT, fill=tk.Y)
        tree_scroll_h.pack(side=tk.BOTTOM, fill=tk.X)

        # Boutons d'export
        export_buttons = ttk.Frame(export_frame)
        export_buttons.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            export_buttons, text="🔄 Actualiser Liste", command=self.refresh_data_list
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            export_buttons,
            text="📊 Export CSV",
            command=lambda: self.export_selected_data("csv"),
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            export_buttons,
            text="📋 Export Excel",
            command=lambda: self.export_selected_data("excel"),
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            export_buttons,
            text="🗄️ Export Parquet",
            command=lambda: self.export_selected_data("parquet"),
        ).pack(side=tk.LEFT, padx=5)

    def create_logs_tab(self, parent):
        """Onglet des logs."""
        logs_frame = ttk.Frame(parent)
        parent.add(logs_frame, text="📋 Logs")

        # Zone de logs
        self.logs_text = scrolledtext.ScrolledText(
            logs_frame, wrap=tk.WORD, width=100, height=35, font=("Consolas", 9)
        )
        self.logs_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tags de coloration
        self.logs_text.tag_config("INFO", foreground="blue")
        self.logs_text.tag_config("WARNING", foreground="orange")
        self.logs_text.tag_config("ERROR", foreground="red")
        self.logs_text.tag_config("SUCCESS", foreground="green")

        # Contrôles logs
        logs_controls = ttk.Frame(logs_frame)
        logs_controls.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(logs_controls, text="🧹 Effacer Logs", command=self.clear_logs).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(
            logs_controls, text="💾 Sauvegarder Logs", command=self.save_logs
        ).pack(side=tk.LEFT, padx=5)

        # Niveau de logs
        level_frame = ttk.Frame(logs_controls)
        level_frame.pack(side=tk.RIGHT)

        ttk.Label(level_frame, text="Niveau:").pack(side=tk.LEFT, padx=5)
        self.log_level_var = tk.StringVar(value="INFO")
        ttk.Combobox(
            level_frame,
            textvariable=self.log_level_var,
            values=["DEBUG", "INFO", "WARNING", "ERROR"],
            width=10,
            state="readonly",
        ).pack(side=tk.LEFT)

    # ========================================
    # Event Handlers - Sélection
    # ========================================

    def select_all_groups(self):
        """Sélectionne tous les groupes."""
        for var in self.group_vars.values():
            var.set(True)
        self.logger.info("✅ Tous les groupes sélectionnés")

    def deselect_all_groups(self):
        """Désélectionne tous les groupes."""
        for var in self.group_vars.values():
            var.set(False)
        self.logger.info("❌ Tous les groupes désélectionnés")

    # ========================================
    # Configuration Management
    # ========================================

    def save_config(self):
        """Sauvegarde la configuration actuelle."""
        config_data = {
            "selected_groups": [
                name for name, var in self.group_vars.items() if var.get()
            ],
            "selected_timeframes": [
                tf for tf, var in self.timeframe_vars.items() if var.get()
            ],
            "selected_indicators": [
                ind for ind, var in self.indicator_vars.items() if var.get()
            ],
            "period": self.period_var.get(),
            "threads": self.threads_var.get(),
            "cache_ttl": self.cache_ttl_var.get(),
        }

        config_file = self.paths["configs"] / "gui_config.json"
        try:
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"💾 Configuration sauvegardée: {config_file}")
            messagebox.showinfo(
                "Configuration", "Configuration sauvegardée avec succès!"
            )
        except Exception as e:
            self.logger.error(f"❌ Erreur sauvegarde config: {e}")
            messagebox.showerror("Erreur", f"Erreur lors de la sauvegarde:\n{e}")

    def load_config(self):
        """Charge la configuration depuis le fichier."""
        config_file = self.paths["configs"] / "gui_config.json"

        if not config_file.exists():
            messagebox.showwarning(
                "Configuration", "Aucun fichier de configuration trouvé."
            )
            return

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)

            # Application de la configuration
            selected_groups = config_data.get("selected_groups", [])
            for name, var in self.group_vars.items():
                var.set(name in selected_groups)

            selected_timeframes = config_data.get("selected_timeframes", [])
            for tf, var in self.timeframe_vars.items():
                var.set(tf in selected_timeframes)

            selected_indicators = config_data.get("selected_indicators", [])
            for ind, var in self.indicator_vars.items():
                var.set(ind in selected_indicators)

            self.period_var.set(config_data.get("period", "30_days"))
            self.threads_var.set(config_data.get("threads", 4))
            self.cache_ttl_var.set(config_data.get("cache_ttl", 15))

            self.logger.info("🔄 Configuration rechargée avec succès")
            messagebox.showinfo("Configuration", "Configuration rechargée!")

        except Exception as e:
            self.logger.error(f"❌ Erreur rechargement config: {e}")
            messagebox.showerror("Erreur", f"Impossible de recharger:\n{e}")

    def open_data_folder(self):
        """Ouvre le dossier de données dans l'explorateur."""
        import subprocess
        import platform

        try:
            if platform.system() == "Windows":
                subprocess.run(["explorer", str(self.paths["processed"])], check=True)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", str(self.paths["processed"])], check=True)
            else:  # Linux
                subprocess.run(["xdg-open", str(self.paths["processed"])], check=True)
        except Exception as e:
            self.logger.error(f"❌ Erreur ouverture dossier: {e}")

    # ========================================
    # Data Update Management
    # ========================================

    def start_data_update(self):
        """Démarre la mise à jour des données."""
        # Validation sélections
        selected_groups = [name for name, var in self.group_vars.items() if var.get()]
        selected_timeframes = [
            tf for tf, var in self.timeframe_vars.items() if var.get()
        ]
        selected_indicators = [
            ind for ind, var in self.indicator_vars.items() if var.get()
        ]

        if not selected_groups:
            messagebox.showwarning(
                "Sélection", "Veuillez sélectionner au moins un groupe."
            )
            return

        if not selected_timeframes:
            messagebox.showwarning(
                "Sélection", "Veuillez sélectionner au moins un timeframe."
            )
            return

        # Préparation des paramètres
        params = {
            "groups": selected_groups,
            "timeframes": selected_timeframes,
            "indicators": selected_indicators,
            "period": self.period_var.get(),
            "threads": self.threads_var.get(),
            "cache_ttl": self.cache_ttl_var.get(),
        }

        # Changement d'état UI
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress_var.set(0)

        # Démarrage thread worker
        self.current_thread = threading.Thread(
            target=self.data_update_worker, args=(params,), daemon=True
        )
        self.current_thread.start()

        self.logger.info(
            f"🚀 Mise à jour démarrée: {len(selected_groups)} groupes, {len(selected_timeframes)} timeframes"
        )

    def stop_data_update(self):
        """Arrête la mise à jour en cours."""
        self.is_running = False
        self.logger.info("⏹️ Arrêt demandé par l'utilisateur")

        # Reset UI
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.progress_label.config(text="Arrêté")

    def data_update_worker(self, params: Dict[str, Any]):
        """Worker thread pour la mise à jour des données."""
        try:
            if not THREADX_AVAILABLE or not self.manager:
                self.simulate_data_update(params)
                return

            # Calcul du nombre total d'opérations
            total_symbols = sum(
                len(DEFAULT_GROUPS[group]) for group in params["groups"]
            )
            total_operations = total_symbols * len(params["timeframes"])
            current_op = 0

            # Configuration période
            period_map = {
                "7_days": 7,
                "30_days": 30,
                "90_days": 90,
                "180_days": 180,
                "365_days": 365,
            }
            days_back = period_map.get(params["period"], 30)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            # Conversion indicateurs
            indicator_specs = self.convert_indicators_to_specs(params["indicators"])

            success_count = 0
            error_count = 0

            # Traitement par groupe
            for group_name in params["groups"]:
                if not self.is_running:
                    break

                symbols = DEFAULT_GROUPS.get(group_name, [])
                self.progress_queue.put(
                    ("status", f"📂 Groupe {group_name}: {len(symbols)} symboles")
                )

                for symbol in symbols:
                    if not self.is_running:
                        break

                    for timeframe in params["timeframes"]:
                        if not self.is_running:
                            break

                        try:
                            # Mise à jour progress
                            progress = (current_op / total_operations) * 100
                            self.progress_queue.put(
                                ("progress", progress, f"⚡ {symbol}@{timeframe}")
                            )

                            # Récupération données ThreadX
                            df, metadata = self.manager.prepare_dataframe(
                                market=symbol,
                                timeframe=timeframe,
                                start=start_date,
                                end=end_date,
                                indicators=indicator_specs,
                                price_source=PriceSourceSpec(name="stub", params={}),
                                cache_ttl_sec=params["cache_ttl"] * 60,
                            )

                            # Sauvegarde organisée
                            self.save_data_organized(
                                df, metadata, group_name, symbol, timeframe
                            )

                            success_count += 1
                            self.progress_queue.put(
                                (
                                    "log",
                                    "INFO",
                                    f"✅ {symbol}@{timeframe}: {len(df)} lignes, {metadata['execution_time_ms']:.1f}ms",
                                )
                            )

                        except Exception as e:
                            error_count += 1
                            self.progress_queue.put(
                                ("log", "ERROR", f"❌ {symbol}@{timeframe}: {e}")
                            )

                        current_op += 1
                        time.sleep(0.01)  # Petite pause pour éviter surcharge

            # Finalisation
            if self.is_running:
                self.progress_queue.put(("progress", 100, "✅ Terminé !"))
                self.progress_queue.put(
                    (
                        "log",
                        "INFO",
                        f"🎉 Mise à jour terminée: {success_count} succès, {error_count} erreurs",
                    )
                )

            self.progress_queue.put(
                (
                    "stats",
                    f"📊 Total: {success_count + error_count} | ✅ {success_count} | ❌ {error_count}",
                )
            )

        except Exception as e:
            self.progress_queue.put(("log", "ERROR", f"💥 Erreur critique: {e}"))
        finally:
            self.progress_queue.put(("finished", None))

    def simulate_data_update(self, params: Dict[str, Any]):
        """Simulation de mise à jour sans ThreadX."""
        total_symbols = sum(len(DEFAULT_GROUPS[group]) for group in params["groups"])
        total_operations = total_symbols * len(params["timeframes"])
        current_op = 0

        self.progress_queue.put(
            ("log", "WARNING", "⚠️  Mode simulation - ThreadX non disponible")
        )

        for group_name in params["groups"]:
            if not self.is_running:
                break

            symbols = DEFAULT_GROUPS.get(group_name, [])
            self.progress_queue.put(
                ("status", f"📂 Simulation {group_name}: {len(symbols)} symboles")
            )

            for symbol in symbols:
                if not self.is_running:
                    break

                for timeframe in params["timeframes"]:
                    if not self.is_running:
                        break

                    progress = (current_op / total_operations) * 100
                    self.progress_queue.put(
                        ("progress", progress, f"🔄 Simulation {symbol}@{timeframe}")
                    )

                    # Simulation avec délai
                    time.sleep(0.1)

                    self.progress_queue.put(
                        ("log", "INFO", f"✅ Simulation {symbol}@{timeframe} OK")
                    )
                    current_op += 1

        self.progress_queue.put(("progress", 100, "✅ Simulation terminée"))
        self.progress_queue.put(("finished", None))

    def convert_indicators_to_specs(self, indicators: List[str]) -> List[IndicatorSpec]:
        """Convertit les noms d'indicateurs GUI en IndicatorSpec."""
        specs = []
        for indicator in indicators:
            if indicator == "RSI":
                specs.append(IndicatorSpec(name="rsi", params={"window": 14}))
            elif indicator == "MACD":
                specs.append(
                    IndicatorSpec(
                        name="macd", params={"fast": 12, "slow": 26, "signal": 9}
                    )
                )
            elif indicator == "Bollinger Bands":
                specs.append(
                    IndicatorSpec(name="bbands", params={"window": 20, "n_std": 2.0})
                )
            elif indicator == "SMA 20":
                specs.append(IndicatorSpec(name="sma", params={"window": 20}))
            elif indicator == "EMA 50":
                specs.append(IndicatorSpec(name="ema", params={"window": 50}))
            elif indicator == "ATR":
                specs.append(IndicatorSpec(name="atr", params={"window": 14}))
            elif indicator == "VWAP":
                specs.append(IndicatorSpec(name="vwap", params={}))
        return specs

    def save_data_organized(
        self, df, metadata: Dict[str, Any], group_name: str, symbol: str, timeframe: str
    ):
        """Sauvegarde les données dans une structure organisée."""
        # Structure: data/processed/{group}/{symbol}/{timeframe}.parquet
        group_dir = self.paths["processed"] / group_name.lower()
        symbol_dir = group_dir / symbol.lower()
        symbol_dir.mkdir(parents=True, exist_ok=True)

        # Sauvegarde Parquet principal
        parquet_path = symbol_dir / f"{timeframe}.parquet"
        df.to_parquet(parquet_path, compression="snappy", index=True)

        # Métadonnées
        meta_path = symbol_dir / f"{timeframe}_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

        # CSV pour compatibilité
        csv_path = symbol_dir / f"{timeframe}.csv"
        df.to_csv(csv_path, index=True)

    # ========================================
    # Export Functions
    # ========================================

    def refresh_data_list(self):
        """Actualise la liste des données disponibles."""
        # Clear existing
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)

        processed_dir = self.paths["processed"]
        if not processed_dir.exists():
            return

        total_files = 0

        for group_dir in processed_dir.iterdir():
            if not group_dir.is_dir():
                continue

            group_name = group_dir.name.upper()

            for symbol_dir in group_dir.iterdir():
                if not symbol_dir.is_dir():
                    continue

                symbol_name = symbol_dir.name.upper()

                for parquet_file in symbol_dir.glob("*.parquet"):
                    try:
                        timeframe = parquet_file.stem

                        # Info fichier
                        stat = parquet_file.stat()
                        size_mb = stat.st_size / (1024 * 1024)
                        mtime = datetime.fromtimestamp(stat.st_mtime).strftime(
                            "%Y-%m-%d %H:%M"
                        )

                        # Métadonnées
                        meta_file = symbol_dir / f"{timeframe}_meta.json"
                        rows = "?"
                        if meta_file.exists():
                            try:
                                with open(meta_file, "r", encoding="utf-8") as f:
                                    meta = json.load(f)
                                    rows = meta.get("rows_processed", "?")
                            except:
                                pass

                        # Insertion dans le tree
                        self.data_tree.insert(
                            "",
                            tk.END,
                            values=(
                                group_name,
                                symbol_name,
                                timeframe,
                                rows,
                                mtime,
                                f"{size_mb:.2f} MB",
                            ),
                        )

                        total_files += 1

                    except Exception as e:
                        self.logger.warning(f"⚠️  Erreur lecture {parquet_file}: {e}")

        self.logger.info(f"📊 Liste actualisée: {total_files} fichiers")

    def export_selected_data(self, format_type: str):
        """Exporte les données sélectionnées."""
        selection = self.data_tree.selection()
        if not selection:
            messagebox.showwarning(
                "Export", "Veuillez sélectionner au moins une ligne."
            )
            return

        export_dir = self.paths["exports"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            if format_type == "excel":
                # Export Excel multi-feuilles
                excel_path = export_dir / f"threadx_export_{timestamp}.xlsx"

                import pandas as pd

                with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                    for item in selection:
                        values = self.data_tree.item(item, "values")
                        group, symbol, timeframe = values[0], values[1], values[2]

                        df = self.load_data_file(group, symbol, timeframe)
                        if df is not None:
                            sheet_name = f"{symbol}_{timeframe}"[:31]  # Limite Excel
                            df.to_excel(writer, sheet_name=sheet_name, index=True)

                self.logger.info(f"📋 Export Excel: {excel_path}")
                messagebox.showinfo("Export", f"Export Excel réussi:\n{excel_path}")

            else:
                # Export CSV/Parquet individuels
                exported_files = []

                for item in selection:
                    values = self.data_tree.item(item, "values")
                    group, symbol, timeframe = values[0], values[1], values[2]

                    df = self.load_data_file(group, symbol, timeframe)
                    if df is not None:
                        filename = f"{group}_{symbol}_{timeframe}_{timestamp}"

                        if format_type == "csv":
                            file_path = export_dir / f"{filename}.csv"
                            df.to_csv(file_path, index=True, encoding="utf-8")
                        elif format_type == "parquet":
                            file_path = export_dir / f"{filename}.parquet"
                            df.to_parquet(file_path, compression="snappy", index=True)

                        exported_files.append(file_path)

                self.logger.info(
                    f"📊 Export {format_type.upper()}: {len(exported_files)} fichiers"
                )
                messagebox.showinfo(
                    "Export",
                    f"Export {format_type.upper()} réussi:\n{len(exported_files)} fichiers dans {export_dir}",
                )

        except Exception as e:
            self.logger.error(f"❌ Erreur export {format_type}: {e}")
            messagebox.showerror("Erreur Export", f"Erreur:\n{e}")

    def load_data_file(self, group: str, symbol: str, timeframe: str):
        """Charge un fichier de données spécifique."""
        try:
            import pandas as pd

            group_dir = self.paths["processed"] / group.lower()
            symbol_dir = group_dir / symbol.lower()
            parquet_path = symbol_dir / f"{timeframe}.parquet"

            if parquet_path.exists():
                return pd.read_parquet(parquet_path)
            else:
                self.logger.warning(f"⚠️  Fichier non trouvé: {parquet_path}")
                return None
        except Exception as e:
            self.logger.error(f"❌ Erreur chargement {group}/{symbol}/{timeframe}: {e}")
            return None

    # ========================================
    # Logs Management
    # ========================================

    def clear_logs(self):
        """Efface les logs."""
        self.logs_text.delete(1.0, tk.END)
        self.logger.info("🧹 Logs effacés")

    def save_logs(self):
        """Sauvegarde les logs dans un fichier."""
        content = self.logs_text.get(1.0, tk.END)
        if not content.strip():
            messagebox.showwarning("Logs", "Aucun log à sauvegarder.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"threadx_logs_{timestamp}.txt"

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfilename=default_name,
        )

        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)

                self.logger.info(f"💾 Logs sauvegardés: {file_path}")
                messagebox.showinfo("Logs", f"Logs sauvegardés:\n{file_path}")
            except Exception as e:
                self.logger.error(f"❌ Erreur sauvegarde logs: {e}")

    # ========================================
    # Queue Monitoring
    # ========================================

    def start_queue_monitoring(self):
        """Démarre la surveillance des queues."""
        self.monitor_queues()

    def monitor_queues(self):
        """Surveille les queues pour mise à jour UI."""
        # Progress queue
        try:
            while True:
                msg_type, *args = self.progress_queue.get_nowait()

                if msg_type == "progress":
                    progress, label = args
                    self.progress_var.set(progress)
                    self.progress_label.config(text=label)

                elif msg_type == "status":
                    status = args[0]
                    self.stats_label.config(text=status)

                elif msg_type == "stats":
                    stats = args[0]
                    self.stats_label.config(text=stats)

                elif msg_type == "log":
                    level, message = args
                    self.add_log_message(level, message)

                elif msg_type == "finished":
                    self.start_button.config(state=tk.NORMAL)
                    self.stop_button.config(state=tk.DISABLED)
                    self.is_running = False

        except queue.Empty:
            pass

        # Log queue
        try:
            while True:
                record = self.log_queue.get_nowait()
                self.display_log_record(record)
        except queue.Empty:
            pass

        # Planifier prochaine vérification
        self.root.after(100, self.monitor_queues)

    def add_log_message(self, level: str, message: str):
        """Ajoute un message de log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {level}: {message}\n"

        self.logs_text.insert(tk.END, formatted, level)
        self.logs_text.see(tk.END)

        # Limiter la taille des logs
        lines = int(self.logs_text.index(tk.END).split(".")[0])
        if lines > 1000:
            self.logs_text.delete(1.0, f"{lines-1000}.0")

    def display_log_record(self, record):
        """Affiche un enregistrement de log."""
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        level = record.levelname
        message = record.getMessage()

        formatted = f"[{timestamp}] {level}: {message}\n"
        self.logs_text.insert(tk.END, formatted, level)
        self.logs_text.see(tk.END)

        # Limiter la taille
        lines = int(self.logs_text.index(tk.END).split(".")[0])
        if lines > 1000:
            self.logs_text.delete(1.0, f"{lines-1000}.0")

    # ========================================
    # Main Application
    # ========================================

    def run(self):
        """Lance l'application."""
        try:
            # Chargement initial
            self.root.after(1000, self.refresh_data_list)

            self.logger.info("🎉 Interface ThreadX prête!")

            # Boucle principale
            self.root.mainloop()

        except KeyboardInterrupt:
            self.logger.info("👋 Fermeture application")
        except Exception as e:
            self.logger.error(f"💥 Erreur fatale: {e}")
            messagebox.showerror("Erreur Fatale", f"Erreur fatale:\n{e}")


def main():
    """Point d'entrée principal."""
    print("🚀 ThreadX Data Manager GUI")
    print("=" * 50)

    try:
        app = ThreadXDataManagerGUI()
        app.run()
    except Exception as e:
        print(f"❌ Erreur fatale: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
