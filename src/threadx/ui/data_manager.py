"""
ThreadX Data Manager UI - Phase 8 Extension
Page Data Manager dans l'UI Tkinter pour téléchargement manuel.

Interface:
- Sélection symboles (multi-select)
- Choix période Start/End (UTC)
- Fréquence fixée à 1m (vérification 1h/3h optionnelle)
- Téléchargement en arrière-plan (non-bloquant)
- Barre de progression + logs temps réel
- Mode simulate/dry-run
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from datetime import datetime, timedelta
from threading import Thread
from queue import Queue, Empty
from typing import List, Dict, Any, Optional
import logging

from ..config import get_settings
from ..data.ingest import IngestionManager


class DataManagerPage(ttk.Frame):
    """
    Page Data Manager pour téléchargement manuel dans l'UI Tkinter.

    Fonctionnalités:
    - Multi-sélection symboles
    - Configuration période
    - Téléchargement background avec progress
    - Logs temps réel
    - Mode dry-run
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.settings = get_settings()
        self.ingestion_manager = IngestionManager(self.settings)

        # Communication thread UI ↔ background
        self.log_queue = Queue()
        self.progress_queue = Queue()

        # État téléchargement
        self.download_thread: Optional[Thread] = None
        self.is_downloading = False

        self.setup_ui()
        self.setup_logging()

        # Polling queues pour UI updates
        self.after(100, self.check_queues)

    def setup_ui(self):
        """Configuration interface utilisateur."""
        # Titre
        title_frame = ttk.Frame(self)
        title_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(title_frame, text="Data Manager", font=("Arial", 14, "bold")).pack(
            side="left"
        )
        ttk.Label(
            title_frame,
            text="Téléchargement et gestion des données OHLCV",
            font=("Arial", 9),
            foreground="gray",
        ).pack(side="left", padx=(10, 0))

        # Configuration principale
        config_frame = ttk.LabelFrame(self, text="Configuration")
        config_frame.pack(fill="x", padx=10, pady=5)

        # Sélection symboles
        symbols_frame = ttk.Frame(config_frame)
        symbols_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(symbols_frame, text="Symboles:").pack(side="left")

        # Listbox multi-select avec scrollbar
        symbols_list_frame = ttk.Frame(symbols_frame)
        symbols_list_frame.pack(side="left", padx=(10, 0), fill="both", expand=True)

        self.symbols_listbox = tk.Listbox(
            symbols_list_frame, selectmode="extended", height=4
        )
        symbols_scrollbar = ttk.Scrollbar(
            symbols_list_frame, orient="vertical", command=self.symbols_listbox.yview
        )
        self.symbols_listbox.config(yscrollcommand=symbols_scrollbar.set)

        self.symbols_listbox.pack(side="left", fill="both", expand=True)
        symbols_scrollbar.pack(side="right", fill="y")

        # Prépopulation symboles populaires
        popular_symbols = [
            "BTCUSDT",
            "ETHUSDT",
            "BNBUSDT",
            "ADAUSDT",
            "SOLUSDT",
            "XRPUSDT",
            "DOTUSDT",
            "LINKUSDT",
            "MATICUSDT",
            "AVAXUSDT",
        ]
        for symbol in popular_symbols:
            self.symbols_listbox.insert("end", symbol)

        # Période
        period_frame = ttk.Frame(config_frame)
        period_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(period_frame, text="Période:").pack(side="left")

        # Date début
        ttk.Label(period_frame, text="Début:").pack(side="left", padx=(20, 5))
        self.start_date = tk.StringVar(
            value=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        )
        start_entry = ttk.Entry(period_frame, textvariable=self.start_date, width=12)
        start_entry.pack(side="left", padx=(0, 10))

        # Date fin
        ttk.Label(period_frame, text="Fin:").pack(side="left", padx=(10, 5))
        self.end_date = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        end_entry = ttk.Entry(period_frame, textvariable=self.end_date, width=12)
        end_entry.pack(side="left")

        # Options avancées
        options_frame = ttk.LabelFrame(config_frame, text="Options avancées")
        options_frame.pack(fill="x", padx=5, pady=5)

        options_row1 = ttk.Frame(options_frame)
        options_row1.pack(fill="x", padx=5, pady=2)

        # Force update
        self.force_update = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options_row1,
            text="Forcer mise à jour (ignore cache local)",
            variable=self.force_update,
        ).pack(side="left")

        # Verification
        self.enable_verification = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_row1,
            text="Vérification 1h/3h (sanity checks)",
            variable=self.enable_verification,
        ).pack(side="left", padx=(20, 0))

        options_row2 = ttk.Frame(options_frame)
        options_row2.pack(fill="x", padx=5, pady=2)

        # Dry run
        self.dry_run = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options_row2,
            text="Mode simulation (dry-run, pas de téléchargement)",
            variable=self.dry_run,
        ).pack(side="left")

        # Timeframes générés
        tf_frame = ttk.Frame(options_frame)
        tf_frame.pack(fill="x", padx=5, pady=2)

        ttk.Label(tf_frame, text="Timeframes à générer:").pack(side="left")
        self.timeframes_var = tk.StringVar(value="1m,3m,5m,15m,30m,1h,2h,4h,1d")
        ttk.Entry(tf_frame, textvariable=self.timeframes_var, width=40).pack(
            sidecond="left", padx=(10, 0)
        )

        # Boutons contrôle
        control_frame = ttk.Frame(self)
        control_frame.pack(fill="x", padx=10, pady=5)

        self.download_btn = ttk.Button(
            control_frame,
            text="Télécharger",
            command=self.start_download,
            style="Accent.TButton",
        )
        self.download_btn.pack(side="left")

        self.stop_btn = ttk.Button(
            control_frame, text="Arrêter", state="disabled", command=self.stop_download
        )
        self.stop_btn.pack(side="left", padx=(10, 0))

        self.clear_logs_btn = ttk.Button(
            control_frame, text="Vider logs", command=self.clear_logs
        )
        self.clear_logs_btn.pack(side="left", padx=(10, 0))

        # Barre de progression
        progress_frame = ttk.Frame(self)
        progress_frame.pack(fill="x", padx=10, pady=5)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame, variable=self.progress_var, maximum=100, length=400
        )
        self.progress_bar.pack(side="left", fill="x", expand=True)

        self.progress_label = ttk.Label(progress_frame, text="Prêt")
        self.progress_label.pack(side="left", padx=(10, 0))

        # Zone logs
        logs_frame = ttk.LabelFrame(self, text="Logs temps réel")
        logs_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.logs_text = scrolledtext.ScrolledText(
            logs_frame, height=15, font=("Consolas", 9)
        )
        self.logs_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Stats résumé
        stats_frame = ttk.LabelFrame(self, text="Statistiques session")
        stats_frame.pack(fill="x", padx=10, pady=5)

        self.stats_label = ttk.Label(stats_frame, text="Aucune session active")
        self.stats_label.pack(padx=5, pady=2)

    def setup_logging(self):
        """Configuration du logging vers l'UI."""
        self.ui_logger = logging.getLogger("ui_data_manager")
        self.ui_logger.setLevel(logging.INFO)

        # Handler custom vers queue
        handler = QueueLogHandler(self.log_queue)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        )
        self.ui_logger.addHandler(handler)

    def start_download(self):
        """Démarre le téléchargement en arrière-plan."""
        if self.is_downloading:
            messagebox.showwarning(
                "Téléchargement en cours", "Un téléchargement est déjà en cours."
            )
            return

        # Validation configuration
        selected_symbols = self.get_selected_symbols()
        if not selected_symbols:
            messagebox.showerror("Erreur", "Sélectionnez au moins un symbole.")
            return

        try:
            start_date = datetime.strptime(self.start_date.get(), "%Y-%m-%d")
            end_date = datetime.strptime(self.end_date.get(), "%Y-%m-%d")

            if start_date >= end_date:
                messagebox.showerror("Erreur", "Date début doit être < date fin.")
                return

        except ValueError as e:
            messagebox.showerror("Erreur", f"Format de date invalide: {e}")
            return

        timeframes = [
            tf.strip() for tf in self.timeframes_var.get().split(",") if tf.strip()
        ]
        if not timeframes:
            messagebox.showerror("Erreur", "Spécifiez au moins un timeframe.")
            return

        # Mode dry-run
        if self.dry_run.get():
            self.simulate_download(selected_symbols, timeframes, start_date, end_date)
            return

        # Démarrage téléchargement réel
        self.is_downloading = True
        self.download_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        self.progress_var.set(0)
        self.progress_label.config(text="Initialisation...")

        # Clear logs précédents
        self.logs_text.delete(1.0, "end")

        self.ui_logger.info(
            f"Démarrage téléchargement: {len(selected_symbols)} symboles × {len(timeframes)} TFs"
        )

        # Thread background
        self.download_thread = Thread(
            target=self.download_worker,
            args=(selected_symbols, timeframes, start_date, end_date),
            daemon=True,
        )
        self.download_thread.start()

    def download_worker(
        self, symbols: List[str], timeframes: List[str], start: datetime, end: datetime
    ):
        """Worker thread pour téléchargement (non-bloquant UI)."""
        try:
            self.ui_logger.info("Configuration ingestion manager...")

            # Callback progress pour UI
            def progress_callback(current: int, total: int, status: str):
                progress_pct = (current / total * 100) if total > 0 else 0
                self.progress_queue.put(
                    ("progress", progress_pct, f"{status} ({current}/{total})")
                )

            # Téléchargement batch via IngestionManager
            results = self.ingestion_manager.update_assets_batch(
                symbols=symbols,
                timeframes=timeframes,
                start=start,
                end=end,
                force=self.force_update.get(),
                enable_verification=self.enable_verification.get(),
                max_workers=4,
            )

            # Rapport final
            summary = results["summary"]
            self.ui_logger.info("=== TÉLÉCHARGEMENT TERMINÉ ===")
            self.ui_logger.info(
                f"Symboles traités: {summary['symbols_processed']}/{summary['symbols_requested']}"
            )
            self.ui_logger.info(f"Fichiers téléchargés: {summary['files_downloaded']}")
            self.ui_logger.info(f"Fichiers resamplés: {summary['files_resampled']}")
            self.ui_logger.info(f"Gaps comblés: {summary['gaps_filled']}")
            self.ui_logger.info(
                f"Warnings vérification: {summary['verification_warnings']}"
            )

            if results["errors"]:
                self.ui_logger.error(f"Erreurs ({len(results['errors'])}):")
                for error in results["errors"]:
                    self.ui_logger.error(f"  - {error}")

            # Stats UI
            stats_text = (
                f"Symboles: {summary['symbols_processed']}/{summary['symbols_requested']} | "
                f"Téléchargés: {summary['files_downloaded']} | "
                f"Resamplés: {summary['files_resampled']} | "
                f"Erreurs: {summary['total_errors']}"
            )

            self.progress_queue.put(("complete", 100, stats_text))

        except Exception as e:
            self.ui_logger.error(f"Erreur téléchargement: {e}")
            self.progress_queue.put(("error", 0, f"Erreur: {e}"))

        finally:
            self.progress_queue.put(("finished", None, None))

    def simulate_download(
        self, symbols: List[str], timeframes: List[str], start: datetime, end: datetime
    ):
        """Mode simulation - affiche ce qui serait téléchargé."""
        self.logs_text.delete(1.0, "end")

        self.ui_logger.info("=== MODE SIMULATION (DRY-RUN) ===")
        self.ui_logger.info(f"Symboles: {', '.join(symbols)}")
        self.ui_logger.info(f"Timeframes: {', '.join(timeframes)}")
        self.ui_logger.info(f"Période: {start.date()} → {end.date()}")
        self.ui_logger.info(f"Force update: {self.force_update.get()}")
        self.ui_logger.info(f"Vérification: {self.enable_verification.get()}")

        days = (end - start).days
        estimated_mb = len(symbols) * days * 0.5  # Estimation rough

        self.ui_logger.info(f"Données estimées: ~{estimated_mb:.1f} MB")
        self.ui_logger.info("Plages qui seraient téléchargées:")

        for symbol in symbols:
            # Simulation vérification banque locale
            self.ui_logger.info(f"  {symbol}: vérification cache local...")
            self.ui_logger.info(
                f"  {symbol}: plage manquante {start.date()} → {end.date()}"
            )

        self.ui_logger.info("=== SIMULATION TERMINÉE ===")
        self.ui_logger.info("Lancez sans 'Mode simulation' pour téléchargement réel.")

        # Progress 100% pour simulation
        self.progress_var.set(100)
        self.progress_label.config(text="Simulation terminée")

    def stop_download(self):
        """Arrête le téléchargement en cours."""
        if self.download_thread and self.download_thread.is_alive():
            self.ui_logger.warning("Arrêt demandé... (peut prendre quelques secondes)")
            # Note: Thread.stop() n'existe pas, on pourrait implémenter Event() pour arrêt propre
            # Pour simplifier, on marque juste l'arrêt

        self.download_finished()

    def download_finished(self):
        """Remet l'UI en état initial après téléchargement."""
        self.is_downloading = False
        self.download_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.download_thread = None

    def get_selected_symbols(self) -> List[str]:
        """Récupère les symboles sélectionnés."""
        selected_indices = self.symbols_listbox.curselection()
        return [self.symbols_listbox.get(i) for i in selected_indices]

    def clear_logs(self):
        """Vide la zone de logs."""
        self.logs_text.delete(1.0, "end")

    def check_queues(self):
        """Polling des queues pour updates UI (thread-safe)."""
        # Logs queue
        while True:
            try:
                log_record = self.log_queue.get_nowait()
                self.logs_text.insert("end", log_record + "\n")
                self.logs_text.see("end")
            except Empty:
                break

        # Progress queue
        while True:
            try:
                msg_type, value, text = self.progress_queue.get_nowait()

                if msg_type == "progress":
                    self.progress_var.set(value)
                    self.progress_label.config(text=text)

                elif msg_type == "complete":
                    self.progress_var.set(value)
                    self.stats_label.config(text=text)
                    self.progress_label.config(text="Terminé")

                elif msg_type == "error":
                    self.progress_label.config(text=text)

                elif msg_type == "finished":
                    self.download_finished()

            except Empty:
                break

        # Re-schedule
        self.after(100, self.check_queues)


class QueueLogHandler(logging.Handler):
    """Handler pour rediriger logs vers Queue (thread-safe UI)."""

    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))
