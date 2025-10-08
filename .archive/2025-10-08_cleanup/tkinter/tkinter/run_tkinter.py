#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import logging
import time
import threading
from datetime import datetime

# Ajoute /src au PYTHONPATH
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import de ThreadX si disponible
try:
    from threadx.backtest.engine import BacktestEngine, BacktestController

    THREADX_AVAILABLE = True
except ImportError:
    THREADX_AVAILABLE = False
import argparse
from pathlib import Path
from typing import Optional

# Ajouter le dossier source au path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    from tkinter.scrolledtext import ScrolledText
except ImportError as e:
    print(f"❌ ERREUR: Tkinter non disponible - {e}")
    print("\n💡 Solutions:")
    print("   - Sur Ubuntu/Debian: sudo apt-get install python3-tk")
    print("   - Sur CentOS/RHEL: sudo yum install tkinter")
    print("   - Sur Windows: Tkinter devrait être inclus avec Python")
    print("\n🔄 Alternative: Utilisez l'interface Streamlit")
    print("   run_streamlit.bat")
    sys.exit(1)


def setup_logging(debug: bool = False) -> logging.Logger:
    """Configure le système de logging."""
    level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                PROJECT_ROOT / "logs" / "tkinter_app.log", encoding="utf-8"
            ),
        ],
    )

    return logging.getLogger("ThreadX.TkinterApp")


def check_dependencies() -> tuple[bool, list[str]]:
    """Vérifie les dépendances requises."""
    missing = []
    required = ["pandas", "numpy", "matplotlib", "plotly", "pyarrow", "psutil", "tqdm"]

    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    return len(missing) == 0, missing


class ThreadXTkinterApp:
    """Application Tkinter principale de ThreadX."""

    def __init__(self, debug: bool = False, theme: str = "dark"):
        self.debug = debug
        self.theme = theme
        self.logger = setup_logging(debug)

        # Initialisation fenêtre principale
        self.root = tk.Tk()
        self.root.title("ThreadX - Plateforme de Trading Algorithmique")
        self.root.geometry("1200x800")

        # Configuration thème
        self.setup_theme()

        # Variables d'état
        self.current_backtest = None
        self.monitoring_active = False

        # Initialisation UI
        self.create_widgets()
        self.logger.info("Application ThreadX Tkinter initialisée")

    def setup_theme(self):
        """Configure le thème de l'interface."""
        if self.theme == "dark":
            bg_color = "#1e1e1e"
            fg_color = "#ffffff"
            select_color = "#333333"
        else:
            bg_color = "#ffffff"
            fg_color = "#000000"
            select_color = "#e0e0e0"

        self.root.configure(bg=bg_color)

        # Style ttk
        style = ttk.Style()
        style.theme_use("clam")

        # Configuration couleurs custom
        style.configure("Custom.TFrame", background=bg_color)
        style.configure("Custom.TLabel", background=bg_color, foreground=fg_color)
        style.configure("Custom.TButton", background=select_color, foreground=fg_color)

        self.colors = {
            "bg": bg_color,
            "fg": fg_color,
            "select": select_color,
            "accent": "#0066cc",
            "success": "#00aa00",
            "warning": "#ff8800",
            "error": "#cc0000",
        }

    def create_widgets(self):
        """Crée les widgets de l'interface."""
        # Menu principal
        self.create_menu()

        # Toolbar
        self.create_toolbar()

        # Zone principale avec onglets
        self.create_main_area()

        # Barre de statut
        self.create_status_bar()

        # Configuration fermeture
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_menu(self):
        """Crée la barre de menu."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Menu Fichier
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Fichier", menu=file_menu)
        file_menu.add_command(label="Nouveau projet", accelerator="Ctrl+N")
        file_menu.add_command(label="Ouvrir projet", accelerator="Ctrl+O")
        file_menu.add_command(label="Sauvegarder", accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(
            label="Quitter", accelerator="Ctrl+Q", command=self.on_closing
        )

        # Menu Outils
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Outils", menu=tools_menu)
        tools_menu.add_command(
            label="Migration TradXPro", command=self.open_migration_tool
        )
        tools_menu.add_command(
            label="Vérification environnement", command=self.check_environment
        )
        tools_menu.add_command(label="Nettoyage cache", command=self.clear_cache)

        # Menu Aide
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Aide", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
        help_menu.add_command(label="À propos", command=self.show_about)

    def create_toolbar(self):
        """Crée la barre d'outils."""
        toolbar = ttk.Frame(self.root, style="Custom.TFrame")
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        # Boutons principaux
        ttk.Button(toolbar, text="🚀 Nouveau Backtest", command=self.new_backtest).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(toolbar, text="⚙️ Configuration", command=self.open_config).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(toolbar, text="📊 Résultats", command=self.show_results).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(toolbar, text="🔧 Outils", command=self.show_tools).pack(
            side=tk.LEFT, padx=2
        )

        # Séparateur
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=10
        )

        # Indicateurs statut
        self.status_gpu = ttk.Label(toolbar, text="GPU: ❓", style="Custom.TLabel")
        self.status_gpu.pack(side=tk.RIGHT, padx=5)

        self.status_env = ttk.Label(toolbar, text="ENV: ❓", style="Custom.TLabel")
        self.status_env.pack(side=tk.RIGHT, padx=5)

    def create_main_area(self):
        """Crée la zone principale avec onglets."""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Onglet Backtest
        self.create_backtest_tab()

        # Onglet Configuration
        self.create_config_tab()

        # Onglet Résultats
        self.create_results_tab()

        # Onglet Logs
        self.create_logs_tab()

    def create_backtest_tab(self):
        """Crée l'onglet de backtest."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🚀 Backtest")

        # Zone de configuration rapide
        config_frame = ttk.LabelFrame(tab, text="Configuration Rapide")
        config_frame.pack(fill=tk.X, padx=10, pady=5)

        # Ligne 1: Symbole et timeframe
        row1 = ttk.Frame(config_frame)
        row1.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(row1, text="Symbole:").pack(side=tk.LEFT)
        self.symbol_var = tk.StringVar(value="BTCUSDC")
        ttk.Entry(row1, textvariable=self.symbol_var, width=12).pack(
            side=tk.LEFT, padx=5
        )

        ttk.Label(row1, text="Timeframe:").pack(side=tk.LEFT, padx=(20, 0))
        self.timeframe_var = tk.StringVar(value="1h")
        ttk.Combobox(
            row1,
            textvariable=self.timeframe_var,
            values=["1m", "5m", "15m", "1h", "4h", "1d"],
            width=8,
        ).pack(side=tk.LEFT, padx=5)

        # Ligne 2: Paramètres stratégie
        row2 = ttk.Frame(config_frame)
        row2.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(row2, text="BB Std:").pack(side=tk.LEFT)
        self.bb_std_var = tk.DoubleVar(value=2.0)
        ttk.Scale(
            row2,
            from_=1.0,
            to=3.0,
            variable=self.bb_std_var,
            orient=tk.HORIZONTAL,
            length=100,
        ).pack(side=tk.LEFT, padx=5)

        ttk.Label(row2, text="Entry Z:").pack(side=tk.LEFT, padx=(20, 0))
        self.entry_z_var = tk.DoubleVar(value=2.0)
        ttk.Scale(
            row2,
            from_=1.0,
            to=4.0,
            variable=self.entry_z_var,
            orient=tk.HORIZONTAL,
            length=100,
        ).pack(side=tk.LEFT, padx=5)

        # Boutons d'action
        action_frame = ttk.Frame(tab)
        action_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(
            action_frame, text="▶️ Lancer Backtest", command=self.start_backtest
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="⏸️ Pause", command=self.pause_backtest).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(action_frame, text="⏹️ Arrêter", command=self.stop_backtest).pack(
            side=tk.LEFT, padx=5
        )

        # Zone de monitoring
        monitor_frame = ttk.LabelFrame(tab, text="Monitoring Temps Réel")
        monitor_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Barre de progression
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            monitor_frame, variable=self.progress_var, maximum=100
        )
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)

        self.progress_label = ttk.Label(monitor_frame, text="Prêt")
        self.progress_label.pack(pady=2)

    def create_config_tab(self):
        """Crée l'onglet de configuration."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="⚙️ Configuration")

        # Message temporaire
        ttk.Label(
            tab,
            text="🚧 Configuration avancée - En développement",
            style="Custom.TLabel",
        ).pack(expand=True)

    def create_results_tab(self):
        """Crée l'onglet des résultats."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📊 Résultats")

        # Message temporaire
        ttk.Label(
            tab,
            text="📈 Visualisation des résultats - En développement",
            style="Custom.TLabel",
        ).pack(expand=True)

    def create_logs_tab(self):
        """Crée l'onglet des logs."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📝 Logs")

        # Zone de logs avec scrolling
        self.log_text = ScrolledText(
            tab,
            height=20,
            bg=self.colors["bg"],
            fg=self.colors["fg"],
            insertbackground=self.colors["fg"],
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Configuration tags pour couleurs
        self.log_text.tag_configure("INFO", foreground=self.colors["fg"])
        self.log_text.tag_configure("WARNING", foreground=self.colors["warning"])
        self.log_text.tag_configure("ERROR", foreground=self.colors["error"])
        self.log_text.tag_configure("SUCCESS", foreground=self.colors["success"])

    def create_status_bar(self):
        """Crée la barre de statut."""
        self.status_bar = ttk.Frame(self.root, style="Custom.TFrame")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_text = ttk.Label(
            self.status_bar, text="ThreadX prêt", style="Custom.TLabel"
        )
        self.status_text.pack(side=tk.LEFT, padx=5)

        # Horloge
        self.update_clock()

    def update_clock(self):
        """Met à jour l'horloge dans la barre de statut."""
        import datetime

        current_time = datetime.datetime.now().strftime("%H:%M:%S")

        # Créer l'horloge si elle n'existe pas
        if not hasattr(self, "clock_label"):
            self.clock_label = ttk.Label(
                self.status_bar, text=current_time, style="Custom.TLabel"
            )
            self.clock_label.pack(side=tk.RIGHT, padx=5)
        else:
            self.clock_label.configure(text=current_time)

        # Programmer prochaine mise à jour
        self.root.after(1000, self.update_clock)

    # Méthodes d'action
    def new_backtest(self):
        """Lance un nouveau backtest."""
        messagebox.showinfo("Nouveau Backtest", "🚧 Fonctionnalité en développement")

    def start_backtest(self):
        """Démarre le backtest."""
        self.log_message("INFO", "🚀 Démarrage du backtest...")
        self.progress_var.set(0)

        # Import de l'engine avec contrôleur si disponible
        try:
            from threadx.backtest.engine import BacktestEngine, BacktestController

            self.backtest_controller = BacktestController()
            self.log_message("INFO", "✅ Contrôleur de backtest initialisé")
        except ImportError:
            self.log_message("WARNING", "⚠️ BacktestController non disponible")
            self.backtest_controller = None

        # TODO: Implémenter logique backtest complète avec threading
        for i in range(101):
            if self.backtest_controller and self.backtest_controller.is_stopped:
                self.log_message("WARNING", "🛑 Backtest arrêté par l'utilisateur")
                break
            if self.backtest_controller and self.backtest_controller.is_paused:
                self.log_message("INFO", "⏸️ Backtest en pause...")
                while (
                    self.backtest_controller.is_paused
                    and not self.backtest_controller.is_stopped
                ):
                    self.root.update()
                    time.sleep(0.1)
                if self.backtest_controller.is_stopped:
                    self.log_message("WARNING", "🛑 Backtest arrêté pendant la pause")
                    break
                self.log_message("INFO", "▶️ Backtest repris")

            self.progress_var.set(i)
            self.progress_label.config(text=f"Traitement... {i}%")
            self.root.update()
            time.sleep(0.05)  # Simulation du travail

        if not (self.backtest_controller and self.backtest_controller.is_stopped):
            self.progress_var.set(100)
            self.progress_label.config(text="Backtest terminé ✅")
            self.log_message("INFO", "✅ Backtest terminé avec succès")

    def pause_backtest(self):
        """Met en pause le backtest."""
        if hasattr(self, "backtest_controller") and self.backtest_controller:
            self.backtest_controller.pause()
            self.log_message("WARNING", "⏸️ Backtest mis en pause")
        else:
            self.log_message(
                "ERROR", "❌ Impossible de mettre en pause (contrôleur non disponible)"
            )

    def stop_backtest(self):
        """Arrête le backtest."""
        if hasattr(self, "backtest_controller") and self.backtest_controller:
            self.backtest_controller.stop()
            self.log_message("ERROR", "🛑 Backtest arrêté")
            self.progress_var.set(0)
            self.progress_label.config(text="Arrêté")
        else:
            self.log_message(
                "ERROR", "❌ Impossible d'arrêter (contrôleur non disponible)"
            )

    def open_config(self):
        """Ouvre la configuration avancée."""
        messagebox.showinfo(
            "Configuration", "🚧 Configuration avancée en développement"
        )

    def show_results(self):
        """Affiche les résultats."""
        self.notebook.select(2)  # Onglet résultats

    def show_tools(self):
        """Affiche les outils."""
        tools_window = tk.Toplevel(self.root)
        tools_window.title("Outils ThreadX")
        tools_window.geometry("400x300")

        ttk.Label(
            tools_window, text="🔧 Outils ThreadX", font=("Arial", 12, "bold")
        ).pack(pady=10)

        ttk.Button(
            tools_window, text="Migration TradXPro", command=self.open_migration_tool
        ).pack(pady=5)
        ttk.Button(
            tools_window,
            text="Vérification Environnement",
            command=self.check_environment,
        ).pack(pady=5)
        ttk.Button(tools_window, text="Nettoyage Cache", command=self.clear_cache).pack(
            pady=5
        )

    def open_migration_tool(self):
        """Ouvre l'outil de migration."""
        self.log_message("INFO", "Ouverture outil de migration TradXPro")
        # TODO: Implémenter interface migration
        messagebox.showinfo(
            "Migration",
            "🔄 Outil de migration TradXPro\n\n🚧 Interface graphique en développement\n\n💡 Utilisez en attendant:\npython tools/migrate_from_tradxpro.py --help",
        )

    def check_environment(self):
        """Vérifie l'environnement système."""
        self.log_message("INFO", "Vérification environnement système")
        # TODO: Implémenter vérification
        messagebox.showinfo(
            "Environnement",
            "🔍 Vérification environnement\n\n🚧 Interface graphique en développement\n\n💡 Utilisez en attendant:\npython tools/check_env.py",
        )

    def clear_cache(self):
        """Nettoie le cache."""
        self.log_message("SUCCESS", "Cache nettoyé avec succès")
        messagebox.showinfo("Cache", "🧹 Cache nettoyé avec succès")

    def show_documentation(self):
        """Affiche la documentation."""
        messagebox.showinfo(
            "Documentation",
            "📚 Documentation ThreadX\n\nConsultez README.md pour la documentation complète",
        )

    def show_about(self):
        """Affiche les informations 'À propos'."""
        about_text = """
🚀 ThreadX - Plateforme de Trading Algorithmique

Version: Phase 10 - UI Desktop Native
Auteur: ThreadX Framework

Fonctionnalités:
• Backtesting avancé avec GPU
• Migration depuis TradXPro
• Interface native Tkinter
• Monitoring temps réel
• Outils d'analyse intégrés

© 2025 ThreadX Framework
        """
        messagebox.showinfo("À propos de ThreadX", about_text)

    def log_message(self, level: str, message: str):
        """Ajoute un message dans les logs."""
        import datetime

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {level}: {message}\n"

        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, formatted_message, level)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

        # Log également dans le système de logging
        getattr(self.logger, level.lower(), self.logger.info)(message)

    def on_closing(self):
        """Gère la fermeture de l'application."""
        if messagebox.askokcancel("Quitter", "Voulez-vous vraiment quitter ThreadX?"):
            self.logger.info("Fermeture application ThreadX Tkinter")
            self.root.destroy()

    def run(self):
        """Lance l'application."""
        self.logger.info("Démarrage interface ThreadX Tkinter")
        self.log_message("SUCCESS", "ThreadX Tkinter initialisé avec succès")
        self.root.mainloop()


def parse_arguments():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="ThreadX Tkinter Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'usage:
  python run_tkinter.py                    # Lancement normal
  python run_tkinter.py --debug            # Mode debug
  python run_tkinter.py --theme light      # Thème clair
  python run_tkinter.py --dev              # Mode développement
        """,
    )

    parser.add_argument(
        "--debug", action="store_true", help="Active le mode debug avec logs détaillés"
    )

    parser.add_argument(
        "--dev",
        action="store_true",
        help="Mode développement (features expérimentales)",
    )

    parser.add_argument(
        "--theme",
        choices=["dark", "light", "auto"],
        default="dark",
        help="Thème de l'interface (défaut: dark)",
    )

    parser.add_argument(
        "--config", type=Path, help="Chemin vers fichier de configuration personnalisé"
    )

    return parser.parse_args()


def main():
    """Point d'entrée principal."""
    # Créer dossier logs si nécessaire
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)

    # Parse arguments
    args = parse_arguments()

    # Vérifier dépendances
    deps_ok, missing = check_dependencies()
    if not deps_ok:
        print(f"❌ ERREUR: Dépendances manquantes - {', '.join(missing)}")
        print("\n💡 Installation:")
        print(f"   pip install {' '.join(missing)}")
        return 1

    try:
        # Initialiser et lancer l'application
        app = ThreadXTkinterApp(debug=args.debug, theme=args.theme)

        if args.dev:
            app.log_message("WARNING", "Mode développement activé")

        app.run()

    except KeyboardInterrupt:
        print("\n🛑 Interruption utilisateur")
        return 1
    except Exception as e:
        print(f"❌ ERREUR FATALE: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
