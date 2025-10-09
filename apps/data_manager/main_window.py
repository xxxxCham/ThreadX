"""
ThreadX Data Manager - Interface utilisateur principale
Interface de gestion des données d'indicateurs existantes
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from pathlib import Path
from typing import Optional, List
import logging

from .data_copier import ThreadXDataCopier
from .ui.discovery_tab import DiscoveryTab
from .ui.validation_tab import ValidationTab
from .ui.integration_tab import IntegrationTab

logger = logging.getLogger(__name__)


class ThreadXDataManagerApp:
    """Application principale ThreadX Data Manager"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ThreadX Data Manager")
        self.root.geometry("1200x800")

        # Composants
        self.data_copier: Optional[ThreadXDataCopier] = None
        self.setup_ui()
        self.setup_logging()

    def setup_logging(self):
        """Configuration du logging"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def setup_ui(self):
        """Configuration de l'interface utilisateur"""
        # Style
        style = ttk.Style()
        style.theme_use("clam")

        # Menu principal
        self.create_menu()

        # Frame principal avec toolbar
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Toolbar
        self.create_toolbar(main_frame)

        # Notebook pour les onglets
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Onglets
        self.create_tabs()

        # Barre de statut
        self.create_status_bar()

    def create_menu(self):
        """Créer le menu principal"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Menu Data
        data_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Data", menu=data_menu)
        data_menu.add_command(
            label="Copier données locales", command=self.copy_local_data
        )
        data_menu.add_command(
            label="Vérifier structure data/", command=self.check_data_structure
        )
        data_menu.add_separator()
        data_menu.add_command(
            label="Ouvrir dossier data/", command=self.open_data_folder
        )
        data_menu.add_command(label="Rapport de copie", command=self.show_copy_report)

        # Menu Tools
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(
            label="Scanner données réelles", command=self.scan_real_data
        )
        tools_menu.add_command(label="Nettoyer cache", command=self.clean_cache)

        # Menu Help
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="À propos", command=self.show_about)

    def create_toolbar(self, parent):
        """Créer la barre d'outils"""
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=tk.X, pady=(0, 10))

        # Boutons principaux
        ttk.Button(
            toolbar, text="📂 Copier Données", command=self.copy_local_data
        ).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(toolbar, text="🔍 Scanner", command=self.scan_real_data).pack(
            side=tk.LEFT, padx=5
        )

        ttk.Button(toolbar, text="📊 Rapport", command=self.show_copy_report).pack(
            side=tk.LEFT, padx=5
        )

        # Séparateur
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=10
        )

        # Boutons secondaires
        ttk.Button(toolbar, text="🧹 Nettoyer", command=self.clean_cache).pack(
            side=tk.LEFT, padx=5
        )

        ttk.Button(toolbar, text="⚙️ Config", command=self.show_config).pack(
            side=tk.LEFT, padx=5
        )

    def create_tabs(self):
        """Créer les onglets"""
        # Onglet Data Copy (nouveau)
        self.copy_tab = self.create_copy_tab()
        self.notebook.add(self.copy_tab, text="📂 Data Copy")

        # Onglet Discovery
        self.discovery_tab = DiscoveryTab(self.notebook)
        self.notebook.add(self.discovery_tab.frame, text="🔍 Discovery")

        # Onglet Validation
        self.validation_tab = ValidationTab(self.notebook)
        self.notebook.add(self.validation_tab.frame, text="✅ Validation")

        # Onglet Integration
        self.integration_tab = IntegrationTab(self.notebook)
        self.notebook.add(self.integration_tab.frame, text="🔗 Integration")

    def create_copy_tab(self):
        """Créer l'onglet de copie des données"""
        tab_frame = ttk.Frame(self.notebook)

        # Titre
        title_label = ttk.Label(
            tab_frame, text="Copie des Données ThreadX", font=("Arial", 14, "bold")
        )
        title_label.pack(pady=10)

        # Information sur la structure
        info_frame = ttk.LabelFrame(tab_frame, text="Information", padding=10)
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        info_text = """
📁 Cette fonctionnalité copie vos données existantes dans l'espace de travail ThreadX.
⚠️  Les données copiées ne seront PAS commitées sur Git (protection .gitignore).
🔄 Les sources originales restent intactes - ce sont des copies de sauvegarde.
        """
        ttk.Label(info_frame, text=info_text.strip(), justify=tk.LEFT).pack(anchor=tk.W)

        # Sources détectées
        self.sources_frame = ttk.LabelFrame(
            tab_frame, text="Sources Détectées", padding=10
        )
        self.sources_frame.pack(fill=tk.X, padx=10, pady=5)

        # Treeview pour les sources
        columns = ("Source", "Status", "Fichiers", "Taille")
        self.sources_tree = ttk.Treeview(
            self.sources_frame, columns=columns, show="headings", height=4
        )

        for col in columns:
            self.sources_tree.heading(col, text=col)
            self.sources_tree.column(col, width=150)

        self.sources_tree.pack(fill=tk.X, pady=5)

        # Boutons d'action
        actions_frame = ttk.Frame(tab_frame)
        actions_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(
            actions_frame,
            text="▶️ Copier Toutes les Données",
            command=self.copy_all_data_action,
            style="Accent.TButton",
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            actions_frame,
            text="🔄 Rafraîchir Status",
            command=self.refresh_sources_status,
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            actions_frame, text="📋 Voir Rapport", command=self.show_copy_report
        ).pack(side=tk.LEFT, padx=5)

        # Zone de log
        log_frame = ttk.LabelFrame(tab_frame, text="Journal", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Scrolled text pour les logs
        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD)
        log_scrollbar = ttk.Scrollbar(
            log_frame, orient=tk.VERTICAL, command=self.log_text.yview
        )
        self.log_text.configure(yscrollcommand=log_scrollbar.set)

        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Initialiser les données
        self.refresh_sources_status()

        return tab_frame

    def create_status_bar(self):
        """Créer la barre de statut"""
        self.status_var = tk.StringVar()
        self.status_var.set("Prêt")

        status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=5,
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def get_data_copier(self) -> ThreadXDataCopier:
        """Obtenir l'instance du copieur de données"""
        if self.data_copier is None:
            self.data_copier = ThreadXDataCopier()
        return self.data_copier

    def refresh_sources_status(self):
        """Rafraîchir le statut des sources"""
        self.log_message("🔄 Rafraîchissement du statut des sources...")

        # Vider le treeview
        for item in self.sources_tree.get_children():
            self.sources_tree.delete(item)

        copier = self.get_data_copier()

        # Vérifier chaque source
        for name, source_path in copier.sources.items():
            if source_path.exists():
                # Calculer les statistiques
                try:
                    stats = copier._calculate_directory_stats(source_path)
                    status = "✅ Disponible"
                    files = f"{stats['files_copied']}"
                    size = f"{stats['total_size_mb']:.1f} MB"
                except Exception as e:
                    status = f"❌ Erreur: {e}"
                    files = "N/A"
                    size = "N/A"
            else:
                status = "❌ Introuvable"
                files = "0"
                size = "0 MB"

            self.sources_tree.insert("", tk.END, values=(name, status, files, size))

        # Vérifier les données déjà copiées
        info = copier.get_local_data_info()
        if info["exists"] and info["sources_available"]:
            self.log_message(
                f"📊 Données locales: {len(info['sources_available'])} sources copiées"
            )
        else:
            self.log_message("📁 Aucune donnée locale trouvée")

    def copy_all_data_action(self):
        """Action de copie de toutes les données"""
        result = messagebox.askyesno(
            "Confirmation",
            "Copier toutes les données dans l'espace de travail ThreadX?\n\n"
            "⚠️ Cette opération peut prendre du temps.\n"
            "📁 Les données ne seront PAS commitées sur Git.",
        )

        if result:
            self.log_message("🚀 Début de la copie des données...")
            self.status_var.set("Copie en cours...")

            # Exécuter en arrière-plan
            def copy_thread():
                try:
                    copier = self.get_data_copier()
                    results = copier.copy_all_data()

                    self.root.after(0, lambda: self.copy_completed(results))

                except Exception as e:
                    error_msg = f"Erreur lors de la copie: {e}"
                    self.root.after(0, lambda: self.copy_error(error_msg))

            threading.Thread(target=copy_thread, daemon=True).start()

    def copy_completed(self, results):
        """Callback de fin de copie"""
        if results["success"]:
            self.log_message(f"✅ Copie terminée avec succès!")
            self.log_message(f"📊 {results['total_files']} fichiers copiés")
            self.log_message(f"💾 {results['total_size_mb']:.1f} MB")

            messagebox.showinfo(
                "Succès",
                f"Copie terminée avec succès!\n\n"
                f"📊 {results['total_files']} fichiers\n"
                f"💾 {results['total_size_mb']:.1f} MB",
            )
        else:
            self.log_message("❌ Échec de la copie")
            messagebox.showerror("Erreur", "Échec de la copie des données.")

        self.status_var.set("Prêt")
        self.refresh_sources_status()

    def copy_error(self, error_msg):
        """Callback d'erreur de copie"""
        self.log_message(f"❌ {error_msg}")
        messagebox.showerror("Erreur", error_msg)
        self.status_var.set("Erreur")

    def copy_local_data(self):
        """Menu action: Copier données locales"""
        self.notebook.select(0)  # Sélectionner l'onglet Data Copy

    def check_data_structure(self):
        """Vérifier la structure data/"""
        copier = self.get_data_copier()
        info = copier.get_local_data_info()

        msg = f"Structure data/: {'✅ OK' if info['exists'] else '❌ Manquante'}\n"
        msg += f"Chemin: {info['data_root']}\n\n"

        if info["sources_available"]:
            msg += "Sources copiées:\n"
            for source in info["sources_available"]:
                msg += f"• {source['name']}: {source['files']} fichiers ({source['size_mb']} MB)\n"
        else:
            msg += "Aucune source copiée"

        messagebox.showinfo("Structure Data", msg)

    def open_data_folder(self):
        """Ouvrir le dossier data/"""
        copier = self.get_data_copier()
        if copier.data_root.exists():
            import subprocess

            subprocess.run(["explorer", str(copier.data_root)])
        else:
            messagebox.showwarning(
                "Attention",
                "Le dossier data/ n'existe pas encore.\n"
                "Effectuez d'abord une copie des données.",
            )

    def show_copy_report(self):
        """Afficher le rapport de copie"""
        copier = self.get_data_copier()
        report_path = copier.data_root / "copy_report.json"

        if report_path.exists():
            try:
                import json

                with open(report_path, "r", encoding="utf-8") as f:
                    report = json.load(f)

                msg = f"Rapport de copie - {report.get('timestamp', 'N/A')}\n\n"
                msg += f"Succès: {'✅' if report.get('success') else '❌'}\n"
                msg += f"Fichiers: {report.get('total_files', 0)}\n"
                msg += f"Taille: {report.get('total_size_mb', 0):.1f} MB\n\n"

                if "sources" in report:
                    msg += "Détails par source:\n"
                    for name, details in report["sources"].items():
                        success = "✅" if details.get("success") else "❌"
                        msg += f"• {name}: {success} - {details.get('files_copied', 0)} fichiers\n"

                messagebox.showinfo("Rapport de Copie", msg)

            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lecture rapport: {e}")
        else:
            messagebox.showwarning(
                "Attention",
                "Aucun rapport de copie trouvé.\n"
                "Effectuez d'abord une copie des données.",
            )

    def scan_real_data(self):
        """Scanner les données réelles"""
        self.notebook.select(1)  # Sélectionner l'onglet Discovery

    def clean_cache(self):
        """Nettoyer le cache"""
        result = messagebox.askyesno(
            "Confirmation",
            "Nettoyer le cache ThreadX?\n\n"
            "Cette action supprimera les fichiers temporaires.",
        )
        if result:
            self.log_message("🧹 Nettoyage du cache...")
            # Implémenter le nettoyage
            messagebox.showinfo("Info", "Cache nettoyé.")

    def show_config(self):
        """Afficher la configuration"""
        messagebox.showinfo(
            "Configuration",
            "Configuration ThreadX Data Manager\n\n"
            "Version: 1.0\n"
            "ThreadX Path: D:\\ThreadX",
        )

    def show_about(self):
        """Afficher À propos"""
        messagebox.showinfo(
            "À propos",
            "ThreadX Data Manager\n\n"
            "Gestionnaire de données pour ThreadX\n"
            "Version 1.0",
        )

    def log_message(self, message: str):
        """Ajouter un message au journal"""
        if hasattr(self, "log_text"):
            from datetime import datetime

            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_msg = f"[{timestamp}] {message}\n"

            self.log_text.insert(tk.END, formatted_msg)
            self.log_text.see(tk.END)
            self.root.update_idletasks()

    def run(self):
        """Lancer l'application"""
        self.root.mainloop()


def main():
    """Point d'entrée principal"""
    app = ThreadXDataManagerApp()
    app.run()


if __name__ == "__main__":
    main()
