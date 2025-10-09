"""
ThreadX Data Manager - Onglet Integration
"""

import tkinter as tk
from tkinter import ttk


class IntegrationTab:
    """Onglet d'intégration des données"""

    def __init__(self, parent):
        self.frame = ttk.Frame(parent)
        self.setup_ui()

    def setup_ui(self):
        """Configuration de l'interface"""
        title = ttk.Label(
            self.frame, text="Integration Tab - En développement", font=("Arial", 12)
        )
        title.pack(pady=20)

        info = ttk.Label(
            self.frame, text="Cette section permet d'intégrer les données."
        )
        info.pack(pady=10)
