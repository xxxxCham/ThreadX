# ThreadX Dashboard

Application web professionnelle pour le backtesting de stratégies de trading de cryptomonnaies.

## 🚀 Fonctionnalités

- **Interface moderne** : Design sombre avec thème cyan professionnel
- **Authentification** : Système de connexion sécurisé avec gestion des sessions
- **Backtesting** : Outils complets pour tester vos stratégies de trading
- **Visualisations** : Graphiques interactifs avec Plotly
- **Responsive** : Interface adaptée aux écrans desktop et mobile
- **Temps réel** : Mise à jour automatique des statistiques

## 📁 Structure du Projet

```
threadx_dashboard/
├── app.py                          # Point d'entrée principal
├── config.py                       # Configuration globale
├── requirements.txt                # Dépendances Python
├── test_app.py                     # Tests de validation
│
├── pages/                          # Pages de l'application
│   ├── home.py                    # Page d'accueil avec statistiques
│   ├── backtesting.py             # Interface de backtesting
│   ├── results.py                 # Résultats et comparaisons
│   └── settings.py                # Paramètres utilisateur
│
├── components/                     # Composants réutilisables
│   ├── navbar.py                  # Barre de navigation
│   ├── sidebar.py                 # Menu latéral avec navigation
│   ├── charts.py                  # Graphiques spécialisés
│   ├── tables.py                  # Tableaux de données
│   └── modals.py                  # Modales et popups
│
├── utils/                          # Utilitaires
│   ├── auth.py                    # Gestion authentification
│   ├── validators.py              # Validation des données
│   └── helpers.py                 # Fonctions d'aide
│
├── assets/                         # Fichiers statiques
│   ├── style.css                  # Styles personnalisés
│   ├── logo.png                   # Logo de l'application
│   └── favicon.ico               # Icône du navigateur
│
└── data/                          # Données de l'application
    └── .gitkeep
```

## 🛠️ Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Étapes d'installation

1. **Cloner le repository** (ou naviguer vers le dossier)
```bash
cd threadx_dashboard
```

2. **Créer un environnement virtuel** (recommandé)
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Configuration** (optionnel)
```bash
# Copier le fichier d'exemple et le modifier selon vos besoins
cp .env.example .env
```

## 🚀 Lancement

### Méthode 1 : Lancement direct
```bash
python app.py
```

### Méthode 2 : Avec variables d'environnement
```bash
# Mode développement
DEBUG=True python app.py

# Mode production
DEBUG=False AUTH_ENABLED=True python app.py
```

L'application sera accessible à l'adresse : `http://localhost:8050`

## 🧪 Tests

Exécuter les tests de validation :

```bash
python test_app.py
```

Ce script vérifie :
- ✅ Imports de tous les modules
- ✅ Configuration correcte
- ✅ Système d'authentification
- ✅ Création des composants
- ✅ Initialisation de l'application

## 🔐 Authentification

### Identifiants par défaut

- **Nom d'utilisateur** : `admin`
- **Mot de passe** : `admin123`

⚠️ **Important** : Changez ces identifiants en production !

### Configuration de la sécurité

Modifiez les variables dans `.env` :

```env
SECRET_KEY=your-super-secret-key-here
DEFAULT_USERNAME=your-username
DEFAULT_PASSWORD=your-secure-password
```

## 🎨 Personnalisation

### Thème et couleurs

Modifiez les couleurs dans `config.py` :

```python
THEME = {
    'primary_bg': '#1a1a1a',        # Fond principal
    'accent_primary': '#00d4ff',    # Couleur d'accent
    # ... autres couleurs
}
```

### Ajout de pages

1. Créer un nouveau fichier dans `pages/`
2. Importer dans `app.py`
3. Ajouter la route dans `update_page_content()`

### Composants personnalisés

Ajouter vos composants dans `components/` et les importer où nécessaire.

## 📊 Fonctionnalités de Backtesting

### Stratégies supportées

- **Bollinger Bands + ATR** : Stratégie basée sur la volatilité
- **RSI + Moving Average** : Stratégie momentum/trend-following
- **MACD Cross** : Détection de changement de tendance

### Métriques calculées

- P&L total et par trade
- Taux de réussite (win rate)
- Ratio de Sharpe
- Drawdown maximum
- Nombre de trades

## 🔧 Configuration Avancée

### Variables d'environnement

| Variable | Description | Défaut |
|----------|-------------|--------|
| `HOST` | Adresse d'écoute | `127.0.0.1` |
| `PORT` | Port d'écoute | `8050` |
| `DEBUG` | Mode debug | `True` |
| `AUTH_ENABLED` | Authentification activée | `True` |
| `SESSION_TIMEOUT` | Timeout session (sec) | `86400` |
| `LOG_LEVEL` | Niveau de log | `INFO` |

### Base de données

Pour le moment, les données sont stockées en mémoire. Pour une version production :

1. Configurer PostgreSQL ou MySQL
2. Modifier `utils/auth.py` pour utiliser SQLAlchemy
3. Créer les schémas de base de données

## 📝 Logs

Les logs sont écrits dans :
- **Console** : Logs temps réel
- **Fichier** : `logs/dashboard.log`

Niveaux disponibles : `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

## 🐛 Résolution de problèmes

### Problème : Application ne se lance pas

1. Vérifier l'installation des dépendances : `pip list`
2. Exécuter les tests : `python test_app.py`
3. Vérifier les logs d'erreur

### Problème : Authentification échoue

1. Vérifier les identifiants par défaut
2. Contrôler `AUTH_ENABLED` dans la config
3. Vérifier la `SECRET_KEY`

### Problème : Erreurs d'import

1. Vérifier que vous êtes dans le bon répertoire
2. Activer l'environnement virtuel
3. Réinstaller les dépendances

## 🚀 Déploiement en Production

### Avec Gunicorn (recommandé)

```bash
pip install gunicorn
gunicorn --workers 4 --bind 0.0.0.0:8050 app:server
```

### Avec Docker

```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8050
CMD ["python", "app.py"]
```

### Variables de production

```env
DEBUG=False
AUTH_ENABLED=True
SECRET_KEY=production-secret-key
HOST=0.0.0.0
PORT=8050
```

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commiter les changements (`git commit -am 'Ajouter nouvelle fonctionnalité'`)
4. Pousser la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. Créer une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 🆘 Support

- **Documentation** : Consulter ce README
- **Issues** : Ouvrir une issue GitHub
- **Tests** : Exécuter `python test_app.py`

---

**ThreadX Dashboard** - Plateforme professionnelle de backtesting crypto 🚀
