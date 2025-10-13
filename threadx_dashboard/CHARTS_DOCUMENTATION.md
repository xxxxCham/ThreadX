# ThreadX Dashboard - Documentation des Graphiques

## Vue d'ensemble

Ce document explique l'utilisation des composants de visualisation créés pour la page de backtesting de ThreadX Dashboard.

## Structure des Composants

### 1. Fichier `components/charts.py`

Ce fichier contient toutes les classes pour créer des graphiques interactifs :

- **PriceAndSignalsChart** : Graphique principal prix + signaux
- **TradingVolumeChart** : Graphique de volume de trading
- **PortfolioBalanceChart** : Graphique d'équité du portefeuille
- **ChartsManager** : Orchestrateur de tous les graphiques

### 2. Fichier `pages/backtesting.py`

Page complète avec layout et callbacks pour afficher les graphiques de backtesting.

## Utilisation des Graphiques

### Exemple Basique

```python
from components.charts import ChartsManager
from config import THEME

# Initialiser le gestionnaire
charts_manager = ChartsManager(THEME)

# Générer tous les graphiques
figures = charts_manager.get_all_figures(backtest_data)

# Accéder aux graphiques individuels
price_figure = figures['price']
volume_figure = figures['volume']
portfolio_figure = figures['portfolio']
```

### Données d'Entrée Requises

Les données de backtest doivent avoir cette structure :

```python
backtest_data = {
    'asset_name': 'BTC-USD',
    'initial_cash': 10000,

    'price_history': {
        'dates': ['2023-01-01', '2023-01-02', ...],  # Format YYYY-MM-DD
        'close': [45000.0, 45500.0, ...],           # Prix de clôture
        'open': [44800.0, 45000.0, ...],            # Prix d'ouverture
        'high': [45200.0, 45800.0, ...],            # Prix max
        'low': [44700.0, 44900.0, ...]              # Prix min
    },

    'buy_signals': [
        {'date': '2023-01-15', 'price': 46000.0, 'quantity': 0.1},
        {'date': '2023-02-20', 'price': 42000.0, 'quantity': 0.2},
        ...
    ],

    'sell_signals': [
        {'date': '2023-03-10', 'price': 48000.0, 'quantity': 0.1},
        ...
    ],

    'volume': {
        'dates': ['2023-01-01', ...],
        'total': [1000000, ...],      # Volume total
        'buy': [600000, ...],         # Volume achat (optionnel)
        'sell': [400000, ...]         # Volume vente (optionnel)
    },

    'portfolio': {
        'dates': ['2023-01-01', ...],
        'equity': [10000, 10200, ...],    # Équité totale
        'cash': [8000, 7800, ...],        # Cash disponible
        'positions': [2000, 2400, ...]    # Valeur des positions
    },

    'buy_hold': {  # Optionnel - pour comparaison
        'dates': ['2023-01-01', ...],
        'equity': [10000, 10150, ...]
    }
}
```

## Fonctionnalités des Graphiques

### PriceAndSignalsChart

- **Affichage** : Prix en chandelier ou ligne + signaux d'achat/vente
- **Signaux** : Triangles verts (achat) et rouges (vente)
- **Support/Résistance** : Lignes horizontales calculées automatiquement
- **Interactivité** : Zoom, pan, rangeslider, hover tooltips

**Styling :**
- Fond sombre : #1a1a1a
- Prix : ligne blanche épaisse
- Support : ligne orange pointillée
- Résistance : ligne verte pointillée
- Signaux : marqueurs avec contour blanc

### TradingVolumeChart

- **Affichage** : Barres de volume + moyenne mobile
- **Données** : Volume total, buy/sell séparés (optionnel)
- **Indicateur** : MA20 par défaut (configurable)

**Styling :**
- Volume total : barres cyan semi-transparentes
- Volume buy : barres vertes
- Volume sell : barres rouges
- Moyenne mobile : ligne orange

### PortfolioBalanceChart

- **Affichage** : Courbe d'équité + drawdown + benchmark
- **Métriques** : Drawdown max, ROI, Sharpe ratio
- **Comparaison** : Stratégie vs Buy & Hold

**Styling :**
- Équité stratégie : ligne cyan épaisse
- Buy & Hold : ligne cyan pointillée
- Zones drawdown : remplissage rouge transparent
- Annotations : métriques en overlay

## Callbacks et Interactions

### Timeframe Selector

Le dropdown timeframe filtre automatiquement toutes les données :

```python
@callback(
    [Output('price-graph', 'figure'), ...],
    [Input('chart-timeframe', 'value')]
)
def update_charts(timeframe):
    filtered_data = filter_data_by_timeframe(data, timeframe)
    # ...
```

**Options disponibles :**
- 1D : Dernier jour
- 1W : Dernière semaine
- 1M : Dernier mois
- 3M : 3 derniers mois
- ALL : Toute la période

### Métriques Dynamiques

Les métriques sont calculées et mises à jour automatiquement :

```python
def calculate_metrics(data):
    equity = pd.Series(data['portfolio']['equity'])

    # Calculs automatiques
    final_equity = equity.iloc[-1]
    total_return = (final_equity - initial_cash) / initial_cash
    max_drawdown = calculate_drawdown(equity)[1]
    sharpe_ratio = calculate_sharpe(equity)

    return {
        'final_equity': final_equity,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio
    }
```

## Fonctions Utilitaires

### Dans `utils/helpers.py`

```python
# Filtrage temporel
filter_data_by_timeframe(data, timeframe)

# Calcul du drawdown
calculate_drawdown(equity_series)

# Calcul support/résistance
calculate_support_resistance(prices, window=20)

# Validation des données
validate_backtest_data(data)
```

## Personnalisation des Thèmes

Tous les graphiques utilisent les couleurs du thème défini dans `config.py` :

```python
THEME = {
    'primary_bg': '#1a1a1a',      # Fond principal
    'secondary_bg': '#242424',    # Fond secondaire
    'text_primary': '#ffffff',    # Texte principal
    'text_secondary': '#b0b0b0',  # Texte secondaire
    'accent': '#00d4ff',          # Couleur d'accent (cyan)
    'success': '#00ff00',         # Vert (gains)
    'danger': '#ff4444',          # Rouge (pertes)
    'warning': '#ffaa00',         # Orange (alertes)
    'border': '#404040'           # Bordures
}
```

## Performance et Optimisation

### Recommandations

1. **Données volumineuses** : Utiliser `scattergl` pour > 1000 points
2. **Mise en cache** : Stocker les figures calculées dans `dcc.Store`
3. **Chargement asynchrone** : Utiliser `dcc.Loading` pour l'UX
4. **Responsive** : Tous les graphiques s'adaptent à la taille d'écran

### Exemple d'optimisation

```python
# Pour de gros datasets
fig.add_trace(go.Scattergl(  # Au lieu de go.Scatter
    x=dates,
    y=prices,
    mode='lines'
))

# Configuration responsive
fig.update_layout(
    autosize=True,
    margin=dict(l=0, r=0, t=30, b=0)
)
```

## Test et Validation

### Script de test

Le fichier `test_charts.py` valide automatiquement :

1. Création du ChartsManager
2. Génération de données test
3. Validation de structure
4. Création des graphiques
5. Tests individuels

### Lancer les tests

```bash
cd threadx_dashboard
python test_charts.py
```

## Intégration avec Pages

### Dans une nouvelle page

```python
from components.charts import ChartsManager
from config import THEME

# Dans le layout
dcc.Graph(id='my-chart'),
dcc.Store(id='chart-data')

# Dans les callbacks
@callback(
    Output('my-chart', 'figure'),
    Input('chart-data', 'data')
)
def update_chart(data):
    charts_manager = ChartsManager(THEME)
    figures = charts_manager.get_all_figures(data)
    return figures['price']
```

## Dépannage

### Erreurs courantes

1. **"'date' key not found"** : Vérifier la structure des données
2. **"Empty figure"** : Données invalides ou manquantes
3. **"Layout error"** : Problème de thème ou configuration

### Solutions

```python
# Validation avant utilisation
if validate_backtest_data(data):
    figures = charts_manager.get_all_figures(data)
else:
    # Utiliser des données par défaut
    figures = {'price': empty_figure, ...}

# Debug des données
print(f"Keys: {list(data.keys())}")
print(f"Price dates: {len(data['price_history']['dates'])}")
```

## Support et Contribution

- **Issues** : Signaler les bugs dans le projet ThreadX
- **Features** : Proposer des améliorations
- **Documentation** : Améliorer cette doc

---

*Cette documentation couvre l'utilisation complète des composants de graphiques ThreadX Dashboard v1.0*
