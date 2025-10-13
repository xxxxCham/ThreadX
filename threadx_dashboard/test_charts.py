"""
Script de test pour les composants de graphiques ThreadX
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from components.charts import ChartsManager
from config import THEME
from utils.helpers import validate_backtest_data
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def test_charts_creation():
    """Test la création des graphiques"""
    print("=== TEST DES GRAPHIQUES THREADX ===\n")

    # Initialiser le ChartsManager
    print("1. Initialisation du ChartsManager...")
    try:
        charts_manager = ChartsManager(THEME)
        print("✓ ChartsManager initialisé avec succès")
    except Exception as e:
        print(f"✗ Erreur lors de l'initialisation: {e}")
        return False

    # Générer des données de test
    print("\n2. Génération des données de test...")
    try:
        test_data = generate_test_data()
        print("✓ Données de test générées")
        print(f"   - {len(test_data['price_history']['dates'])} jours de données")
        print(f"   - {len(test_data['buy_signals'])} signaux d'achat")
        print(f"   - {len(test_data['sell_signals'])} signaux de vente")
    except Exception as e:
        print(f"✗ Erreur lors de la génération des données: {e}")
        return False

    # Valider les données
    print("\n3. Validation des données...")
    try:
        is_valid = validate_backtest_data(test_data)
        if is_valid:
            print("✓ Structure des données valide")
        else:
            print("✗ Structure des données invalide")
            return False
    except Exception as e:
        print(f"✗ Erreur lors de la validation: {e}")
        return False

    # Tester la création des graphiques
    print("\n4. Création des graphiques...")
    try:
        figures = charts_manager.get_all_figures(test_data)

        # Vérifier que tous les graphiques sont créés
        expected_keys = ["price", "volume", "portfolio"]
        for key in expected_keys:
            if key in figures:
                print(f"✓ Graphique {key} créé avec succès")
            else:
                print(f"✗ Graphique {key} manquant")
                return False

    except Exception as e:
        print(f"✗ Erreur lors de la création des graphiques: {e}")
        return False

    # Test des graphiques individuels
    print("\n5. Test des graphiques individuels...")
    try:
        # Test price chart
        price_history = test_data["price_history"]
        df_price = pd.DataFrame(
            {
                "date": pd.to_datetime(price_history["dates"]),
                "close": price_history["close"],
                "open": price_history["open"],
                "high": price_history["high"],
                "low": price_history["low"],
            }
        )

        price_fig = charts_manager.price_chart.create_chart(
            df_price=df_price,
            buy_signals=test_data["buy_signals"],
            sell_signals=test_data["sell_signals"],
            asset_name=test_data["asset_name"],
        )
        print("✓ Graphique de prix créé")

        # Test volume chart
        df_volume = pd.DataFrame(
            {
                "date": test_data["volume"]["dates"],
                "volume": test_data["volume"]["total"],
            }
        )

        volume_fig = charts_manager.volume_chart.create_chart(df_volume)
        print("✓ Graphique de volume créé")

        # Test portfolio chart
        equity_series = pd.Series(
            test_data["portfolio"]["equity"],
            index=pd.to_datetime(test_data["portfolio"]["dates"]),
        )

        portfolio_fig = charts_manager.portfolio_chart.create_chart(
            equity_curve=equity_series, initial_cash=test_data["initial_cash"]
        )
        print("✓ Graphique de portfolio créé")

    except Exception as e:
        print(f"✗ Erreur lors du test individuel: {e}")
        return False

    print("\n✅ TOUS LES TESTS RÉUSSIS !")
    return True


def generate_test_data():
    """Génère des données de test réalistes"""
    # Paramètres
    days = 365
    start_date = datetime.now() - timedelta(days=days)
    dates = pd.date_range(start=start_date, periods=days, freq="D")
    date_strings = [d.strftime("%Y-%m-%d") for d in dates]

    # Prix avec marche aléatoire
    np.random.seed(42)
    initial_price = 45000
    returns = np.random.normal(0, 0.02, days)
    prices = [initial_price]

    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, initial_price * 0.5))

    # OHLC
    opens = [prices[0]] + prices[:-1]
    highs = [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices]
    lows = [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]

    # Volumes
    volumes = np.random.lognormal(15, 1, days)

    # Signaux de trading
    buy_signals = []
    sell_signals = []

    for i in range(20, days - 20, 45):  # Signaux espacés
        if i % 90 == 20:  # Signal d'achat
            buy_signals.append(
                {"date": date_strings[i], "price": prices[i], "quantity": 0.1}
            )
        elif i % 90 == 65:  # Signal de vente
            sell_signals.append(
                {"date": date_strings[i], "price": prices[i], "quantity": 0.1}
            )

    # Simulation de portfolio
    equity_values = []
    cash = 10000
    position = 0

    for i, price in enumerate(prices):
        # Vérifier signaux
        buy_signal = next(
            (s for s in buy_signals if s["date"] == date_strings[i]), None
        )
        sell_signal = next(
            (s for s in sell_signals if s["date"] == date_strings[i]), None
        )

        if buy_signal and cash > buy_signal["price"] * buy_signal["quantity"]:
            position += buy_signal["quantity"]
            cash -= buy_signal["price"] * buy_signal["quantity"]
        elif sell_signal and position >= sell_signal["quantity"]:
            position -= sell_signal["quantity"]
            cash += sell_signal["price"] * sell_signal["quantity"]

        total_equity = cash + position * price
        equity_values.append(total_equity)

    # Buy & Hold
    initial_btc = 10000 / prices[0]
    buy_hold_values = [initial_btc * p for p in prices]

    return {
        "asset_name": "BTC-USD",
        "period_start": date_strings[0],
        "period_end": date_strings[-1],
        "initial_cash": 10000,
        "final_cash": equity_values[-1],
        "total_return": (equity_values[-1] - 10000) / 10000,
        "price_history": {
            "dates": date_strings,
            "close": prices,
            "open": opens,
            "high": highs,
            "low": lows,
        },
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "volume": {
            "dates": date_strings,
            "total": volumes.tolist(),
            "buy": (volumes * 0.6).tolist(),
            "sell": (volumes * 0.4).tolist(),
        },
        "portfolio": {
            "dates": date_strings,
            "equity": equity_values,
            "cash": [cash] * days,
            "positions": [position * p for p in prices],
        },
        "buy_hold": {
            "dates": date_strings,
            "equity": buy_hold_values,
        },
        "metrics": {
            "total_trades": len(buy_signals) + len(sell_signals),
            "win_rate": 0.65,
            "avg_win": 500,
            "avg_loss": -300,
            "max_drawdown": -0.15,
            "sharpe_ratio": 0.89,
        },
    }


if __name__ == "__main__":
    test_charts_creation()
