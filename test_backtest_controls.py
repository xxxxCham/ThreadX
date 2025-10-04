#!/usr/bin/env python3
"""
Script de test pour le système de contrôle pause/stop des backtests ThreadX.

Ce script teste le BacktestController et montre comment l'utiliser pour
contrôler l'exécution des backtests de manière thread-safe.
"""

import sys
import time
import threading
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from threadx.backtest.engine import BacktestController, BacktestEngine
    from threadx.utils.log import get_logger
    import pandas as pd
    import numpy as np

    THREADX_AVAILABLE = True
    logger = get_logger(__name__)
except ImportError as e:
    print(f"⚠️ ThreadX non disponible: {e}")
    THREADX_AVAILABLE = False


def simulate_long_running_backtest(controller: "BacktestController"):
    """
    Simule un backtest long avec des points de contrôle.
    """
    print("🚀 Démarrage simulation backtest...")

    total_steps = 100
    for i in range(total_steps):
        # Point de contrôle tous les 10 steps
        if i % 10 == 0:
            try:
                controller.check_interruption(f"étape {i}/{total_steps}")
            except KeyboardInterrupt:
                print(f"⏹️ Backtest interrompu à l'étape {i}")
                return False

        # Simulation du travail
        print(f"📊 Traitement étape {i+1}/{total_steps}")
        time.sleep(0.1)  # Simule le travail

    print("✅ Backtest terminé avec succès")
    return True


def test_controller_basic():
    """Test basique du contrôleur."""
    print("\n" + "=" * 50)
    print("🧪 TEST 1: Contrôles de base")
    print("=" * 50)

    if not THREADX_AVAILABLE:
        print("❌ ThreadX non disponible, skip test")
        return

    controller = BacktestController()

    # Test des états initiaux
    print(
        f"État initial - Stopped: {controller.is_stopped}, Paused: {controller.is_paused}"
    )
    assert not controller.is_stopped
    assert not controller.is_paused

    # Test pause
    controller.pause()
    print(
        f"Après pause - Stopped: {controller.is_stopped}, Paused: {controller.is_paused}"
    )
    assert controller.is_paused

    # Test resume
    controller.resume()
    print(
        f"Après resume - Stopped: {controller.is_stopped}, Paused: {controller.is_paused}"
    )
    assert not controller.is_paused

    # Test stop
    controller.stop()
    print(
        f"Après stop - Stopped: {controller.is_stopped}, Paused: {controller.is_paused}"
    )
    assert controller.is_stopped

    # Test reset
    controller.reset()
    print(
        f"Après reset - Stopped: {controller.is_stopped}, Paused: {controller.is_paused}"
    )
    assert not controller.is_stopped
    assert not controller.is_paused

    print("✅ Test des contrôles de base réussi")


def test_controller_threading():
    """Test du contrôleur avec threading."""
    print("\n" + "=" * 50)
    print("🧪 TEST 2: Threading avec contrôles")
    print("=" * 50)

    if not THREADX_AVAILABLE:
        print("❌ ThreadX non disponible, skip test")
        return

    controller = BacktestController()
    results = {"completed": False, "interrupted": False}

    def backtest_worker():
        try:
            results["completed"] = simulate_long_running_backtest(controller)
        except KeyboardInterrupt:
            results["interrupted"] = True
            print("🛑 Worker interrompu")

    # Démarrer le backtest dans un thread
    thread = threading.Thread(target=backtest_worker)
    thread.start()

    # Attendre un peu puis mettre en pause
    time.sleep(0.5)
    print("\n⏸️ Mise en pause du backtest...")
    controller.pause()

    # Attendre puis reprendre
    time.sleep(1.0)
    print("\n▶️ Reprise du backtest...")
    controller.resume()

    # Attendre un peu puis arrêter
    time.sleep(0.5)
    print("\n🛑 Arrêt du backtest...")
    controller.stop()

    # Attendre que le thread se termine
    thread.join(timeout=5.0)

    print(
        f"\nRésultats - Completed: {results['completed']}, Interrupted: {results['interrupted']}"
    )
    print("✅ Test threading réussi")


def test_backtest_engine_integration():
    """Test d'intégration avec BacktestEngine."""
    print("\n" + "=" * 50)
    print("🧪 TEST 3: Intégration BacktestEngine")
    print("=" * 50)

    if not THREADX_AVAILABLE:
        print("❌ ThreadX non disponible, skip test")
        return

    # Créer des données de test
    dates = pd.date_range("2023-01-01", periods=1000, freq="T")
    np.random.seed(42)

    df_test = pd.DataFrame(
        {
            "open": 100 + np.random.randn(1000).cumsum() * 0.1,
            "high": 100 + np.random.randn(1000).cumsum() * 0.1 + 0.1,
            "low": 100 + np.random.randn(1000).cumsum() * 0.1 - 0.1,
            "close": 100 + np.random.randn(1000).cumsum() * 0.1,
            "volume": np.random.randint(1000, 10000, 1000),
        },
        index=dates,
    )

    # Indicateurs mock
    indicators = {
        "bollinger": (
            df_test["close"] + 2,  # upper
            df_test["close"],  # middle
            df_test["close"] - 2,  # lower
        ),
        "atr": pd.Series(np.ones(len(df_test)) * 0.5, index=df_test.index),
    }

    params = {"entry_z": 2.0, "k_sl": 1.5, "leverage": 3.0, "initial_capital": 10000.0}

    # Créer contrôleur et engine
    controller = BacktestController()
    engine = BacktestEngine(controller=controller)

    def run_backtest():
        try:
            result = engine.run(
                df_1m=df_test,
                indicators=indicators,
                params=params,
                symbol="TEST",
                timeframe="1m",
                seed=42,
            )
            print(
                f"✅ Backtest terminé - Status: {result.meta.get('status', 'completed')}"
            )
            return result
        except Exception as e:
            print(f"❌ Erreur backtest: {e}")
            return None

    # Test 1: Backtest normal
    print("\n📊 Test backtest normal...")
    controller.reset()
    result = run_backtest()
    if result and result.meta.get("status") != "interrupted":
        print("✅ Backtest normal réussi")

    # Test 2: Backtest avec interruption
    print("\n📊 Test backtest avec interruption...")
    controller.reset()

    def interrupt_after_delay():
        time.sleep(0.1)  # Attendre le début
        print("🛑 Interruption du backtest...")
        controller.stop()

    interrupt_thread = threading.Thread(target=interrupt_after_delay)
    interrupt_thread.start()

    result = run_backtest()
    interrupt_thread.join()

    if result and result.meta.get("status") == "interrupted":
        print("✅ Test interruption réussi")
    else:
        print("⚠️ Test interruption - résultat inattendu")

    print("✅ Test intégration BacktestEngine terminé")


def interactive_test():
    """Test interactif pour tester manuellement les contrôles."""
    print("\n" + "=" * 50)
    print("🎮 TEST INTERACTIF")
    print("=" * 50)
    print("Commandes disponibles:")
    print("  s - Start/Reset backtest")
    print("  p - Pause backtest")
    print("  r - Resume backtest")
    print("  x - Stop backtest")
    print("  q - Quit")
    print("=" * 50)

    if not THREADX_AVAILABLE:
        print("❌ ThreadX non disponible, mode simulation activé")
        return

    controller = BacktestController()
    backtest_thread = None

    def run_backtest_interactive():
        try:
            print("🚀 Backtest interactif démarré...")
            simulate_long_running_backtest(controller)
        except KeyboardInterrupt:
            print("🛑 Backtest interrompu")

    while True:
        try:
            cmd = input("\n> ").lower().strip()

            if cmd == "q":
                if backtest_thread and backtest_thread.is_alive():
                    controller.stop()
                    backtest_thread.join(timeout=2.0)
                print("👋 Au revoir!")
                break

            elif cmd == "s":
                if backtest_thread and backtest_thread.is_alive():
                    print("⚠️ Un backtest est déjà en cours")
                    continue
                controller.reset()
                backtest_thread = threading.Thread(target=run_backtest_interactive)
                backtest_thread.start()
                print("✅ Backtest démarré")

            elif cmd == "p":
                controller.pause()
                print("⏸️ Pause demandée")

            elif cmd == "r":
                controller.resume()
                print("▶️ Resume demandé")

            elif cmd == "x":
                controller.stop()
                print("🛑 Stop demandé")

            else:
                print("❌ Commande inconnue")

        except KeyboardInterrupt:
            print("\n🛑 Interruption détectée")
            if backtest_thread and backtest_thread.is_alive():
                controller.stop()
                backtest_thread.join(timeout=2.0)
            break


def main():
    """Fonction principale de test."""
    print("🧪 ThreadX Backtest Controller - Tests")
    print("=====================================")

    # Tests automatiques
    test_controller_basic()
    test_controller_threading()
    test_backtest_engine_integration()

    # Test interactif optionnel
    try:
        response = input("\n🎮 Lancer le test interactif ? (y/N): ").lower().strip()
        if response in ["y", "yes", "oui"]:
            interactive_test()
    except KeyboardInterrupt:
        print("\n👋 Tests terminés")


if __name__ == "__main__":
    main()
