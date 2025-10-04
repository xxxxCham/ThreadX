"""
ThreadX Phase 2 Validation Script
Script de validation complète pour Phase 2: Data Foundations.

Exécute tous les tests critiques et valide les critères de succès:
- Resample 1m → 1h avec agrégations correctes
- Gestion gaps < 5% avec forward-fill
- I/O + Validation schéma
- Registry rapide
- Déterminisme synthétique
- Batch mode
"""

import sys
import time
import traceback
from pathlib import Path
import tempfile
import pandas as pd

# Ajout du chemin source pour imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def print_section(title: str):
    """Affiche une section avec séparateur."""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def print_test(test_name: str, success: bool, details: str = ""):
    """Affiche le résultat d'un test."""
    status = "✅" if success else "❌"
    print(f"{status} {test_name}")
    if details:
        print(f"   → {details}")

def validate_phase2():
    """Validation complète Phase 2 selon critères de succès."""
    
    print("🚀 VALIDATION PHASE 2 : DATA FOUNDATIONS")
    print("ThreadX - Validation automatisée des critères de succès")
    
    total_tests = 0
    passed_tests = 0
    start_time = time.perf_counter()
    
    # ========================================================================
    # TEST 1: IMPORT MODULES
    # ========================================================================
    
    print_section("Imports Modules ThreadX Data")
    
    try:
        from threadx.data import (
            read_frame, write_frame, normalize_ohlcv,
            resample_from_1m, resample_batch,
            dataset_exists, scan_symbols, quick_inventory, file_checksum,
            make_synth_ohlcv
        )
        print_test("Import modules ThreadX data", True, "Tous les modules importés")
        passed_tests += 1
    except Exception as e:
        print_test("Import modules ThreadX data", False, f"Erreur: {e}")
    
    total_tests += 1
    
    # ========================================================================
    # TEST 2: DONNÉES SYNTHÉTIQUES DÉTERMINISTES
    # ========================================================================
    
    print_section("Déterminisme Données Synthétiques")
    
    try:
        # Génération 2x avec même seed
        df1 = make_synth_ohlcv(n=100, seed=42)
        df2 = make_synth_ohlcv(n=100, seed=42)
        
        is_identical = df1.equals(df2)
        print_test("Déterminisme synthétique (seed=42)", is_identical)
        
        if is_identical:
            passed_tests += 1
            
            # Vérification structure OHLCV
            has_ohlcv = all(col in df1.columns for col in ["open", "high", "low", "close", "volume"])
            is_datetime_index = isinstance(df1.index, pd.DatetimeIndex)
            is_utc = str(df1.index.tz) == "UTC" if df1.index.tz else False
            
            structure_ok = has_ohlcv and is_datetime_index and is_utc
            print_test("Structure OHLCV synthétique", structure_ok, 
                      f"Colonnes: {has_ohlcv}, DatetimeIndex: {is_datetime_index}, UTC: {is_utc}")
            
            if structure_ok:
                passed_tests += 1
        
        total_tests += 2
        
    except Exception as e:
        print_test("Déterminisme synthétique", False, f"Erreur: {e}")
        total_tests += 2
    
    # ========================================================================
    # TEST 3: RESAMPLING 1M → 1H CANONIQUE
    # ========================================================================
    
    print_section("Resampling Canonique 1m → 1h")
    
    try:
        # Données 1m (24h = 1440 minutes → 24 barres 1h)
        df_1m = make_synth_ohlcv(n=1440, freq="1min", seed=123, start="2024-01-01")
        
        # Resampling
        df_1h = resample_from_1m(df_1m, "1h")
        
        # Vérifications
        correct_length = len(df_1h) == 24
        print_test("Resampling 1440min → 24h", correct_length, f"Obtenu: {len(df_1h)} barres")
        
        if correct_length:
            passed_tests += 1
            
            # Vérification agrégations première barre
            first_hour_1m = df_1m.iloc[:60]  # Première heure
            first_1h = df_1h.iloc[0]
            
            import numpy as np
            aggregations_ok = (
                np.allclose(first_1h["open"], first_hour_1m["open"].iloc[0], rtol=1e-12, atol=1e-12) and  # Premier open
                np.allclose(first_1h["high"], first_hour_1m["high"].max(), rtol=1e-12, atol=1e-12) and    # Max high  
                np.allclose(first_1h["low"], first_hour_1m["low"].min(), rtol=1e-12, atol=1e-12) and      # Min low
                np.allclose(first_1h["close"], first_hour_1m["close"].iloc[-1], rtol=1e-12, atol=1e-12) and  # Dernier close
                np.allclose(first_1h["volume"], first_hour_1m["volume"].sum(), rtol=1e-12, atol=1e-12)  # Somme volume
            )
            
            if not aggregations_ok:
                print(f"   Debug agrégations:")
                print(f"     Open: {first_1h['open']:.6f} vs {first_hour_1m['open'].iloc[0]:.6f} (diff: {abs(first_1h['open'] - first_hour_1m['open'].iloc[0]):.8f})")
                print(f"     High: {first_1h['high']:.6f} vs {first_hour_1m['high'].max():.6f} (diff: {abs(first_1h['high'] - first_hour_1m['high'].max()):.8f})")
                print(f"     Low: {first_1h['low']:.6f} vs {first_hour_1m['low'].min():.6f} (diff: {abs(first_1h['low'] - first_hour_1m['low'].min()):.8f})")
                print(f"     Close: {first_1h['close']:.6f} vs {first_hour_1m['close'].iloc[-1]:.6f} (diff: {abs(first_1h['close'] - first_hour_1m['close'].iloc[-1]):.8f})")
            
            print_test("Agrégations OHLCV correctes", aggregations_ok,
                      f"Open: {first_1h['open']:.2f}, High: {first_1h['high']:.2f}, Low: {first_1h['low']:.2f}, Close: {first_1h['close']:.2f}")
            
            if aggregations_ok:
                passed_tests += 1
        
        total_tests += 2
        
    except Exception as e:
        print_test("Resampling 1m → 1h", False, f"Erreur: {e}")
        total_tests += 2
    
    # ========================================================================
    # TEST 4: GESTION GAPS < 5%
    # ========================================================================
    
    print_section("Gestion Gaps Forward-Fill")
    
    try:
        # Données avec gaps artificiels
        df_base = make_synth_ohlcv(n=100, freq="1min", seed=456)
        
        # Suppression 3 timestamps (3% < 5%)
        df_with_gaps = df_base.drop(df_base.index[[10, 25, 60]])
        original_len = len(df_with_gaps)
        
        # Resampling avec gap filling
        df_resampled = resample_from_1m(df_with_gaps, "15m", gap_ffill_threshold=0.05)
        
        # Vérifications
        no_nans = not df_resampled.isnull().any().any()
        has_data = len(df_resampled) > 0
        
        gaps_handled = no_nans and has_data
        print_test("Forward-fill gaps < 5%", gaps_handled, 
                  f"Entrée: {original_len} lignes, Sortie: {len(df_resampled)} barres, NaN: {not no_nans}")
        
        if gaps_handled:
            passed_tests += 1
        
        total_tests += 1
        
    except Exception as e:
        print_test("Gestion gaps", False, f"Erreur: {e}")
        total_tests += 1
    
    # ========================================================================
    # TEST 5: I/O + VALIDATION
    # ========================================================================
    
    print_section("I/O et Validation Schéma")
    
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Génération données test
            df_test = make_synth_ohlcv(n=200, seed=789)
            
            # Test écriture/lecture Parquet
            parquet_path = Path(tmp_dir) / "test.parquet"
            write_frame(df_test, parquet_path)
            df_read_parquet = read_frame(parquet_path, validate=True)
            
            parquet_ok = len(df_read_parquet) == 200
            print_test("I/O Parquet avec validation", parquet_ok, 
                      f"Écrit: {len(df_test)}, Lu: {len(df_read_parquet)}")
            
            if parquet_ok:
                passed_tests += 1
            
            # Test écriture/lecture JSON
            json_path = Path(tmp_dir) / "test.json"
            write_frame(df_test, json_path)
            df_read_json = read_frame(json_path, validate=True)
            
            json_ok = len(df_read_json) == 200
            print_test("I/O JSON avec validation", json_ok,
                      f"Écrit: {len(df_test)}, Lu: {len(df_read_json)}")
            
            if json_ok:
                passed_tests += 1
            
            total_tests += 2
    
    except Exception as e:
        print_test("I/O et validation", False, f"Erreur: {e}")
        total_tests += 2
    
    # ========================================================================
    # TEST 6: REGISTRY RAPIDE
    # ========================================================================
    
    print_section("Registry et Checksums")
    
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Création structure test
            processed_dir = tmp_path / "processed"
            
            # Symboles test
            test_data = {
                "BTCUSDC": ["1m", "15m", "1h"],
                "ETHUSDC": ["1m", "5m"],
                "ADAUSDC": ["15m"]
            }
            
            for symbol, timeframes in test_data.items():
                symbol_dir = processed_dir / symbol
                symbol_dir.mkdir(parents=True)
                
                for tf in timeframes:
                    df_small = make_synth_ohlcv(n=10, seed=hash(symbol+tf) % 1000)
                    tf_path = symbol_dir / f"{tf}.parquet"
                    write_frame(df_small, tf_path)
            
            # Test scan rapide
            start_scan = time.perf_counter()
            inventory = quick_inventory(root=tmp_path)
            scan_elapsed = time.perf_counter() - start_scan
            
            # Normalisation des timeframes triés pour comparaison
            test_data_sorted = {k: sorted(v) for k, v in test_data.items()}
            inventory_sorted = {k: sorted(v) for k, v in inventory.items()}
            
            inventory_correct = inventory_sorted == test_data_sorted
            if not inventory_correct:
                print(f"   Attendu: {test_data_sorted}")
                print(f"   Obtenu: {inventory_sorted}")
            print_test("Quick inventory", inventory_correct, 
                      f"Trouvé: {len(inventory)} symboles en {scan_elapsed:.3f}s")
            
            if inventory_correct:
                passed_tests += 1
            
            # Test checksum
            test_file = processed_dir / "BTCUSDC" / "1m.parquet"
            checksum = file_checksum(test_file)
            
            checksum_ok = len(checksum) == 32 and checksum.isalnum()  # MD5 hex
            print_test("Checksum MD5", checksum_ok, f"Hash: {checksum[:16]}...")
            
            if checksum_ok:
                passed_tests += 1
            
            total_tests += 2
    
    except Exception as e:
        print_test("Registry rapide", False, f"Erreur: {e}")
        total_tests += 2
    
    # ========================================================================
    # TEST 7: BATCH MODE
    # ========================================================================
    
    print_section("Batch Resampling")
    
    try:
        # Préparation données multi-symboles
        frames_by_symbol = {
            f"SYM{i:02d}USDC": make_synth_ohlcv(n=240, seed=i+100, base_price=1000*(i+1))
            for i in range(12)  # 12 symboles → mode parallèle
        }
        
        # Batch resampling
        start_batch = time.perf_counter()
        results = resample_batch(frames_by_symbol, "1h", batch_threshold=5, parallel=True)
        batch_elapsed = time.perf_counter() - start_batch
        
        # Vérifications
        all_processed = len(results) == 12
        all_correct_length = all(len(df) == 4 for df in results.values())  # 240min / 60min = 4h
        order_preserved = list(results.keys()) == list(frames_by_symbol.keys())
        
        batch_ok = all_processed and all_correct_length and order_preserved
        print_test("Batch resampling parallèle", batch_ok,
                  f"{len(results)} symboles, {batch_elapsed:.3f}s, ordre préservé: {order_preserved}")
        
        if batch_ok:
            passed_tests += 1
        
        total_tests += 1
        
    except Exception as e:
        print_test("Batch resampling", False, f"Erreur: {e}")
        total_tests += 1
    
    # ========================================================================
    # RÉSUMÉ FINAL
    # ========================================================================
    
    elapsed = time.perf_counter() - start_time
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print_section("RÉSULTAT VALIDATION PHASE 2")
    
    print(f"📊 RÉSULTAT: {passed_tests}/{total_tests} tests réussis ({success_rate:.1f}%)")
    print(f"⏱️  DURÉE: {elapsed:.2f} secondes")
    
    if passed_tests == total_tests:
        print("🎉 PHASE 2 VALIDÉE - Data Foundations opérationnelles!")
        print("\n✅ Critères de succès Phase 2:")
        print("   ✓ Resample 1m → 1h avec agrégations correctes")
        print("   ✓ Gestion gaps < 5% avec forward-fill") 
        print("   ✓ I/O + validation OHLCV_SCHEMA")
        print("   ✓ Registry rapide (scan O(n) fichiers)")
        print("   ✓ Déterminisme synthétique (seed fixe)")
        print("   ✓ Batch mode parallèle")
        
        print(f"\n🚀 Prêt pour Phase 3: Indicators Layer")
        return True
    else:
        print("❌ VALIDATION INCOMPLÈTE")
        missing = total_tests - passed_tests
        print(f"   → {missing} test{'s' if missing > 1 else ''} échoué{'s' if missing > 1 else ''}")
        
        print("\n🔧 Actions requises:")
        print("   • Vérifier imports modules ThreadX")
        print("   • Corriger erreurs validation schéma")
        print("   • Tester resampling et gap filling")
        print("   • Valider registry et checksums")
        
        return False


if __name__ == "__main__":
    try:
        success = validate_phase2()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 ERREUR CRITIQUE: {e}")
        print("\nTraceback complet:")
        traceback.print_exc()
        sys.exit(2)