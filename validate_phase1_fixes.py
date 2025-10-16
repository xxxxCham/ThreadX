"""
Quick Validation - Phase 1 Fixes
=================================

Tests simples qui vérifient que les fixes ont été appliquées
sans dépendances complexes à l'architecture.
"""

import sys
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_fix1_get_state_with_lock():
    """✅ FIX #1: Vérifie que queue_size() est appelé DANS le lock."""
    import inspect
    from threadx.bridge.async_coordinator import ThreadXBridge

    # Obtenir le code source de get_state
    source = inspect.getsource(ThreadXBridge.get_state)

    # Vérifier que qsize() est appelé dans le lock
    lines = source.split("\n")

    # Trouver la section "with self.state_lock:"
    in_lock = False
    qsize_in_lock = False

    for i, line in enumerate(lines):
        if "with self.state_lock:" in line:
            in_lock = True

        if in_lock and "queue_size" in line and "qsize()" in line:
            qsize_in_lock = True
            break

        if in_lock and ("return" in line or line.strip().startswith("def ")):
            break

    assert qsize_in_lock, "❌ FIX #1 FAILED: qsize() not called inside lock"
    print("✅ FIX #1 PASSED: qsize() called inside lock")


def test_fix2_finalize_helper_exists():
    """✅ FIX #2: Vérifie que le helper _finalize_task_result existe."""
    from threadx.bridge.async_coordinator import ThreadXBridge

    # Vérifier que la méthode existe
    assert hasattr(
        ThreadXBridge, "_finalize_task_result"
    ), "❌ FIX #2 FAILED: _finalize_task_result method missing"

    # Vérifier signature
    import inspect

    sig = inspect.signature(ThreadXBridge._finalize_task_result)
    params = set(sig.parameters.keys())

    expected = {"self", "task_id", "result", "error", "event_type_success", "callback"}
    assert (
        expected <= params
    ), f"❌ FIX #2 FAILED: Missing parameters. Expected {expected}, got {params}"

    # Vérifier qu'il utilise les lock correctement
    source = inspect.getsource(ThreadXBridge._finalize_task_result)
    assert (
        "with self.state_lock:" in source
    ), "❌ FIX #2 FAILED: Lock not used in _finalize_task_result"

    print(
        "✅ FIX #2 PASSED: Helper _finalize_task_result exists with correct signature"
    )


def test_fix3_parse_timestamps_exists():
    """✅ FIX #3: Vérifie que _parse_timestamps_to_utc existe."""
    from threadx.data.ingest import IngestionManager

    # Vérifier que la méthode existe
    assert hasattr(
        IngestionManager, "_parse_timestamps_to_utc"
    ), "❌ FIX #3 FAILED: _parse_timestamps_to_utc method missing"

    # Vérifier signature
    import inspect

    sig = inspect.signature(IngestionManager._parse_timestamps_to_utc)
    params = set(sig.parameters.keys())

    expected = {"self", "start", "end"}
    assert (
        expected <= params
    ), f"❌ FIX #3 FAILED: Missing parameters. Expected {expected}, got {params}"

    print(
        "✅ FIX #3 PASSED: Helper _parse_timestamps_to_utc exists with correct signature"
    )


def test_fix3_timezone_handling():
    """✅ FIX #3: Vérifie que get_1m() utilise le helper de normalisation."""
    import inspect
    from threadx.data.ingest import IngestionManager

    source = inspect.getsource(IngestionManager.get_1m)

    # Vérifier que _parse_timestamps_to_utc est appelé
    assert (
        "_parse_timestamps_to_utc" in source
    ), "❌ FIX #3 FAILED: _parse_timestamps_to_utc not called in get_1m()"

    # Vérifier qu'il y a du tz_localize ou tz_convert
    assert (
        "tz_localize" in source or "tz_convert" in source
    ), "❌ FIX #3 FAILED: Timezone normalization not found"

    print("✅ FIX #3 PASSED: get_1m() uses timezone normalization helper")


def main():
    print("=" * 70)
    print("PHASE 1 FIXES - QUICK VALIDATION")
    print("=" * 70)

    try:
        test_fix1_get_state_with_lock()
        test_fix2_finalize_helper_exists()
        test_fix3_parse_timestamps_exists()
        test_fix3_timezone_handling()

        print("=" * 70)
        print("✅ ALL FIXES VALIDATED SUCCESSFULLY")
        print("=" * 70)
        return 0

    except AssertionError as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
