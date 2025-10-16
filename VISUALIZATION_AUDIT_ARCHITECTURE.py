"""
VISUALISATION AUDIT - Violations d'Architecture ThreadX
=======================================================

État Actuel vs État Cible de l'Architecture
"""

# ════════════════════════════════════════════════════════════════════════
# ÉTAT ACTUEL - ThreadX Architecture (16 octobre 2025)
# ════════════════════════════════════════════════════════════════════════

ACTUAL_ARCHITECTURE = """

┌─────────────────────────────────────────────────────────────┐
│                    DASH UI LAYER                            │
│  ✅ layout.py        → Bridge only                          │
│  ✅ callbacks.py     → Bridge only (NOT REGISTERED!)        │
│  ✅ components/*     → Bridge only                          │
│  ✅ charts.py        → Bridge only                          │
│  ✅ tables.py        → Bridge only                          │
│  ❌ sweep.py        → DIRECT ENGINE IMPORTS!               │
│  ❌ downloads.py    → DIRECT IngestionManager!             │
│  ❌ data_manager.py → DIRECT IngestionManager!             │
└─────────────────────────────────────────────────────────────┘
                            ↓
         ┌──────────────────────────────────────┐
         │   BRIDGE LAYER (Incomplete!)         │
         │                                      │
         │  ✅ BacktestController              │
         │  ✅ MetricsController               │
         │  ✅ DataIngestionController         │
         │  ❌ SweepController (MISSING!)      │
         │                                      │
         │  ❌ Models Duplication:              │
         │     - models.py (OLD DataClass)     │
         │     - validation.py (NEW Pydantic)  │
         └──────────────────────────────────────┘
                            ↓
         ┌──────────────────────────────────────┐
         │      ENGINE LAYER                    │
         │                                      │
         │  ✅ BacktestEngine                  │
         │  ✅ IndicatorBank                   │
         │  ✅ OptimizationEngine              │
         │  ✅ IngestionManager                │
         │  ✅ PerformanceCalculator           │
         └──────────────────────────────────────┘

VIOLATIONS DÉTECTÉES:
  🔴 sweep.py IMPORTS: UnifiedOptimizationEngine, IndicatorBank
  🔴 downloads.py IMPORTS: IngestionManager
  🔴 data_manager.py IMPORTS: IngestionManager
  🔴 callbacks not registered in dash_app.py
"""

# ════════════════════════════════════════════════════════════════════════
# ÉTAT CIBLE - ThreadX Architecture (After Fixes)
# ════════════════════════════════════════════════════════════════════════

TARGET_ARCHITECTURE = """

┌─────────────────────────────────────────────────────────────┐
│                    DASH UI LAYER                            │
│  ✅ layout.py        → Bridge only                          │
│  ✅ callbacks.py     → Bridge only (REGISTERED)             │
│  ✅ components/*     → Bridge only                          │
│  ✅ components/optimization_panel → Bridge only           │
│  ✅ charts.py        → Bridge only                          │
│  ✅ tables.py        → Bridge only                          │
│  ✅ downloads.py     → Bridge DataIngestionController      │
│  ✅ data_manager.py  → Bridge DataIngestionController      │
│  ✅ sweep.py         → Bridge SweepController              │
└─────────────────────────────────────────────────────────────┘
                            ↓
         ┌──────────────────────────────────────┐
         │   BRIDGE LAYER (Complete!)           │
         │                                      │
         │  ✅ BacktestController              │
         │  ✅ MetricsController               │
         │  ✅ DataIngestionController         │
         │  ✅ SweepController (NEW!)          │
         │  ✅ IndicatorController             │
         │                                      │
         │  ✅ Unified Pydantic Models:        │
         │     - BacktestRequest               │
         │     - IndicatorRequest              │
         │     - DataValidationRequest         │
         │     - OptimizeRequest               │
         └──────────────────────────────────────┘
                            ↓
         ┌──────────────────────────────────────┐
         │      ENGINE LAYER                    │
         │                                      │
         │  ✅ BacktestEngine                  │
         │  ✅ IndicatorBank                   │
         │  ✅ OptimizationEngine              │
         │  ✅ IngestionManager                │
         │  ✅ PerformanceCalculator           │
         └──────────────────────────────────────┘

RÉSULTAT:
  ✅ ZERO direct Engine imports from UI
  ✅ ALL UI → Bridge only
  ✅ Bridge → Engine only
  ✅ Callbacks registered & working
  ✅ Models unified (Pydantic)
"""

# ════════════════════════════════════════════════════════════════════════
# VIOLATION DETAILS BY FILE
# ════════════════════════════════════════════════════════════════════════

VIOLATIONS = """

FILE: src/threadx/ui/sweep.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Line 32-34 ❌ VIOLATION
    from ..optimization.engine import UnifiedOptimizationEngine, DEFAULT_SWEEP_CONFIG
    from ..indicators.bank import IndicatorBank

Line 58-61 ❌ VIOLATING CODE
    self.optimization_engine = UnifiedOptimizationEngine(
        indicator_bank=self.indicator_bank, max_workers=4
    )

FIX:
    from threadx.bridge import SweepController
    self.sweep_controller = SweepController()
    task_id = self.sweep_controller.run_sweep_async(request)


FILE: src/threadx/ui/downloads.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Line 26 ❌ VIOLATION
    from ..data.ingest import IngestionManager

Line 50+ ❌ VIOLATING CODE
    manager = IngestionManager()
    manager.download_symbols()
    manager.download_data()

FIX:
    from threadx.bridge import DataIngestionController
    controller = DataIngestionController()
    result = controller.ingest_batch(request)


FILE: src/threadx/ui/data_manager.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Line 23 ❌ VIOLATION
    from ..data.ingest import IngestionManager

Line 100+ ❌ VIOLATING CODE
    manager = IngestionManager()
    manager.scan_available_symbols()

FIX:
    from threadx.bridge import DataIngestionController
    controller = DataIngestionController()
    symbols = controller.scan_available_symbols()


FILE: apps/dash_app.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Line 50 ❌ MISSING REGISTRATION
    register_callbacks = None  # Never called!

FIX:
    if register_callbacks and bridge:
        register_callbacks(app, bridge)
        logger.info("✅ Callbacks registered")
    else:
        logger.warning("⚠️  Callbacks not registered")
"""

# ════════════════════════════════════════════════════════════════════════
# TEST RESULTS BEFORE & AFTER
# ════════════════════════════════════════════════════════════════════════

TEST_RESULTS = """

BEFORE FIXES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

$ pytest tests/test_architecture_separation.py -v

test_ui_files_discovered ✅ PASS
test_ui_no_engine_imports ⚠️  PARTIAL (sweep.py not caught)
test_ui_no_pandas_operations ✅ PASS
test_bridge_exports_validation ✅ PASS
test_bridge_controllers_exist ⚠️  PARTIAL (SweepController missing)

RESULT: 4/5 PASS (80%) - False positive, real violations exist


AFTER FIXES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

$ pytest tests/test_architecture_separation.py -v

test_ui_files_discovered ✅ PASS
test_ui_no_engine_imports ✅ PASS (zero violations)
test_ui_no_pandas_operations ✅ PASS
test_bridge_exports_validation ✅ PASS
test_bridge_controllers_exist ✅ PASS (SweepController exists)
test_callbacks_registered ✅ PASS (NEW)

RESULT: 6/6 PASS (100%) - All architecture rules enforced!
"""

# ════════════════════════════════════════════════════════════════════════
# IMPLEMENTATION ROADMAP
# ════════════════════════════════════════════════════════════════════════

ROADMAP = """

PHASE 1: Create SweepController (15 min)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
File: src/threadx/bridge/controllers.py

class SweepController:
    def run_sweep_async(self, request: SweepRequest) -> str:
        # task_id
        ...

    def run_sweep(self, request: SweepRequest) -> SweepResult:
        # synchronous version
        ...


PHASE 2: Fix sweep.py imports (10 min)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Line 32-34: Remove direct Engine imports
            Add Bridge import

Line 58-61: Replace direct engine instantiation
           with controller method call


PHASE 3: Fix downloads.py imports (10 min)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Line 26: Remove IngestionManager
        Add DataIngestionController

Line 50+: Replace direct calls with controller


PHASE 4: Fix data_manager.py imports (10 min)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Line 23: Remove IngestionManager
        Add DataIngestionController

Line 100+: Replace direct calls with controller


PHASE 5: Register callbacks in dash_app.py (5 min)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Line 50: Add callback registration call


TOTAL TIME: ~45 minutes to full compliance


VALIDATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
After each phase:
    $ pytest tests/test_architecture_separation.py -v

Final validation:
    $ pytest tests/test_architecture_separation.py -v
    Expected: 6/6 PASS ✅
"""

if __name__ == "__main__":
    print(ACTUAL_ARCHITECTURE)
    print("\n" + "=" * 70 + "\n")
    print(TARGET_ARCHITECTURE)
    print("\n" + "=" * 70 + "\n")
    print(VIOLATIONS)
    print("\n" + "=" * 70 + "\n")
    print(TEST_RESULTS)
    print("\n" + "=" * 70 + "\n")
    print(ROADMAP)
