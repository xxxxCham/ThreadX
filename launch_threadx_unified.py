#!/usr/bin/env python3
"""
ThreadX Unified Interface Launcher
Inspired by TradXPro architecture
"""

import sys
from pathlib import Path

# Add ThreadX to path
threadx_root = Path(__file__).parent.parent
sys.path.insert(0, str(threadx_root))

if __name__ == "__main__":
    try:
        from apps.threadx_unified_interface import main

        main()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("🔧 Please ensure ThreadX is properly installed")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
