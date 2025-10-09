#!/usr/bin/env python3
"""
ThreadX Unified Simple Launcher
"""

import sys
from pathlib import Path

# Add ThreadX to path
threadx_root = Path(__file__).parent.parent
sys.path.insert(0, str(threadx_root))

if __name__ == "__main__":
    try:
        from apps.threadx_unified_simple import main

        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        input("Press Enter to exit...")
        sys.exit(1)
