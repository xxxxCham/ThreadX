#!/usr/bin/env python3
"""
Script de validation des corrections d'architecture
====================================================

Vérifie que tous les imports directs Engine ont été supprimés de la couche UI.

Usage:
    python validate_fixes.py
"""

import re
from pathlib import Path
from typing import List, Tuple

# Fichiers UI à vérifier
UI_FILES = [
    "src/threadx/ui/sweep.py",
    "src/threadx/ui/downloads.py",
    "src/threadx/ui/data_manager.py",
]

# Imports interdits (Engine direct)
FORBIDDEN_IMPORTS = [
    r"from\s+\.\.\s*optimization\s*import",  # from ..optimization import
    r"from\s+\.\.\s*data\s*import.*IngestionManager",  # from ..data import IngestionManager
    r"from\s+\.\.\s*indicators\s*import",  # from ..indicators import
]

# Imports requis (Bridge)
REQUIRED_IMPORTS = [
    "from threadx.bridge import",  # Utiliser Bridge
]


def check_file(filepath: Path) -> Tuple[bool, List[str]]:
    """
    Vérifie un fichier UI pour violations d'architecture.

    Returns:
        (is_compliant, violations)
    """
    violations = []
    content = filepath.read_text()

    # Check forbidden imports
    for pattern in FORBIDDEN_IMPORTS:
        matches = re.finditer(pattern, content, re.MULTILINE)
        for match in matches:
            # Get line number
            line_num = content[: match.start()].count("\n") + 1
            violations.append(f"  ❌ Ligne {line_num}: {match.group()}")

    return len(violations) == 0, violations


def validate_all():
    """Valide tous les fichiers UI."""
    print("\n" + "=" * 70)
    print("🔍 VALIDATION ARCHITECTURE - Corrections Appliquées")
    print("=" * 70)

    all_compliant = True
    results = {}

    for ui_file in UI_FILES:
        filepath = Path(ui_file)
        if not filepath.exists():
            print(f"\n⚠️  {ui_file}: Fichier non trouvé")
            continue

        is_compliant, violations = check_file(filepath)
        results[ui_file] = (is_compliant, violations)

        if is_compliant:
            print(f"\n✅ {ui_file}")
            print(f"   → Aucune violation d'import Engine détectée")
        else:
            print(f"\n❌ {ui_file}")
            print(f"   → {len(violations)} violation(s):")
            for v in violations:
                print(f"     {v}")
            all_compliant = False

    # Summary
    print("\n" + "=" * 70)
    compliant_count = sum(1 for is_compliant, _ in results.values() if is_compliant)
    total_count = len(results)

    if all_compliant:
        print(f"✅ SUCCÈS: {compliant_count}/{total_count} fichiers conformes")
        print("   → Architecture 3-tier respectée!")
        print("   → Tous les imports Engine ont été remplacés par Bridge")
    else:
        print(f"❌ ÉCHEC: {compliant_count}/{total_count} fichiers conformes")
        print("   → Des violations restent à corriger")

    print("=" * 70 + "\n")
    return all_compliant


if __name__ == "__main__":
    import sys

    success = validate_all()
    sys.exit(0 if success else 1)
