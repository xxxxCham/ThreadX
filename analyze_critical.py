#!/usr/bin/env python3
"""Script pour analyser les problèmes critiques et haute priorité"""

import json

with open("AUDIT_THREADX_FINDINGS.json", encoding="utf-8") as f:
    data = json.load(f)

critical_high = [f for f in data["findings"] if f["severity"] in ["critical", "high"]]

print(
    f"Problèmes CRITICAL: {len([f for f in critical_high if f['severity'] == 'critical'])}"
)
print(f"Problèmes HIGH: {len([f for f in critical_high if f['severity'] == 'high'])}")
print("\n" + "=" * 80)
print("DÉTAILS DES PROBLÈMES CRITIQUES ET HIGH")
print("=" * 80 + "\n")

for idx, finding in enumerate(critical_high, 1):
    print(f"{idx}. [{finding['severity'].upper()}] {finding['category'].upper()}")
    print(f"   Fichier: {finding['file']}:{finding['line']}")
    print(f"   Problème: {finding['description']}")
    print(f"   Action: {finding['recommendation']}")
    print()
