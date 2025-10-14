#!/usr/bin/env python3
"""
Audit ThreadX : Identifie code métier mélangé dans UI
Exécution : python scripts/audit_ui_engine.py
Sortie : AUDIT_THREADX.md
"""

import os
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Configuration
UI_PATHS = [Path("src/threadx/ui"), Path("apps"), Path("threadx_dashboard")]
OUTPUT_FILE = Path("AUDIT_THREADX.md")

# Imports métier à rechercher
DANGEROUS_IMPORTS = {
    "create_engine": "Moteur backtest",
    "IndicatorBank": "Banque indicateurs",
    "PerformanceCalculator": "Calcul perf",
    "PerformanceMetrics": "Métriques perf",
    "IngestionManager": "Ingestion données",
    "UnifiedOptimizationEngine": "Engine optimisation",
    "DeviceManager": "Gestionnaire GPU",
    "MultiGPUDispatcher": "Dispatcher GPU",
    "BacktestEngine": "Engine backtest",
}

DANGEROUS_MODULES = [
    "threadx.backtest",
    "threadx.indicators",
    "threadx.optimization",
    "threadx.data.ingest",
    "threadx.utils.gpu",
]

# Patterns de calcul en UI
CALC_PATTERNS = {
    r"engine\.run\(": "Exécution backtest",
    r"create_engine": "Création engine",
    r"indicator_bank\.calculate": "Calcul indicateurs",
    r"sweep_engine\.run": "Exécution sweep",
    r"\.fit\(": "Entraînement ML",
    r"\.predict\(": "Prédiction ML",
    r"\.resample\(": "Transformation données",
    r"\.dropna\(": "Nettoyage données",
    r"\.fillna\(": "Imputation données",
}


def scan_file(filepath):
    """Scan un fichier pour imports/calculs métier"""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            lines = content.split("\n")
    except Exception as e:
        print(f"Erreur lecture {filepath}: {e}")
        return {"imports": [], "calculations": [], "severity": "🟢 VERT"}

    findings = {
        "imports": [],
        "calculations": [],
        "severity": "🟢 VERT",
    }

    # Scan imports
    for line_num, line in enumerate(lines, 1):
        for module in DANGEROUS_MODULES:
            if re.search(rf"from {re.escape(module)}", line) or re.search(
                rf"import {re.escape(module)}", line
            ):
                findings["imports"].append(
                    {
                        "line": line_num,
                        "code": line.strip(),
                        "module": module,
                    }
                )

        for import_name, description in DANGEROUS_IMPORTS.items():
            if re.search(rf"\b{import_name}\b", line) and "import" in line:
                findings["imports"].append(
                    {
                        "line": line_num,
                        "code": line.strip(),
                        "name": import_name,
                        "desc": description,
                    }
                )

    # Scan calculs
    for line_num, line in enumerate(lines, 1):
        for pattern, description in CALC_PATTERNS.items():
            if re.search(pattern, line):
                findings["calculations"].append(
                    {
                        "line": line_num,
                        "code": line.strip(),
                        "pattern": pattern,
                        "desc": description,
                    }
                )

    # Déterminer sévérité
    if findings["imports"] or findings["calculations"]:
        if len(findings["calculations"]) > 2 or len(findings["imports"]) > 3:
            findings["severity"] = "🔴 CRITIQUE"
        elif findings["calculations"] or findings["imports"]:
            findings["severity"] = "🟡 MOYEN"

    return findings


def scan_directory(path):
    """Scanner récursivement un répertoire pour fichiers Python"""
    results = {}

    if not path.exists():
        return results

    for py_file in path.rglob("*.py"):
        # Ignorer les fichiers de test et cache
        if any(part.startswith(".") or part == "__pycache__" for part in py_file.parts):
            continue
        if "test" in py_file.name or "conftest" in py_file.name:
            continue

        try:
            relative_path = py_file.relative_to(Path.cwd())
        except ValueError:
            relative_path = py_file
        findings = scan_file(py_file)

        if findings["imports"] or findings["calculations"]:
            results[str(relative_path)] = findings

    return results


def main():
    """Lancer audit complet"""
    print("🔍 Audit ThreadX en cours...\n")

    all_findings = {}
    total_issues = 0
    files_scanned = 0

    # Scanner tous les répertoires UI
    for ui_path in UI_PATHS:
        if ui_path.exists():
            print(f"📁 Scanning {ui_path}...")
            results = scan_directory(ui_path)
            all_findings.update(results)

            # Compter les fichiers
            for py_file in ui_path.rglob("*.py"):
                if not any(
                    part.startswith(".") or part == "__pycache__"
                    for part in py_file.parts
                ):
                    files_scanned += 1

    # Calculer totaux
    for filepath, findings in all_findings.items():
        issue_count = len(findings["imports"]) + len(findings["calculations"])
        total_issues += issue_count
        print(f"  ❌ {filepath}: {issue_count} issues ({findings['severity']})")

    # Générer rapport
    report = generate_report(all_findings, total_issues, files_scanned)

    # Sauvegarder
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✅ Audit terminé. Rapport : {OUTPUT_FILE}")
    print(
        f"📊 Résumé : {total_issues} issues détectées dans {len(all_findings)} fichiers"
    )
    print(f"📂 {files_scanned} fichiers Python analysés\n")


def generate_report(findings, total, files_scanned):
    """Générer rapport Markdown"""
    md = f"""# 🔍 AUDIT THREADX : Séparation UI / Métier

**Date** : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Auditeur** : Script automatisé
**Statut** : Audit complet

---

## 📊 Résumé

| Métrique | Valeur |
|----------|--------|
| Fichiers Python analysés | {files_scanned} |
| Fichiers problématiques | {len(findings)} |
| Issues totales | {total} |
| Imports métier trouvés | {sum(len(f['imports']) for f in findings.values())} |
| Calculs en UI détectés | {sum(len(f['calculations']) for f in findings.values())} |

**Priorité globale** : {"🔴 CRITIQUE" if total > 10 else "🟡 MOYENNE" if total > 0 else "🟢 BASSE"}

---

## 📁 Détail par fichier

"""

    # Trier par sévérité
    sorted_files = sorted(
        findings.items(),
        key=lambda x: (
            0
            if x[1]["severity"] == "🔴 CRITIQUE"
            else 1 if x[1]["severity"] == "🟡 MOYEN" else 2
        ),
    )

    for filepath, f in sorted_files:
        md += f"\n### {filepath}\n"
        md += f"**Sévérité** : {f['severity']}\n\n"

        if f["imports"]:
            md += "**Imports métier** :\n"
            for imp in f["imports"]:
                module_info = imp.get("module", imp.get("name", "unknown"))
                desc = imp.get("desc", "Import métier")
                md += f"- L{imp['line']}: {imp['code']}\n"
                md += f"  → *{desc}*\n"
            md += "\n"

        if f["calculations"]:
            md += "**Calculs détectés** :\n"
            for calc in f["calculations"]:
                md += f"- L{calc['line']}: {calc['desc']}\n"
                md += f"  ```python\n  {calc['code']}\n  ```\n"
            md += "\n"

        # Actions recommandées
        md += "**Actions requises** :\n"
        if f["imports"]:
            md += "  1. [ ] Supprimer les imports métier\n"
            md += "  2. [ ] Utiliser des appels bridge au lieu des imports directs\n"
        if f["calculations"]:
            md += "  3. [ ] Extraire la logique de calcul vers le moteur\n"
            md += "  4. [ ] Remplacer par des appels asynchrones via bridge\n"
        md += f"  5. [ ] Créer les dataclasses de requête appropriées\n\n"

        # Complexité estimée
        complexity_score = len(f["imports"]) + len(f["calculations"]) * 2
        complexity = (
            "🟢 FAIBLE"
            if complexity_score < 3
            else "🟡 MOYENNE" if complexity_score < 6 else "🔴 HAUTE"
        )
        md += f"**Complexité** : {complexity}\n\n"

    md += f"\n---\n\n## 🔧 Extractions recommandées\n\n"

    extraction_count = 1
    for filepath, f in sorted_files:
        if f["calculations"]:
            for calc in f["calculations"]:
                md += f"### Extraction #{extraction_count} : {calc['desc']}\n\n"
                md += f"**Fichier source** : `{filepath}:{calc['line']}`\n\n"
                md += f"**Code problématique** :\n"
                md += f"```python\n{calc['code']}\n```\n\n"
                md += f"**Action** : Extraire vers moteur de calcul + appel bridge\n\n"
                md += f"**Priorité** : {'🔴 HAUTE' if f['severity'] == '🔴 CRITIQUE' else '🟡 MOYENNE'}\n\n"
                extraction_count += 1

    md += f"\n---\n\n## ✅ Checklist de validation\n\n"
    md += f"""Après refactorisation, vérifier :

- [ ] Aucun `import create_engine` dans les fichiers UI
- [ ] Aucun `import IndicatorBank` dans les fichiers UI
- [ ] Aucun `import PerformanceCalculator` dans les fichiers UI
- [ ] Tous les appels métier passent par self.bridge
- [ ] Tous les widgets UI restent dans les fichiers UI
- [ ] Tests unitaires UI passent (avec mocks bridge)
- [ ] Tests intégration Bridge ↔ Engine passent

---

## 🚀 Prochaines étapes

1. ✅ **Audit complet** (TERMINÉ - ce prompt)
2. ⏳ **Créer src/threadx/bridge/** (Prompt 2)
   - BacktestController
   - IndicatorController
   - DataController
   - ThreadXBridge (orchestrateur principal)
3. ⏳ **Refactoriser les fichiers UI** (Prompts 3-N)
   - Supprimer imports métier
   - Ajouter appels bridge
   - Créer dataclasses de requête
4. ⏳ **Tests et validation**
   - Tests unitaires bridge
   - Tests intégration UI-Bridge-Engine
   - Validation fonctionnelle complète

---

*Rapport généré automatiquement le {datetime.now().strftime('%Y-%m-%d à %H:%M:%S')}*
"""

    return md


if __name__ == "__main__":
    main()
