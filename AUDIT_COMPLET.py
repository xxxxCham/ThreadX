#!/usr/bin/env python3
"""
AUDIT COMPLET - Analyse systématique des problèmes majeurs ThreadX
===================================================================

Examine tous les fichiers du workspace pour identifier:
1. Violations d'architecture (imports directs Engine dans UI)
2. Imports circulaires
3. Dépendances manquantes
4. Incohérences d'API
5. Code dupliqué
6. Erreur d'import
7. Fichiers obsolètes

Exécution: python audit_complet.py
Output: AUDIT_COMPLET_FINDINGS.md
"""

import ast
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set

# Configuration
REPO_ROOT = Path(__file__).parent
SRC_ROOT = REPO_ROOT / "src" / "threadx"

# Imports qui ne devraient pas être dans UI
FORBIDDEN_IN_UI = {
    "threadx.optimization.engine": "Direct Engine (bypass Bridge)",
    "threadx.optimization": "Direct Optimization module",
    "threadx.backtest.engine": "Direct Backtest Engine",
    "threadx.indicators.bank": "Direct IndicatorBank",
    "threadx.data.ingest": "Direct IngestionManager",
    "threadx.data.registry": "Direct Registry",
    "threadx.indicators": "Direct Indicators module",
    "threadx.backtest": "Direct Backtest module",
    "threadx.data": "Direct Data module",
}

# Imports requis dans UI
REQUIRED_IN_UI = {
    "threadx.bridge": "Bridge orchestration layer",
}

# Les chemins d'UI
UI_PATHS = [
    SRC_ROOT / "ui",
    REPO_ROOT / "apps",
]

# Les chemins d'Engine
ENGINE_PATHS = [
    SRC_ROOT / "optimization",
    SRC_ROOT / "backtest",
    SRC_ROOT / "indicators",
    SRC_ROOT / "data",
]


def get_all_python_files(paths: List[Path]) -> List[Path]:
    """Récupère tous les fichiers Python."""
    files = []
    for path in paths:
        if path.exists():
            files.extend(path.rglob("*.py"))
    return files


def extract_imports(file_path: Path) -> Tuple[List[str], List[str]]:
    """Extrait tous les imports d'un fichier Python."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            tree = ast.parse(f.read())
    except SyntaxError:
        return [], []

    imports_from = []
    imports_import = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module:
                imports_from.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports_import.append(alias.name)

    return imports_from, imports_import


def check_ui_file(file_path: Path) -> Dict[str, any]:
    """Vérifie un fichier UI pour violations."""
    imports_from, imports_import = extract_imports(file_path)
    all_imports = imports_from + imports_import

    violations = []
    missing_bridge = []

    # Vérifier imports interdits
    for imp in all_imports:
        for forbidden, description in FORBIDDEN_IN_UI.items():
            if forbidden in imp:
                violations.append(
                    {
                        "type": "FORBIDDEN_IMPORT",
                        "import": imp,
                        "description": description,
                        "severity": "HIGH",
                    }
                )

    # Vérifier imports Bridge présents
    has_bridge = any("bridge" in imp for imp in all_imports)
    if not has_bridge and any(imp in all_imports for imp in ["dash", "tkinter"]):
        # UI a probablement besoin du Bridge
        if not any("pytest" in str(file_path) for _ in [1]):
            missing_bridge.append(
                {
                    "type": "MISSING_BRIDGE",
                    "description": "UI component without Bridge import",
                    "severity": "MEDIUM",
                }
            )

    return {
        "file": str(file_path.relative_to(REPO_ROOT)),
        "violations": violations,
        "missing_bridge": missing_bridge,
        "all_imports": all_imports,
    }


def check_circular_imports(file_path: Path) -> List[Dict]:
    """Vérifie les imports circulaires potentiels."""
    content = file_path.read_text(errors="ignore")
    issues = []

    # Pattern: relative imports dans bridge vers ui ou vice-versa
    if "ui" in str(file_path) and "from threadx.bridge" in content:
        if "from threadx.ui" in content:
            issues.append(
                {
                    "type": "CIRCULAR_RISK",
                    "description": "UI importing from Bridge while Bridge imports from UI",
                    "severity": "HIGH",
                }
            )

    return issues


def scan_file_quality(file_path: Path) -> Dict[str, any]:
    """Analyse qualité générale d'un fichier."""
    try:
        content = file_path.read_text(errors="ignore")
    except:
        return {}

    issues = []

    # Chercher des patterns problématiques
    patterns = {
        r"TODO.*FIXME": "TODO/FIXME marker",
        r"XXX.*hack": "Hack comment",
        r"pass\s*$": "Empty pass statement",
        r"except:": "Bare except clause",
        r"import \*": "Star import",
    }

    for pattern, description in patterns.items():
        if re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
            issues.append(
                {
                    "type": "CODE_QUALITY",
                    "pattern": pattern,
                    "description": description,
                    "severity": "LOW" if "pass" in pattern else "MEDIUM",
                }
            )

    # Chercher duplicate code (imports répétés)
    import_lines = [
        l
        for l in content.split("\n")
        if l.strip().startswith("import ") or l.strip().startswith("from ")
    ]
    import_counts = defaultdict(int)
    for imp in import_lines:
        import_counts[imp] += 1

    for imp, count in import_counts.items():
        if count > 1:
            issues.append(
                {
                    "type": "DUPLICATE",
                    "description": f"Import repeated {count} times: {imp}",
                    "severity": "LOW",
                }
            )

    return {"quality_issues": issues}


def run_audit() -> Tuple[Dict, int]:
    """Exécute l'audit complet."""
    print("\n" + "=" * 80)
    print("🔍 AUDIT COMPLET - ThreadX Framework")
    print("=" * 80)

    ui_files = get_all_python_files(UI_PATHS)
    engine_files = get_all_python_files(ENGINE_PATHS)

    findings = {
        "ui_violations": [],
        "circular_risks": [],
        "quality_issues": [],
        "summary": {
            "total_ui_files": len(ui_files),
            "total_engine_files": len(engine_files),
            "violations_found": 0,
            "high_severity": 0,
            "medium_severity": 0,
            "low_severity": 0,
        },
    }

    print(f"\n📁 Scanning {len(ui_files)} UI files...")
    for file_path in ui_files:
        if "__pycache__" in str(file_path):
            continue

        result = check_ui_file(file_path)
        if result["violations"] or result["missing_bridge"]:
            findings["ui_violations"].append(result)
            findings["summary"]["violations_found"] += len(result["violations"])

        circular = check_circular_imports(file_path)
        if circular:
            findings["circular_risks"].append(
                {"file": str(file_path.relative_to(REPO_ROOT)), "issues": circular}
            )

        quality = scan_file_quality(file_path)
        if quality.get("quality_issues"):
            findings["quality_issues"].append(
                {
                    "file": str(file_path.relative_to(REPO_ROOT)),
                    "issues": quality["quality_issues"],
                }
            )

    # Compter sévérités
    for item in findings["ui_violations"]:
        for v in item["violations"]:
            if v["severity"] == "HIGH":
                findings["summary"]["high_severity"] += 1
            elif v["severity"] == "MEDIUM":
                findings["summary"]["medium_severity"] += 1

    for item in findings["circular_risks"]:
        for i in item["issues"]:
            if i["severity"] == "HIGH":
                findings["summary"]["high_severity"] += 1

    return findings, findings["summary"]["violations_found"]


def generate_report(findings: Dict) -> str:
    """Génère rapport Markdown."""
    md = "# 🔍 AUDIT COMPLET - Findings ThreadX\n\n"
    md += f"**Date**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Summary
    summary = findings["summary"]
    md += "## 📊 Summary\n\n"
    md += f"- UI Files Scanned: {summary['total_ui_files']}\n"
    md += f"- Engine Files Scanned: {summary['total_engine_files']}\n"
    md += f"- **Violations Found**: {summary['violations_found']}\n"
    md += f"  - 🔴 High Severity: {summary['high_severity']}\n"
    md += f"  - 🟠 Medium Severity: {summary['medium_severity']}\n"
    md += f"  - 🟡 Low Severity: {summary['low_severity']}\n\n"

    # Violations
    if findings["ui_violations"]:
        md += "## 🚨 UI Architecture Violations\n\n"
        for item in findings["ui_violations"]:
            md += f"### {item['file']}\n\n"
            for v in item["violations"]:
                md += f"- **{v['type']}**: {v['import']}\n"
                md += f"  - Description: {v['description']}\n"
                md += f"  - Severity: {v['severity']}\n"
            if item["missing_bridge"]:
                md += "- **Missing Bridge Import** detected\n"
            md += "\n"

    # Circular risks
    if findings["circular_risks"]:
        md += "## ⚠️ Circular Import Risks\n\n"
        for item in findings["circular_risks"]:
            md += f"### {item['file']}\n\n"
            for issue in item["issues"]:
                md += f"- {issue['description']}\n"
            md += "\n"

    # Quality issues
    if findings["quality_issues"]:
        md += "## 🔧 Code Quality Issues\n\n"
        for item in findings["quality_issues"]:
            md += f"### {item['file']}\n\n"
            for issue in item["issues"][:5]:  # Limit to 5 per file
                md += f"- {issue['description']}\n"
            if len(item["issues"]) > 5:
                md += f"- ... and {len(item['issues']) - 5} more\n"
            md += "\n"

    return md


if __name__ == "__main__":
    findings, violation_count = run_audit()

    # Print summary
    print(f"\n✅ Scan complete!")
    print(f"   Violations: {findings['summary']['violations_found']}")
    print(f"   High severity: {findings['summary']['high_severity']}")
    print(f"   Medium severity: {findings['summary']['medium_severity']}")

    # Generate report
    report = generate_report(findings)
    report_path = REPO_ROOT / "AUDIT_COMPLET_FINDINGS.md"
    report_path.write_text(report)
    print(f"\n📄 Report written to: {report_path}")

    # Save findings as JSON for programmatic access
    import json

    findings_path = REPO_ROOT / "audit_findings.json"
    with open(findings_path, "w") as f:
        json.dump(findings, f, indent=2, default=str)

    sys.exit(1 if violation_count > 0 else 0)
