#!/usr/bin/env python3
"""
Audit complet ThreadX - Phase 1: Préparation et Auditing
=========================================================

Ce script implémente un audit exhaustif du projet ThreadX selon les bonnes pratiques
pour les applications de trading quantitatif.

Catégories d'audit:
1. Logic Errors (erreurs de logique trading)
2. Code Duplication (duplication de code)
3. Structural Issues (problèmes structurels)
4. Security Issues (vulnérabilités)
5. Performance Issues (problèmes de performance)

Auteur: ThreadX Audit System
Date: Octobre 2025
"""

import ast
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Configuration
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src" / "threadx"
TESTS_DIR = PROJECT_ROOT / "tests"
REPORT_FILE = PROJECT_ROOT / "AUDIT_THREADX_FINDINGS.json"
MARKDOWN_REPORT = PROJECT_ROOT / "AUDIT_THREADX_REPORT.md"

# Seuils de qualité
DUPLICATION_THRESHOLD = 10  # % max de duplication acceptable
COMPLEXITY_THRESHOLD = 10  # Complexité cyclomatique max par fonction
MIN_DOCSTRING_COVERAGE = 70  # % minimum de fonctions documentées


@dataclass
class Finding:
    """Représente une découverte d'audit"""

    category: str  # 'logic', 'duplication', 'structural', 'security', 'performance'
    severity: str  # 'critical', 'high', 'medium', 'low'
    file_path: str
    line_number: int
    description: str
    recommendation: str
    code_snippet: str = ""


@dataclass
class AuditStats:
    """Statistiques globales de l'audit"""

    total_files: int = 0
    total_lines: int = 0
    total_functions: int = 0
    total_classes: int = 0
    duplicated_lines: int = 0
    duplication_percentage: float = 0.0
    findings_by_severity: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    findings_by_category: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )


class ThreadXAuditor:
    """Auditeur principal pour ThreadX"""

    def __init__(self):
        self.findings: List[Finding] = []
        self.stats = AuditStats()
        self.code_blocks: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.imports_by_file: Dict[str, Set[str]] = {}

    def run_full_audit(self) -> None:
        """Exécute l'audit complet"""
        print("🔍 Démarrage de l'audit complet ThreadX...\n")

        # Phase 1: Collecte des fichiers
        python_files = self._collect_python_files()
        print(f"📁 {len(python_files)} fichiers Python trouvés\n")

        # Phase 2: Analyse de chaque fichier
        for file_path in python_files:
            self._analyze_file(file_path)

        # Phase 3: Audits spécifiques
        print("🔬 Audit des erreurs logiques de trading...")
        self._audit_trading_logic()

        print("🔍 Détection de la duplication de code...")
        self._detect_code_duplication()

        print("🏗️ Analyse structurelle...")
        self._audit_structure()

        print("🛡️ Audit de sécurité...")
        self._audit_security()

        print("⚡ Audit de performance GPU...")
        self._audit_gpu_performance()

        # Phase 4: Génération des rapports
        print("\n📊 Génération des rapports...")
        self._generate_reports()

        print(f"\n✅ Audit terminé! {len(self.findings)} problèmes détectés.")
        print(f"📄 Rapport JSON: {REPORT_FILE}")
        print(f"📄 Rapport Markdown: {MARKDOWN_REPORT}")

    def _collect_python_files(self) -> List[Path]:
        """Collecte tous les fichiers Python du projet"""
        files = []
        for directory in [SRC_DIR, TESTS_DIR]:
            if directory.exists():
                files.extend(directory.rglob("*.py"))

        # Exclure les fichiers générés et caches
        exclude_patterns = ["__pycache__", ".egg-info", "build", "dist"]
        return [f for f in files if not any(p in str(f) for p in exclude_patterns)]

    def _analyze_file(self, file_path: Path) -> None:
        """Analyse un fichier Python"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")

            self.stats.total_files += 1
            self.stats.total_lines += len(lines)

            # Parse AST
            try:
                tree = ast.parse(content, filename=str(file_path))
                self._analyze_ast(tree, file_path, content)
            except SyntaxError as e:
                self._add_finding(
                    category="structural",
                    severity="critical",
                    file_path=str(file_path),
                    line_number=e.lineno or 0,
                    description=f"Erreur de syntaxe: {e.msg}",
                    recommendation="Corriger l'erreur de syntaxe immédiatement",
                )

            # Analyse des imports
            self._analyze_imports(file_path, content)

            # Stockage des blocs de code pour détection de duplication
            self._store_code_blocks(file_path, lines)

        except Exception as e:
            print(f"⚠️ Erreur lors de l'analyse de {file_path}: {e}")

    def _analyze_ast(self, tree: ast.AST, file_path: Path, content: str) -> None:
        """Analyse l'AST pour détecter les problèmes"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self.stats.total_functions += 1
                self._check_function_complexity(node, file_path)
                self._check_function_docstring(node, file_path)

            elif isinstance(node, ast.ClassDef):
                self.stats.total_classes += 1
                self._check_class_structure(node, file_path)

    def _check_function_complexity(
        self, node: ast.FunctionDef, file_path: Path
    ) -> None:
        """Vérifie la complexité cyclomatique d'une fonction"""
        complexity = self._calculate_complexity(node)

        if complexity > COMPLEXITY_THRESHOLD:
            self._add_finding(
                category="structural",
                severity="medium",
                file_path=str(file_path),
                line_number=node.lineno,
                description=f"Fonction '{node.name}' trop complexe (complexité: {complexity})",
                recommendation=f"Refactoriser en fonctions plus petites. Cible: <{COMPLEXITY_THRESHOLD}",
            )

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calcule la complexité cyclomatique (McCabe)"""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity

    def _check_function_docstring(self, node: ast.FunctionDef, file_path: Path) -> None:
        """Vérifie la présence de docstrings"""
        if not ast.get_docstring(node) and not node.name.startswith("_"):
            self._add_finding(
                category="structural",
                severity="low",
                file_path=str(file_path),
                line_number=node.lineno,
                description=f"Fonction publique '{node.name}' sans docstring",
                recommendation="Ajouter une docstring décrivant le comportement",
            )

    def _check_class_structure(self, node: ast.ClassDef, file_path: Path) -> None:
        """Vérifie la structure des classes"""
        methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]

        # Classes trop grandes
        if len(methods) > 20:
            self._add_finding(
                category="structural",
                severity="medium",
                file_path=str(file_path),
                line_number=node.lineno,
                description=f"Classe '{node.name}' trop grande ({len(methods)} méthodes)",
                recommendation="Considérer de diviser en classes plus petites",
            )

    def _analyze_imports(self, file_path: Path, content: str) -> None:
        """Analyse les imports pour détecter les duplications"""
        imports = set()
        for line in content.split("\n"):
            if line.strip().startswith(("import ", "from ")):
                imports.add(line.strip())

        self.imports_by_file[str(file_path)] = imports

    def _store_code_blocks(self, file_path: Path, lines: List[str]) -> None:
        """Stocke les blocs de code pour détection de duplication"""
        # Blocs de 5+ lignes non vides
        for i in range(len(lines) - 4):
            block_lines = [
                l.strip()
                for l in lines[i : i + 5]
                if l.strip() and not l.strip().startswith("#")
            ]
            if len(block_lines) >= 5:
                block_hash = hash(tuple(block_lines))
                self.code_blocks[str(block_hash)].append((str(file_path), i + 1))

    def _audit_trading_logic(self) -> None:
        """Audit des erreurs logiques spécifiques au trading"""

        # 1. Vérifier les backtests pour look-ahead bias
        backtest_files = list(SRC_DIR.glob("backtest/**/*.py"))
        for file_path in backtest_files:
            self._check_lookahead_bias(file_path)
            self._check_overfitting_indicators(file_path)

        # 2. Vérifier les stratégies pour risk management
        strategy_files = list(SRC_DIR.glob("strategy/**/*.py"))
        for file_path in strategy_files:
            self._check_risk_controls(file_path)

        # 3. Vérifier data quality checks
        data_files = list(SRC_DIR.glob("data/**/*.py"))
        for file_path in data_files:
            self._check_data_validation(file_path)

    def _check_lookahead_bias(self, file_path: Path) -> None:
        """Détecte le look-ahead bias dans les backtests"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Patterns suspects
            suspicious_patterns = [
                (
                    r"\.iloc\[\s*-?\d+:\s*\]",
                    "Utilisation de iloc sans garantie temporelle",
                ),
                (
                    r"\.sort_values\([^)]*ascending\s*=\s*False",
                    "Tri descendant peut causer look-ahead",
                ),
                (
                    r"\.shift\(\s*-\d+\s*\)",
                    "shift() négatif accède aux données futures",
                ),
            ]

            for pattern, message in suspicious_patterns:
                for match in re.finditer(pattern, content):
                    line_num = content[: match.start()].count("\n") + 1
                    self._add_finding(
                        category="logic",
                        severity="critical",
                        file_path=str(file_path),
                        line_number=line_num,
                        description=f"Risque de look-ahead bias: {message}",
                        recommendation="Vérifier que seules les données passées sont utilisées",
                        code_snippet=content[match.start() : match.end()],
                    )
        except Exception as e:
            print(f"⚠️ Erreur check_lookahead_bias pour {file_path}: {e}")

    def _check_overfitting_indicators(self, file_path: Path) -> None:
        """Détecte les signes d'overfitting"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Trop de paramètres optimisés
            param_pattern = (
                r"def\s+\w+\([^)]*,\s*[^)]*,\s*[^)]*,\s*[^)]*,\s*[^)]*,\s*[^)]*,"
            )
            if re.search(param_pattern, content):
                self._add_finding(
                    category="logic",
                    severity="high",
                    file_path=str(file_path),
                    line_number=1,
                    description="Fonction avec trop de paramètres - risque d'overfitting",
                    recommendation="Réduire le nombre de paramètres. Utiliser walk-forward ou cross-validation",
                )

            # Absence de validation out-of-sample
            if (
                "backtest" in content.lower()
                and "train_test_split" not in content
                and "walk_forward" not in content
            ):
                self._add_finding(
                    category="logic",
                    severity="high",
                    file_path=str(file_path),
                    line_number=1,
                    description="Backtest sans validation out-of-sample apparente",
                    recommendation="Implémenter train/test split ou walk-forward validation",
                )
        except Exception as e:
            print(f"⚠️ Erreur check_overfitting pour {file_path}: {e}")

    def _check_risk_controls(self, file_path: Path) -> None:
        """Vérifie les contrôles de risque dans les stratégies"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Vérifier présence de stops
            if (
                "position" in content.lower()
                and "stop" not in content.lower()
                and "risk" not in content.lower()
            ):
                self._add_finding(
                    category="logic",
                    severity="high",
                    file_path=str(file_path),
                    line_number=1,
                    description="Stratégie sans contrôle de risque apparent (stop-loss, position sizing)",
                    recommendation="Ajouter stop-loss et position sizing basé sur volatilité",
                )

            # Position sizing hardcodé
            if re.search(r"position_size\s*=\s*\d+\.?\d*", content):
                match = re.search(r"position_size\s*=\s*\d+\.?\d*", content)
                line_num = content[: match.start()].count("\n") + 1
                self._add_finding(
                    category="logic",
                    severity="medium",
                    file_path=str(file_path),
                    line_number=line_num,
                    description="Position size hardcodée - devrait être basée sur volatilité",
                    recommendation="Calculer position size avec ATR ou volatilité",
                )
        except Exception as e:
            print(f"⚠️ Erreur check_risk_controls pour {file_path}: {e}")

    def _check_data_validation(self, file_path: Path) -> None:
        """Vérifie la validation des données"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Manque de validation NaN
            if "pd.read" in content or "DataFrame" in content:
                if (
                    ".isnull()" not in content
                    and ".isna()" not in content
                    and ".dropna()" not in content
                ):
                    self._add_finding(
                        category="logic",
                        severity="medium",
                        file_path=str(file_path),
                        line_number=1,
                        description="Chargement de données sans vérification de valeurs manquantes",
                        recommendation="Ajouter: assert not df.isnull().any().any() ou df.fillna()",
                    )

                # Manque de vérification de duplicates
                if ".duplicated()" not in content:
                    self._add_finding(
                        category="logic",
                        severity="low",
                        file_path=str(file_path),
                        line_number=1,
                        description="Pas de vérification de timestamps dupliqués",
                        recommendation="Ajouter: assert not df.duplicated().any()",
                    )
        except Exception as e:
            print(f"⚠️ Erreur check_data_validation pour {file_path}: {e}")

    def _detect_code_duplication(self) -> None:
        """Détecte la duplication de code"""

        # 1. Blocs de code dupliqués
        for block_hash, locations in self.code_blocks.items():
            if len(locations) > 1:
                self.stats.duplicated_lines += len(locations) * 5

                for file_path, line_num in locations:
                    self._add_finding(
                        category="duplication",
                        severity="medium",
                        file_path=file_path,
                        line_number=line_num,
                        description=f"Bloc de code dupliqué ({len(locations)} occurrences)",
                        recommendation="Extraire dans une fonction réutilisable",
                    )

        # 2. Imports dupliqués
        import_groups = defaultdict(list)
        for file_path, imports in self.imports_by_file.items():
            import_key = frozenset(imports)
            import_groups[import_key].append(file_path)

        for imports, files in import_groups.items():
            if len(files) > 3 and len(imports) > 5:
                for file_path in files:
                    self._add_finding(
                        category="duplication",
                        severity="low",
                        file_path=file_path,
                        line_number=1,
                        description=f"Groupe d'imports dupliqué dans {len(files)} fichiers",
                        recommendation="Créer un module utils/common_imports.py",
                    )

        # Calcul du pourcentage de duplication
        if self.stats.total_lines > 0:
            self.stats.duplication_percentage = (
                self.stats.duplicated_lines / self.stats.total_lines
            ) * 100

    def _audit_structure(self) -> None:
        """Audit de la structure architecturale"""

        # 1. Vérifier séparation UI/Bridge/Engine
        self._check_layer_separation()

        # 2. Vérifier dépendances circulaires
        self._check_circular_dependencies()

        # 3. Vérifier cohésion des modules
        self._check_module_cohesion()

    def _check_layer_separation(self) -> None:
        """Vérifie la séparation des couches UI/Bridge/Engine"""
        ui_files = list(SRC_DIR.glob("ui/**/*.py"))

        for ui_file in ui_files:
            try:
                with open(ui_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # UI ne devrait pas importer directement Engine
                engine_imports = [
                    "from threadx.backtest.engine import",
                    "from threadx.indicators.engine import",
                    "from threadx.optimization.engine import",
                ]

                for pattern in engine_imports:
                    if pattern in content:
                        self._add_finding(
                            category="structural",
                            severity="high",
                            file_path=str(ui_file),
                            line_number=1,
                            description="UI importe directement Engine - violation de l'architecture 3-tier",
                            recommendation="Utiliser Bridge comme interface: from threadx.bridge import ...",
                        )
            except Exception as e:
                print(f"⚠️ Erreur check_layer_separation pour {ui_file}: {e}")

    def _check_circular_dependencies(self) -> None:
        """Détecte les dépendances circulaires"""
        # Implémentation simplifiée - peut être étendue avec networkx
        pass

    def _check_module_cohesion(self) -> None:
        """Vérifie la cohésion des modules"""
        # Vérifier que les fichiers __init__.py exportent correctement
        init_files = list(SRC_DIR.rglob("__init__.py"))

        for init_file in init_files:
            try:
                with open(init_file, "r", encoding="utf-8") as f:
                    content = f.read()

                if len(content.strip()) == 0:
                    self._add_finding(
                        category="structural",
                        severity="low",
                        file_path=str(init_file),
                        line_number=1,
                        description="Fichier __init__.py vide - exports non définis",
                        recommendation="Définir __all__ pour exports explicites",
                    )
            except Exception as e:
                print(f"⚠️ Erreur check_module_cohesion pour {init_file}: {e}")

    def _audit_security(self) -> None:
        """Audit de sécurité basique"""
        python_files = self._collect_python_files()

        for file_path in python_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Patterns de sécurité
                security_patterns = [
                    (
                        r"eval\s*\(",
                        "critical",
                        "Utilisation de eval() - injection de code possible",
                    ),
                    (
                        r"exec\s*\(",
                        "critical",
                        "Utilisation de exec() - injection de code possible",
                    ),
                    (
                        r"pickle\.loads?\(",
                        "high",
                        "Pickle non sécurisé - risque de désérialisation",
                    ),
                    (
                        r"os\.system\(",
                        "high",
                        "os.system() - préférer subprocess avec shell=False",
                    ),
                    (
                        r"shell\s*=\s*True",
                        "medium",
                        "subprocess avec shell=True - risque d'injection",
                    ),
                ]

                for pattern, severity, message in security_patterns:
                    for match in re.finditer(pattern, content):
                        line_num = content[: match.start()].count("\n") + 1
                        self._add_finding(
                            category="security",
                            severity=severity,
                            file_path=str(file_path),
                            line_number=line_num,
                            description=message,
                            recommendation="Utiliser des alternatives sécurisées",
                        )
            except Exception as e:
                print(f"⚠️ Erreur audit_security pour {file_path}: {e}")

    def _audit_gpu_performance(self) -> None:
        """Audit des performances GPU"""
        gpu_files = list(SRC_DIR.glob("**/gpu*.py")) + list(
            SRC_DIR.glob("indicators/**/*.py")
        )

        for file_path in gpu_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Vérifier fallback CPU
                if "cupy" in content or "cp." in content:
                    if "try:" not in content or "except ImportError" not in content:
                        self._add_finding(
                            category="performance",
                            severity="medium",
                            file_path=str(file_path),
                            line_number=1,
                            description="Code GPU sans fallback CPU",
                            recommendation="Ajouter try/except ImportError avec fallback numpy",
                        )

                    # Vérifier shape checks
                    if "vector_checks" not in content and ".shape" not in content:
                        self._add_finding(
                            category="performance",
                            severity="low",
                            file_path=str(file_path),
                            line_number=1,
                            description="Opérations GPU sans vérification de shape",
                            recommendation="Utiliser utils/gpu/vector_checks.py",
                        )
            except Exception as e:
                print(f"⚠️ Erreur audit_gpu_performance pour {file_path}: {e}")

    def _add_finding(
        self,
        category: str,
        severity: str,
        file_path: str,
        line_number: int,
        description: str,
        recommendation: str,
        code_snippet: str = "",
    ) -> None:
        """Ajoute une découverte d'audit"""
        finding = Finding(
            category=category,
            severity=severity,
            file_path=file_path,
            line_number=line_number,
            description=description,
            recommendation=recommendation,
            code_snippet=code_snippet,
        )
        self.findings.append(finding)
        self.stats.findings_by_severity[severity] += 1
        self.stats.findings_by_category[category] += 1

    def _generate_reports(self) -> None:
        """Génère les rapports d'audit"""

        # Rapport JSON
        report_data = {
            "stats": {
                "total_files": self.stats.total_files,
                "total_lines": self.stats.total_lines,
                "total_functions": self.stats.total_functions,
                "total_classes": self.stats.total_classes,
                "duplication_percentage": round(self.stats.duplication_percentage, 2),
                "findings_by_severity": dict(self.stats.findings_by_severity),
                "findings_by_category": dict(self.stats.findings_by_category),
            },
            "findings": [
                {
                    "category": f.category,
                    "severity": f.severity,
                    "file": f.file_path,
                    "line": f.line_number,
                    "description": f.description,
                    "recommendation": f.recommendation,
                    "code_snippet": f.code_snippet,
                }
                for f in sorted(
                    self.findings,
                    key=lambda x: (
                        {"critical": 0, "high": 1, "medium": 2, "low": 3}[x.severity],
                        x.file_path,
                    ),
                )
            ],
        }

        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        # Rapport Markdown
        self._generate_markdown_report()

    def _generate_markdown_report(self) -> None:
        """Génère le rapport Markdown détaillé"""

        with open(MARKDOWN_REPORT, "w", encoding="utf-8") as f:
            f.write("# 🔍 Rapport d'Audit ThreadX - Analyse Complète\n\n")
            f.write(f"**Date:** {self._get_timestamp()}\n\n")
            f.write("---\n\n")

            # Executive Summary
            f.write("## 📊 Résumé Exécutif\n\n")
            f.write(f"- **Fichiers analysés:** {self.stats.total_files}\n")
            f.write(f"- **Lignes de code:** {self.stats.total_lines:,}\n")
            f.write(f"- **Fonctions:** {self.stats.total_functions}\n")
            f.write(f"- **Classes:** {self.stats.total_classes}\n")
            f.write(f"- **Duplication:** {self.stats.duplication_percentage:.1f}%\n")
            f.write(f"- **Problèmes détectés:** {len(self.findings)}\n\n")

            # Score de qualité
            quality_score = self._calculate_quality_score()
            f.write(f"### 🎯 Score de Qualité Global: **{quality_score}/10**\n\n")

            if quality_score >= 8:
                f.write("✅ **Excellente qualité** - Continuez ainsi!\n\n")
            elif quality_score >= 6:
                f.write("⚠️ **Qualité acceptable** - Améliorations recommandées\n\n")
            else:
                f.write("🚨 **Qualité préoccupante** - Action immédiate requise\n\n")

            # Répartition par sévérité
            f.write("## 🚨 Répartition par Sévérité\n\n")
            f.write("| Sévérité | Nombre | Pourcentage |\n")
            f.write("|----------|--------|-------------|\n")

            total = len(self.findings)
            for severity in ["critical", "high", "medium", "low"]:
                count = self.stats.findings_by_severity.get(severity, 0)
                pct = (count / total * 100) if total > 0 else 0
                icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}[
                    severity
                ]
                f.write(f"| {icon} {severity.capitalize()} | {count} | {pct:.1f}% |\n")
            f.write("\n")

            # Répartition par catégorie
            f.write("## 📁 Répartition par Catégorie\n\n")
            f.write("| Catégorie | Nombre | Description |\n")
            f.write("|-----------|--------|-------------|\n")

            categories = {
                "logic": "Erreurs logiques de trading",
                "duplication": "Duplication de code",
                "structural": "Problèmes structurels",
                "security": "Vulnérabilités de sécurité",
                "performance": "Problèmes de performance",
            }

            for cat, desc in categories.items():
                count = self.stats.findings_by_category.get(cat, 0)
                f.write(f"| {cat.capitalize()} | {count} | {desc} |\n")
            f.write("\n")

            # Détails des findings par sévérité
            for severity in ["critical", "high", "medium", "low"]:
                findings = [f for f in self.findings if f.severity == severity]
                if findings:
                    icon = {
                        "critical": "🔴",
                        "high": "🟠",
                        "medium": "🟡",
                        "low": "🟢",
                    }[severity]
                    f.write(f"## {icon} Problèmes de Sévérité {severity.upper()}\n\n")

                    for idx, finding in enumerate(findings, 1):
                        f.write(f"### {idx}. {finding.description}\n\n")
                        f.write(
                            f"**Fichier:** `{finding.file_path}:{finding.line_number}`\n\n"
                        )
                        f.write(f"**Catégorie:** {finding.category}\n\n")
                        f.write(f"**Recommandation:** {finding.recommendation}\n\n")

                        if finding.code_snippet:
                            f.write("**Code concerné:**\n```python\n")
                            f.write(finding.code_snippet)
                            f.write("\n```\n\n")

                        f.write("---\n\n")

            # Plan d'action
            f.write("## 🎯 Plan d'Action Recommandé\n\n")

            critical = self.stats.findings_by_severity.get("critical", 0)
            high = self.stats.findings_by_severity.get("high", 0)

            if critical > 0:
                f.write(f"### 🔴 URGENT: {critical} problèmes critiques\n")
                f.write(
                    "**Action immédiate requise** - Ces problèmes peuvent causer des pertes financières\n\n"
                )

            if high > 0:
                f.write(f"### 🟠 PRIORITAIRE: {high} problèmes haute priorité\n")
                f.write(
                    "**À traiter dans les 48h** - Impact significatif sur la fiabilité\n\n"
                )

            if self.stats.duplication_percentage > DUPLICATION_THRESHOLD:
                f.write(
                    f"### ♻️ REFACTORING: {self.stats.duplication_percentage:.1f}% de duplication\n"
                )
                f.write(f"**Objectif:** Réduire à <{DUPLICATION_THRESHOLD}%\n\n")

            # Recommandations générales
            f.write("## 💡 Recommandations Générales\n\n")
            f.write("1. **Installer les outils d'analyse:**\n")
            f.write("   ```bash\n")
            f.write("   pip install pylint flake8 mypy bandit black\n")
            f.write("   ```\n\n")

            f.write(
                "2. **Configurer pre-commit hooks** pour prévenir les régressions\n\n"
            )

            f.write(
                "3. **Augmenter la couverture de tests** pour validation continue\n\n"
            )

            f.write("4. **Implémenter CI/CD** avec vérifications automatiques\n\n")

            # Footer
            f.write("---\n\n")
            f.write("*Généré par ThreadX Audit System*\n")

    def _calculate_quality_score(self) -> float:
        """Calcule le score de qualité global (0-10)"""
        score = 10.0

        # Pénalités par sévérité
        score -= self.stats.findings_by_severity.get("critical", 0) * 1.0
        score -= self.stats.findings_by_severity.get("high", 0) * 0.5
        score -= self.stats.findings_by_severity.get("medium", 0) * 0.2
        score -= self.stats.findings_by_severity.get("low", 0) * 0.05

        # Pénalité duplication
        if self.stats.duplication_percentage > DUPLICATION_THRESHOLD:
            score -= (self.stats.duplication_percentage - DUPLICATION_THRESHOLD) * 0.1

        return max(0.0, min(10.0, score))

    @staticmethod
    def _get_timestamp() -> str:
        """Retourne timestamp formaté"""
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    """Point d'entrée principal"""
    print("=" * 70)
    print("  THREADX AUDIT SYSTEM - Analyse Complète de Qualité")
    print("=" * 70)
    print()

    auditor = ThreadXAuditor()

    try:
        auditor.run_full_audit()

        # Afficher résumé
        print("\n" + "=" * 70)
        print("  RÉSUMÉ DE L'AUDIT")
        print("=" * 70)
        print(f"\n📊 Statistiques:")
        print(f"   - Fichiers: {auditor.stats.total_files}")
        print(f"   - Lignes: {auditor.stats.total_lines:,}")
        print(f"   - Duplication: {auditor.stats.duplication_percentage:.1f}%")

        print(f"\n🚨 Problèmes par sévérité:")
        for severity in ["critical", "high", "medium", "low"]:
            count = auditor.stats.findings_by_severity.get(severity, 0)
            icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}[
                severity
            ]
            print(f"   {icon} {severity.capitalize()}: {count}")

        quality_score = auditor._calculate_quality_score()
        print(f"\n🎯 Score de Qualité: {quality_score:.1f}/10")

        print("\n" + "=" * 70)

        # Code de sortie basé sur sévérité
        if auditor.stats.findings_by_severity.get("critical", 0) > 0:
            sys.exit(2)
        elif auditor.stats.findings_by_severity.get("high", 0) > 0:
            sys.exit(1)
        else:
            sys.exit(0)

    except Exception as e:
        print(f"\n❌ Erreur fatale: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()
