"""
ThreadX Data Manager - Modèles de données
Structures pour cataloguer et analyser les données d'indicateurs existantes
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path
from datetime import datetime
from enum import Enum


class DataQuality(Enum):
    """Niveaux de qualité des données"""

    PENDING = "pending"  # ⏳ En attente d'analyse
    EXCELLENT = "excellent"  # ✅ Aucun problème détecté
    GOOD = "good"  # ✅ Quelques warnings mineurs
    WARNING = "warning"  # ⚠️ Problèmes non-critiques
    POOR = "poor"  # ❌ Problèmes significatifs
    CORRUPTED = "corrupted"  # 💥 Données inutilisables


class ValidationStatus(Enum):
    """Statuts de validation"""

    PENDING = "pending"  # En attente de validation
    RUNNING = "running"  # Validation en cours
    COMPLETED = "completed"  # Validation terminée
    FAILED = "failed"  # Échec de validation


@dataclass
class IndicatorFile:
    """Informations sur un fichier d'indicateur"""

    path: Path
    symbol: str  # e.g., "ETHUSDC"
    timeframe: str  # e.g., "5m"
    indicator: str  # e.g., "atr", "bollinger"
    parameters: Dict[str, Any]  # e.g., {"period": 14}, {"period": 20, "sigma": 2.0}

    # Métadonnées fichier
    size_bytes: int = 0
    modified_time: Optional[datetime] = None

    # Informations contenu
    row_count: Optional[int] = None
    date_range: Optional[Tuple[datetime, datetime]] = None
    columns: Optional[List[str]] = None

    # Qualité données
    quality: DataQuality = DataQuality.PENDING
    validation_issues: List[str] = field(default_factory=list)

    @property
    def file_name(self) -> str:
        """Nom du fichier"""
        return self.path.name

    @property
    def indicator_key(self) -> str:
        """Clé unique pour cet indicateur+paramètres"""
        param_str = "_".join(f"{k}{v}" for k, v in sorted(self.parameters.items()))
        return f"{self.indicator}_{param_str}"


@dataclass
class SymbolData:
    """Données regroupées pour un symbole"""

    symbol: str  # e.g., "ETHUSDC"
    timeframes: Dict[str, "TimeframeData"]  # e.g., {"5m": TimeframeData(...)}
    total_files: int = 0
    total_size_bytes: int = 0
    overall_quality: DataQuality = DataQuality.PENDING

    @property
    def indicators_summary(self) -> Dict[str, int]:
        """Résumé des indicateurs par timeframe"""
        summary = {}
        for tf, tf_data in self.timeframes.items():
            summary[tf] = len(tf_data.indicators)
        return summary


@dataclass
class TimeframeData:
    """Données pour un timeframe spécifique"""

    timeframe: str  # e.g., "5m"
    indicators: Dict[str, List[IndicatorFile]]  # e.g., {"atr": [file1, file2]}
    file_count: int = 0
    size_bytes: int = 0
    quality: DataQuality = DataQuality.PENDING

    @property
    def unique_indicators(self) -> Set[str]:
        """Indicateurs uniques dans ce timeframe"""
        return set(self.indicators.keys())

    @property
    def parameter_variations(self) -> Dict[str, Set[str]]:
        """Variations de paramètres par indicateur"""
        variations = {}
        for indicator, files in self.indicators.items():
            param_sets = set()
            for file in files:
                param_str = "_".join(
                    f"{k}={v}" for k, v in sorted(file.parameters.items())
                )
                param_sets.add(param_str)
            variations[indicator] = param_sets
        return variations


@dataclass
class DataCatalog:
    """Catalogue complet des données découvertes"""

    root_paths: List[Path]  # Chemins scannés
    symbols: Dict[str, SymbolData]  # Données par symbole
    scan_time: datetime = field(default_factory=datetime.now)

    # Statistiques globales
    total_files: int = 0
    total_size_bytes: int = 0
    unique_symbols: Set[str] = field(default_factory=set)
    unique_timeframes: Set[str] = field(default_factory=set)
    unique_indicators: Set[str] = field(default_factory=set)

    @property
    def size_mb(self) -> float:
        """Taille totale en MB"""
        return self.total_size_bytes / (1024 * 1024)

    @property
    def quality_distribution(self) -> Dict[DataQuality, int]:
        """Distribution des niveaux de qualité"""
        distribution = {quality: 0 for quality in DataQuality}
        for symbol_data in self.symbols.values():
            for tf_data in symbol_data.timeframes.values():
                for files in tf_data.indicators.values():
                    for file in files:
                        distribution[file.quality] += 1
        return distribution


@dataclass
class ValidationIssue:
    """Problème détecté lors de la validation"""

    file_path: Path
    severity: str  # "error", "warning", "info"
    category: str  # "schema", "data", "performance"
    message: str
    details: Optional[str] = None

    @property
    def is_critical(self) -> bool:
        """Problème critique qui empêche l'utilisation"""
        return self.severity == "error"


@dataclass
class ValidationReport:
    """Rapport de validation des données"""

    catalog: DataCatalog
    validation_time: datetime = field(default_factory=datetime.now)
    status: ValidationStatus = ValidationStatus.PENDING

    # Issues par fichier
    issues: List[ValidationIssue] = field(default_factory=list)

    # Statistiques validation
    files_validated: int = 0
    files_passed: int = 0
    files_with_warnings: int = 0
    files_with_errors: int = 0

    @property
    def success_rate(self) -> float:
        """Taux de réussite (0.0 - 1.0)"""
        if self.files_validated == 0:
            return 0.0
        return self.files_passed / self.files_validated

    @property
    def critical_issues(self) -> List[ValidationIssue]:
        """Issues critiques uniquement"""
        return [issue for issue in self.issues if issue.is_critical]

    @property
    def summary(self) -> str:
        """Résumé textuel du rapport"""
        if self.status == ValidationStatus.PENDING:
            return "⏳ Validation en attente"
        elif self.status == ValidationStatus.RUNNING:
            return (
                f"🔄 Validation en cours... ({self.files_validated} fichiers traités)"
            )
        elif self.status == ValidationStatus.FAILED:
            return "❌ Échec de la validation"
        else:
            success_rate = self.success_rate * 100
            if success_rate >= 95:
                icon = "✅"
            elif success_rate >= 80:
                icon = "⚠️"
            else:
                icon = "❌"
            return f"{icon} {success_rate:.1f}% réussite ({self.files_passed}/{self.files_validated})"


@dataclass
class IntegrationAction:
    """Action d'intégration à effectuer"""

    action_type: str  # "import", "merge", "skip", "delete"
    source_files: List[Path]
    target_path: Optional[Path] = None
    reason: str = ""
    estimated_time_sec: float = 0.0

    @property
    def action_description(self) -> str:
        """Description de l'action"""
        if self.action_type == "import":
            return f"Importer {len(self.source_files)} fichier(s)"
        elif self.action_type == "merge":
            return f"Fusionner {len(self.source_files)} fichier(s)"
        elif self.action_type == "skip":
            return f"Ignorer {len(self.source_files)} fichier(s)"
        elif self.action_type == "delete":
            return f"Supprimer {len(self.source_files)} fichier(s)"
        else:
            return (
                f"Action '{self.action_type}' sur {len(self.source_files)} fichier(s)"
            )


@dataclass
class IntegrationPlan:
    """Plan d'intégration des données"""

    validation_report: ValidationReport
    actions: List[IntegrationAction] = field(default_factory=list)
    target_structure: Dict[str, Any] = field(default_factory=dict)

    # Estimations
    estimated_duration_sec: float = 0.0
    estimated_space_mb: float = 0.0

    @property
    def total_files_affected(self) -> int:
        """Nombre total de fichiers affectés"""
        return sum(len(action.source_files) for action in self.actions)

    @property
    def actions_by_type(self) -> Dict[str, List[IntegrationAction]]:
        """Actions regroupées par type"""
        by_type = {}
        for action in self.actions:
            if action.action_type not in by_type:
                by_type[action.action_type] = []
            by_type[action.action_type].append(action)
        return by_type

    @property
    def summary(self) -> str:
        """Résumé du plan"""
        if not self.actions:
            return "Aucune action requise"

        summary_parts = []
        for action_type, actions in self.actions_by_type.items():
            files_count = sum(len(a.source_files) for a in actions)
            summary_parts.append(f"{action_type}: {files_count} fichiers")

        duration_min = self.estimated_duration_sec / 60
        return f"{', '.join(summary_parts)} (~{duration_min:.1f}min)"


@dataclass
class IntegrationProgress:
    """Progression de l'intégration"""

    plan: IntegrationPlan
    current_action: Optional[IntegrationAction] = None
    completed_actions: List[IntegrationAction] = field(default_factory=list)
    failed_actions: List[Tuple[IntegrationAction, str]] = field(default_factory=list)

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def progress_percent(self) -> float:
        """Pourcentage de progression (0.0 - 1.0)"""
        if not self.plan.actions:
            return 1.0
        return len(self.completed_actions) / len(self.plan.actions)

    @property
    def is_running(self) -> bool:
        """Intégration en cours"""
        return self.start_time is not None and self.end_time is None

    @property
    def is_completed(self) -> bool:
        """Intégration terminée"""
        return self.end_time is not None

    @property
    def success_rate(self) -> float:
        """Taux de réussite des actions"""
        total_completed = len(self.completed_actions) + len(self.failed_actions)
        if total_completed == 0:
            return 0.0
        return len(self.completed_actions) / total_completed

    @property
    def status_summary(self) -> str:
        """Résumé du statut"""
        if not self.is_running and not self.is_completed:
            return "⏳ En attente"
        elif self.is_running:
            progress = self.progress_percent * 100
            return f"🔄 En cours ({progress:.1f}%)"
        else:
            success_rate = self.success_rate * 100
            if success_rate >= 95:
                icon = "✅"
            elif success_rate >= 80:
                icon = "⚠️"
            else:
                icon = "❌"
            return f"{icon} Terminé ({success_rate:.1f}% réussite)"
