"""
ThreadX Bridge - Pydantic Validation Layer
===========================================

Modèles de validation pour les requêtes vers l'Engine.

Author: ThreadX Framework
Version: Prompt 2 - Bridge Foundation
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class BacktestRequest(BaseModel):
    """Requête de backtest validée."""

    symbol: str = Field(..., min_length=1, description="Symbole de trading")
    timeframe: str = Field(
        ...,
        pattern=r"^(\d+m|1h|2h|4h|6h|8h|12h|1d|1w|1M)$",
        description="Timeframe: 1m,5m,15m,30m,45m,1h,2h,4h,6h,8h,12h,1d,1w,1M",
    )
    strategy: str = Field(..., min_length=1, description="Nom de la stratégie")
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Paramètres de stratégie"
    )
    start_date: Optional[str] = Field(None, description="Date de début")
    end_date: Optional[str] = Field(None, description="Date de fin")

    @field_validator("params")
    @classmethod
    def validate_params(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Valider les paramètres de stratégie."""
        if not isinstance(v, dict):
            raise ValueError("params doit être un dictionnaire")
        return v


class IndicatorRequest(BaseModel):
    """Requête de calcul d'indicateur validée."""

    symbol: str = Field(..., min_length=1, description="Symbole de trading")
    timeframe: str = Field(
        ...,
        pattern=r"^(\d+m|1h|2h|4h|6h|8h|12h|1d|1w|1M)$",
        description="Timeframe: 1m,5m,15m,30m,45m,1h,2h,4h,6h,8h,12h,1d,1w,1M",
    )
    indicator: str = Field(..., min_length=1, description="Nom de l'indicateur")
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Paramètres d'indicateur"
    )

    @field_validator("params")
    @classmethod
    def validate_params(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Valider les paramètres d'indicateur."""
        if not isinstance(v, dict):
            raise ValueError("params doit être un dictionnaire")
        return v


class OptimizeRequest(BaseModel):
    """Requête d'optimisation de paramètres validée."""

    symbol: str = Field(..., min_length=1, description="Symbole de trading")
    timeframe: str = Field(
        ...,
        pattern=r"^(\d+m|1h|2h|4h|6h|8h|12h|1d|1w|1M)$",
        description="Timeframe: 1m,5m,15m,30m,45m,1h,2h,4h,6h,8h,12h,1d,1w,1M",
    )
    strategy: str = Field(..., min_length=1, description="Nom de la stratégie")
    param_ranges: Dict[str, List[Any]] = Field(
        ..., description="Plages de paramètres à tester"
    )
    objective: str = Field(
        default="sharpe_ratio", description="Métrique d'optimisation"
    )

    @field_validator("param_ranges")
    @classmethod
    def validate_param_ranges(cls, v: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Valider les plages de paramètres."""
        if not isinstance(v, dict):
            raise ValueError("param_ranges doit être un dictionnaire")
        for key, value in v.items():
            if not isinstance(value, list):
                raise ValueError(f"param_ranges[{key}] doit être une liste")
        return v


class DataValidationRequest(BaseModel):
    """Requête de validation de données validée."""

    symbol: str = Field(..., min_length=1, description="Symbole de trading")
    timeframe: str = Field(
        ...,
        pattern=r"^(\d+m|1h|2h|4h|6h|8h|12h|1d|1w|1M)$",
        description="Timeframe: 1m,5m,15m,30m,45m,1h,2h,4h,6h,8h,12h,1d,1w,1M",
    )
    start_date: Optional[str] = Field(None, description="Date de début")
    end_date: Optional[str] = Field(None, description="Date de fin")
    checks: List[str] = Field(
        default_factory=lambda: ["completeness", "duplicates", "outliers"],
        description="Types de vérifications",
    )

    @field_validator("checks")
    @classmethod
    def validate_checks(cls, v: List[str]) -> List[str]:
        """Valider les types de vérifications."""
        if not isinstance(v, list):
            raise ValueError("checks doit être une liste")
        valid_checks = {"completeness", "duplicates", "outliers", "gaps"}
        for check in v:
            if check not in valid_checks:
                raise ValueError(f"Type de vérification invalide: {check}")
        return v
