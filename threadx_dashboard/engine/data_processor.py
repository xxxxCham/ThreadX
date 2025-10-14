"""
Processeur de Données ThreadX
=============================

Moteur de traitement et validation des données de marché.
Cette classe contient uniquement la logique de traitement,
sans aucune dépendance vers l'interface utilisateur.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class DataQuality(Enum):
    """Niveaux de qualité des données"""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    INVALID = "invalid"


@dataclass
class DataValidationResult:
    """Résultat de la validation des données"""

    is_valid: bool
    quality: DataQuality
    missing_data_count: int
    duplicates_count: int
    outliers_count: int
    date_gaps_count: int
    issues: List[str]
    recommendations: List[str]


@dataclass
class ProcessedData:
    """Données traitées et nettoyées"""

    raw_data: pd.DataFrame
    cleaned_data: pd.DataFrame
    validation_result: DataValidationResult
    processing_metadata: Dict[str, Any]


class DataProcessor:
    """
    Processeur de données pur - logique métier uniquement

    Cette classe traite et valide les données de marché,
    sans aucune dépendance vers l'interface utilisateur.
    """

    def __init__(self):
        """Initialise le processeur de données"""
        self.required_columns = ["date", "open", "high", "low", "close", "volume"]
        self.optional_columns = ["symbol", "adj_close"]

    def process_market_data(
        self,
        raw_data: pd.DataFrame,
        symbol: Optional[str] = None,
        auto_clean: bool = True,
    ) -> ProcessedData:
        """
        Traite et valide des données de marché

        Args:
            raw_data: DataFrame brut avec données OHLCV
            symbol: Symbole de l'actif (optionnel)
            auto_clean: Nettoyer automatiquement les données

        Returns:
            ProcessedData: Données traitées et validées
        """
        # Copier les données pour ne pas modifier l'original
        data = raw_data.copy()

        # Ajouter le symbole si fourni
        if symbol and "symbol" not in data.columns:
            data["symbol"] = symbol

        # Validation initiale
        validation_result = self.validate_data_quality(data)

        # Nettoyage automatique si demandé
        if auto_clean and validation_result.is_valid:
            cleaned_data = self.clean_data(data)
        else:
            cleaned_data = data

        # Métadonnées de traitement
        processing_metadata = {
            "processed_at": datetime.now().isoformat(),
            "original_rows": len(raw_data),
            "cleaned_rows": len(cleaned_data),
            "auto_clean_applied": auto_clean,
            "symbol": symbol,
        }

        return ProcessedData(
            raw_data=raw_data,
            cleaned_data=cleaned_data,
            validation_result=validation_result,
            processing_metadata=processing_metadata,
        )

    def validate_data_quality(self, data: pd.DataFrame) -> DataValidationResult:
        """
        Valide la qualité des données de marché

        Args:
            data: DataFrame à valider

        Returns:
            DataValidationResult: Résultat de la validation
        """
        issues = []
        recommendations = []

        # Vérifier les colonnes obligatoires
        missing_columns = [
            col for col in self.required_columns if col not in data.columns
        ]
        if missing_columns:
            issues.append(f"Colonnes manquantes: {missing_columns}")
            return DataValidationResult(
                is_valid=False,
                quality=DataQuality.INVALID,
                missing_data_count=0,
                duplicates_count=0,
                outliers_count=0,
                date_gaps_count=0,
                issues=issues,
                recommendations=[],
            )

        # Compter les données manquantes
        missing_data_count = data.isnull().sum().sum()
        if missing_data_count > 0:
            issues.append(f"{missing_data_count} valeurs manquantes détectées")
            recommendations.append(
                "Considérer l'interpolation ou la suppression des lignes incomplètes"
            )

        # Compter les doublons
        duplicates_count = data.duplicated().sum()
        if duplicates_count > 0:
            issues.append(f"{duplicates_count} lignes dupliquées détectées")
            recommendations.append("Supprimer les doublons")

        # Détecter les valeurs aberrantes (outliers)
        outliers_count = self._detect_outliers(data)
        if outliers_count > 0:
            issues.append(f"{outliers_count} valeurs aberrantes détectées")
            recommendations.append("Vérifier la cohérence des prix extrêmes")

        # Vérifier les écarts de dates
        date_gaps_count = self._detect_date_gaps(data)
        if date_gaps_count > 0:
            issues.append(f"{date_gaps_count} écarts de dates détectés")
            recommendations.append("Vérifier la continuité temporelle des données")

        # Validation OHLC
        ohlc_issues = self._validate_ohlc_consistency(data)
        if ohlc_issues:
            issues.extend(ohlc_issues)
            recommendations.append("Corriger les incohérences OHLC")

        # Déterminer la qualité globale
        quality = self._determine_data_quality(
            missing_data_count,
            duplicates_count,
            outliers_count,
            date_gaps_count,
            len(ohlc_issues),
            len(data),
        )

        is_valid = quality != DataQuality.INVALID

        return DataValidationResult(
            is_valid=is_valid,
            quality=quality,
            missing_data_count=missing_data_count,
            duplicates_count=duplicates_count,
            outliers_count=outliers_count,
            date_gaps_count=date_gaps_count,
            issues=issues,
            recommendations=recommendations,
        )

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie les données de marché

        Args:
            data: DataFrame à nettoyer

        Returns:
            pd.DataFrame: Données nettoyées
        """
        cleaned = data.copy()

        # Supprimer les doublons
        cleaned = cleaned.drop_duplicates()

        # Convertir la colonne date
        if "date" in cleaned.columns:
            cleaned["date"] = pd.to_datetime(cleaned["date"])
            cleaned = cleaned.sort_values("date")

        # Supprimer les lignes avec trop de valeurs manquantes
        threshold = len(self.required_columns) * 0.8  # 80% des colonnes requises
        cleaned = cleaned.dropna(thresh=threshold)

        # Interpoler les valeurs manquantes restantes pour les colonnes numériques
        numeric_columns = ["open", "high", "low", "close", "volume"]
        for col in numeric_columns:
            if col in cleaned.columns:
                cleaned[col] = cleaned[col].interpolate(method="linear")

        # Corriger les valeurs OHLC incohérentes
        cleaned = self._fix_ohlc_inconsistencies(cleaned)

        # Supprimer les valeurs négatives ou nulles pour le volume
        if "volume" in cleaned.columns:
            cleaned["volume"] = cleaned["volume"].clip(lower=0)

        return cleaned

    def resample_data(
        self, data: pd.DataFrame, timeframe: str = "1D", date_column: str = "date"
    ) -> pd.DataFrame:
        """
        Rééchantillonne les données à une fréquence différente

        Args:
            data: DataFrame avec données temporelles
            timeframe: Fréquence cible ('1D', '1H', '4H', '1W', etc.)
            date_column: Nom de la colonne de date

        Returns:
            pd.DataFrame: Données rééchantillonnées
        """
        if date_column not in data.columns:
            raise ValueError(f"Colonne de date '{date_column}' introuvable")

        # Préparer les données
        resampled_data = data.copy()
        resampled_data[date_column] = pd.to_datetime(resampled_data[date_column])
        resampled_data = resampled_data.set_index(date_column)

        # Définir les règles d'agrégation
        agg_rules = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }

        # Appliquer le rééchantillonnage
        available_columns = {
            col: rule
            for col, rule in agg_rules.items()
            if col in resampled_data.columns
        }

        resampled = resampled_data.resample(timeframe).agg(available_columns)

        # Remettre la date en colonne
        resampled = resampled.reset_index()

        return resampled

    def _detect_outliers(self, data: pd.DataFrame) -> int:
        """Détecte les valeurs aberrantes dans les prix"""
        outliers_count = 0
        price_columns = ["open", "high", "low", "close"]

        for col in price_columns:
            if col in data.columns:
                values = data[col].dropna()
                if len(values) > 0:
                    # Méthode IQR
                    Q1 = values.quantile(0.25)
                    Q3 = values.quantile(0.75)
                    IQR = Q3 - Q1

                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outliers = values[(values < lower_bound) | (values > upper_bound)]
                    outliers_count += len(outliers)

        return outliers_count

    def _detect_date_gaps(self, data: pd.DataFrame) -> int:
        """Détecte les écarts dans les dates"""
        if "date" not in data.columns:
            return 0

        dates = pd.to_datetime(data["date"]).sort_values()

        if len(dates) < 2:
            return 0

        # Calculer les écarts entre dates consécutives
        date_diffs = dates.diff().dt.days

        # Un écart de plus de 3 jours est considéré comme un gap
        gaps = date_diffs[date_diffs > 3]

        return len(gaps)

    def _validate_ohlc_consistency(self, data: pd.DataFrame) -> List[str]:
        """Valide la cohérence des données OHLC"""
        issues = []

        required_ohlc = ["open", "high", "low", "close"]
        if not all(col in data.columns for col in required_ohlc):
            return issues

        # Vérifier que high >= max(open, close) et low <= min(open, close)
        high_issues = (
            (data["high"] < data["open"]) | (data["high"] < data["close"])
        ).sum()

        low_issues = (
            (data["low"] > data["open"]) | (data["low"] > data["close"])
        ).sum()

        if high_issues > 0:
            issues.append(f"{high_issues} incohérences détectées: high < open ou close")

        if low_issues > 0:
            issues.append(f"{low_issues} incohérences détectées: low > open ou close")

        return issues

    def _fix_ohlc_inconsistencies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Corrige les incohérences OHLC"""
        fixed_data = data.copy()

        required_ohlc = ["open", "high", "low", "close"]
        if not all(col in fixed_data.columns for col in required_ohlc):
            return fixed_data

        # Corriger les valeurs high et low
        for index, row in fixed_data.iterrows():
            open_price = row["open"]
            close_price = row["close"]
            high_price = row["high"]
            low_price = row["low"]

            # Ajuster high si nécessaire
            correct_high = max(open_price, close_price, high_price)
            fixed_data.at[index, "high"] = correct_high

            # Ajuster low si nécessaire
            correct_low = min(open_price, close_price, low_price)
            fixed_data.at[index, "low"] = correct_low

        return fixed_data

    def _determine_data_quality(
        self,
        missing_count: int,
        duplicates_count: int,
        outliers_count: int,
        gaps_count: int,
        ohlc_issues_count: int,
        total_rows: int,
    ) -> DataQuality:
        """Détermine la qualité globale des données"""
        if total_rows == 0:
            return DataQuality.INVALID

        # Calculer les pourcentages de problèmes
        missing_pct = missing_count / (total_rows * len(self.required_columns))
        duplicates_pct = duplicates_count / total_rows
        outliers_pct = outliers_count / total_rows
        gaps_pct = gaps_count / total_rows
        ohlc_issues_pct = ohlc_issues_count / total_rows

        # Score de qualité (plus c'est bas, mieux c'est)
        quality_score = (
            missing_pct * 0.4
            + duplicates_pct * 0.2
            + outliers_pct * 0.2
            + gaps_pct * 0.1
            + ohlc_issues_pct * 0.1
        )

        if quality_score > 0.2:
            return DataQuality.POOR
        elif quality_score > 0.1:
            return DataQuality.ACCEPTABLE
        elif quality_score > 0.05:
            return DataQuality.GOOD
        else:
            return DataQuality.EXCELLENT
