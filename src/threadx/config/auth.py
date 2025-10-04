"""
ThreadX Authentication Module
============================

Module d'authentification centralisé pour gérer les clés API.
Utilise les variables d'environnement pour une sécurité renforcée.

Variables d'environnement supportées:
- BINANCE_API_KEY: Clé API Binance
- BINANCE_API_SECRET: Secret API Binance
- COINGECKO_API_KEY: Clé API CoinGecko (optionnelle)
- ALPHA_VANTAGE_API_KEY: Clé API Alpha Vantage (optionnelle)
- POLYGON_API_KEY: Clé API Polygon (optionnelle)

Usage:
    from threadx.config.auth import AuthManager
    
    auth = AuthManager()
    if auth.has_binance_credentials():
        api_key, api_secret = auth.get_binance_credentials()
        # Utiliser les credentials pour les appels API

Author: ThreadX Framework
Version: 1.0
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any
import warnings

from threadx.utils.log import get_logger

logger = get_logger(__name__)


@dataclass
class APICredentials:
    """Structure pour stocker les credentials d'une API"""
    api_key: str
    api_secret: Optional[str] = None
    api_url: Optional[str] = None
    rate_limit: Optional[int] = None
    is_testnet: bool = False


@dataclass
class AuthConfig:
    """Configuration d'authentification globale"""
    enable_api_auth: bool = True
    fallback_to_public: bool = True
    log_auth_attempts: bool = True
    mask_credentials_in_logs: bool = True
    validate_on_startup: bool = False
    
    # Timeouts et retry
    api_timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


class AuthManager:
    """
    Gestionnaire centralisé d'authentification pour les APIs externes.
    
    Gère les credentials de manière sécurisée via variables d'environnement
    avec fallback gracieux vers les endpoints publics si nécessaire.
    """
    
    def __init__(self, config: Optional[AuthConfig] = None):
        """
        Initialise le gestionnaire d'authentification.
        
        Args:
            config: Configuration d'authentification (optionnelle)
        """
        self.config = config or AuthConfig()
        self.logger = logger
        self._credentials_cache: Dict[str, APICredentials] = {}
        
        # Variables d'environnement supportées
        self.env_mappings = {
            'binance': {
                'api_key': 'BINANCE_API_KEY',
                'api_secret': 'BINANCE_API_SECRET',
                'testnet_key': 'BINANCE_TESTNET_API_KEY',
                'testnet_secret': 'BINANCE_TESTNET_API_SECRET'
            },
            'coingecko': {
                'api_key': 'COINGECKO_API_KEY'
            },
            'alpha_vantage': {
                'api_key': 'ALPHA_VANTAGE_API_KEY'
            },
            'polygon': {
                'api_key': 'POLYGON_API_KEY'
            },
            'finnhub': {
                'api_key': 'FINNHUB_API_KEY'
            }
        }
        
        # URLs des APIs
        self.api_urls = {
            'binance': 'https://api.binance.com',
            'binance_testnet': 'https://testnet.binance.vision',
            'coingecko': 'https://api.coingecko.com/api/v3',
            'alpha_vantage': 'https://www.alphavantage.co',
            'polygon': 'https://api.polygon.io',
            'finnhub': 'https://finnhub.io/api/v1'
        }
        
        # Rate limits par défaut (requêtes/minute)
        self.rate_limits = {
            'binance': 1200,
            'binance_testnet': 1200,
            'coingecko': 10,  # Public: 10-50/min, avec clé: 500/min
            'alpha_vantage': 5,
            'polygon': 5,
            'finnhub': 60
        }
        
        if self.config.log_auth_attempts:
            self.logger.info("AuthManager initialisé - Variables d'environnement chargées")
    
    def _get_env_variable(self, var_name: str, required: bool = False) -> Optional[str]:
        """
        Récupère une variable d'environnement de manière sécurisée.
        
        Args:
            var_name: Nom de la variable d'environnement
            required: Si True, lève une exception si la variable n'existe pas
            
        Returns:
            Valeur de la variable ou None
            
        Raises:
            ValueError: Si la variable est requise mais absente
        """
        value = os.getenv(var_name)
        
        if required and not value:
            raise ValueError(f"Variable d'environnement requise manquante: {var_name}")
        
        if value and self.config.log_auth_attempts:
            masked_value = f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "***"
            self.logger.debug(f"Variable {var_name} trouvée: {masked_value}")
        elif not value and self.config.log_auth_attempts:
            self.logger.debug(f"Variable {var_name} non définie")
            
        return value
    
    def has_credentials(self, provider: str) -> bool:
        """
        Vérifie si les credentials sont disponibles pour un provider.
        
        Args:
            provider: Nom du provider ('binance', 'coingecko', etc.)
            
        Returns:
            True si les credentials sont disponibles
        """
        if provider not in self.env_mappings:
            return False
        
        mapping = self.env_mappings[provider]
        api_key = self._get_env_variable(mapping['api_key'])
        
        # Pour Binance, on a besoin aussi du secret
        if provider == 'binance' and api_key:
            api_secret = self._get_env_variable(mapping['api_secret'])
            return api_key is not None and api_secret is not None
        
        return api_key is not None
    
    def get_credentials(self, provider: str, use_testnet: bool = False) -> Optional[APICredentials]:
        """
        Récupère les credentials pour un provider.
        
        Args:
            provider: Nom du provider
            use_testnet: Utiliser les credentials testnet (Binance uniquement)
            
        Returns:
            APICredentials ou None si non disponible
        """
        cache_key = f"{provider}_{'testnet' if use_testnet else 'mainnet'}"
        
        # Vérifier le cache
        if cache_key in self._credentials_cache:
            return self._credentials_cache[cache_key]
        
        if provider not in self.env_mappings:
            self.logger.warning(f"Provider non supporté: {provider}")
            return None
        
        mapping = self.env_mappings[provider]
        
        try:
            if provider == 'binance':
                if use_testnet:
                    api_key = self._get_env_variable(mapping.get('testnet_key'))
                    api_secret = self._get_env_variable(mapping.get('testnet_secret'))
                    api_url = self.api_urls['binance_testnet']
                else:
                    api_key = self._get_env_variable(mapping['api_key'])
                    api_secret = self._get_env_variable(mapping['api_secret'])
                    api_url = self.api_urls['binance']
                
                if not (api_key and api_secret):
                    return None
                
                credentials = APICredentials(
                    api_key=api_key,
                    api_secret=api_secret,
                    api_url=api_url,
                    rate_limit=self.rate_limits.get(provider, 60),
                    is_testnet=use_testnet
                )
            else:
                # Autres providers (clé API uniquement)
                api_key = self._get_env_variable(mapping['api_key'])
                if not api_key:
                    return None
                
                credentials = APICredentials(
                    api_key=api_key,
                    api_url=self.api_urls.get(provider),
                    rate_limit=self.rate_limits.get(provider, 60)
                )
            
            # Cache et retour
            self._credentials_cache[cache_key] = credentials
            
            if self.config.log_auth_attempts:
                self.logger.info(f"✅ Credentials {provider} {'(testnet)' if use_testnet else ''} chargés")
            
            return credentials
            
        except Exception as e:
            self.logger.error(f"❌ Erreur chargement credentials {provider}: {e}")
            return None
    
    def get_binance_credentials(self, use_testnet: bool = False) -> Optional[Tuple[str, str]]:
        """
        Récupère les credentials Binance (helper method).
        
        Args:
            use_testnet: Utiliser les credentials testnet
            
        Returns:
            Tuple (api_key, api_secret) ou None
        """
        credentials = self.get_credentials('binance', use_testnet)
        if credentials and credentials.api_secret:
            return credentials.api_key, credentials.api_secret
        return None
    
    def has_binance_credentials(self, use_testnet: bool = False) -> bool:
        """Vérifie si les credentials Binance sont disponibles."""
        return self.get_binance_credentials(use_testnet) is not None
    
    def get_coingecko_key(self) -> Optional[str]:
        """Récupère la clé API CoinGecko."""
        credentials = self.get_credentials('coingecko')
        return credentials.api_key if credentials else None
    
    def has_coingecko_key(self) -> bool:
        """Vérifie si la clé CoinGecko est disponible."""
        return self.get_coingecko_key() is not None
    
    def get_api_url(self, provider: str, use_testnet: bool = False) -> Optional[str]:
        """
        Récupère l'URL de base pour un provider.
        
        Args:
            provider: Nom du provider
            use_testnet: Utiliser l'URL testnet (si applicable)
            
        Returns:
            URL de base ou None
        """
        if provider == 'binance' and use_testnet:
            return self.api_urls['binance_testnet']
        return self.api_urls.get(provider)
    
    def get_rate_limit(self, provider: str) -> int:
        """Récupère la limite de taux pour un provider."""
        return self.rate_limits.get(provider, 60)
    
    def validate_all_credentials(self) -> Dict[str, bool]:
        """
        Valide tous les credentials disponibles.
        
        Returns:
            Dictionnaire {provider: is_valid}
        """
        results = {}
        
        for provider in self.env_mappings.keys():
            try:
                credentials = self.get_credentials(provider)
                results[provider] = credentials is not None
                
                if credentials and self.config.log_auth_attempts:
                    self.logger.info(f"✅ {provider}: Credentials valides")
                elif self.config.log_auth_attempts:
                    self.logger.warning(f"⚠️ {provider}: Credentials manquants")
                    
            except Exception as e:
                results[provider] = False
                self.logger.error(f"❌ {provider}: Erreur validation - {e}")
        
        return results
    
    def print_auth_status(self):
        """Affiche le statut d'authentification pour debugging."""
        print("\n🔐 STATUT D'AUTHENTIFICATION THREADX")
        print("=" * 50)
        
        validation_results = self.validate_all_credentials()
        
        for provider, is_valid in validation_results.items():
            status = "✅ Configuré" if is_valid else "❌ Manquant"
            env_vars = self.env_mappings[provider]
            
            print(f"{provider.upper():15s}: {status}")
            for var_type, var_name in env_vars.items():
                value = os.getenv(var_name)
                if value:
                    masked = f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "***"
                    print(f"  └─ {var_name}: {masked}")
                else:
                    print(f"  └─ {var_name}: Non défini")
        
        print(f"\nConfiguration: Fallback public = {self.config.fallback_to_public}")
    
    def get_auth_headers(self, provider: str, use_testnet: bool = False) -> Dict[str, str]:
        """
        Génère les headers d'authentification pour les requêtes API.
        
        Args:
            provider: Nom du provider
            use_testnet: Utiliser testnet
            
        Returns:
            Dictionnaire des headers
        """
        credentials = self.get_credentials(provider, use_testnet)
        if not credentials:
            return {}
        
        headers = {}
        
        if provider == 'coingecko' and credentials.api_key:
            headers['x-cg-demo-api-key'] = credentials.api_key
        elif provider == 'alpha_vantage' and credentials.api_key:
            # Alpha Vantage utilise un paramètre URL plutôt qu'un header
            pass
        elif provider == 'polygon' and credentials.api_key:
            headers['Authorization'] = f'Bearer {credentials.api_key}'
        elif provider == 'finnhub' and credentials.api_key:
            headers['X-Finnhub-Token'] = credentials.api_key
        
        return headers


# Instance globale du gestionnaire d'authentification
_auth_manager: Optional[AuthManager] = None


def get_auth_manager(force_reload: bool = False) -> AuthManager:
    """
    Récupère l'instance globale du gestionnaire d'authentification (singleton).
    
    Args:
        force_reload: Force la recharge de l'instance
        
    Returns:
        Instance AuthManager
    """
    global _auth_manager
    if _auth_manager is None or force_reload:
        _auth_manager = AuthManager()
    return _auth_manager


def has_api_credentials(provider: str) -> bool:
    """Helper function pour vérifier rapidement les credentials."""
    return get_auth_manager().has_credentials(provider)


def get_api_credentials(provider: str, use_testnet: bool = False) -> Optional[APICredentials]:
    """Helper function pour récupérer rapidement les credentials."""
    return get_auth_manager().get_credentials(provider, use_testnet)


# Export des classes et fonctions principales
__all__ = [
    "AuthManager",
    "APICredentials", 
    "AuthConfig",
    "get_auth_manager",
    "has_api_credentials",
    "get_api_credentials"
]


if __name__ == "__main__":
    # Script de test pour vérifier l'authentification
    auth = AuthManager()
    auth.print_auth_status()