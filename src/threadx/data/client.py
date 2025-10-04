"""
ThreadX Data Client with Authentication
======================================

Client unifié pour télécharger des données crypto avec support d'authentification.
Intègre le système AuthManager pour gérer les clés API automatiquement.

Features:
- Support multi-API : Binance, CoinGecko, Alpha Vantage, etc.
- Authentification automatique via variables d'environnement
- Fallback gracieux vers endpoints publics
- Rate limiting et retry automatique
- Cache des données avec TTL
- Logging détaillé

Variables d'environnement requises:
- BINANCE_API_KEY, BINANCE_API_SECRET : Pour Binance API (privée)
- COINGECKO_API_KEY : Pour CoinGecko API (optionnelle, améliore les limits)

Usage:
    from threadx.data.client import DataClient
    
    client = DataClient()
    
    # Télécharge avec authentification si disponible
    ohlcv_data = client.get_binance_klines("BTCUSDC", "1h", days=30)
    marketcap_data = client.get_coingecko_coins(limit=100)

Author: ThreadX Framework
Version: 1.0 - Authentication Support
"""

import time
import requests
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging

from threadx.config.auth import get_auth_manager, AuthManager, APICredentials
from threadx.config.settings import get_settings
from threadx.utils.log import get_logger

logger = get_logger(__name__)


class DataClientError(Exception):
    """Exception de base pour les erreurs du client de données."""
    pass


class APIRateLimitError(DataClientError):
    """Exception levée quand les limites de taux sont atteintes."""
    pass


class AuthenticationError(DataClientError):
    """Exception levée en cas de problème d'authentification."""
    pass


class DataClient:
    """
    Client unifié pour télécharger des données crypto avec authentification.
    
    Gère automatiquement:
    - Authentification via AuthManager
    - Rate limiting par API
    - Retry automatique sur échec
    - Cache local des données
    - Fallback vers endpoints publics
    """
    
    def __init__(self, auth_manager: Optional[AuthManager] = None):
        """
        Initialise le client de données.
        
        Args:
            auth_manager: Gestionnaire d'authentification (optionnel)
        """
        self.auth = auth_manager or get_auth_manager()
        self.settings = get_settings()
        self.logger = logger
        
        # Configuration des timeouts et retry
        self.timeout = self.settings.API_TIMEOUT_SECONDS
        self.max_retries = self.settings.API_MAX_RETRIES
        self.retry_delay = 1.0
        
        # Cache des appels API pour éviter les doublons
        self._request_cache: Dict[str, Tuple[Any, float]] = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Compteurs de rate limiting
        self._rate_counters: Dict[str, List[float]] = {}
        
        self.logger.info("DataClient initialisé avec support d'authentification")
    
    def _generate_cache_key(self, method: str, **kwargs) -> str:
        """Génère une clé de cache pour la requête."""
        key_data = f"{method}_{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Vérifie si le cache est valide."""
        if cache_key not in self._request_cache:
            return False
        
        _, timestamp = self._request_cache[cache_key]
        return time.time() - timestamp < self.cache_ttl
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Récupère une valeur du cache."""
        if self._is_cache_valid(cache_key):
            data, _ = self._request_cache[cache_key]
            return data
        return None
    
    def _store_in_cache(self, cache_key: str, data: Any):
        """Stocke une valeur dans le cache."""
        self._request_cache[cache_key] = (data, time.time())
    
    def _check_rate_limit(self, provider: str, limit_per_minute: int) -> bool:
        """
        Vérifie si on peut faire une requête sans dépasser les limites.
        
        Args:
            provider: Nom du provider (binance, coingecko, etc.)
            limit_per_minute: Limite de requêtes par minute
            
        Returns:
            True si on peut faire la requête
        """
        now = time.time()
        minute_ago = now - 60
        
        # Nettoyer les anciens appels
        if provider not in self._rate_counters:
            self._rate_counters[provider] = []
        
        self._rate_counters[provider] = [
            timestamp for timestamp in self._rate_counters[provider]
            if timestamp > minute_ago
        ]
        
        # Vérifier la limite
        if len(self._rate_counters[provider]) >= limit_per_minute:
            return False
        
        # Enregistrer cet appel
        self._rate_counters[provider].append(now)
        return True
    
    def _wait_for_rate_limit(self, provider: str, limit_per_minute: int):
        """Attend si nécessaire pour respecter les limites de taux."""
        if not self._check_rate_limit(provider, limit_per_minute):
            wait_time = 60 / limit_per_minute
            self.logger.info(f"Rate limit atteint pour {provider}, attente {wait_time:.1f}s")
            time.sleep(wait_time)
    
    def _make_authenticated_request(
        self, 
        url: str, 
        params: Dict[str, Any] = None,
        headers: Dict[str, str] = None,
        method: str = "GET",
        provider: str = "unknown"
    ) -> requests.Response:
        """
        Effectue une requête HTTP avec gestion d'authentification et retry.
        
        Args:
            url: URL de la requête
            params: Paramètres de la requête
            headers: Headers HTTP
            method: Méthode HTTP
            provider: Nom du provider pour rate limiting
            
        Returns:
            Response de la requête
            
        Raises:
            DataClientError: En cas d'erreur
        """
        params = params or {}
        headers = headers or {}
        
        # Ajouter les headers d'authentification si disponibles
        auth_headers = self.auth.get_auth_headers(provider)
        headers.update(auth_headers)
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Vérifier rate limiting
                rate_limit = self.auth.get_rate_limit(provider)
                self._wait_for_rate_limit(provider, rate_limit)
                
                if method.upper() == "GET":
                    response = requests.get(
                        url, 
                        params=params, 
                        headers=headers, 
                        timeout=self.timeout
                    )
                elif method.upper() == "POST":
                    response = requests.post(
                        url, 
                        json=params, 
                        headers=headers, 
                        timeout=self.timeout
                    )
                else:
                    raise DataClientError(f"Méthode HTTP non supportée: {method}")
                
                # Vérifier la réponse
                if response.status_code == 429:  # Rate limit
                    if attempt < self.max_retries:
                        wait_time = 2 ** attempt  # Exponential backoff
                        self.logger.warning(f"Rate limit {provider}, retry dans {wait_time}s")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise APIRateLimitError(f"Rate limit dépassé pour {provider}")
                
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** attempt)
                    self.logger.warning(f"Erreur requête {provider} (tentative {attempt+1}), retry dans {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    break
        
        raise DataClientError(f"Échec requête {provider} après {self.max_retries} tentatives: {last_exception}")
    
    def get_binance_klines(
        self, 
        symbol: str, 
        interval: str, 
        days: int = 30,
        use_testnet: bool = False
    ) -> pd.DataFrame:
        """
        Télécharge les données OHLCV depuis Binance.
        
        Args:
            symbol: Symbole de trading (ex: BTCUSDC)
            interval: Interval de temps (1m, 5m, 1h, 1d, etc.)
            days: Nombre de jours d'historique
            use_testnet: Utiliser l'environnement testnet
            
        Returns:
            DataFrame avec colonnes OHLCV
        """
        cache_key = self._generate_cache_key(
            "binance_klines", 
            symbol=symbol, 
            interval=interval, 
            days=days, 
            testnet=use_testnet
        )
        
        # Vérifier le cache
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            self.logger.debug(f"Données Binance {symbol}_{interval} trouvées en cache")
            return cached_data
        
        # Déterminer l'URL selon testnet/mainnet
        if use_testnet:
            base_url = "https://testnet.binance.vision/api/v3"
        else:
            base_url = "https://api.binance.com/api/v3"
        
        url = f"{base_url}/klines"
        
        # Calcul des timestamps
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000
        }
        
        try:
            # Si on a des credentials Binance, les utiliser pour des limites supérieures
            if self.auth.has_binance_credentials(use_testnet):
                # Pour Binance, les endpoints publics ne nécessitent pas de signature
                # Mais avoir une API key peut augmenter les limits
                credentials = self.auth.get_credentials('binance', use_testnet)
                if credentials:
                    params['apikey'] = credentials.api_key
                    self.logger.debug(f"Utilisation API key Binance pour {symbol}")
            
            response = self._make_authenticated_request(
                url, 
                params=params, 
                provider='binance'
            )
            
            data = response.json()
            
            if not data:
                raise DataClientError(f"Aucune donnée reçue pour {symbol}")
            
            # Conversion en DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'count', 'taker_buy_volume',
                'taker_buy_quote_volume', 'ignore'
            ])
            
            # Nettoyage et formatage
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            
            # Conversion des types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df = df.set_index('timestamp').sort_index()
            
            # Suppression des doublons et NaN
            df = df[~df.index.duplicated(keep='last')]
            df = df.dropna()
            
            if len(df) == 0:
                raise DataClientError(f"DataFrame vide après nettoyage pour {symbol}")
            
            # Mise en cache
            self._store_in_cache(cache_key, df)
            
            self.logger.info(f"✅ Données Binance {symbol}_{interval}: {len(df)} lignes téléchargées")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Erreur téléchargement Binance {symbol}_{interval}: {e}")
            raise DataClientError(f"Échec téléchargement Binance: {e}")
    
    def get_coingecko_coins(self, limit: int = 100, vs_currency: str = "usd") -> List[Dict]:
        """
        Récupère la liste des cryptos depuis CoinGecko.
        
        Args:
            limit: Nombre de coins à récupérer
            vs_currency: Devise de référence
            
        Returns:
            Liste des données de coins
        """
        cache_key = self._generate_cache_key(
            "coingecko_coins", 
            limit=limit, 
            vs_currency=vs_currency
        )
        
        # Vérifier le cache
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            self.logger.debug(f"Données CoinGecko trouvées en cache ({limit} coins)")
            return cached_data
        
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": vs_currency,
            "order": "market_cap_desc",
            "per_page": limit,
            "page": 1,
            "sparkline": False
        }
        
        try:
            # Utiliser la clé API si disponible
            headers = {}
            if self.auth.has_coingecko_key():
                coingecko_key = self.auth.get_coingecko_key()
                headers["x-cg-demo-api-key"] = coingecko_key
                self.logger.debug("Utilisation clé API CoinGecko")
            else:
                self.logger.debug("Utilisation endpoint public CoinGecko")
            
            response = self._make_authenticated_request(
                url,
                params=params,
                headers=headers,
                provider='coingecko'
            )
            
            data = response.json()
            
            if not data:
                raise DataClientError("Aucune donnée reçue de CoinGecko")
            
            # Formatage des données
            formatted_data = []
            for coin in data:
                formatted_data.append({
                    "symbol": coin["symbol"].upper(),
                    "name": coin["name"],
                    "market_cap": coin.get("market_cap", 0),
                    "market_cap_rank": coin.get("market_cap_rank", 999),
                    "volume": coin.get("total_volume", 0),
                    "price": coin.get("current_price", 0),
                    "price_change_24h": coin.get("price_change_percentage_24h", 0)
                })
            
            # Mise en cache
            self._store_in_cache(cache_key, formatted_data)
            
            self.logger.info(f"✅ Données CoinGecko: {len(formatted_data)} coins récupérés")
            return formatted_data
            
        except Exception as e:
            self.logger.error(f"❌ Erreur téléchargement CoinGecko: {e}")
            raise DataClientError(f"Échec téléchargement CoinGecko: {e}")
    
    def get_binance_24hr_ticker(self) -> List[Dict]:
        """
        Récupère les statistiques 24h de tous les symboles Binance.
        
        Returns:
            Liste des données de ticker 24h
        """
        cache_key = self._generate_cache_key("binance_24hr_ticker")
        
        # Vérifier le cache
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            self.logger.debug("Données ticker 24h Binance trouvées en cache")
            return cached_data
        
        url = "https://api.binance.com/api/v3/ticker/24hr"
        
        try:
            response = self._make_authenticated_request(url, provider='binance')
            data = response.json()
            
            if not data:
                raise DataClientError("Aucune donnée ticker reçue de Binance")
            
            # Filtrer et formater les données
            formatted_data = []
            for ticker in data:
                if ticker["symbol"].endswith("USDC"):
                    base_asset = ticker["symbol"].replace("USDC", "")
                    formatted_data.append({
                        "symbol": base_asset,
                        "volume": float(ticker["quoteVolume"]) if ticker["quoteVolume"] else 0,
                        "price_change": float(ticker["priceChangePercent"]) if ticker["priceChangePercent"] else 0,
                        "price": float(ticker["lastPrice"]) if ticker["lastPrice"] else 0
                    })
            
            # Trier par volume décroissant
            formatted_data.sort(key=lambda x: x["volume"], reverse=True)
            
            # Mise en cache
            self._store_in_cache(cache_key, formatted_data)
            
            self.logger.info(f"✅ Données ticker Binance: {len(formatted_data)} symboles USDC récupérés")
            return formatted_data
            
        except Exception as e:
            self.logger.error(f"❌ Erreur téléchargement ticker Binance: {e}")
            raise DataClientError(f"Échec téléchargement ticker Binance: {e}")
    
    def test_all_connections(self) -> Dict[str, bool]:
        """
        Teste la connexion à toutes les APIs configurées.
        
        Returns:
            Dictionnaire {provider: is_working}
        """
        results = {}
        
        # Test Binance
        try:
            self.get_binance_klines("BTCUSDC", "1d", days=1)
            results["binance"] = True
            self.logger.info("✅ Connexion Binance OK")
        except Exception as e:
            results["binance"] = False
            self.logger.error(f"❌ Connexion Binance KO: {e}")
        
        # Test CoinGecko
        try:
            self.get_coingecko_coins(limit=10)
            results["coingecko"] = True
            self.logger.info("✅ Connexion CoinGecko OK")
        except Exception as e:
            results["coingecko"] = False
            self.logger.error(f"❌ Connexion CoinGecko KO: {e}")
        
        return results
    
    def print_status(self):
        """Affiche le statut du client de données."""
        print("\n📡 STATUT CLIENT DE DONNÉES THREADX")
        print("=" * 50)
        
        # Statut authentification
        print("🔐 Authentification:")
        if self.auth.has_binance_credentials():
            print("  ✅ Binance API: Configurée")
        else:
            print("  ⚠️ Binance API: Endpoints publics uniquement")
        
        if self.auth.has_coingecko_key():
            print("  ✅ CoinGecko API: Configurée (limites étendues)")
        else:
            print("  ⚠️ CoinGecko API: Endpoints publics (limites réduites)")
        
        # Test des connexions
        print("\n🌐 Test des connexions:")
        connections = self.test_all_connections()
        for provider, status in connections.items():
            emoji = "✅" if status else "❌"
            print(f"  {emoji} {provider.capitalize()}: {'Fonctionnel' if status else 'Erreur'}")
        
        # Statistiques cache
        print(f"\n💾 Cache: {len(self._request_cache)} entrées")


# Instance globale du client de données
_data_client: Optional[DataClient] = None


def get_data_client(force_reload: bool = False) -> DataClient:
    """
    Récupère l'instance globale du client de données (singleton).
    
    Args:
        force_reload: Force la recharge de l'instance
        
    Returns:
        Instance DataClient
    """
    global _data_client
    if _data_client is None or force_reload:
        _data_client = DataClient()
    return _data_client


# Export des classes et fonctions principales
__all__ = [
    "DataClient",
    "DataClientError",
    "APIRateLimitError", 
    "AuthenticationError",
    "get_data_client"
]


if __name__ == "__main__":
    # Script de test
    client = DataClient()
    client.print_status()