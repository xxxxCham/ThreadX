"""
Gestion de l'authentification pour ThreadX Dashboard
===================================================

Ce module fournit toutes les fonctionnalités d'authentification et de gestion
des sessions pour l'application Dash.
"""

import hashlib
import hmac
import time
import uuid
from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, Optional, Any

import bcrypt
import jwt
from flask import session, request

from config import (
    AUTH_ENABLED,
    SECRET_KEY,
    SESSION_TIMEOUT,
    DEFAULT_USERNAME,
    DEFAULT_PASSWORD,
)


class AuthManager:
    """
    Gestionnaire d'authentification pour ThreadX Dashboard.

    Gère les connexions, déconnexions, vérification des tokens
    et gestion des sessions utilisateur.
    """

    def __init__(self):
        """Initialise le gestionnaire d'authentification."""
        self.secret_key = SECRET_KEY
        self.session_timeout = SESSION_TIMEOUT
        self.users_db = self._init_users_db()

    def _init_users_db(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialise la base de données des utilisateurs en mémoire.

        Returns:
            Dict: Base de données des utilisateurs
        """
        # Hash du mot de passe par défaut
        hashed_password = bcrypt.hashpw(
            DEFAULT_PASSWORD.encode("utf-8"), bcrypt.gensalt()
        )

        return {
            DEFAULT_USERNAME: {
                "username": DEFAULT_USERNAME,
                "password_hash": hashed_password,
                "role": "admin",
                "created_at": datetime.utcnow(),
                "last_login": None,
                "is_active": True,
            }
        }

    def _hash_password(self, password: str) -> bytes:
        """
        Hash un mot de passe avec bcrypt.

        Args:
            password: Mot de passe en texte clair

        Returns:
            bytes: Mot de passe hashé
        """
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    def _verify_password(self, password: str, password_hash: bytes) -> bool:
        """
        Vérifie un mot de passe contre son hash.

        Args:
            password: Mot de passe en texte clair
            password_hash: Hash à vérifier

        Returns:
            bool: True si le mot de passe est correct
        """
        return bcrypt.checkpw(password.encode("utf-8"), password_hash)

    def login(self, username: str, password: str) -> bool:
        """
        Authentifie un utilisateur avec username/password.

        Args:
            username: Nom d'utilisateur
            password: Mot de passe

        Returns:
            bool: True si l'authentification réussit
        """
        if not AUTH_ENABLED:
            return True

        user = self.users_db.get(username)
        if not user or not user["is_active"]:
            return False

        if self._verify_password(password, user["password_hash"]):
            # Créer une session
            session_id = str(uuid.uuid4())
            token = self._generate_token(username, session_id)

            session["user_id"] = username
            session["session_id"] = session_id
            session["token"] = token
            session["login_time"] = time.time()
            session["last_activity"] = time.time()

            # Mettre à jour la dernière connexion
            user["last_login"] = datetime.utcnow()

            return True

        return False

    def logout(self) -> None:
        """Déconnecte l'utilisateur actuel."""
        session.clear()

    def is_authenticated(self) -> bool:
        """
        Vérifie si l'utilisateur actuel est authentifié.

        Returns:
            bool: True si l'utilisateur est authentifié
        """
        if not AUTH_ENABLED:
            return True

        if "user_id" not in session or "token" not in session:
            return False

        # Vérifier l'expiration de la session
        if self._is_session_expired():
            self.logout()
            return False

        # Vérifier le token
        if not self.verify_token(session.get("token")):
            self.logout()
            return False

        # Mettre à jour la dernière activité
        session["last_activity"] = time.time()

        return True

    def get_current_user(self) -> Optional[Dict[str, Any]]:
        """
        Récupère les informations de l'utilisateur actuel.

        Returns:
            Dict: Informations utilisateur ou None
        """
        if not self.is_authenticated():
            return None

        username = session.get("user_id")
        if username in self.users_db:
            user = self.users_db[username].copy()
            user.pop("password_hash", None)  # Ne pas exposer le hash
            return user

        return None

    def verify_token(self, token: str) -> bool:
        """
        Vérifie la validité d'un token JWT.

        Args:
            token: Token à vérifier

        Returns:
            bool: True si le token est valide
        """
        if not token:
            return False

        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload.get("username") == session.get("user_id")
        except jwt.InvalidTokenError:
            return False

    def _generate_token(self, username: str, session_id: str) -> str:
        """
        Génère un token JWT pour l'utilisateur.

        Args:
            username: Nom d'utilisateur
            session_id: ID de session

        Returns:
            str: Token JWT
        """
        payload = {
            "username": username,
            "session_id": session_id,
            "exp": datetime.utcnow() + timedelta(seconds=self.session_timeout),
            "iat": datetime.utcnow(),
        }

        return jwt.encode(payload, self.secret_key, algorithm="HS256")

    def _is_session_expired(self) -> bool:
        """
        Vérifie si la session actuelle est expirée.

        Returns:
            bool: True si la session est expirée
        """
        last_activity = session.get("last_activity", 0)
        return time.time() - last_activity > self.session_timeout


# Instance globale du gestionnaire d'authentification
auth_manager = AuthManager()


# =============================================================================
# DÉCORATEURS
# =============================================================================


def require_login(f):
    """
    Décorateur pour protéger les routes nécessitant une authentification.

    Args:
        f: Fonction à protéger

    Returns:
        function: Fonction décorée
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not auth_manager.is_authenticated():
            return {"error": "Authentication required"}, 401
        return f(*args, **kwargs)

    return decorated_function


def admin_only(f):
    """
    Décorateur pour les routes nécessitant des privilèges administrateur.

    Args:
        f: Fonction à protéger

    Returns:
        function: Fonction décorée
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not auth_manager.is_authenticated():
            return {"error": "Authentication required"}, 401

        user = auth_manager.get_current_user()
        if not user or user.get("role") != "admin":
            return {"error": "Admin privileges required"}, 403

        return f(*args, **kwargs)

    return decorated_function


def rate_limit(max_calls: int = 100, window_seconds: int = 3600):
    """
    Décorateur pour limiter le nombre d'appels par fenêtre de temps.

    Args:
        max_calls: Nombre maximum d'appels
        window_seconds: Taille de la fenêtre en secondes

    Returns:
        function: Décorateur
    """

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Simple rate limiting basé sur l'IP
            client_ip = request.remote_addr
            current_time = time.time()

            # Ici on devrait utiliser Redis ou une base de données
            # Pour la démo, on utilise la session Flask
            rate_key = f"rate_limit_{client_ip}_{f.__name__}"

            if rate_key not in session:
                session[rate_key] = []

            # Nettoyer les anciens appels
            session[rate_key] = [
                call_time
                for call_time in session[rate_key]
                if current_time - call_time < window_seconds
            ]

            # Vérifier la limite
            if len(session[rate_key]) >= max_calls:
                return {"error": "Rate limit exceeded"}, 429

            # Enregistrer cet appel
            session[rate_key].append(current_time)

            return f(*args, **kwargs)

        return decorated_function

    return decorator


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================


def get_session_info() -> Dict[str, Any]:
    """
    Récupère les informations de session actuelles.

    Returns:
        Dict: Informations de session
    """
    if not auth_manager.is_authenticated():
        return {"authenticated": False}

    user = auth_manager.get_current_user()
    login_time = session.get("login_time", 0)
    last_activity = session.get("last_activity", 0)

    return {
        "authenticated": True,
        "user": user,
        "login_time": (
            datetime.fromtimestamp(login_time).isoformat() if login_time else None
        ),
        "last_activity": (
            datetime.fromtimestamp(last_activity).isoformat() if last_activity else None
        ),
        "session_expires_in": max(0, SESSION_TIMEOUT - (time.time() - last_activity)),
    }


def create_secure_token(data: str) -> str:
    """
    Crée un token sécurisé pour des données arbitraires.

    Args:
        data: Données à sécuriser

    Returns:
        str: Token sécurisé
    """
    return hmac.new(SECRET_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()


def verify_secure_token(data: str, token: str) -> bool:
    """
    Vérifie un token sécurisé.

    Args:
        data: Données originales
        token: Token à vérifier

    Returns:
        bool: True si le token est valide
    """
    expected_token = create_secure_token(data)
    return hmac.compare_digest(expected_token, token)
