"""Configuration-related exceptions for ThreadX."""
from __future__ import annotations

from typing import Optional


class ConfigurationError(Exception):
    """Structured configuration error with optional context."""

    def __init__(
        self,
        path: Optional[str] | None,
        reason: Optional[str] = None,
        *,
        details: Optional[str] = None,
    ) -> None:
        if reason is None:
            # Allow shorthand initialisation ``ConfigurationError("message")``
            reason = "Unknown configuration error" if path is None else str(path)
            path = None

        super().__init__(reason)
        self.path: Optional[str] = path
        self.reason: str = reason
        self.details: Optional[str] = details

    @property
    def user_message(self) -> str:
        location = f" (file: {self.path})" if self.path else ""
        return f"Configuration error{location}: {self.reason}"

    def __str__(self) -> str:
        message = self.user_message
        if self.details:
            message = f"{message}\n{self.details}"
        return message


class PathValidationError(Exception):
    """Raised when configuration paths do not pass validation."""

    def __init__(self, message: str):
        super().__init__(message)
