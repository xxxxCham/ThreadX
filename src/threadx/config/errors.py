"""Configuration-related exceptions for ThreadX."""
from __future__ import annotations

from typing import Optional


class ConfigurationError(Exception):
    """Raised when configuration loading or validation fails."""

    def __init__(
        self,
        path_or_reason: Optional[str] = None,
        reason: Optional[str] = None,
        *,
        details: Optional[str] = None,
    ) -> None:
        if reason is None:
            self.path = None
            self.reason = path_or_reason or "Unknown configuration error"
        else:
            self.path = path_or_reason
            self.reason = reason
        self.details = details
        message = self.user_message
        if self.details:
            message = f"{message}\n{self.details}"
        super().__init__(message)

    @property
    def user_message(self) -> str:
        location = f" (file: {self.path})" if self.path else ""
        return f"Configuration error{location}: {self.reason}"


class PathValidationError(Exception):
    """Raised when configuration paths do not pass validation."""

    def __init__(self, message: str):
        super().__init__(message)
