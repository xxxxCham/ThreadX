"""Configuration-related exceptions for ThreadX."""
from __future__ import annotations

from typing import Optional


class ConfigurationError(Exception):
    """Generic configuration exception with optional context."""

    def __init__(
        self,
        path_or_reason: Optional[str] = None,
        reason: Optional[str] = None,
        *,
        details: Optional[str] = None,
    ) -> None:
        if reason is None:
            self.path = None
            self.reason = path_or_reason or "Configuration error"
        else:
            self.path = path_or_reason
            self.reason = reason
        self.details = details
        super().__init__(self.__str__())

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
