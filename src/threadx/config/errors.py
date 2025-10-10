"""Configuration-related exceptions for ThreadX."""
from __future__ import annotations

from typing import Optional


class ConfigurationError(Exception):
    """Domain-specific exception carrying contextual information."""

    def __init__(
        self,
        path_or_reason: Optional[str],
        reason: Optional[str] = None,
        *,
        details: Optional[str] = None,
        path: Optional[str] = None,
    ) -> None:
        if reason is None:
            self.reason: str = (
                str(path_or_reason)
                if path_or_reason is not None
                else "Unknown configuration error"
            )
        else:
            self.reason = str(reason)

        if path is not None:
            self.path: Optional[str] = str(path)
        elif reason is not None:
            self.path = str(path_or_reason) if path_or_reason is not None else None
        else:
            self.path = None
        self.details = str(details) if details is not None else None
        self._message = self._compose_full_message()
        super().__init__(self._message)

    def _compose_full_message(self) -> str:
        base_message = self.user_message
        if self.details:
            return f"{base_message}\n{self.details}"
        return base_message

    @property
    def user_message(self) -> str:
        location = f" (file: {self.path})" if self.path else ""
        return f"Configuration error{location}: {self.reason}"

    def __str__(self) -> str:
        return self._message


class PathValidationError(Exception):
    """Raised when configuration paths do not pass validation."""

    def __init__(self, message: str):
        super().__init__(message)
