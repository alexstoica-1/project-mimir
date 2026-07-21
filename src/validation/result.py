"""Shared validation result models."""

from __future__ import annotations

from dataclasses import dataclass, field


class DataValidationError(RuntimeError):
    """Raised when validation cannot be executed safely."""


@dataclass(frozen=True)
class ValidationResult:
    """Structured validation outcome for collected data."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    row_count: int = 0
