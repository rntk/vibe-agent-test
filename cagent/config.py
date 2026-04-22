"""Configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """LLM provider configuration."""

    host: str | None = None
    token: str | None = None
    type: str | None = None
    model: str | None = None


def load_provider_config(prefix: str) -> ProviderConfig:
    """Load provider configuration from environment variables.

    Reads ``{PREFIX}_HOST``, ``{PREFIX}_TOKEN``, ``{PREFIX}_TYPE``, and
    ``{PREFIX}_MODEL``.
    """
    return ProviderConfig(
        host=os.getenv(f"{prefix}_HOST") or None,
        token=os.getenv(f"{prefix}_TOKEN") or None,
        type=os.getenv(f"{prefix}_TYPE") or None,
        model=os.getenv(f"{prefix}_MODEL") or None,
    )


def load_fast_api_config() -> ProviderConfig:
    """Load FAST_API configuration from environment variables."""
    return load_provider_config("FAST_API")


def load_smart_api_config() -> ProviderConfig:
    """Load SMART_API configuration from environment variables."""
    return load_provider_config("SMART_API")
