"""LLM client creation logic."""

from __future__ import annotations

from cagent.config import ProviderConfig, load_fast_api_config, load_smart_api_config
from cagent.llm import LLMClient, LLMRequest, LLMResponse
from cagent.llm.anthropic import AnthropicClient
from cagent.llm.gemini import GeminiClient
from cagent.llm.llamacpp import LLamaCPP
from cagent.llm.openai import OpenAIChatCompletionsClient


class EchoLLMClient(LLMClient):
    """Local provider used by the skeleton entry point."""

    def _complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(content=request.user_prompt)


def create_fast_api_client() -> LLMClient | None:
    """Create a FAST_API LLM client when provider configuration is set."""
    config = load_fast_api_config()
    if not _has_usable_provider_config(config):
        return None
    return _create_client_from_config(config)


def create_smart_api_client() -> LLMClient | None:
    """Create a SMART_API LLM client when provider configuration is set."""
    config = load_smart_api_config()
    if not _has_usable_provider_config(config):
        return None
    return _create_client_from_config(config)


def _has_usable_provider_config(config: ProviderConfig) -> bool:
    """Return whether a provider config is sufficient to create a client."""
    provider_type = (config.type or "llamacpp").lower()
    if provider_type == "llamacpp":
        return bool(config.host)
    return bool(config.type)


def _create_client_from_config(config: ProviderConfig) -> LLMClient:
    """Instantiate the correct LLM client based on provider configuration."""
    provider_type = (config.type or "llamacpp").lower()

    if provider_type == "llamacpp":
        if not config.host:
            raise ValueError("HOST is required for TYPE=llamacpp.")
        return LLamaCPP(
            host=config.host,
            token=config.token,
            model=config.model or "moonshotai/Kimi-K2.5",
        )

    if provider_type == "openai":
        try:
            import openai as openai_sdk
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for TYPE=openai. "
                "Install it with: pip install openai"
            ) from exc
        openai_kwargs: dict[str, str] = {}
        if config.host:
            openai_kwargs["base_url"] = config.host
        if config.token:
            openai_kwargs["api_key"] = config.token
        client = openai_sdk.OpenAI(**openai_kwargs)
        return OpenAIChatCompletionsClient(
            client=client,
            default_model=config.model or "gpt-4o",
        )

    if provider_type == "anthropic":
        try:
            import anthropic as anthropic_sdk
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for TYPE=anthropic. "
                "Install it with: pip install anthropic"
            ) from exc
        anthropic_kwargs: dict[str, str] = {}
        if config.host:
            anthropic_kwargs["base_url"] = config.host
        if config.token:
            anthropic_kwargs["api_key"] = config.token
        client = anthropic_sdk.Anthropic(**anthropic_kwargs)
        return AnthropicClient(
            client=client,
            model=config.model or "claude-3-5-sonnet-20241022",
        )

    if provider_type == "gemini":
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "The 'google-genai' package is required for TYPE=gemini. "
                "Install it with: pip install google-genai"
            ) from exc
        
        client = genai.Client(
            api_key=config.token,
            http_options={"api_version": "v1alpha"} if "thinking" in (config.model or "") else None
        )
        return GeminiClient(
            client=client,
            model=config.model or "gemini-2.0-flash",
        )

    raise ValueError(f"Unsupported LLM provider type: {config.type}")


__all__ = [
    "EchoLLMClient",
    "create_fast_api_client",
    "create_smart_api_client",
]
