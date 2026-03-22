"""
Model registry — maps friendly/short names to official API call names.

Usage:
    from model_registry import resolve_model, ModelProvider

    api_name = resolve_model("qwen-flash")       # → "qwen-flash"
    api_name = resolve_model("gpt4o")            # → "gpt-4o"
    api_name = resolve_model("claude-sonnet")    # → "claude-sonnet-4-6"
"""

from __future__ import annotations
from enum import Enum


class ModelProvider(str, Enum):
    OPENAI    = "openai"
    ANTHROPIC = "anthropic"
    QWEN      = "qwen"          # Alibaba DashScope
    DEEPSEEK  = "deepseek"
    GEMINI    = "gemini"        # Google
    MISTRAL   = "mistral"


# ---------------------------------------------------------------------------
# Canonical model names (official API identifiers)
# ---------------------------------------------------------------------------

# fmt: off
_REGISTRY: dict[str, tuple[str, ModelProvider]] = {

    # ── OpenAI ──────────────────────────────────────────────────────────────
    "gpt-4o"                  : ("gpt-4o",                        ModelProvider.OPENAI),
    "gpt-4o-mini"             : ("gpt-4o-mini",                   ModelProvider.OPENAI),
    "gpt-4-turbo"             : ("gpt-4-turbo",                   ModelProvider.OPENAI),
    "gpt-4"                   : ("gpt-4",                         ModelProvider.OPENAI),
    "gpt-3.5-turbo"           : ("gpt-3.5-turbo",                 ModelProvider.OPENAI),
    "o1"                      : ("o1",                            ModelProvider.OPENAI),
    "o1-mini"                 : ("o1-mini",                       ModelProvider.OPENAI),
    "o3-mini"                 : ("o3-mini",                       ModelProvider.OPENAI),

    # ── Anthropic Claude ────────────────────────────────────────────────────
    "claude-opus"             : ("claude-opus-4-6",               ModelProvider.ANTHROPIC),
    "claude-sonnet"           : ("claude-sonnet-4-6",             ModelProvider.ANTHROPIC),
    "claude-haiku"            : ("claude-haiku-4-5-20251001",     ModelProvider.ANTHROPIC),
    "claude-3-5-sonnet"       : ("claude-3-5-sonnet-20241022",    ModelProvider.ANTHROPIC),
    "claude-3-5-haiku"        : ("claude-3-5-haiku-20241022",     ModelProvider.ANTHROPIC),
    "claude-3-opus"           : ("claude-3-opus-20240229",        ModelProvider.ANTHROPIC),

    # ── Alibaba Qwen (DashScope / compatible mode) ──────────────────────────
    "qwen-max"                : ("qwen-max",                      ModelProvider.QWEN),
    "qwen-plus"               : ("qwen-plus",                     ModelProvider.QWEN),
    "qwen-turbo"              : ("qwen-turbo",                    ModelProvider.QWEN),
    "qwen-flash"              : ("qwen-flash",                    ModelProvider.QWEN),
    "qwen-long"               : ("qwen-long",                     ModelProvider.QWEN),
    "qwen2.5-72b"             : ("qwen2.5-72b-instruct",          ModelProvider.QWEN),
    "qwen2.5-32b"             : ("qwen2.5-32b-instruct",          ModelProvider.QWEN),
    "qwen2.5-14b"             : ("qwen2.5-14b-instruct",          ModelProvider.QWEN),
    "qwen2.5-7b"              : ("qwen2.5-7b-instruct",           ModelProvider.QWEN),

    # ── DeepSeek ────────────────────────────────────────────────────────────
    "deepseek-v3"             : ("deepseek-chat",                 ModelProvider.DEEPSEEK),
    "deepseek-chat"           : ("deepseek-chat",                 ModelProvider.DEEPSEEK),
    "deepseek-r1"             : ("deepseek-reasoner",             ModelProvider.DEEPSEEK),
    "deepseek-reasoner"       : ("deepseek-reasoner",             ModelProvider.DEEPSEEK),

    # ── Google Gemini ────────────────────────────────────────────────────────
    "gemini-2.0-flash"        : ("gemini-2.0-flash",              ModelProvider.GEMINI),
    "gemini-2.5-pro"          : ("gemini-2.5-pro-preview-03-25",  ModelProvider.GEMINI),
    "gemini-1.5-pro"          : ("gemini-1.5-pro",                ModelProvider.GEMINI),
    "gemini-1.5-flash"        : ("gemini-1.5-flash",              ModelProvider.GEMINI),

    # ── Mistral ──────────────────────────────────────────────────────────────
    "mistral-large"           : ("mistral-large-latest",          ModelProvider.MISTRAL),
    "mistral-small"           : ("mistral-small-latest",          ModelProvider.MISTRAL),
    "codestral"               : ("codestral-latest",              ModelProvider.MISTRAL),
    "mixtral-8x7b"            : ("open-mixtral-8x7b",             ModelProvider.MISTRAL),
}

# Alias table — alternative short names that map to a canonical key above
_ALIASES: dict[str, str] = {
    # OpenAI
    "gpt4o"          : "gpt-4o",
    "gpt4o-mini"     : "gpt-4o-mini",
    "gpt4"           : "gpt-4",
    "gpt35"          : "gpt-3.5-turbo",
    "gpt-35-turbo"   : "gpt-3.5-turbo",
    # Anthropic
    "opus"           : "claude-opus",
    "sonnet"         : "claude-sonnet",
    "haiku"          : "claude-haiku",
    # Qwen
    "qwen"           : "qwen-plus",
    # DeepSeek
    "deepseek"       : "deepseek-v3",
    "r1"             : "deepseek-r1",
    # Gemini
    "gemini"         : "gemini-2.0-flash",
    "gemini-flash"   : "gemini-2.0-flash",
    # Mistral
    "mistral"        : "mistral-large",
}
# fmt: on

DEFAULT_MODEL = "qwen-flash"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def resolve_model(name: str | None = None) -> str:
    """Return the official API model identifier for *name*.

    Resolution order:
      1. Exact match in registry
      2. Alias → registry lookup
      3. Pass-through (assume *name* is already a valid API id)
      4. Fall back to DEFAULT_MODEL when *name* is None / empty
    """
    if not name:
        return _REGISTRY[DEFAULT_MODEL][0]

    key = name.strip().lower()

    if key in _REGISTRY:
        return _REGISTRY[key][0]

    canonical_key = _ALIASES.get(key)
    if canonical_key:
        return _REGISTRY[canonical_key][0]

    # Unknown name — pass through and let the API surface the error
    return name


def get_provider(name: str) -> ModelProvider | None:
    """Return the ModelProvider for a given model name (or alias), or None."""
    key = name.strip().lower()
    if key in _REGISTRY:
        return _REGISTRY[key][1]
    canonical_key = _ALIASES.get(key)
    if canonical_key:
        return _REGISTRY[canonical_key][1]
    return None


def list_models(provider: ModelProvider | None = None) -> list[str]:
    """Return canonical API names, optionally filtered by provider."""
    if provider is None:
        return [v[0] for v in _REGISTRY.values()]
    return [v[0] for v in _REGISTRY.values() if v[1] == provider]

if __name__ == "__main__":
    print(resolve_model("qwen-flash"))
    print(get_provider("qwen"))
    print(list_models(ModelProvider.QWEN))
