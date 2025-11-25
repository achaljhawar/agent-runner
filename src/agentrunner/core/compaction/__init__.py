"""Context compaction strategies for managing token limits.

Implements pluggable compaction strategies to reduce message history size
while preserving important information for continued conversation.
See INTERFACES/COMPACTION_SUMMARIZATION.md for specification.
"""

from typing import Any

from agentrunner.core.compaction.base import CompactionContext, CompactionResult, CompactionStrategy
from agentrunner.core.compaction.strategies import (
    AggressiveCompactor,
    ClaudeStyleCompactor,
    NoOpCompactor,
)

# Global registry of compaction strategies
_STRATEGY_REGISTRY: dict[str, type[CompactionStrategy]] = {}

# Register built-in strategies
_BUILT_IN_STRATEGIES: dict[str, type[CompactionStrategy]] = {
    "claude_style": ClaudeStyleCompactor,
    "aggressive": AggressiveCompactor,
    "noop": NoOpCompactor,
}


def register_compactor(name: str, strategy_class: type[CompactionStrategy]) -> None:
    """Register a compaction strategy.

    Args:
        name: Strategy name (must be unique)
        strategy_class: CompactionStrategy subclass

    Raises:
        ValueError: If name is already registered or strategy_class is invalid
    """
    if not name:
        raise ValueError("Strategy name cannot be empty")

    if not issubclass(strategy_class, CompactionStrategy):
        raise ValueError(
            f"Strategy class must inherit from CompactionStrategy, got {strategy_class}"
        )

    if name in _STRATEGY_REGISTRY:
        raise ValueError(f"Strategy '{name}' is already registered")

    _STRATEGY_REGISTRY[name] = strategy_class


def get_compactor(name: str, **kwargs: Any) -> CompactionStrategy:
    """Get a compaction strategy instance by name.

    Args:
        name: Strategy name
        **kwargs: Additional arguments passed to strategy constructor

    Returns:
        CompactionStrategy instance

    Raises:
        ValueError: If strategy name is not found
    """
    if not name:
        raise ValueError("Strategy name cannot be empty")

    # Check built-ins first
    if name in _BUILT_IN_STRATEGIES:
        strategy_class: type[CompactionStrategy] = _BUILT_IN_STRATEGIES[name]
        return strategy_class(**kwargs)

    # Check registry
    if name in _STRATEGY_REGISTRY:
        strategy_class = _STRATEGY_REGISTRY[name]
        return strategy_class(**kwargs)

    # Not found
    available = list(_BUILT_IN_STRATEGIES.keys()) + list(_STRATEGY_REGISTRY.keys())
    raise ValueError(f"Unknown compaction strategy: {name}. Available: {available}")


def list_compactors() -> list[str]:
    """List all available compaction strategy names.

    Returns:
        List of strategy names
    """
    return list(_BUILT_IN_STRATEGIES.keys()) + list(_STRATEGY_REGISTRY.keys())


def clear_registry() -> None:
    """Clear the strategy registry (for testing).

    Note: Built-in strategies are not affected.
    """
    _STRATEGY_REGISTRY.clear()


def get_default_compactor(**kwargs: Any) -> CompactionStrategy:
    """Get the default compaction strategy.

    Args:
        **kwargs: Additional arguments passed to strategy constructor

    Returns:
        Default CompactionStrategy instance (ClaudeStyleCompactor)
    """
    return get_compactor("claude_style", **kwargs)


# Auto-register built-in strategies on import
def _register_built_ins() -> None:
    """Register built-in strategies."""
    for _name, _strategy_class in _BUILT_IN_STRATEGIES.items():
        # Use internal registry to avoid validation for built-ins
        pass  # Built-ins are handled directly in get_compactor()


_register_built_ins()

__all__ = [
    "CompactionContext",
    "CompactionResult",
    "CompactionStrategy",
    "AggressiveCompactor",
    "ClaudeStyleCompactor",
    "NoOpCompactor",
    "register_compactor",
    "get_compactor",
    "list_compactors",
    "clear_registry",
    "get_default_compactor",
]
