"""Core modules for Agent Runner.

This package contains the fundamental components and utilities used throughout
the Agent Runner system including exceptions, messages, tokens, and tool protocols.
"""

# Note: ModelInfo moved to agentrunner.providers.base to avoid circular import
from .compaction import (
    AggressiveCompactor,
    ClaudeStyleCompactor,
    CompactionContext,
    CompactionResult,
    CompactionStrategy,
    NoOpCompactor,
    get_compactor,
    get_default_compactor,
    list_compactors,
    register_compactor,
)
from .exceptions import (
    # Error codes
    E_CONFIRMED_DENY,
    E_NOT_FOUND,
    E_NOT_UNIQUE,
    E_PERMISSIONS,
    E_TIMEOUT,
    E_TOOL_UNKNOWN,
    E_UNSAFE,
    E_VALIDATION,
    AgentRunnerException,
    ConfigurationError,
    ModelResponseError,
    TokenLimitExceededError,
    ToolExecutionError,
    WorkspaceSecurityError,
    format_error_for_log,
    format_error_for_user,
)
from .messages import Message, MessageHistory
from .tokens import ContextManager, TokenCounter
from .tool_protocol import ToolDefinition

__all__ = [
    # Error codes
    "E_CONFIRMED_DENY",
    "E_NOT_FOUND",
    "E_NOT_UNIQUE",
    "E_PERMISSIONS",
    "E_TIMEOUT",
    "E_TOOL_UNKNOWN",
    "E_UNSAFE",
    "E_VALIDATION",
    # Exception classes
    "AgentRunnerException",
    "ConfigurationError",
    # Compaction system
    "AggressiveCompactor",
    "ClaudeStyleCompactor",
    "CompactionContext",
    "CompactionResult",
    "CompactionStrategy",
    "NoOpCompactor",
    # Other core components
    "ContextManager",
    "Message",
    "MessageHistory",
    "ModelInfo",
    "ModelResponseError",
    "TokenCounter",
    "TokenLimitExceededError",
    "ToolDefinition",
    "ToolExecutionError",
    "WorkspaceSecurityError",
    # Error formatting utilities
    "format_error_for_log",
    "format_error_for_user",
    # Compaction registry functions
    "get_compactor",
    "get_default_compactor",
    "list_compactors",
    "register_compactor",
]
