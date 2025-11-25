"""Token counting and context window management.

Implements TokenCounter and ContextManager per INTERFACES/CONTEXT_MANAGEMENT.md.
"""

from typing import TYPE_CHECKING, Any

import tiktoken

from .messages import Message

if TYPE_CHECKING:
    from agentrunner.core.tool_protocol import ToolDefinition
    from agentrunner.providers.base import ModelInfo


class TokenCounter:
    """Counts tokens using tiktoken for various model inputs."""

    def __init__(self) -> None:
        """Initialize token counter with encoding cache."""
        self._encoding_cache: dict[str, tiktoken.Encoding] = {}

    def _get_encoding(self, model: str) -> tiktoken.Encoding:
        """Get tiktoken encoding for model with caching."""
        if model not in self._encoding_cache:
            try:
                # Try to get model-specific encoding first
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fall back to cl100k_base for unknown models
                encoding = tiktoken.get_encoding("cl100k_base")
            self._encoding_cache[model] = encoding
        return self._encoding_cache[model]

    def count_text(self, text: str, model: str) -> int:
        """Count tokens in text string.

        Args:
            text: Text to count tokens for
            model: Model name for encoding selection

        Returns:
            Number of tokens in the text
        """
        if not text:
            return 0

        encoding = self._get_encoding(model)
        return len(encoding.encode(text))

    def count_messages(self, messages: list[Message], model: str) -> int:
        """Count tokens in message list including formatting overhead.

        Args:
            messages: List of messages to count
            model: Model name for encoding selection

        Returns:
            Total tokens including message formatting overhead
        """
        if not messages:
            return 0

        encoding = self._get_encoding(model)
        total_tokens = 0

        for message in messages:
            # Count role and content tokens
            total_tokens += len(encoding.encode(message.role))
            total_tokens += len(encoding.encode(message.content))

            # Add formatting overhead for message structure
            # Based on OpenAI's token counting: 3 tokens per message for role/content
            total_tokens += 3

            # Count tool calls if present
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    # Count tool call name and arguments
                    if "name" in tool_call:
                        total_tokens += len(encoding.encode(str(tool_call["name"])))
                    if "arguments" in tool_call:
                        total_tokens += len(encoding.encode(str(tool_call["arguments"])))
                    # Add formatting overhead for tool call structure
                    total_tokens += 5

            # Count tool_call_id if present (for tool result messages)
            if message.tool_call_id:
                total_tokens += len(encoding.encode(message.tool_call_id))
                total_tokens += 2  # formatting overhead

        # Add conversation-level formatting overhead
        total_tokens += 3

        return total_tokens

    def count_tools(self, tools: list["ToolDefinition"]) -> int:
        """Count tokens for tool definitions.

        Args:
            tools: List of tool definitions

        Returns:
            Total tokens for all tool definitions
        """
        if not tools:
            return 0

        # Use cl100k_base encoding for tool definitions
        encoding = tiktoken.get_encoding("cl100k_base")
        total_tokens = 0

        for tool in tools:
            # Count name and description
            total_tokens += len(encoding.encode(tool.name))
            total_tokens += len(encoding.encode(tool.description))

            # Count parameters schema (convert to string representation)
            if tool.parameters:
                params_str = str(tool.parameters)
                total_tokens += len(encoding.encode(params_str))

            # Add formatting overhead for tool definition structure
            total_tokens += 10

        return total_tokens


class ContextManager:
    """Manages context window and token budgeting."""

    def __init__(self, counter: TokenCounter, model_info: "ModelInfo") -> None:
        """Initialize context manager.

        Args:
            counter: Token counter instance
            model_info: Information about the model's limits
        """
        self.counter = counter
        self.model_info = model_info
        self.input_tokens_used = 0
        self.output_tokens_used = 0

    def total_tokens(
        self, messages: list[Message], tools: list["ToolDefinition"], response_buffer: int = 1536
    ) -> int:
        """Calculate total tokens needed for request.

        Args:
            messages: Conversation messages
            tools: Available tools
            response_buffer: Tokens reserved for response

        Returns:
            Total tokens needed
        """
        message_tokens = self.counter.count_messages(messages, self.model_info.name)
        tool_tokens = self.counter.count_tools(tools)
        return message_tokens + tool_tokens + response_buffer

    def is_near_limit(self, threshold: float = 0.8) -> bool:
        """Check if context usage is near the limit.

        Args:
            threshold: Fraction of max tokens (0.0 to 1.0)

        Returns:
            True if usage exceeds threshold
        """
        current_usage = self.input_tokens_used + self.output_tokens_used
        limit = int(self.model_info.max_tokens * threshold)
        return current_usage >= limit

    def available_for_response(self) -> int:
        """Calculate tokens available for model response.

        Returns:
            Number of tokens available for response
        """
        used_tokens = self.input_tokens_used + self.output_tokens_used
        return max(0, self.model_info.max_tokens - used_tokens)

    def record_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Record token usage from a model request.

        Args:
            input_tokens: Tokens sent to model (prompt + tools + messages)
            output_tokens: Tokens generated by model (completion)
        """
        self.input_tokens_used += input_tokens
        self.output_tokens_used += output_tokens

    def reset_usage(self) -> None:
        """Reset tracked token usage."""
        self.input_tokens_used = 0
        self.output_tokens_used = 0

    def get_usage_stats(self) -> dict[str, Any]:
        """Get current usage statistics.

        Returns:
            Dictionary with usage statistics
        """
        total_used = self.input_tokens_used + self.output_tokens_used
        max_tokens = self.model_info.max_tokens
        utilization = total_used / max_tokens if max_tokens > 0 else 0
        return {
            "input_tokens": self.input_tokens_used,
            "output_tokens": self.output_tokens_used,
            "total_tokens": total_used,
            "max_tokens": max_tokens,
            "utilization": utilization,
            "available": max(0, max_tokens - total_used),
        }
