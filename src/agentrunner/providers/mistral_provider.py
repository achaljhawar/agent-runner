"""Mistral AI provider implementation for agentrunner."""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import tiktoken
from openai import APIError, APITimeoutError, AsyncOpenAI, OpenAI, RateLimitError

from agentrunner.core.config import AgentConfig
from agentrunner.core.exceptions import ModelResponseError
from agentrunner.core.messages import Message
from agentrunner.core.tool_protocol import ToolCall, ToolDefinition
from agentrunner.providers.base import (
    BaseLLMProvider,
    ModelInfo,
    ProviderConfig,
    ProviderResponse,
    StreamChunk,
)
from agentrunner.providers.registry import ModelRegistry

# Mistral AI base URL (OpenAI-compatible)
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"


class MistralProvider(BaseLLMProvider):
    """Mistral AI LLM provider implementation.

    Supports Mistral Large, Medium, Small, Codestral, Pixtral, and other models
    through OpenAI-compatible API. Handles tool calling, streaming, and context management.
    """

    def __init__(
        self,
        api_key: str,
        config: ProviderConfig,
        base_url: str | None = None,
        max_retries: int = 3,
    ) -> None:
        """Initialize Mistral provider.

        Args:
            api_key: Mistral API key (MISTRAL_API_KEY)
            config: Provider configuration
            base_url: Optional custom API base URL (defaults to MISTRAL_BASE_URL)
            max_retries: Maximum number of retries for rate limits
        """
        super().__init__(api_key, config)
        self.max_retries = max_retries

        # Use custom base URL or default Mistral endpoint
        effective_base_url = base_url or MISTRAL_BASE_URL

        # Initialize sync and async clients with Mistral base URL
        self.client = OpenAI(api_key=api_key, base_url=effective_base_url)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=effective_base_url)

        # Initialize tokenizer for approximate token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("o200k_base")

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        config: AgentConfig | None = None,
    ) -> ProviderResponse:
        """Send messages to Mistral and get response.

        Args:
            messages: Conversation history
            tools: Available tools for the LLM to call
            config: Agent configuration (deprecated, not used)

        Returns:
            ProviderResponse with assistant's reply and usage info

        Raises:
            ModelResponseError: If Mistral returns an error
            TokenLimitExceededError: If input exceeds context window
        """
        # Convert messages to Mistral format
        mistral_messages = self._convert_messages_to_mistral(messages)

        # Convert tools to Mistral format
        mistral_tools = self._convert_tools_to_mistral(tools) if tools else None

        # Prepare API parameters
        api_params: dict[str, Any] = {
            "model": self.config.model,
            "messages": mistral_messages,
            "temperature": self.config.temperature,
        }

        if mistral_tools:
            api_params["tools"] = mistral_tools
            api_params["tool_choice"] = "auto"

        if self.config.max_tokens:
            api_params["max_tokens"] = self.config.max_tokens

        # Call Mistral API with retries
        for attempt in range(self.max_retries):
            try:
                response = await self.async_client.chat.completions.create(**api_params)

                # Parse response
                assistant_message = self._parse_response_message(response)
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                }

                return ProviderResponse(messages=[assistant_message], usage=usage)

            except RateLimitError as e:
                if attempt < self.max_retries - 1:
                    # Exponential backoff with jitter
                    wait_time = (2**attempt) + (0.1 * attempt)
                    await asyncio.sleep(wait_time)
                    continue
                raise ModelResponseError(
                    message=f"Rate limit exceeded after {self.max_retries} retries",
                    provider="mistral",
                    status_code=429,
                ) from e

            except APITimeoutError as e:
                raise ModelResponseError(
                    message="Request timeout",
                    provider="mistral",
                    error_code="E_TIMEOUT",
                ) from e

            except APIError as e:
                status_code = getattr(e, "status_code", None)
                raise ModelResponseError(
                    message=str(e),
                    provider="mistral",
                    status_code=status_code,
                ) from e

        # Should not reach here
        raise ModelResponseError(
            message="Max retries exceeded",
            provider="mistral",
        )

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        config: AgentConfig | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream messages from Mistral incrementally.

        Args:
            messages: Conversation history
            tools: Available tools for the LLM to call
            config: Agent configuration (deprecated, not used)

        Yields:
            StreamChunk objects with incremental response data

        Raises:
            ModelResponseError: If Mistral returns an error
        """
        # Convert messages and tools
        mistral_messages = self._convert_messages_to_mistral(messages)
        mistral_tools = self._convert_tools_to_mistral(tools) if tools else None

        # Prepare API parameters
        api_params: dict[str, Any] = {
            "model": self.config.model,
            "messages": mistral_messages,
            "temperature": self.config.temperature,
            "stream": True,
        }

        if mistral_tools:
            api_params["tools"] = mistral_tools
            api_params["tool_choice"] = "auto"

        if self.config.max_tokens:
            api_params["max_tokens"] = self.config.max_tokens

        try:
            stream = await self.async_client.chat.completions.create(**api_params)

            async for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # Handle text content
                if delta.content:
                    yield StreamChunk(
                        type="token",
                        payload={"content": delta.content},
                    )

                # Handle tool calls
                if delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        if tool_call.function:
                            yield StreamChunk(
                                type="tool_call",
                                payload={
                                    "id": tool_call.id,
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            )

                # Handle finish reason
                if chunk.choices[0].finish_reason:
                    yield StreamChunk(
                        type="status",
                        payload={"finish_reason": chunk.choices[0].finish_reason},
                    )

        except RateLimitError as e:
            yield StreamChunk(
                type="error",
                payload={"error": "rate_limit", "message": str(e)},
            )

        except APITimeoutError as e:
            yield StreamChunk(
                type="error",
                payload={"error": "timeout", "message": str(e)},
            )

        except APIError as e:
            yield StreamChunk(
                type="error",
                payload={"error": "api_error", "message": str(e)},
            )

    def get_model_info(self) -> ModelInfo:
        """Get information about the model from registry.

        Returns:
            ModelInfo with context window size and pricing
        """
        model_spec = ModelRegistry.get_model_spec(self.config.model)
        return model_spec.to_model_info()

    def get_system_prompt(
        self, workspace_root: str, tools: list["ToolDefinition"] | None = None
    ) -> str:
        """Build Mistral-optimized system prompt.

        Uses default prompt sections with Mistral-specific optimizations.

        Args:
            workspace_root: Path to workspace directory
            tools: Optional tool definitions for dynamic tool section

        Returns:
            Formatted system prompt string
        """
        from agentrunner.core.prompts.utils import build_prompt

        return build_prompt(
            workspace_root=workspace_root,
            model_name=self.config.model,
            tool_definitions=tools,
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tokenizer approximation.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens (approximate)
        """
        return len(self.tokenizer.encode(text))

    def _convert_messages_to_mistral(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert agentrunner messages to Mistral format.

        Args:
            messages: agentrunner messages

        Returns:
            List of messages in Mistral format
        """
        mistral_messages = []

        for msg in messages:
            mistral_msg: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }

            # Add tool_call_id and name for tool messages
            if msg.role == "tool":
                if msg.tool_call_id:
                    mistral_msg["tool_call_id"] = msg.tool_call_id
                if msg.meta.get("tool_name"):
                    mistral_msg["name"] = msg.meta["tool_name"]

            # Add tool_calls for assistant messages
            if msg.role == "assistant" and msg.tool_calls:
                formatted_calls = []
                for tc in msg.tool_calls:
                    # Extract function info
                    if "function" in tc:
                        func = tc["function"]
                        func_name = func.get("name", "")
                        func_args = func.get("arguments", "{}")
                    else:
                        func_name = tc.get("name", "")
                        func_args = tc.get("arguments", {})

                    # Ensure arguments is a JSON string
                    if not isinstance(func_args, str):
                        func_args = json.dumps(func_args)

                    formatted_calls.append(
                        {
                            "id": tc.get("id", f"call_{uuid.uuid4()}"),
                            "type": "function",
                            "function": {
                                "name": func_name,
                                "arguments": func_args,
                            },
                        }
                    )

                if formatted_calls:
                    mistral_msg["tool_calls"] = formatted_calls

            mistral_messages.append(mistral_msg)

        return mistral_messages

    def _convert_tools_to_mistral(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Convert agentrunner tool definitions to Mistral format.

        Args:
            tools: agentrunner tool definitions

        Returns:
            List of tools in Mistral format
        """
        mistral_tools = []

        for tool in tools:
            mistral_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            mistral_tools.append(mistral_tool)

        return mistral_tools

    def _parse_response_message(self, response: Any) -> Message:  # noqa: ANN401
        """Parse Mistral response into agentrunner Message.

        Args:
            response: Mistral chat completion response

        Returns:
            agentrunner Message object
        """
        choice = response.choices[0]
        message = choice.message

        # Extract content
        content = message.content or ""

        # Extract tool calls if present
        tool_calls = None
        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                # Parse arguments safely
                try:
                    if isinstance(tc.function.arguments, str):
                        arguments = json.loads(tc.function.arguments)
                    else:
                        arguments = tc.function.arguments
                except json.JSONDecodeError:
                    # If JSON parsing fails, use empty dict
                    arguments = {}

                tool_calls.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": arguments,
                        },
                    }
                )

        return Message(
            id=str(uuid.uuid4()),
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            meta={"ts": datetime.now(UTC).isoformat()},
        )

    def parse_tool_calls(self, assistant_message: Message) -> list[ToolCall]:
        """Parse tool calls from assistant message.

        Args:
            assistant_message: Assistant message with tool_calls

        Returns:
            List of normalized ToolCall objects
        """
        if not assistant_message.tool_calls:
            return []

        tool_calls = []
        for tc in assistant_message.tool_calls:
            # Parse arguments JSON
            try:
                if isinstance(tc["function"]["arguments"], str):
                    arguments = json.loads(tc["function"]["arguments"])
                else:
                    arguments = tc["function"]["arguments"]
            except json.JSONDecodeError:
                # If arguments aren't valid JSON, use empty dict
                arguments = {}

            tool_calls.append(
                ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=arguments,
                )
            )

        return tool_calls
