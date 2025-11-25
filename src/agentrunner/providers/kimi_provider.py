"""Kimi (Moonshot AI) provider implementation for agentrunner."""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import tiktoken
from openai import (
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

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


class KimiProvider(BaseLLMProvider):
    """Kimi (Moonshot AI) provider implementation.

    Supports Kimi K2 and K2 thinking models via OpenAI-compatible API.
    Uses Moonshot's API endpoint with OpenAI SDK.
    """

    KIMI_BASE_URL = "https://api.moonshot.ai/v1"

    def __init__(
        self,
        api_key: str,
        config: ProviderConfig,
        base_url: str | None = None,
        max_retries: int = 3,
    ) -> None:
        """Initialize Kimi provider.

        Args:
            api_key: Moonshot API key
            config: Provider configuration
            base_url: Optional custom API base URL (defaults to Moonshot endpoint)
            max_retries: Maximum number of retries for rate limits
        """
        super().__init__(api_key, config)
        self.max_retries = max_retries

        # Use Moonshot API endpoint
        api_base = base_url or self.KIMI_BASE_URL

        # Initialize sync and async clients
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=api_base)

        # Use cl100k_base tokenizer (compatible with Kimi models)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        config: AgentConfig | None = None,
    ) -> ProviderResponse:
        """Send messages to Kimi and get response.

        Args:
            messages: Conversation history
            tools: Available tools for the LLM to call
            config: Agent configuration (deprecated, not used)

        Returns:
            ProviderResponse with assistant's reply and usage info

        Raises:
            ModelResponseError: If Kimi returns an error
            TokenLimitExceededError: If input exceeds context window
        """
        # Convert messages to OpenAI format
        kimi_messages = self._convert_messages_to_kimi(messages)

        # Convert tools to OpenAI format
        kimi_tools = self._convert_tools_to_kimi(tools) if tools else None

        # Prepare API parameters
        api_params: dict[str, Any] = {
            "model": self.config.model,
            "messages": kimi_messages,
            "temperature": self.config.temperature,
        }

        if kimi_tools:
            api_params["tools"] = kimi_tools
            api_params["tool_choice"] = "auto"

        if self.config.max_tokens:
            api_params["max_tokens"] = self.config.max_tokens

        # Call Kimi API with retries
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

            except AuthenticationError as e:
                raise ModelResponseError(
                    message=f"Authentication failed: {str(e)}. Check MOONSHOT_API_KEY environment variable.",
                    provider="kimi",
                    status_code=401,
                ) from e

            except RateLimitError as e:
                if attempt < self.max_retries - 1:
                    # Exponential backoff with jitter
                    wait_time = (2**attempt) + (0.1 * attempt)
                    await asyncio.sleep(wait_time)
                    continue
                raise ModelResponseError(
                    message=f"Rate limit exceeded after {self.max_retries} retries",
                    provider="kimi",
                    status_code=429,
                ) from e

            except APITimeoutError as e:
                raise ModelResponseError(
                    message="Request timeout",
                    provider="kimi",
                    error_code="E_TIMEOUT",
                ) from e

            except APIError as e:
                status_code = getattr(e, "status_code", None)
                raise ModelResponseError(
                    message=str(e),
                    provider="kimi",
                    status_code=status_code,
                ) from e

        # Should not reach here
        raise ModelResponseError(
            message="Max retries exceeded",
            provider="kimi",
        )

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        config: AgentConfig | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream messages from Kimi incrementally.

        Args:
            messages: Conversation history
            tools: Available tools for the LLM to call
            config: Agent configuration (deprecated, not used)

        Yields:
            StreamChunk objects with incremental response data

        Raises:
            ModelResponseError: If Kimi returns an error
        """
        # Convert messages and tools
        kimi_messages = self._convert_messages_to_kimi(messages)
        kimi_tools = self._convert_tools_to_kimi(tools) if tools else None

        # Prepare API parameters
        api_params: dict[str, Any] = {
            "model": self.config.model,
            "messages": kimi_messages,
            "temperature": self.config.temperature,
            "stream": True,
        }

        if kimi_tools:
            api_params["tools"] = kimi_tools
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

        except AuthenticationError as e:
            yield StreamChunk(
                type="error",
                payload={
                    "error": "authentication_error",
                    "message": f"Authentication failed: {str(e)}. Check MOONSHOT_API_KEY environment variable.",
                },
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
        """Build Kimi-optimized system prompt.

        Uses default prompt sections with Kimi-specific optimizations.

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
        """Count tokens in text using cl100k_base tokenizer.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))

    def _convert_messages_to_kimi(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert agentrunner messages to Kimi format.

        Args:
            messages: agentrunner messages

        Returns:
            List of messages in Kimi (OpenAI-compatible) format
        """
        kimi_messages = []

        for msg in messages:
            kimi_msg: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }

            # Add tool_call_id and name for tool messages
            if msg.role == "tool":
                if msg.tool_call_id:
                    kimi_msg["tool_call_id"] = msg.tool_call_id
                if msg.meta.get("tool_name"):
                    kimi_msg["name"] = msg.meta["tool_name"]

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
                    kimi_msg["tool_calls"] = formatted_calls

            kimi_messages.append(kimi_msg)

        return kimi_messages

    def _convert_tools_to_kimi(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Convert agentrunner tool definitions to Kimi format.

        Args:
            tools: agentrunner tool definitions

        Returns:
            List of tools in Kimi (OpenAI-compatible) format
        """
        kimi_tools = []

        for tool in tools:
            kimi_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            kimi_tools.append(kimi_tool)

        return kimi_tools

    def _parse_response_message(self, response: Any) -> Message:  # noqa: ANN401
        """Parse Kimi response into agentrunner Message.

        Args:
            response: Kimi chat completion response

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
                arguments = {}

            tool_calls.append(
                ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=arguments,
                )
            )

        return tool_calls
