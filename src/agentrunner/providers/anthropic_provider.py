"""Anthropic (Claude) provider implementation with tool_use normalization."""

import uuid
from collections.abc import AsyncIterator
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import (
    Message as AnthropicMessage,
)
from anthropic.types import (
    MessageParam,
    ToolParam,
)

from agentrunner.core.config import AgentConfig
from agentrunner.core.exceptions import ModelResponseError
from agentrunner.core.messages import Message
from agentrunner.core.tool_protocol import ToolDefinition
from agentrunner.providers.base import (
    BaseLLMProvider,
    ModelInfo,
    ProviderConfig,
    ProviderResponse,
    StreamChunk,
)
from agentrunner.providers.registry import ModelRegistry


class AnthropicProvider(BaseLLMProvider):
    """Anthropic (Claude) provider with tool_use normalization.

    Supports Claude 3.5 Sonnet, Opus, and Haiku models.
    Handles Anthropic's unique requirements:
    - System message as separate parameter
    - tool_use blocks in content array
    - input_schema for tool definitions

    Normalizes all tool calls to Agent Runner's ToolCall format.
    """

    def __init__(self, api_key: str, config: ProviderConfig) -> None:
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key
            config: Provider configuration
        """
        super().__init__(api_key, config)
        self.client = AsyncAnthropic(api_key=api_key)

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        config: AgentConfig | None = None,
    ) -> ProviderResponse:
        """Execute chat completion with Anthropic API.

        Args:
            messages: Conversation history
            tools: Available tools (None if no tools)
            config: Agent configuration

        Returns:
            ProviderResponse with assistant message(s) and usage stats

        Raises:
            ModelResponseError: On API errors
        """
        try:
            # Extract system message (Anthropic requires it separate)
            system_msg, anthropic_messages = self._convert_messages(messages)

            # Convert tools to Anthropic format
            anthropic_tools = []
            if tools:
                anthropic_tools = [self._tool_to_anthropic_format(tool) for tool in tools]

            # Build request parameters
            request_params: dict[str, Any] = {
                "model": self.config.model,
                "messages": anthropic_messages,
                "max_tokens": self.config.max_tokens or 4096,
            }

            if system_msg:
                request_params["system"] = system_msg

            if anthropic_tools:
                request_params["tools"] = anthropic_tools

            if self.config.temperature is not None:
                request_params["temperature"] = self.config.temperature

            # Make API call
            response: AnthropicMessage = await self.client.messages.create(**request_params)

            # Parse response into Agent Runner format
            assistant_messages = self._parse_response(response)

            # Extract usage information
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }

            return ProviderResponse(messages=assistant_messages, usage=usage)

        except Exception as e:
            # Map Anthropic errors to ModelResponseError
            error_msg = f"{type(e).__name__}: {str(e)}" if str(e) else f"{type(e).__name__}"
            status_code = getattr(e, "status_code", None)

            raise ModelResponseError(
                message=f"Anthropic API error: {error_msg}",
                provider="anthropic",
                status_code=status_code,
            ) from e

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        config: AgentConfig | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Execute streaming chat completion.

        Args:
            messages: Conversation history
            tools: Available tools (None if no tools)
            config: Agent configuration

        Yields:
            StreamChunk events (token deltas, tool calls, status updates, errors)

        Raises:
            ModelResponseError: On API errors
        """
        try:
            # Extract system message
            system_msg, anthropic_messages = self._convert_messages(messages)

            # Convert tools to Anthropic format
            anthropic_tools = []
            if tools:
                anthropic_tools = [self._tool_to_anthropic_format(tool) for tool in tools]

            # Build request parameters
            # Note: stream() method doesn't need stream=True parameter
            request_params: dict[str, Any] = {
                "model": self.config.model,
                "messages": anthropic_messages,
                "max_tokens": self.config.max_tokens or 4096,
            }

            if system_msg:
                request_params["system"] = system_msg

            if anthropic_tools:
                request_params["tools"] = anthropic_tools

            if self.config.temperature is not None:
                request_params["temperature"] = self.config.temperature

            # Stream API call
            async with self.client.messages.stream(**request_params) as stream:
                async for event in stream:
                    # Parse stream events
                    chunk = self._parse_stream_event(event)
                    if chunk:
                        yield chunk

        except Exception as e:
            # Emit error event
            error_msg = f"{type(e).__name__}: {str(e)}" if str(e) else f"{type(e).__name__}"
            status_code = getattr(e, "status_code", None)

            yield StreamChunk(
                type="error",
                payload={
                    "error": error_msg,
                    "provider": "anthropic",
                    "status_code": status_code,
                },
            )

    def get_model_info(self) -> ModelInfo:
        """Get model information from registry.

        Returns:
            ModelInfo with name, context window, pricing
        """
        model_spec = ModelRegistry.get_model_spec(self.config.model)
        return model_spec.to_model_info()

    def get_system_prompt(
        self, workspace_root: str, tools: list["ToolDefinition"] | None = None
    ) -> str:
        """Build system prompt.

        Uses default prompt sections with baseline optimizations.

        Args:
            workspace_root: Path to workspace directory
            tools: Optional tool definitions for dynamic tool section

        Returns:
            Formatted system prompt string
        """
        from agentrunner.core.prompts.base import SystemPromptBuilder
        from agentrunner.core.prompts.sections import get_default_sections
        from agentrunner.core.prompts.utils import generate_tool_section_from_definitions

        builder = SystemPromptBuilder()

        # Add default sections
        for section in get_default_sections():
            if section.name == "available_tools" and tools is not None:
                continue
            builder.add_section(section)

        # Add dynamic tool section if provided
        if tools is not None:
            tool_section = generate_tool_section_from_definitions(tools)
            builder.add_section(tool_section)

        # Load custom sections if they exist
        builder.load_custom_sections(workspace_root)

        # Build with variables
        variables = {
            "workspace_root": workspace_root,
            "model_name": self.config.model,
        }

        return builder.build(variables=variables)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text for this provider's model.

        Note: Anthropic doesn't provide a public tokenizer.
        This is an approximation based on character count.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        # Rough approximation: ~4 chars per token for English
        # This is a conservative estimate
        return len(text) // 4

    def _convert_messages(self, messages: list[Message]) -> tuple[str | None, list[MessageParam]]:
        """Convert Agent Runner messages to Anthropic format.

        Anthropic requires:
        - System message as separate parameter
        - Messages must alternate user/assistant
        - Tool results as user messages with tool_result content

        Args:
            messages: agentrunner message list

        Returns:
            Tuple of (system_message, anthropic_messages)
        """
        system_msg = None
        anthropic_messages: list[MessageParam] = []

        for msg in messages:
            if msg.role == "system":
                # Extract system message (only use first one)
                if system_msg is None:
                    system_msg = msg.content
                continue

            if msg.role == "user":
                anthropic_messages.append({"role": "user", "content": msg.content})

            elif msg.role == "assistant":
                # Assistant message may have tool_use blocks
                if msg.tool_calls:
                    # Build content array with text and tool_use blocks
                    content: list[dict[str, Any]] = []

                    # Add text content if present (Anthropic allows empty text before tool_use)
                    # But we must have at least tool_use blocks
                    if msg.content:
                        content.append({"type": "text", "text": msg.content})

                    # Add tool_use blocks
                    for tool_call in msg.tool_calls:
                        content.append(
                            {
                                "type": "tool_use",
                                "id": tool_call["id"],
                                "name": tool_call["name"],
                                "input": tool_call["arguments"],
                            }
                        )

                    # Only append if we have content (tool_use blocks at minimum)
                    if content:
                        anthropic_messages.append({"role": "assistant", "content": content})  # type: ignore[typeddict-item]
                else:
                    # Simple text response - skip if empty
                    if msg.content:
                        anthropic_messages.append({"role": "assistant", "content": msg.content})

            elif msg.role == "tool":
                # Tool results must be sent as user messages
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {  # type: ignore[list-item]
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.content,
                            }
                        ],
                    }
                )

        return system_msg, anthropic_messages

    def _tool_to_anthropic_format(self, tool: ToolDefinition) -> ToolParam:
        """Convert ToolDefinition to Anthropic tool format.

        Args:
            tool: agentrunner tool definition

        Returns:
            Anthropic ToolParam dict
        """
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }

    def _parse_response(self, response: AnthropicMessage) -> list[Message]:
        """Parse Anthropic response into AgentRunner messages.

        Args:
            response: Anthropic API response

        Returns:
            List of AgentRunner Message objects
        """
        messages: list[Message] = []

        # Extract text content and tool_use blocks
        text_parts = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_use_block = block
                tool_calls.append(
                    {
                        "id": tool_use_block.id,
                        "name": tool_use_block.name,
                        "arguments": tool_use_block.input,
                    }
                )

        # Create assistant message
        content = "\n".join(text_parts) if text_parts else ""
        msg = Message(
            id=str(uuid.uuid4()),
            role="assistant",
            content=content,
            tool_calls=tool_calls if tool_calls else None,
        )
        messages.append(msg)

        return messages

    def _parse_stream_event(self, event: Any) -> StreamChunk | None:  # noqa: ANN401
        """Parse streaming event into StreamChunk.

        Args:
            event: Anthropic stream event

        Returns:
            StreamChunk or None if event should be skipped
        """
        event_type = event.type

        if event_type == "content_block_start":
            # Start of content block
            block = event.content_block
            if block.type == "text":
                return StreamChunk(type="status", payload={"status": "streaming_text"})
            if block.type == "tool_use":
                return StreamChunk(
                    type="tool_call",
                    payload={
                        "tool_call_id": block.id,
                        "tool_name": block.name,
                        "status": "start",
                    },
                )

        elif event_type == "content_block_delta":
            # Delta update
            delta = event.delta
            if delta.type == "text_delta":
                return StreamChunk(type="token", payload={"content": delta.text})
            if delta.type == "input_json_delta":
                return StreamChunk(
                    type="tool_call",
                    payload={"json_delta": delta.partial_json},
                )

        elif event_type == "content_block_stop":
            # End of content block
            return StreamChunk(type="status", payload={"status": "block_complete"})

        elif event_type == "message_start":
            # Message started
            usage = event.message.usage
            return StreamChunk(
                type="status",
                payload={
                    "status": "started",
                    "input_tokens": usage.input_tokens,
                },
            )

        elif event_type == "message_delta":
            # Message delta (usage update)
            usage = event.usage
            return StreamChunk(
                type="status",
                payload={
                    "output_tokens": usage.output_tokens,
                },
            )

        elif event_type == "message_stop":
            # Message complete
            return StreamChunk(type="status", payload={"status": "complete"})

        # Skip other event types
        return None
