"""Google Gemini provider implementation."""

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any

from google import genai
from google.api_core import exceptions as google_exceptions
from google.genai import types

from agentrunner.core.config import AgentConfig
from agentrunner.core.exceptions import ModelResponseError, TokenLimitExceededError
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


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider implementation.

    Supports Gemini Pro and Flash models with tool calling, streaming,
    and message format normalization.
    """

    # Map deprecated model names to current API model names
    MODEL_NAME_MAP = {
        "gemini-1.5-pro": "gemini-2.5-pro",
        "gemini-1.5-flash": "gemini-2.5-flash",
        "gemini-pro": "gemini-2.5-pro",  # Legacy name
    }

    def __init__(self, api_key: str, config: ProviderConfig) -> None:
        """Initialize Gemini provider.

        Args:
            api_key: Google API key
            config: Provider configuration
        """
        super().__init__(api_key, config)

        # Use the new google.genai SDK
        self.client = genai.Client(api_key=api_key)

        # Map deprecated model names to current API names
        api_model_name = self.MODEL_NAME_MAP.get(config.model, config.model)
        self._api_model_name = api_model_name

    def _get_thinking_level(self, model: str) -> str | None:
        """Extract thinking level from model ID for Gemini 3 models.

        Args:
            model: Model identifier (e.g., "gemini-3-pro-preview-low")

        Returns:
            Thinking level ("low", "medium", "high") or None
        """
        if "gemini-3" not in model.lower():
            return None

        # Check for thinking level suffix
        if model.endswith("-low"):
            return "low"
        elif model.endswith("-medium"):
            return "medium"
        elif model.endswith("-high"):
            return "high"
        else:
            # Gemini 3 defaults to "high" if not specified
            return "high" if "gemini-3" in model.lower() else None

    def _is_gemini_3(self, model: str) -> bool:
        """Check if model is Gemini 3 (requires thought signatures).

        Args:
            model: Model identifier

        Returns:
            True if Gemini 3 model
        """
        # Be exact: only gemini-3-pro-preview requires thought signatures
        return "gemini-3-pro-preview" in model.lower()

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        config: AgentConfig | None = None,
    ) -> ProviderResponse:
        """Execute chat completion with Gemini.

        Args:
            messages: Conversation history
            tools: Available tools (None if no tools)
            config: Agent configuration

        Returns:
            ProviderResponse with assistant message(s) and usage stats

        Raises:
            ModelResponseError: On API errors
            TokenLimitExceededError: If request exceeds context window
        """
        try:
            # Convert messages to Gemini format
            gemini_contents, system_instruction = self._convert_messages(messages)

            # Convert tools to Gemini format
            gemini_tools = None
            if tools:
                gemini_tools = self._convert_tools(tools)

            # Extract thinking level if applicable (Gemini 3 only)
            thinking_level = self._get_thinking_level(self._api_model_name)

            # Prepare generation config
            config_dict: dict[str, Any] = {
                "temperature": self.config.temperature,
            }

            if self.config.max_tokens:
                config_dict["max_output_tokens"] = self.config.max_tokens

            if system_instruction:
                config_dict["system_instruction"] = system_instruction

            if gemini_tools:
                config_dict["tools"] = gemini_tools

            # Add thinking_level for Gemini 3 models
            if thinking_level:
                # Cast to ThinkingLevel type to satisfy mypy
                config_dict["thinking_config"] = types.ThinkingConfig(
                    thinking_level=thinking_level  # type: ignore[arg-type]
                )

            generation_config = types.GenerateContentConfig(**config_dict)

            # Execute chat completion using new SDK
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self._api_model_name,
                contents=gemini_contents,
                config=generation_config,
            )

            # Parse response
            assistant_message, usage = self._parse_response(response)

            return ProviderResponse(messages=[assistant_message], usage=usage)

        except google_exceptions.InvalidArgument as e:
            if "context_window" in str(e).lower() or "token" in str(e).lower():
                raise TokenLimitExceededError(f"Token limit exceeded: {e}") from e
            raise ModelResponseError(f"Invalid request: {e}") from e
        except google_exceptions.GoogleAPIError as e:
            raise ModelResponseError(f"Gemini API error: {e}") from e
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}" if str(e) else f"{type(e).__name__}"
            raise ModelResponseError(f"Unexpected error: {error_msg}") from e

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        config: AgentConfig | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Execute streaming chat completion with Gemini.

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
            # Convert messages to Gemini format
            gemini_contents, system_instruction = self._convert_messages(messages)

            # Convert tools to Gemini format
            gemini_tools = None
            if tools:
                gemini_tools = self._convert_tools(tools)

            # Extract thinking level if applicable (Gemini 3 only)
            thinking_level = self._get_thinking_level(self._api_model_name)

            # Prepare generation config
            config_dict: dict[str, Any] = {
                "temperature": self.config.temperature,
            }

            if self.config.max_tokens:
                config_dict["max_output_tokens"] = self.config.max_tokens

            if system_instruction:
                config_dict["system_instruction"] = system_instruction

            if gemini_tools:
                config_dict["tools"] = gemini_tools

            # Add thinking_level for Gemini 3 models
            if thinking_level:
                # Cast to ThinkingLevel type to satisfy mypy
                config_dict["thinking_config"] = types.ThinkingConfig(
                    thinking_level=thinking_level  # type: ignore[arg-type]
                )

            generation_config = types.GenerateContentConfig(**config_dict)

            # Execute streaming chat completion using new SDK
            response_stream = self.client.models.generate_content_stream(
                model=self._api_model_name,
                contents=gemini_contents,
                config=generation_config,
            )

            # Process stream chunks
            accumulated_text = ""
            accumulated_calls: list[dict[str, Any]] = []

            for chunk in response_stream:
                if not chunk.candidates:
                    continue

                candidate = chunk.candidates[0]

                # Handle text content - check if content exists and has parts
                if (
                    candidate.content
                    and hasattr(candidate.content, "parts")
                    and candidate.content.parts
                ):
                    for part in candidate.content.parts:
                        if hasattr(part, "text") and part.text:
                            text_delta = part.text[len(accumulated_text) :]
                            accumulated_text = part.text
                            if text_delta:
                                yield StreamChunk(type="token", payload={"content": text_delta})

                        # Handle function calls
                        if hasattr(part, "function_call") and part.function_call:
                            # With new SDK, convert to dict directly
                            args_dict: dict[str, Any] = {}
                            if (
                                hasattr(part.function_call, "args")
                                and part.function_call.args is not None
                            ):
                                args_dict = dict(part.function_call.args)
                            call_data = {
                                "name": part.function_call.name,
                                "args": args_dict,
                            }
                            if call_data not in accumulated_calls:
                                accumulated_calls.append(call_data)
                                yield StreamChunk(
                                    type="tool_call",
                                    payload={
                                        "id": str(uuid.uuid4()),
                                        "name": call_data["name"],
                                        "arguments": call_data["args"],
                                    },
                                )

            # Send completion status
            yield StreamChunk(type="status", payload={"status": "complete"})

        except google_exceptions.GoogleAPIError as e:
            yield StreamChunk(type="error", payload={"error": str(e)})
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}" if str(e) else f"{type(e).__name__}"
            yield StreamChunk(type="error", payload={"error": error_msg})

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
        """Build Gemini-optimized system prompt.

        Uses default prompt sections with Gemini-specific optimizations.

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
        """Count tokens in text for Gemini model.

        Args:
            text: Text to count tokens for

        Returns:
            Token count
        """
        try:
            result = self.client.models.count_tokens(model=self._api_model_name, contents=text)
            if result.total_tokens is not None:
                return int(result.total_tokens)
            return len(text) // 4
        except Exception:
            # Fallback: rough estimate (4 chars per token)
            return len(text) // 4

    def _convert_messages(
        self, messages: list[Message]
    ) -> tuple[list[types.Content] | str, str | None]:
        """Convert AgentRunner messages to Gemini format using SDK types.

        Args:
            messages: agentrunner Message objects

        Returns:
            Tuple of (gemini_contents, system_instruction)
            gemini_contents can be a string (for simple case) or list of Content objects
        """
        system_instruction: str | None = None
        gemini_contents: list[types.Content] = []

        for msg in messages:
            # Extract system instruction separately
            if msg.role == "system":
                system_instruction = msg.content
                continue

            # Convert role
            role = "user" if msg.role in ("user", "tool") else "model"

            # Build parts using SDK types
            parts: list[types.Part] = []

            # Add text content first
            if msg.content:
                parts.append(types.Part(text=msg.content))

            # Add function calls for assistant messages
            if msg.role == "assistant" and msg.tool_calls:
                for idx, call in enumerate(msg.tool_calls):
                    func_call = types.FunctionCall(name=call["name"], args=call["arguments"])

                    part_dict: dict[str, Any] = {"function_call": func_call}

                    # For Gemini 3: Use the thought_signature from the original model response
                    if self._is_gemini_3(self._api_model_name):
                        # Check if this specific call has a thought_signature
                        if "thought_signature" in call:
                            part_dict["thought_signature"] = call["thought_signature"]
                        # Otherwise check if it's stored in message metadata
                        elif msg.meta and "thought_signatures" in msg.meta:
                            signatures = msg.meta["thought_signatures"]
                            if idx < len(signatures) and signatures[idx] is not None:
                                part_dict["thought_signature"] = signatures[idx]

                    parts.append(types.Part(**part_dict))

            # Add function responses for tool messages
            if msg.role == "tool" and msg.tool_call_id:
                # Extract tool name from metadata if available
                tool_name = msg.tool_call_id
                if msg.meta and "tool_name" in msg.meta:
                    tool_name = msg.meta["tool_name"]

                # Don't add text content for tool messages, only function_response
                func_response = types.FunctionResponse(
                    name=tool_name, response={"result": msg.content}
                )
                parts.append(types.Part(function_response=func_response))

            if parts:
                gemini_contents.append(types.Content(role=role, parts=parts))

        # If there's only one user message with text, return it as a string
        if len(gemini_contents) == 1 and gemini_contents[0].role == "user":
            first_content = gemini_contents[0]
            if (
                first_content.parts
                and len(first_content.parts) == 1
                and first_content.parts[0].text
            ):
                return first_content.parts[0].text, system_instruction

        return gemini_contents, system_instruction

    def _sanitize_schema_for_gemini(self, schema: dict[str, Any]) -> Any:
        """Remove JSON Schema fields not supported by Gemini API.

        Gemini doesn't support: additionalProperties, $schema, $defs, etc.

        Args:
            schema: JSON Schema object

        Returns:
            Sanitized schema with only Gemini-supported fields
        """
        if not isinstance(schema, dict):
            return schema

        # Fields to remove (not supported by Gemini)
        unsupported_fields = {
            "additionalProperties",
            "additional_properties",
            "$schema",
            "$defs",
            "$id",
            "definitions",
            "patternProperties",
            "minProperties",
            "maxProperties",
            "dependencies",
            "propertyNames",
            "const",
            "examples",
            "default",
            "title",
        }

        sanitized = {}
        for key, value in schema.items():
            # Skip unsupported fields
            if key in unsupported_fields:
                continue

            # Recursively sanitize nested objects
            if isinstance(value, dict):
                sanitized[key] = self._sanitize_schema_for_gemini(value)
            elif isinstance(value, list):
                sanitized_list = [
                    self._sanitize_schema_for_gemini(item) if isinstance(item, dict) else item
                    for item in value
                ]
                sanitized[key] = sanitized_list
            else:
                sanitized[key] = value

        return sanitized

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[types.Tool]:
        """Convert agentrunner tools to Gemini function declarations.

        Args:
            tools: agentrunner ToolDefinition objects

        Returns:
            List of Gemini Tool objects
        """
        function_declarations = []

        for tool in tools:
            # Sanitize parameters to remove unsupported JSON Schema fields
            sanitized_params = self._sanitize_schema_for_gemini(tool.parameters)

            function_declaration = types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=sanitized_params,
            )
            function_declarations.append(function_declaration)

        # Wrap in a Tool object
        return [types.Tool(function_declarations=function_declarations)]

    def _parse_response(self, response: Any) -> tuple[Message, dict[str, int]]:  # noqa: ANN401
        """Parse Gemini response to agentrunner format.

        Args:
            response: Gemini response object

        Returns:
            Tuple of (Message, usage dict)
        """
        # Extract text and tool calls
        text_content = ""
        tool_calls: list[dict[str, Any]] = []
        thought_signatures: list[bytes | None] = []  # Store thought signatures

        if response.candidates:
            candidate = response.candidates[0]

            # Check if content exists and has parts
            if candidate.content and hasattr(candidate.content, "parts"):
                for part in candidate.content.parts:
                    # Extract text
                    if hasattr(part, "text") and part.text:
                        text_content += part.text

                    # Extract function calls with thought signatures
                    if hasattr(part, "function_call") and part.function_call:
                        # With new SDK, convert to dict directly
                        args_dict = (
                            dict(part.function_call.args)
                            if hasattr(part.function_call, "args")
                            else {}
                        )

                        tool_call_dict = {
                            "id": str(uuid.uuid4()),  # Gemini doesn't provide IDs
                            "name": part.function_call.name,
                            "arguments": args_dict,
                        }

                        # Extract thought_signature if present (Gemini 3)
                        thought_sig = None
                        if hasattr(part, "thought_signature") and part.thought_signature:
                            thought_sig = part.thought_signature
                            # Store in the tool call metadata
                            tool_call_dict["thought_signature"] = thought_sig

                        tool_calls.append(tool_call_dict)
                        thought_signatures.append(thought_sig)

        # Build message with thought signatures in metadata
        meta: dict[str, Any] = {}
        if thought_signatures and any(sig is not None for sig in thought_signatures):
            meta["thought_signatures"] = thought_signatures

        message = Message(
            id=str(uuid.uuid4()),
            role="assistant",
            content=text_content,
            tool_calls=tool_calls if tool_calls else None,
            meta=meta,
        )

        # Extract usage stats - handle both old and new SDK formats
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", None)
            completion_tokens = getattr(response.usage_metadata, "candidates_token_count", None)
            total_tokens = getattr(response.usage_metadata, "total_token_count", None)

            # Ensure we have integers, not None
            usage["prompt_tokens"] = int(prompt_tokens) if prompt_tokens is not None else 0
            usage["completion_tokens"] = (
                int(completion_tokens) if completion_tokens is not None else 0
            )
            usage["total_tokens"] = int(total_tokens) if total_tokens is not None else 0

        return message, usage
