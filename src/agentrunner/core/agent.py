"""AgentRunnerAgent: Main agent orchestrator with agentic loop.

Implements the core agent loop per INTERFACES/AGENT.md:
message history → provider response → tool execution → updated history
"""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from agentrunner.core.config import AgentConfig
from agentrunner.core.events import StreamEvent
from agentrunner.core.exceptions import AgentRunnerException
from agentrunner.core.logger import AgentRunnerLogger
from agentrunner.core.messages import Message, MessageHistory
from agentrunner.core.session import SessionManager
from agentrunner.core.tokens import ContextManager
from agentrunner.core.tool_protocol import ToolCall, ToolResult, format_tool_result
from agentrunner.core.workspace import Workspace
from agentrunner.security.confirmation import ActionDescriptor, ConfirmationLevel

if TYPE_CHECKING:
    from agentrunner.core.events import EventBus
    from agentrunner.providers.base import BaseLLMProvider
    from agentrunner.security.command_validator import CommandValidator
    from agentrunner.security.confirmation import ConfirmationService
    from agentrunner.tools.base import ToolRegistry


class AgentState(Enum):
    """Agent lifecycle states."""

    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING_TOOLS = "executing_tools"
    COMPLETED = "completed"
    ABORTING = "aborting"
    ABORTED = "aborted"
    ERROR = "error"


@dataclass
class AssistantResult:
    """Result from process_message."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    rounds: int = 0
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    stopped_reason: str = "completed"


class AgentRunnerAgent:
    """Main agent orchestrator with agentic loop.

    Manages conversation history, executes tools, and interfaces with LLM providers.
    """

    def __init__(
        self,
        provider: "BaseLLMProvider",
        workspace: Workspace,
        config: AgentConfig,
        context_manager: ContextManager,
        logger: AgentRunnerLogger,
        tools: "ToolRegistry | None" = None,
        confirmation: "ConfirmationService | None" = None,
        command_validator: "CommandValidator | None" = None,
        event_bus: "EventBus | None" = None,
    ) -> None:
        """Initialize agent with dependencies.

        Args:
            provider: LLM provider implementation
            workspace: Workspace for file operations
            config: Agent configuration
            context_manager: Token counting and context management
            logger: Structured logger
            tools: Tool registry (optional)
            confirmation: Confirmation service (optional, defaults to auto-approve)
            command_validator: Command validator (optional, defaults to permissive)
            event_bus: Event bus for streaming (optional)
        """
        self.provider = provider
        self.workspace = workspace
        self.config = config
        self.context_manager = context_manager
        self.logger = logger
        self.tools = tools
        self.confirmation = confirmation
        self.command_validator = command_validator
        self.event_bus = event_bus

        self.history = MessageHistory()
        self.state = AgentState.IDLE
        self._abort_requested = False
        self._system_prompt_initialized = False
        self._created_at = datetime.now(UTC).isoformat()

    async def process_message(self, user_message: str) -> AssistantResult:
        """Process a user message through the agentic loop.

        Args:
            user_message: User's input message

        Returns:
            AssistantResult with final response and metadata

        Raises:
            TokenLimitExceededError: If context window is exhausted
            AgentRunnerException: For other agent errors
        """
        self.logger.info("Processing user message", message_length=len(user_message))
        self._abort_requested = False

        # Initialize system prompt FIRST (must be message[0] in history)
        if not self._system_prompt_initialized:
            tool_defs = self.tools.get_definitions() if self.tools else []

            # Build system prompt with actual tools
            system_prompt = self.provider.get_system_prompt(
                str(self.workspace.root_path), tools=tool_defs
            )
            self.history.add_system(system_prompt)
            self._system_prompt_initialized = True

            self.logger.debug("System prompt initialized", tool_count=len(tool_defs))

        # Add user message to history (after system prompt)
        self.history.add_user(user_message)

        # Emit user_message event for UI (v2 architecture)
        if self.event_bus:
            user_msg = self.history.messages[-1]
            self.event_bus.publish(
                StreamEvent(
                    type="user_message",
                    data={
                        "message_id": user_msg.id,  # For worker event_loader.py
                        "id": user_msg.id,  # For UI (expects 'id' field)
                        "content": user_msg.content,
                        "role": "user",
                    },
                    model_id=self.provider.config.model,
                    ts=datetime.now(UTC).isoformat(),
                )
            )

        rounds = 0
        final_content = ""

        try:
            while rounds < self.config.max_rounds:
                if self._abort_requested:
                    self.state = AgentState.ABORTED
                    self.logger.info("Agent aborted by user", rounds=rounds)
                    return AssistantResult(
                        content=final_content,
                        rounds=rounds,
                        input_tokens=0,
                        output_tokens=0,
                        total_tokens=0,
                        stopped_reason="aborted",
                    )

                # Get all available tools
                tool_defs = self.tools.get_definitions() if self.tools else []
                self.logger.info(
                    "Calling provider with tools",
                    tool_count=len(tool_defs),
                    tool_names=(
                        [t["name"] if isinstance(t, dict) else t.name for t in tool_defs]
                        if tool_defs
                        else []
                    ),
                )

                # Get provider response
                self.state = AgentState.THINKING
                self.logger.debug(
                    "Requesting provider response",
                    round=rounds + 1,
                    tool_count=len(tool_defs),
                )

                # Emit status event
                if self.event_bus:
                    self.event_bus.publish(
                        StreamEvent(
                            type="status_update",
                            data={"status": "thinking", "detail": "Waiting for LLM response"},
                            model_id=self.provider.config.model,
                            ts=datetime.now(UTC).isoformat(),
                        )
                    )

                if not hasattr(self.provider, "chat"):
                    raise AgentRunnerException("Provider not yet implemented")

                provider_response = await self.provider.chat(
                    messages=self.history.get(),
                    tools=tool_defs,
                )

                # Record token usage immediately after provider response
                usage = provider_response.usage
                self.context_manager.record_usage(
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                )

                assistant_msg = provider_response.messages[-1]
                self.history.add(assistant_msg)
                final_content = assistant_msg.content or ""

                # Check for tool calls
                if not assistant_msg.tool_calls:
                    self.state = AgentState.COMPLETED
                    self.logger.info(
                        "Agent completed",
                        rounds=rounds + 1,
                        reason="no_tool_calls",
                    )

                    # Calculate cost (usage already extracted above)
                    model_info = self.provider.get_model_info()
                    input_cost_per_1k = model_info.pricing.get("input_per_1k", 0)
                    output_cost_per_1k = model_info.pricing.get("output_per_1k", 0)
                    cost = (usage.get("prompt_tokens", 0) / 1000.0) * input_cost_per_1k + (
                        usage.get("completion_tokens", 0) / 1000.0
                    ) * output_cost_per_1k

                    if self.event_bus:
                        import time

                        # Publish assistant message first
                        self.event_bus.publish(
                            StreamEvent(
                                type="assistant_message",
                                data={
                                    "message_id": str(uuid.uuid4()),
                                    "content": final_content,
                                    "metadata": {
                                        "rounds": rounds + 1,
                                        "input_tokens": usage.get("prompt_tokens", 0),
                                        "output_tokens": usage.get("completion_tokens", 0),
                                        "total_tokens": usage.get("total_tokens", 0),
                                        "cost": cost,
                                    },
                                },
                                model_id=self.provider.config.model,
                                ts=datetime.now(UTC).isoformat(),
                            )
                        )

                        # Tiny delay ensures status_update gets a later timestamp
                        time.sleep(0.005)

                        # Publish status update after
                        self.event_bus.publish(
                            StreamEvent(
                                type="status_update",
                                data={"status": "idle", "detail": "Completed"},
                                model_id=self.provider.config.model,
                                ts=datetime.now(UTC).isoformat(),
                            )
                        )

                    return AssistantResult(
                        content=final_content,
                        rounds=rounds + 1,
                        total_tokens=usage.get("total_tokens", 0),
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0),
                        stopped_reason="completed",
                    )

                # Parse tool calls from dict format to ToolCall objects
                tool_calls = []
                for tc in assistant_msg.tool_calls:
                    # Handle nested OpenAI format: {id, type, function: {name, arguments}}
                    if "function" in tc:
                        func = tc["function"]
                        args = func.get("arguments", {})
                        # Parse JSON string arguments if needed
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                args = {}
                        tool_calls.append(
                            ToolCall(
                                id=tc.get("id", ""),
                                name=func.get("name", ""),
                                arguments=args,
                            )
                        )
                    # Handle flat agentrunner format: {id, name, arguments}
                    else:
                        tool_calls.append(
                            ToolCall(
                                id=tc.get("id", ""),
                                name=tc.get("name", ""),
                                arguments=tc.get("arguments", {}),
                            )
                        )

                # Execute tools
                self.state = AgentState.EXECUTING_TOOLS

                # Emit status event
                if self.event_bus:
                    self.event_bus.publish(
                        StreamEvent(
                            type="status_update",
                            data={
                                "status": "executing_tools",
                                "detail": f"Executing {len(tool_calls)} tool(s)",
                            },
                            model_id=self.provider.config.model,
                            ts=datetime.now(UTC).isoformat(),
                        )
                    )

                await self._execute_tool_loop(tool_calls)

                rounds += 1

            # Max rounds reached
            self.state = AgentState.COMPLETED
            self.logger.warn("Max rounds reached", max_rounds=self.config.max_rounds)

            if self.event_bus:
                self.event_bus.publish(
                    StreamEvent(
                        type="assistant_message",
                        data={
                            "message_id": str(uuid.uuid4()),
                            "content": final_content,
                            "metadata": {
                                "rounds": rounds,
                                "input_tokens": 0,
                                "output_tokens": 0,
                                "total_tokens": 0,
                                "cost": 0.0,
                            },
                        },
                        model_id=self.provider.config.model,
                        ts=datetime.now(UTC).isoformat(),
                    )
                )
                self.event_bus.publish(
                    StreamEvent(
                        type="status_update",
                        data={"status": "idle", "detail": "Max rounds reached"},
                        model_id=self.provider.config.model,
                        ts=datetime.now(UTC).isoformat(),
                    )
                )

            return AssistantResult(
                content=final_content,
                rounds=rounds,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                stopped_reason="max_rounds",
            )

        except Exception as e:
            self.state = AgentState.ERROR
            error_msg = f"{type(e).__name__}: {str(e)}" if str(e) else f"{type(e).__name__}"
            self.logger.error("Agent error", error=error_msg, rounds=rounds)

            # Publish error event to update UI
            if self.event_bus:
                self.event_bus.publish(
                    StreamEvent(
                        type="error",
                        data={
                            "error": error_msg,
                            "error_type": type(e).__name__,
                            "details": str(e) if str(e) else None,
                            "message": error_msg,  # Backward compatibility
                        },
                        model_id=self.provider.config.model,
                        ts=datetime.now(UTC).isoformat(),
                    )
                )

            raise

    def _ensure_prior_read(self, call: ToolCall) -> bool:
        """Ensure file has been read before editing (read-before-edit safety).

        Args:
            call: Tool call that requires read-first

        Returns:
            True if file was previously read, False if violation (error added to history)
        """
        file_path = call.arguments.get("file_path")
        if not file_path:
            # No file_path argument, safety check doesn't apply
            return True

        # Look for successful read operations on this file by matching tool calls with results
        messages = self.history.get()
        has_read = False

        for i, msg in enumerate(messages):
            # Look for assistant messages with read_file tool calls
            if msg.role == "assistant" and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if (
                        tool_call.get("name") == "read_file"
                        and tool_call.get("arguments", {}).get("file_path") == file_path
                    ):
                        # Found a read_file call for this file, check if it succeeded
                        tool_call_id = tool_call.get("id")
                        if tool_call_id and self._find_successful_tool_result(
                            messages[i + 1 :], tool_call_id
                        ):
                            has_read = True
                            break
            if has_read:
                break

        if has_read:
            return True

        # Read-before-edit violation
        self.logger.warn(
            "Read-before-edit violation",
            tool=call.name,
            file=file_path,
        )
        error_result = ToolResult(
            success=False,
            error=f"Must read {file_path} before editing",
            error_code="E_VALIDATION",
        )
        tool_msg_dict = format_tool_result(call.id, error_result)
        self.history.add_tool(
            content=tool_msg_dict["content"],
            tool_call_id=call.id,
        )
        return False

    def _find_successful_tool_result(self, messages: list[Message], tool_call_id: str) -> bool:
        """Find successful tool result for the given tool call ID.

        Args:
            messages: List of messages to search (typically from current position forward)
            tool_call_id: Tool call ID to match

        Returns:
            True if successful tool result found, False otherwise
        """
        for msg in messages:
            if msg.role == "tool" and msg.tool_call_id == tool_call_id:
                # Found the tool result message, check if it indicates success
                content = msg.content or ""
                content_lower = content.lower()

                # Check for error indicators
                if any(
                    error_phrase in content_lower
                    for error_phrase in [
                        "error:",
                        "failed",
                        "not found",
                        "permission denied",
                        "no such file",
                    ]
                ):
                    return False

                # For read operations, success is indicated by line-numbered content
                has_line_numbers = any(f"{i}|" in content for i in range(1, 6))
                return has_line_numbers

        return False

    async def process_message_stream(self, user_message: str) -> AsyncIterator[dict[str, Any]]:
        """Process message with streaming events.

        Args:
            user_message: User's input message

        Yields:
            StreamEvent dicts with type and payload
        """
        if not self.event_bus:
            # Fallback to non-streaming if no event bus
            result = await self.process_message(user_message)
            yield {
                "type": "assistant_message",
                "payload": {"content": result.content, "rounds": result.rounds},
            }
            yield {"type": "completed", "payload": {"reason": result.stopped_reason}}
            return

        self.logger.info("Processing message (streaming)", message_length=len(user_message))

        # Start consuming events from event bus
        async def consume_events() -> AsyncIterator[dict[str, Any]]:
            if self.event_bus is None:
                return
            async for event in self.event_bus:
                yield {
                    "type": event.type,
                    "payload": event.data,
                    "ts": event.ts,
                }

        # Process message (which will publish events to bus)
        process_task = asyncio.create_task(self.process_message(user_message))

        # Stream events while processing
        async for event_dict in consume_events():
            yield event_dict

            # Stop consuming when completed
            if event_dict["type"] == "completed":
                break

        # Wait for process to complete
        await process_task

    async def _execute_tool_loop(self, tool_calls: list[ToolCall]) -> None:
        """Execute a batch of tool calls with safety checks.

        Args:
            tool_calls: List of tool calls to execute
        """
        self.logger.debug("Executing tools", tool_count=len(tool_calls))

        for call in tool_calls:
            # Check if tool exists
            if self.tools and not self.tools.has(call.name):
                self.logger.warn("Unknown tool", tool_name=call.name)
                error_result = ToolResult(
                    success=False,
                    error=f"Unknown tool: {call.name}",
                    error_code="E_TOOL_UNKNOWN",
                )
                tool_msg_dict = format_tool_result(call.id, error_result)
                self.history.add_tool(
                    content=tool_msg_dict["content"],
                    tool_call_id=call.id,
                )
                continue

            # Security checks
            if self.tools:
                tool_instance = self.tools.get(call.name)
                if tool_instance is None:
                    self.logger.warn(f"Tool {call.name} not found in registry")
                    continue
                tool_def = tool_instance.get_definition()

                # Check read-before-edit requirement
                if tool_def.safety.get("requires_read_first", False):
                    if not self._ensure_prior_read(call):
                        continue

                # Check if confirmation required
                if (
                    tool_def.safety.get("requires_confirmation", False)
                    and self.confirmation
                    and hasattr(self.confirmation, "approve")
                ):
                    action = ActionDescriptor(
                        level=ConfirmationLevel.DESTRUCTIVE,
                        operation=call.name,
                        description=f"Execute {call.name}",
                        target=str(call.arguments.get("file_path", call.arguments)),
                    )

                    if not self.confirmation.approve(action):
                        self.logger.info("User denied operation", tool=call.name)
                        error_result = ToolResult(
                            success=False,
                            error="User denied operation",
                            error_code="E_CONFIRMED_DENY",
                        )
                        tool_msg_dict = format_tool_result(call.id, error_result)
                        self.history.add_tool(
                            content=tool_msg_dict["content"],
                            tool_call_id=call.id,
                        )
                        continue

                # Check command safety for bash tool
                if (
                    call.name == "bash"
                    and self.command_validator
                    and hasattr(self.command_validator, "is_safe_tool")
                    and not self.command_validator.is_safe_tool(call)
                ):
                    self.logger.warn("Unsafe command blocked", tool=call.name)
                    error_result = ToolResult(
                        success=False,
                        error="Command not whitelisted or dangerous",
                        error_code="E_UNSAFE",
                    )
                    tool_msg_dict = format_tool_result(call.id, error_result)
                    self.history.add_tool(
                        content=tool_msg_dict["content"],
                        tool_call_id=call.id,
                    )
                    continue

            # Execute tool
            try:
                if not self.tools:
                    raise AgentRunnerException("ToolRegistry not initialized")

                # Emit tool start event
                if self.event_bus:
                    self.event_bus.publish(
                        StreamEvent(
                            type="tool_call_started",
                            data={
                                "tool_name": call.name,
                                "call_id": call.id,
                                "arguments": call.arguments,
                            },
                            model_id=self.provider.config.model,
                            ts=datetime.now(UTC).isoformat(),
                        )
                    )

                import time

                start_time = time.time()

                result = await asyncio.wait_for(
                    self.tools.execute(call),
                    timeout=self.config.tool_timeout_s,
                )

                duration_ms = int((time.time() - start_time) * 1000)

                self.logger.info(
                    "Tool executed",
                    tool_name=call.name,
                    success=result.success,
                    duration_ms=duration_ms,
                )

                # Emit tool complete event
                if self.event_bus:
                    # Include FULL output/error for event sourcing
                    output_data = ""
                    if result.output:
                        output_data = result.output
                    elif result.error:
                        output_data = result.error

                    self.event_bus.publish(
                        StreamEvent(
                            type="tool_call_completed",
                            data={
                                "tool_name": call.name,
                                "call_id": call.id,
                                "success": result.success,
                                "output": output_data,
                                "duration_ms": duration_ms,
                                "files_changed": result.files_changed or [],
                            },
                            model_id=self.provider.config.model,
                            ts=datetime.now(UTC).isoformat(),
                        )
                    )

                tool_msg_dict = format_tool_result(call.id, result)
                self.history.add_tool(
                    content=tool_msg_dict["content"],
                    tool_call_id=call.id,
                    tool_name=call.name,
                )

            except TimeoutError:
                self.logger.error(
                    "Tool timeout",
                    tool_name=call.name,
                    timeout_s=self.config.tool_timeout_s,
                )
                error_result = ToolResult(
                    success=False,
                    error=f"Tool execution timed out after {self.config.tool_timeout_s}s",
                    error_code="E_TIMEOUT",
                )

                # Emit tool complete event with failure
                if self.event_bus:
                    self.event_bus.publish(
                        StreamEvent(
                            type="tool_call_completed",
                            data={
                                "tool_name": call.name,
                                "call_id": call.id,
                                "success": False,
                                "output": error_result.error,
                                "duration_ms": int(self.config.tool_timeout_s * 1000),
                            },
                            model_id=self.provider.config.model,
                            ts=datetime.now(UTC).isoformat(),
                        )
                    )

                tool_msg_dict = format_tool_result(call.id, error_result)
                self.history.add_tool(
                    content=tool_msg_dict["content"],
                    tool_call_id=call.id,
                )

            except Exception as e:
                self.logger.error("Tool execution failed", tool_name=call.name, error=str(e))
                error_result = ToolResult(
                    success=False,
                    error=f"Tool execution failed: {e}",
                    error_code="E_VALIDATION",
                )

                # Emit tool complete event with failure
                if self.event_bus:
                    import time

                    duration_ms = (
                        int((time.time() - start_time) * 1000) if "start_time" in locals() else 0
                    )
                    self.event_bus.publish(
                        StreamEvent(
                            type="tool_call_completed",
                            data={
                                "tool_name": call.name,
                                "call_id": call.id,
                                "success": False,
                                "output": str(e),
                                "duration_ms": duration_ms,
                            },
                            model_id=self.provider.config.model,
                            ts=datetime.now(UTC).isoformat(),
                        )
                    )

                tool_msg_dict = format_tool_result(call.id, error_result)
                self.history.add_tool(
                    content=tool_msg_dict["content"],
                    tool_call_id=call.id,
                )

    def abort(self) -> None:
        """Request agent to abort current processing."""
        self.logger.info("Abort requested")
        self._abort_requested = True
        self.state = AgentState.ABORTING

    def reset(self) -> None:
        """Reset agent state and clear history."""
        self.logger.info("Resetting agent")
        self.history.clear()

        # Re-add system prompt from provider
        system_prompt = self.provider.get_system_prompt(str(self.workspace.root_path))
        self.history.add_system(system_prompt)

        self.state = AgentState.IDLE
        self._abort_requested = False

    def set_configuration(self, config: AgentConfig) -> None:
        """Update agent configuration.

        Args:
            config: New configuration
        """
        self.logger.info("Updating configuration", max_rounds=config.max_rounds)
        self.config = config

        # Note: System prompt is provider-specific and rebuilt on reset()
        # Model changes require creating a new agent with new provider

    def get_messages(self) -> list[Message]:
        """Get all messages in conversation history.

        Returns:
            List of messages
        """
        return self.history.get()

    async def save_session(self, session_id: str) -> None:
        """Save session to disk.

        Args:
            session_id: Unique session identifier

        Raises:
            AgentRunnerException: If save fails
        """
        self.logger.info("Saving session", session_id=session_id)

        manager = SessionManager(self.workspace)

        meta = {
            "state": self.state.value,
            "total_rounds": len([m for m in self.history.get() if m.role == "user"]),
        }

        await manager.save(
            session_id=session_id,
            messages=self.history.get(),
            config=self.config,
            meta=meta,
        )

        self.logger.info("Session saved", session_id=session_id)

    async def load_session(self, session_id: str) -> None:
        """Load session from disk.

        Args:
            session_id: Session identifier to load

        Raises:
            AgentRunnerException: If load fails
            FileNotFoundError: If session doesn't exist
        """
        self.logger.info("Loading session", session_id=session_id)

        manager = SessionManager(self.workspace)

        messages, config, _meta = await manager.load(session_id)

        # Replace current history with loaded messages
        self.history.clear()
        for msg in messages:
            self.history.add(msg)

        # Update config
        self.config = config

        self.logger.info(
            "Session loaded",
            session_id=session_id,
            message_count=len(messages),
        )

    def snapshot(self) -> dict[str, Any]:
        """Get serializable agent state.

        Pure method with no storage dependencies.
        Backend is responsible for persisting this.

        Returns:
            dict: Agent state (messages, tokens, rounds, config)
        """
        return {
            "model": self.provider.config.model,
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "sequence": idx,  # Add sequence number for correct ordering
                    # Include tool fields when present to enable faithful restore
                    **({"tool_calls": msg.tool_calls} if getattr(msg, "tool_calls", None) else {}),
                    **(
                        {"tool_call_id": msg.tool_call_id}
                        if getattr(msg, "tool_call_id", None)
                        else {}
                    ),
                    "meta": msg.meta or {},
                }
                for idx, msg in enumerate(self.history.messages)
            ],
            "input_tokens": self.context_manager.input_tokens_used,
            "output_tokens": self.context_manager.output_tokens_used,
            "rounds": len([m for m in self.history.messages if m.role == "assistant"]),
            "workspace_root": str(self.workspace.root_path),
            "created_at": getattr(self, "_created_at", None),
            "state": self.state.value,
        }

    @classmethod
    def from_snapshot(
        cls,
        snapshot: dict[str, Any],
        provider: "BaseLLMProvider",
        workspace: Workspace,
        tools: "ToolRegistry",
        event_bus: "EventBus | None" = None,
        config: AgentConfig | None = None,
        context_manager: ContextManager | None = None,
        logger: "AgentRunnerLogger | None" = None,
    ) -> "AgentRunnerAgent":
        """Create agent from saved state.

        Pure method with no storage dependencies.
        Backend is responsible for loading the snapshot.

        Args:
            snapshot: State dict from snapshot()
            provider: LLM provider instance
            workspace: Workspace instance
            tools: Tool registry
            event_bus: Optional event bus
            config: Optional agent config (creates default if not provided)
            context_manager: Optional context manager (creates from provider if not provided)
            logger: Optional logger (creates default if not provided)

        Returns:
            AgentRunnerAgent instance with restored state
        """
        from agentrunner.core.tokens import TokenCounter

        # Create missing dependencies with defaults
        if config is None:
            config = AgentConfig()
        if context_manager is None:
            counter = TokenCounter()
            context_manager = ContextManager(counter, provider.get_model_info())
        if logger is None:
            logger = AgentRunnerLogger()

        # Create agent with basic dependencies
        agent = cls(
            provider=provider,
            workspace=workspace,
            config=config,
            context_manager=context_manager,
            logger=logger,
            tools=tools,
            event_bus=event_bus,
        )

        # Clear the default system message (will be replaced by snapshot messages)
        agent.history.messages.clear()

        # Restore messages from snapshot
        from uuid import uuid4

        for msg_data in snapshot.get("messages", []):
            role = msg_data.get("role")
            # Build Message kwargs safely, including tool fields if valid
            msg_kwargs: dict[str, Any] = {
                "id": msg_data.get("id", str(uuid4())),
                "role": role,
                "content": msg_data.get("content", ""),
                "meta": msg_data.get("meta", {}),
            }
            # Only assistant messages may have tool_calls
            if role == "assistant" and msg_data.get("tool_calls") is not None:
                msg_kwargs["tool_calls"] = msg_data["tool_calls"]
            # Only tool messages may have tool_call_id
            if role == "tool":
                tool_call_id = msg_data.get("tool_call_id")
                if not tool_call_id:
                    # Skip invalid historical tool messages lacking required ID
                    continue
                msg_kwargs["tool_call_id"] = tool_call_id

            agent.history.add(Message(**msg_kwargs))

        # Restore context manager token counts
        agent.context_manager.input_tokens_used = snapshot.get("input_tokens", 0)
        agent.context_manager.output_tokens_used = snapshot.get("output_tokens", 0)

        # Restore state
        state_value = snapshot.get("state", "idle")
        try:
            agent.state = AgentState(state_value)
        except ValueError:
            agent.state = AgentState.IDLE

        # Restore creation timestamp
        if snapshot.get("created_at"):
            agent._created_at = snapshot["created_at"]

        return agent
