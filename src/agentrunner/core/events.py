"""Streaming event system for real-time UI updates.

Implements event bus and streaming infrastructure per INTERFACES/STREAMING_UI.md.
Enables real-time updates during agent execution for CLI, web UI, and other consumers.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from agentrunner.core.logger import AgentRunnerLogger

if TYPE_CHECKING:
    pass

logger = AgentRunnerLogger()

# When adding new event types:
# 1. Add to EventType literal below
# 2. Run: pytest tests/unit/test_events.py

EventType = Literal[
    # Agent/tool execution events
    "token_delta",
    "assistant_delta",
    "user_message",  # User's input message (v2 worker)
    "assistant_message",  # Final assistant message when not streaming
    "tool_call_started",
    "tool_output",
    "tool_call_completed",
    "status_update",
    "usage_update",
    "error",
    "compaction",  # Context compaction event
    # File system events
    "file_created",
    "file_modified",
    "file_tree_update",
    # Scaffold event
    "scaffold_complete",
    # Preview/Deployment events
    "preview_update",
    "preview_multi",
    "preview_ready",
    "deployment_ready",  # Vercel deployment complete
    # Server/Screenshot events
    "server_starting",
    "server_ready",
    "screenshot_taken",
    "server_stopped",
    # Bash execution events
    "bash_started",
    "bash_executed",
    # Session/workspace events (sent manually by backend)
    "session_created",
    "session_restored",
    "workspace_updated",
    # Multi-agent events (sent manually by backend)
    "execution_summary",
    "agent_error",
]


@dataclass
class StreamEvent:
    """Event emitted during agent execution.

    Represents a single event in the agent's execution stream,
    with standardized format for UI consumption.

    Contract tests: tests/unit/test_events.py
    """

    type: EventType  # Event type identifier (type-safe literal)
    data: dict[str, Any]  # Event payload/data
    model_id: str  # REQUIRED: Which model/agent created this event (e.g., "gpt-4-turbo")
    ts: str  # ISO timestamp when event was created
    id: str = field(default_factory=lambda: str(uuid.uuid4()))  # Unique event ID for deduplication

    def __post_init__(self) -> None:
        """Validate event data structure."""
        if not isinstance(self.data, dict):
            raise ValueError(f"Event data must be dict, got {type(self.data)}")

        if not self.model_id:
            raise ValueError("model_id is required for all events")


class EventBus:
    """Thread-safe event bus for streaming agent updates.

    Provides publish/subscribe pattern with async iteration support.
    Multiple subscribers can listen for events simultaneously.

    Buffers recent events so new subscribers can receive missed events.
    """

    def __init__(self, max_history: int = 100) -> None:
        """Initialize event bus with empty subscriber list and async queue.

        Args:
            max_history: Maximum number of events to buffer for replay (default: 100)
        """
        self._subscribers: list[Callable[[StreamEvent], None]] = []
        self._queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
        self._closed = False
        self._event_history: list[StreamEvent] = []  # Buffer for missed events
        self._max_history = max_history

    def publish(self, event: StreamEvent) -> None:
        """Publish event to all subscribers and queue.

        Args:
            event: StreamEvent to publish

        Note:
            Thread-safe operation using asyncio queue.
        """
        if self._closed:
            logger.warn("Publishing to closed event bus", event_type=event.type)
            return

        logger.debug(
            "EventBus publishing event",
            event_type=event.type,
            event_id=event.id,
            subscriber_count=len(self._subscribers),
        )

        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warn(
                "Event queue full, dropping event", event_type=event.type, event_id=event.id
            )

        for handler in self._subscribers[:]:
            try:
                handler(event)
                logger.debug(
                    "EventBus: Handler called successfully",
                    handler=handler.__name__,
                    event_type=event.type,
                    event_id=event.id,
                )
            except Exception as e:
                logger.error(
                    "EventBus: Handler failed",
                    error=str(e),
                    handler=handler.__name__,
                    event_type=event.type,
                    event_id=event.id,
                )

    def subscribe(self, handler: Callable[[StreamEvent], None]) -> None:
        """Subscribe to events.

        Args:
            handler: Callback function to receive events
        """
        if handler not in self._subscribers:
            self._subscribers.append(handler)
            logger.debug("Added event subscriber", handler=handler.__name__)

    def unsubscribe(self, handler: Callable[[StreamEvent], None]) -> None:
        """Unsubscribe from events.

        Args:
            handler: Callback function to remove
        """
        if handler in self._subscribers:
            self._subscribers.remove(handler)
            logger.debug("Removed event subscriber", handler=handler.__name__)

    async def __aiter__(self) -> AsyncIterator[StreamEvent]:
        """Async iteration over events.

        Yields events from queue as they arrive.
        Continues until bus is closed and queue is empty.
        """
        while not self._closed or not self._queue.empty():
            try:
                # Wait for next event with timeout to allow checking closed status
                event = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                yield event
            except TimeoutError:
                # Continue to check if bus is closed
                continue
            except Exception as e:
                logger.error("Error in event iteration", error=str(e))
                break

    def clear(self) -> None:
        """Clear all subscribers and queue."""
        self._subscribers.clear()

        # Drain the queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        logger.debug("Event bus cleared")

    def close(self) -> None:
        """Close the event bus, stopping async iteration."""
        self._closed = True
        logger.debug("Event bus closed")

    @property
    def subscriber_count(self) -> int:
        """Get number of active subscribers."""
        return len(self._subscribers)

    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()


def create_token_delta_event(delta: str, model_id: str) -> StreamEvent:
    """Create token delta event for streaming token output.

    Args:
        delta: Token text delta
        model_id: Model identifier (e.g., "gpt-4-turbo")

    Returns:
        StreamEvent with token_delta type
    """
    return StreamEvent(
        type="token_delta",
        data={"text": delta},
        model_id=model_id,
        ts=datetime.now(UTC).isoformat(),
    )


def create_usage_event(
    prompt_tokens: int, completion_tokens: int, total_tokens: int, model_id: str
) -> StreamEvent:
    """Create token usage event.

    Args:
        prompt_tokens: Number of prompt tokens used
        completion_tokens: Number of completion tokens generated
        total_tokens: Total tokens used
        model_id: Model identifier (e.g., "gpt-4-turbo")

    Returns:
        StreamEvent with usage_update type
    """
    return StreamEvent(
        type="usage_update",
        data={
            "usage": {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": total_tokens,
            }
        },
        model_id=model_id,
        ts=datetime.now(UTC).isoformat(),
    )


def create_error_event(error: str, model_id: str, error_code: str = "") -> StreamEvent:
    """Create error event.

    Args:
        error: Error message
        model_id: Model identifier (e.g., "gpt-4-turbo")
        error_code: Optional error code

    Returns:
        StreamEvent with error type
    """
    data = {"error": error}
    if error_code:
        data["error_code"] = error_code

    return StreamEvent(type="error", data=data, model_id=model_id, ts=datetime.now(UTC).isoformat())


def create_file_tree_event(
    model_id: str,
    created: list[str] | None = None,
    modified: list[str] | None = None,
    deleted: list[str] | None = None,
) -> StreamEvent:
    """Create file tree update event.

    Args:
        model_id: Model identifier (e.g., "gpt-4-turbo")
        created: List of created file paths
        modified: List of modified file paths
        deleted: List of deleted file paths

    Returns:
        StreamEvent with file_tree_update type
    """
    data = {}
    if created:
        data["created"] = created
    if modified:
        data["modified"] = modified
    if deleted:
        data["deleted"] = deleted

    return StreamEvent(
        type="file_tree_update", data=data, model_id=model_id, ts=datetime.now(UTC).isoformat()
    )


def create_preview_event(kind: str, url: str, model_id: str, frame_id: str = "") -> StreamEvent:
    """Create preview update event.

    Args:
        kind: Preview kind (web, content)
        url: Preview URL
        model_id: Model identifier (e.g., "gpt-4-turbo")
        frame_id: Optional frame identifier

    Returns:
        StreamEvent with preview_update type
    """
    data = {"kind": kind, "url": url}
    if frame_id:
        data["frame_id"] = frame_id

    return StreamEvent(
        type="preview_update", data=data, model_id=model_id, ts=datetime.now(UTC).isoformat()
    )


def create_preview_ready_event(
    url: str,
    port: int,
    command: str,
    model_id: str,
) -> StreamEvent:
    """Create preview_ready event.

    Emitted when preview server URL is detected and ready to use.

    Args:
        url: Preview URL path for frontend (e.g., "/preview/abc123/gpt-4-turbo/")
        port: Port number the server is listening on (e.g., 5173)
        command: Command that was run (e.g., "npm run dev")
        model_id: Model identifier (e.g., "gpt-4-turbo")

    Returns:
        StreamEvent with preview_ready type
    """
    return StreamEvent(
        type="preview_ready",
        data={
            "url": url,
            "port": port,
            "command": command,
        },
        model_id=model_id,
        ts=datetime.now(UTC).isoformat(),
    )
