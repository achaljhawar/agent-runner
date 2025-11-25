"""CLI session management with conversation history persistence.

Provides local storage for CLI sessions using JSONL format.
Stores messages and events (tool calls, outputs, etc.) for full replay.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from agentrunner.core.events import StreamEvent
from agentrunner.core.messages import Message


class CLISession:
    """Manages CLI session history with local file persistence.

    Stores conversation history and events in JSONL format (one item per line)
    in platform-appropriate location (~/.local/share/agentrunner/sessions/).

    JSONL format:
        {"_type": "metadata", "session_id": "...", ...}
        {"_type": "message", "role": "user", "content": "...", ...}
        {"_type": "event", "type": "tool_call_started", "data": {...}, ...}
        {"_type": "message", "role": "assistant", "content": "...", ...}
    """

    def __init__(self, session_id: str | None = None, workspace_root: str | None = None):
        """Initialize CLI session.

        Args:
            session_id: Existing session ID to resume, or None for new session
            workspace_root: Workspace directory path (for metadata)
        """
        self.session_id = session_id or f"cli_{uuid.uuid4().hex[:12]}"
        self.workspace_root = workspace_root or "."

        # Use XDG Base Directory standard for Unix-like systems
        if Path.home().joinpath(".local/share").exists():
            # Linux/macOS with XDG
            self.sessions_dir = Path.home() / ".local/share/agentrunner/sessions"
        else:
            # Fallback for other systems
            self.sessions_dir = Path.home() / ".agentrunner/sessions"

        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.session_file = self.sessions_dir / f"{self.session_id}.jsonl"

        # Create session file if it doesn't exist
        if not self.session_file.exists():
            self._write_metadata()

    def _write_metadata(self) -> None:
        """Write session metadata as first line."""
        metadata = {
            "_type": "metadata",
            "session_id": self.session_id,
            "workspace_root": self.workspace_root,
            "created_at": datetime.now().isoformat(),
        }
        with open(self.session_file, "w") as f:
            f.write(json.dumps(metadata) + "\n")

    def save_message(self, message: Message) -> None:
        """Append message to session history.

        Args:
            message: Message to persist
        """
        message_data = {
            "_type": "message",
            "id": message.id,
            "role": message.role,
            "content": message.content,
            "tool_calls": message.tool_calls,
            "tool_call_id": message.tool_call_id,
            "timestamp": datetime.now().isoformat(),
        }

        with open(self.session_file, "a") as f:
            f.write(json.dumps(message_data) + "\n")

    def save_event(self, event: StreamEvent) -> None:
        """Append event to session history.

        Events include tool calls, outputs, token deltas, file changes, etc.
        This provides full execution trace for debugging and replay.

        Args:
            event: StreamEvent to persist
        """
        event_data = {
            "_type": "event",
            "event_id": event.id,
            "type": event.type,
            "data": event.data,
            "model_id": event.model_id,
            "timestamp": event.ts,
        }

        with open(self.session_file, "a") as f:
            f.write(json.dumps(event_data) + "\n")

    def load_messages(self, limit: int | None = None) -> list[Message]:
        """Load messages from session history.

        Args:
            limit: Maximum number of messages to load (None = all)

        Returns:
            List of Message objects
        """
        if not self.session_file.exists():
            return []

        messages: list[Message] = []
        with open(self.session_file) as f:
            for line in f:
                data = json.loads(line.strip())

                # Skip non-message lines
                if data.get("_type") != "message":
                    continue

                # Reconstruct Message
                messages.append(
                    Message(
                        id=data["id"],
                        role=data["role"],
                        content=data["content"],
                        tool_calls=data.get("tool_calls"),
                        tool_call_id=data.get("tool_call_id"),
                    )
                )

        # Apply limit if specified
        if limit:
            messages = messages[-limit:]

        return messages

    def load_events(
        self, limit: int | None = None, event_types: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Load events from session history.

        Args:
            limit: Maximum number of events to load (None = all)
            event_types: Filter by specific event types (e.g., ["tool_call_started", "bash_executed"])

        Returns:
            List of event dicts with keys: event_id, type, data, model_id, timestamp
        """
        if not self.session_file.exists():
            return []

        events: list[dict[str, Any]] = []
        with open(self.session_file) as f:
            for line in f:
                data = json.loads(line.strip())

                # Skip non-event lines
                if data.get("_type") != "event":
                    continue

                # Filter by event type if specified
                if event_types and data.get("type") not in event_types:
                    continue

                events.append(
                    {
                        "event_id": data.get("event_id"),
                        "type": data.get("type"),
                        "data": data.get("data", {}),
                        "model_id": data.get("model_id"),
                        "timestamp": data.get("timestamp"),
                    }
                )

        # Apply limit if specified
        if limit:
            events = events[-limit:]

        return events

    def get_metadata(self) -> dict[str, Any] | None:
        """Get session metadata.

        Returns:
            Metadata dict or None if not found
        """
        if not self.session_file.exists():
            return None

        with open(self.session_file) as f:
            first_line = f.readline().strip()
            if first_line:
                data = cast(dict[str, Any], json.loads(first_line))
                if data.get("_type") == "metadata":
                    return data

        return None

    @classmethod
    def list_sessions(cls) -> list[dict[str, Any]]:
        """List all saved CLI sessions.

        Returns:
            List of session metadata dicts with message_count and event_count
        """
        sessions_dir = Path.home() / ".local/share/agentrunner/sessions"
        if not sessions_dir.exists():
            sessions_dir = Path.home() / ".agentrunner/sessions"

        if not sessions_dir.exists():
            return []

        sessions = []
        for session_file in sessions_dir.glob("cli_*.jsonl"):
            try:
                with open(session_file) as f:
                    first_line = f.readline().strip()
                    if first_line:
                        metadata = json.loads(first_line)
                        if metadata.get("_type") == "metadata":
                            # Count messages and events
                            message_count = 0
                            event_count = 0
                            for line in f:
                                data = json.loads(line)
                                if data.get("_type") == "message":
                                    message_count += 1
                                elif data.get("_type") == "event":
                                    event_count += 1

                            metadata["message_count"] = message_count
                            metadata["event_count"] = event_count
                            sessions.append(metadata)
            except (json.JSONDecodeError, OSError):
                continue

        # Sort by created_at (newest first)
        sessions.sort(key=lambda s: s.get("created_at", ""), reverse=True)
        return sessions

    @classmethod
    def delete_session(cls, session_id: str) -> bool:
        """Delete a session file.

        Args:
            session_id: Session ID to delete

        Returns:
            True if deleted, False if not found
        """
        sessions_dir = Path.home() / ".local/share/agentrunner/sessions"
        if not sessions_dir.exists():
            sessions_dir = Path.home() / ".agentrunner/sessions"

        session_file = sessions_dir / f"{session_id}.jsonl"
        if session_file.exists():
            session_file.unlink()
            return True
        return False
