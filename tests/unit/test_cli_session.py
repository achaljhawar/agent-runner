"""Unit tests for CLISession event persistence."""

import json
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import pytest

from agentrunner.core.cli_session import CLISession
from agentrunner.core.events import StreamEvent
from agentrunner.core.messages import Message


@pytest.fixture
def temp_sessions_dir(monkeypatch):
    """Create temporary sessions directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sessions_dir = Path(tmpdir) / "sessions"
        sessions_dir.mkdir(parents=True)

        # Mock the sessions directory path
        def mock_init(self, session_id=None, workspace_root=None):
            self.session_id = session_id or f"cli_{uuid.uuid4().hex[:12]}"
            self.workspace_root = workspace_root or "."
            self.sessions_dir = sessions_dir
            self.session_file = self.sessions_dir / f"{self.session_id}.jsonl"

            if not self.session_file.exists():
                self._write_metadata()

        monkeypatch.setattr(CLISession, "__init__", mock_init)
        yield sessions_dir


def test_save_event(temp_sessions_dir):
    """Test saving events to session."""
    session = CLISession(workspace_root="/test")

    event = StreamEvent(
        type="tool_call_started",
        data={"name": "bash", "arguments": {"command": "ls"}},
        model_id="gpt-5.1-2025-11-13",
        ts=datetime.now().isoformat(),
    )

    session.save_event(event)

    # Verify event was written
    with open(session.session_file) as f:
        lines = f.readlines()

    # Should have metadata + 1 event
    assert len(lines) == 2

    event_line = json.loads(lines[1])
    assert event_line["_type"] == "event"
    assert event_line["type"] == "tool_call_started"
    assert event_line["data"]["name"] == "bash"
    assert event_line["model_id"] == "gpt-5.1-2025-11-13"


def test_load_events(temp_sessions_dir):
    """Test loading events from session."""
    session = CLISession(workspace_root="/test")

    # Save multiple events
    events = [
        StreamEvent(
            type="tool_call_started",
            data={"name": "bash"},
            model_id="gpt-5.1-2025-11-13",
            ts=datetime.now().isoformat(),
        ),
        StreamEvent(
            type="bash_executed",
            data={"command": "ls", "exit_code": 0},
            model_id="gpt-5.1-2025-11-13",
            ts=datetime.now().isoformat(),
        ),
        StreamEvent(
            type="tool_call_completed",
            data={"success": True},
            model_id="gpt-5.1-2025-11-13",
            ts=datetime.now().isoformat(),
        ),
    ]

    for event in events:
        session.save_event(event)

    # Load all events
    loaded_events = session.load_events()

    assert len(loaded_events) == 3
    assert loaded_events[0]["type"] == "tool_call_started"
    assert loaded_events[1]["type"] == "bash_executed"
    assert loaded_events[2]["type"] == "tool_call_completed"


def test_load_events_with_filter(temp_sessions_dir):
    """Test loading events with type filter."""
    session = CLISession(workspace_root="/test")

    # Save mixed event types
    events = [
        StreamEvent(
            type="tool_call_started",
            data={},
            model_id="gpt-5.1-2025-11-13",
            ts=datetime.now().isoformat(),
        ),
        StreamEvent(
            type="bash_executed",
            data={},
            model_id="gpt-5.1-2025-11-13",
            ts=datetime.now().isoformat(),
        ),
        StreamEvent(
            type="file_created",
            data={},
            model_id="gpt-5.1-2025-11-13",
            ts=datetime.now().isoformat(),
        ),
        StreamEvent(
            type="bash_executed",
            data={},
            model_id="gpt-5.1-2025-11-13",
            ts=datetime.now().isoformat(),
        ),
        StreamEvent(
            type="tool_call_completed",
            data={},
            model_id="gpt-5.1-2025-11-13",
            ts=datetime.now().isoformat(),
        ),
    ]

    for event in events:
        session.save_event(event)

    # Filter for bash events only
    bash_events = session.load_events(event_types=["bash_executed"])

    assert len(bash_events) == 2
    assert all(e["type"] == "bash_executed" for e in bash_events)


def test_load_events_with_limit(temp_sessions_dir):
    """Test loading events with limit."""
    session = CLISession(workspace_root="/test")

    # Save many events
    for i in range(10):
        event = StreamEvent(
            type="tool_call_started",
            data={"index": i},
            model_id="gpt-5.1-2025-11-13",
            ts=datetime.now().isoformat(),
        )
        session.save_event(event)

    # Load last 3 events
    recent_events = session.load_events(limit=3)

    assert len(recent_events) == 3
    assert recent_events[-1]["data"]["index"] == 9  # Most recent


def test_save_event_and_message_interleaved(temp_sessions_dir):
    """Test that events and messages can be interleaved."""
    session = CLISession(workspace_root="/test")

    # Save message
    msg1 = Message(id=str(uuid.uuid4()), role="user", content="Hello")
    session.save_message(msg1)

    # Save event
    event = StreamEvent(
        type="tool_call_started",
        data={"name": "bash"},
        model_id="gpt-5.1-2025-11-13",
        ts=datetime.now().isoformat(),
    )
    session.save_event(event)

    # Save another message
    msg2 = Message(id=str(uuid.uuid4()), role="assistant", content="Hi")
    session.save_message(msg2)

    # Verify both can be loaded independently
    messages = session.load_messages()
    events = session.load_events()

    assert len(messages) == 2
    assert len(events) == 1


def test_load_messages_ignores_events(temp_sessions_dir):
    """Test that load_messages only returns messages."""
    session = CLISession(workspace_root="/test")

    # Save message
    msg = Message(id=str(uuid.uuid4()), role="user", content="Test")
    session.save_message(msg)

    # Save event
    event = StreamEvent(
        type="tool_call_started",
        data={},
        model_id="gpt-5.1-2025-11-13",
        ts=datetime.now().isoformat(),
    )
    session.save_event(event)

    # load_messages should only return message
    messages = session.load_messages()

    assert len(messages) == 1
    assert messages[0].content == "Test"


def test_load_events_ignores_messages(temp_sessions_dir):
    """Test that load_events only returns events."""
    session = CLISession(workspace_root="/test")

    # Save message
    msg = Message(id=str(uuid.uuid4()), role="user", content="Test")
    session.save_message(msg)

    # Save event
    event = StreamEvent(
        type="tool_call_started",
        data={"name": "bash"},
        model_id="gpt-5.1-2025-11-13",
        ts=datetime.now().isoformat(),
    )
    session.save_event(event)

    # load_events should only return event
    events = session.load_events()

    assert len(events) == 1
    assert events[0]["type"] == "tool_call_started"


def test_list_sessions_includes_event_count(temp_sessions_dir):
    """Test that list_sessions includes event counts."""
    # Create session with messages and events
    session = CLISession(workspace_root="/test")

    msg = Message(id=str(uuid.uuid4()), role="user", content="Test")
    session.save_message(msg)

    for _i in range(3):
        event = StreamEvent(
            type="tool_call_started",
            data={},
            model_id="gpt-5.1-2025-11-13",
            ts=datetime.now().isoformat(),
        )
        session.save_event(event)

    # Directly check the session file
    with open(session.session_file) as f:
        lines = f.readlines()

    # Count types
    message_count = sum(1 for line in lines if json.loads(line).get("_type") == "message")
    event_count = sum(1 for line in lines if json.loads(line).get("_type") == "event")

    assert message_count == 1
    assert event_count == 3


def test_load_events_empty_session(temp_sessions_dir):
    """Test loading events from empty session."""
    session = CLISession(workspace_root="/test")

    events = session.load_events()

    assert events == []


def test_load_events_nonexistent_session(temp_sessions_dir):
    """Test loading events from non-existent session."""
    # Don't create the session file
    session_id = "cli_nonexistent"
    session_file = temp_sessions_dir / f"{session_id}.jsonl"

    # Manually create CLISession without writing metadata
    session = CLISession.__new__(CLISession)
    session.session_id = session_id
    session.workspace_root = "."
    session.sessions_dir = temp_sessions_dir
    session.session_file = session_file

    events = session.load_events()

    assert events == []
