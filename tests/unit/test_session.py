"""Unit tests for session management."""

import asyncio
import json
from pathlib import Path

import pytest

from agentrunner.core.config import AgentConfig
from agentrunner.core.messages import Message
from agentrunner.core.session import SessionManager
from agentrunner.core.workspace import Workspace


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    """Create a temporary workspace."""
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    return Workspace(str(workspace_dir))


@pytest.fixture
def session_manager(tmp_path: Path, workspace: Workspace) -> SessionManager:
    """Create a SessionManager with temporary sessions directory."""
    manager = SessionManager(workspace)
    manager.sessions_dir = tmp_path / "sessions"
    manager.sessions_dir.mkdir()
    return manager


@pytest.fixture
def sample_messages() -> list[Message]:
    """Create sample messages for testing."""
    return [
        Message(id="1", role="system", content="You are a helpful assistant."),
        Message(id="2", role="user", content="Hello!"),
        Message(id="3", role="assistant", content="Hi! How can I help you?"),
    ]


@pytest.fixture
def sample_config() -> AgentConfig:
    """Create sample config for testing."""
    return AgentConfig(max_rounds=50)


@pytest.fixture
def sample_meta() -> dict:
    """Create sample metadata for testing."""
    return {
        "total_tokens": 150,
        "duration_ms": 1234,
    }


@pytest.mark.asyncio
async def test_save_creates_session_directory(
    session_manager: SessionManager,
    sample_messages: list[Message],
    sample_config: AgentConfig,
    sample_meta: dict,
) -> None:
    """Test that save creates the session directory."""
    session_id = "test-session"
    await session_manager.save(session_id, sample_messages, sample_config, sample_meta)

    session_dir = session_manager._get_session_dir(session_id)
    assert session_dir.exists()
    assert session_dir.is_dir()


@pytest.mark.asyncio
async def test_save_creates_required_files(
    session_manager: SessionManager,
    sample_messages: list[Message],
    sample_config: AgentConfig,
    sample_meta: dict,
) -> None:
    """Test that save creates all required files."""
    session_id = "test-session"
    await session_manager.save(session_id, sample_messages, sample_config, sample_meta)

    session_dir = session_manager._get_session_dir(session_id)
    assert (session_dir / "messages.jsonl").exists() or (session_dir / "messages.jsonl.gz").exists()
    assert (session_dir / "config.json").exists()
    assert (session_dir / "meta.json").exists()


@pytest.mark.asyncio
async def test_save_and_load_roundtrip(
    session_manager: SessionManager,
    sample_messages: list[Message],
    sample_config: AgentConfig,
    sample_meta: dict,
) -> None:
    """Test that save and load work together correctly."""
    session_id = "test-session"
    await session_manager.save(session_id, sample_messages, sample_config, sample_meta)

    loaded_messages, loaded_config, loaded_meta = await session_manager.load(session_id)

    # Check messages
    assert len(loaded_messages) == len(sample_messages)
    for original, loaded in zip(sample_messages, loaded_messages, strict=True):
        assert loaded.id == original.id
        assert loaded.role == original.role
        assert loaded.content == original.content
        assert loaded.tool_calls == original.tool_calls
        assert loaded.tool_call_id == original.tool_call_id

    # Check config
    assert loaded_config.max_rounds == sample_config.max_rounds
    # Note: model and temperature moved to ProviderConfig

    # Check meta
    assert loaded_meta["total_tokens"] == sample_meta["total_tokens"]
    assert loaded_meta["duration_ms"] == sample_meta["duration_ms"]
    assert "schema_version" in loaded_meta
    assert "created_at" in loaded_meta
    assert "updated_at" in loaded_meta


@pytest.mark.asyncio
async def test_save_preserves_message_structure(
    session_manager: SessionManager,
    sample_config: AgentConfig,
    sample_meta: dict,
) -> None:
    """Test that save preserves complex message structure."""
    messages = [
        Message(
            id="1",
            role="assistant",
            content="Let me help you with that.",
            tool_calls=[{"id": "call_123", "name": "read_file", "arguments": {"path": "file.py"}}],
            meta={"thinking": "some thought"},
        ),
        Message(
            id="2",
            role="tool",
            content="file contents here",
            tool_call_id="call_123",
        ),
    ]

    session_id = "test-session"
    await session_manager.save(session_id, messages, sample_config, sample_meta)
    loaded_messages, _, _ = await session_manager.load(session_id)

    assert len(loaded_messages) == 2
    assert loaded_messages[0].tool_calls == messages[0].tool_calls
    assert loaded_messages[0].meta == messages[0].meta
    assert loaded_messages[1].tool_call_id == messages[1].tool_call_id


@pytest.mark.asyncio
async def test_save_compresses_large_messages(
    session_manager: SessionManager,
    sample_config: AgentConfig,
    sample_meta: dict,
) -> None:
    """Test that large message files are compressed."""
    # Create messages with enough content to trigger compression (>1KB)
    large_content = "x" * 500
    messages = [Message(id=str(i), role="user", content=large_content) for i in range(10)]

    session_id = "test-session"
    await session_manager.save(session_id, messages, sample_config, sample_meta)

    session_dir = session_manager._get_session_dir(session_id)
    assert (session_dir / "messages.jsonl.gz").exists()
    assert not (session_dir / "messages.jsonl").exists()


@pytest.mark.asyncio
async def test_save_no_compression_for_small_messages(
    session_manager: SessionManager,
    sample_messages: list[Message],
    sample_config: AgentConfig,
    sample_meta: dict,
) -> None:
    """Test that small message files are not compressed."""
    session_id = "test-session"
    await session_manager.save(session_id, sample_messages, sample_config, sample_meta)

    session_dir = session_manager._get_session_dir(session_id)
    assert (session_dir / "messages.jsonl").exists()
    assert not (session_dir / "messages.jsonl.gz").exists()


@pytest.mark.asyncio
async def test_load_nonexistent_session_raises(session_manager: SessionManager) -> None:
    """Test that loading a nonexistent session raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Session not found"):
        await session_manager.load("nonexistent-session")


@pytest.mark.asyncio
async def test_load_compressed_messages(
    session_manager: SessionManager,
    sample_config: AgentConfig,
    sample_meta: dict,
) -> None:
    """Test loading sessions with compressed messages."""
    large_content = "x" * 500
    messages = [Message(id=str(i), role="user", content=large_content) for i in range(10)]

    session_id = "test-session"
    await session_manager.save(session_id, messages, sample_config, sample_meta)
    loaded_messages, _, _ = await session_manager.load(session_id)

    assert len(loaded_messages) == len(messages)
    assert all(msg.content == large_content for msg in loaded_messages)


@pytest.mark.asyncio
async def test_load_missing_messages_file_raises(
    session_manager: SessionManager,
    sample_messages: list[Message],
    sample_config: AgentConfig,
    sample_meta: dict,
) -> None:
    """Test that loading with missing messages file raises error."""
    session_id = "test-session"
    await session_manager.save(session_id, sample_messages, sample_config, sample_meta)

    # Delete messages file
    session_dir = session_manager._get_session_dir(session_id)
    (session_dir / "messages.jsonl").unlink()

    with pytest.raises(FileNotFoundError, match="Messages file not found"):
        await session_manager.load(session_id)


@pytest.mark.asyncio
async def test_load_invalid_json_raises(
    session_manager: SessionManager,
    sample_messages: list[Message],
    sample_config: AgentConfig,
    sample_meta: dict,
) -> None:
    """Test that loading with invalid JSON raises error."""
    session_id = "test-session"
    await session_manager.save(session_id, sample_messages, sample_config, sample_meta)

    # Corrupt config file
    session_dir = session_manager._get_session_dir(session_id)
    (session_dir / "config.json").write_text("invalid json {")

    with pytest.raises(json.JSONDecodeError):
        await session_manager.load(session_id)


@pytest.mark.asyncio
async def test_list_empty_returns_empty_list(session_manager: SessionManager) -> None:
    """Test that listing with no sessions returns empty list."""
    sessions = await session_manager.list()
    assert sessions == []


@pytest.mark.asyncio
async def test_list_returns_all_sessions(
    session_manager: SessionManager,
    sample_messages: list[Message],
    sample_config: AgentConfig,
    sample_meta: dict,
) -> None:
    """Test that list returns all saved sessions."""
    await session_manager.save("session-1", sample_messages, sample_config, sample_meta)
    await session_manager.save("session-2", sample_messages, sample_config, sample_meta)
    await session_manager.save("session-3", sample_messages, sample_config, sample_meta)

    sessions = await session_manager.list()
    assert len(sessions) == 3
    session_ids = {s["id"] for s in sessions}
    assert session_ids == {"session-1", "session-2", "session-3"}


@pytest.mark.asyncio
async def test_list_includes_metadata(
    session_manager: SessionManager,
    sample_messages: list[Message],
    sample_config: AgentConfig,
    sample_meta: dict,
) -> None:
    """Test that list includes all required metadata."""
    session_id = "test-session"
    await session_manager.save(session_id, sample_messages, sample_config, sample_meta)

    sessions = await session_manager.list()
    assert len(sessions) == 1

    session_info = sessions[0]
    assert session_info["id"] == session_id
    assert "created_at" in session_info
    assert "updated_at" in session_info
    # Note: model is no longer in AgentConfig (moved to ProviderConfig)
    assert session_info["tokens"] == sample_meta["total_tokens"]


@pytest.mark.asyncio
async def test_list_sorted_by_updated_at(
    session_manager: SessionManager,
    sample_messages: list[Message],
    sample_config: AgentConfig,
    sample_meta: dict,
) -> None:
    """Test that list returns sessions sorted by updated_at (most recent first)."""
    # Create sessions with slight delays to ensure different timestamps
    await session_manager.save("session-1", sample_messages, sample_config, sample_meta)
    await asyncio.sleep(0.01)
    await session_manager.save("session-2", sample_messages, sample_config, sample_meta)
    await asyncio.sleep(0.01)
    await session_manager.save("session-3", sample_messages, sample_config, sample_meta)

    sessions = await session_manager.list()
    assert len(sessions) == 3

    # Most recent should be first
    assert sessions[0]["id"] == "session-3"
    assert sessions[1]["id"] == "session-2"
    assert sessions[2]["id"] == "session-1"


@pytest.mark.asyncio
async def test_list_skips_corrupted_sessions(
    session_manager: SessionManager,
    sample_messages: list[Message],
    sample_config: AgentConfig,
    sample_meta: dict,
) -> None:
    """Test that list skips sessions with corrupted files."""
    await session_manager.save("good-session", sample_messages, sample_config, sample_meta)

    # Create corrupted session
    bad_dir = session_manager._get_session_dir("bad-session")
    bad_dir.mkdir()
    (bad_dir / "config.json").write_text("invalid json")
    (bad_dir / "meta.json").write_text("invalid json")

    sessions = await session_manager.list()
    assert len(sessions) == 1
    assert sessions[0]["id"] == "good-session"


@pytest.mark.asyncio
async def test_delete_removes_session(
    session_manager: SessionManager,
    sample_messages: list[Message],
    sample_config: AgentConfig,
    sample_meta: dict,
) -> None:
    """Test that delete removes a session."""
    session_id = "test-session"
    await session_manager.save(session_id, sample_messages, sample_config, sample_meta)

    session_dir = session_manager._get_session_dir(session_id)
    assert session_dir.exists()

    await session_manager.delete(session_id)
    assert not session_dir.exists()


@pytest.mark.asyncio
async def test_delete_nonexistent_session_raises(session_manager: SessionManager) -> None:
    """Test that deleting a nonexistent session raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Session not found"):
        await session_manager.delete("nonexistent-session")


@pytest.mark.asyncio
async def test_atomic_write_creates_file(
    session_manager: SessionManager,
    tmp_path: Path,
) -> None:
    """Test that atomic write creates the target file."""
    file_path = tmp_path / "test.txt"
    content = b"test content"

    session_manager._atomic_write(file_path, content)

    assert file_path.exists()
    assert file_path.read_bytes() == content


@pytest.mark.asyncio
async def test_atomic_write_cleans_up_temp_file(
    session_manager: SessionManager,
    tmp_path: Path,
) -> None:
    """Test that atomic write cleans up temporary file."""
    file_path = tmp_path / "test.txt"
    content = b"test content"

    session_manager._atomic_write(file_path, content)

    # Check no .tmp files left behind
    temp_files = list(tmp_path.glob("*.tmp"))
    assert len(temp_files) == 0


@pytest.mark.asyncio
async def test_save_updates_existing_session(
    session_manager: SessionManager,
    sample_messages: list[Message],
    sample_config: AgentConfig,
    sample_meta: dict,
) -> None:
    """Test that saving to an existing session updates it."""
    session_id = "test-session"

    # Save initial version
    await session_manager.save(session_id, sample_messages, sample_config, sample_meta)
    _, _, meta1 = await session_manager.load(session_id)
    updated_at_1 = meta1["updated_at"]

    # Small delay to ensure different timestamp
    await asyncio.sleep(0.01)

    # Save updated version
    new_messages = [*sample_messages, Message(id="4", role="user", content="Another message")]
    await session_manager.save(session_id, new_messages, sample_config, sample_meta)
    loaded_messages, _, meta2 = await session_manager.load(session_id)

    assert len(loaded_messages) == 4
    assert meta2["updated_at"] > updated_at_1
    assert meta2["created_at"] == meta1["created_at"]  # created_at should not change


@pytest.mark.asyncio
async def test_schema_version_in_meta(
    session_manager: SessionManager,
    sample_messages: list[Message],
    sample_config: AgentConfig,
    sample_meta: dict,
) -> None:
    """Test that schema version is included in saved metadata."""
    session_id = "test-session"
    await session_manager.save(session_id, sample_messages, sample_config, sample_meta)

    _, _, loaded_meta = await session_manager.load(session_id)
    assert loaded_meta["schema_version"] == SessionManager.SCHEMA_VERSION


@pytest.mark.asyncio
async def test_save_empty_messages(
    session_manager: SessionManager,
    sample_config: AgentConfig,
    sample_meta: dict,
) -> None:
    """Test that saving with empty messages works."""
    session_id = "test-session"
    await session_manager.save(session_id, [], sample_config, sample_meta)

    loaded_messages, _, _ = await session_manager.load(session_id)
    assert loaded_messages == []


@pytest.mark.asyncio
async def test_list_with_tokens_meta_variations(
    session_manager: SessionManager,
    sample_messages: list[Message],
    sample_config: AgentConfig,
) -> None:
    """Test that list handles different token metadata field names."""
    # Session with total_tokens
    await session_manager.save("session-1", sample_messages, sample_config, {"total_tokens": 100})

    # Session with tokens
    await session_manager.save("session-2", sample_messages, sample_config, {"tokens": 200})

    # Session with neither
    await session_manager.save("session-3", sample_messages, sample_config, {})

    sessions = await session_manager.list()
    sessions_by_id = {s["id"]: s for s in sessions}

    assert sessions_by_id["session-1"]["tokens"] == 100
    assert sessions_by_id["session-2"]["tokens"] == 200
    assert sessions_by_id["session-3"]["tokens"] is None
