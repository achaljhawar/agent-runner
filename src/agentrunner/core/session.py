"""Session management for saving and loading conversation state.

Implements SessionManager per INTERFACES/SESSION_MANAGEMENT.md.
"""

import gzip
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentrunner.core.config import AgentConfig
from agentrunner.core.messages import Message
from agentrunner.core.workspace import Workspace

# Compression threshold: compress messages files larger than 1KB
COMPRESSION_THRESHOLD = 1024


class SessionManager:
    """Manages session persistence to ~/.agentrunner/sessions/.

    Provides atomic saves, optional compression, and session listing.
    Each session is stored in its own directory with:
    - messages.jsonl (one message per line)
    - config.json (agent configuration)
    - meta.json (timestamps, tokens, versions)
    """

    SCHEMA_VERSION = 1

    def __init__(self, workspace: Workspace) -> None:
        """Initialize session manager.

        Args:
            workspace: Workspace instance for path validation
        """
        self.workspace = workspace
        self.sessions_dir = Path("~/.agentrunner/sessions").expanduser()
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_dir(self, session_id: str) -> Path:
        """Get path to session directory.

        Args:
            session_id: Unique session identifier

        Returns:
            Path to session directory
        """
        return self.sessions_dir / session_id

    def _atomic_write(self, path: Path, content: bytes) -> None:
        """Write file atomically using temp file and rename.

        Args:
            path: Target file path
            content: Bytes to write
        """
        temp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            # Write to temp file
            temp_path.write_bytes(content)
            # Ensure data is flushed to disk
            with temp_path.open("rb") as f:
                os.fsync(f.fileno())
            # Atomic rename
            temp_path.replace(path)
        finally:
            # Clean up temp file if it still exists
            temp_path.unlink(missing_ok=True)

    async def save(
        self,
        session_id: str,
        messages: list[Message],
        config: AgentConfig,
        meta: dict[str, Any],
    ) -> None:
        """Save session to disk.

        Args:
            session_id: Unique session identifier
            messages: List of conversation messages
            config: Agent configuration
            meta: Metadata (tokens, durations, etc.)

        Raises:
            OSError: If write fails
        """
        session_dir = self._get_session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        # Load existing meta to preserve created_at
        existing_created_at = None
        meta_path = session_dir / "meta.json"
        if meta_path.exists():
            try:
                existing_meta = json.loads(meta_path.read_text())
                existing_created_at = existing_meta.get("created_at")
            except (json.JSONDecodeError, OSError):
                pass  # If we can't read existing meta, just proceed without it

        # Add schema version and timestamps to meta
        now = datetime.now(UTC).isoformat()
        full_meta = {
            "schema_version": self.SCHEMA_VERSION,
            "created_at": existing_created_at or meta.get("created_at", now),
            "updated_at": now,
            **meta,
        }

        # Save messages as JSONL
        messages_path = session_dir / "messages.jsonl"
        messages_content = "\n".join(
            json.dumps(
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "tool_calls": msg.tool_calls,
                    "tool_call_id": msg.tool_call_id,
                    "meta": msg.meta,
                }
            )
            for msg in messages
        ).encode("utf-8")

        # Check if compression is beneficial (>1KB)
        if len(messages_content) > COMPRESSION_THRESHOLD:
            messages_path = messages_path.with_suffix(".jsonl.gz")
            messages_content = gzip.compress(messages_content)

        self._atomic_write(messages_path, messages_content)

        # Save config
        config_path = session_dir / "config.json"
        config_content = json.dumps(config.to_dict(), indent=2).encode("utf-8")
        self._atomic_write(config_path, config_content)

        # Save meta
        meta_path = session_dir / "meta.json"
        meta_content = json.dumps(full_meta, indent=2).encode("utf-8")
        self._atomic_write(meta_path, meta_content)

    async def load(self, session_id: str) -> tuple[list[Message], AgentConfig, dict[str, Any]]:
        """Load session from disk.

        Args:
            session_id: Unique session identifier

        Returns:
            Tuple of (messages, config, meta)

        Raises:
            FileNotFoundError: If session doesn't exist
            json.JSONDecodeError: If session data is corrupted
        """
        session_dir = self._get_session_dir(session_id)
        if not session_dir.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")

        # Load messages (check both compressed and uncompressed)
        messages_path_gz = session_dir / "messages.jsonl.gz"
        messages_path = session_dir / "messages.jsonl"

        if messages_path_gz.exists():
            messages_content = gzip.decompress(messages_path_gz.read_bytes()).decode("utf-8")
        elif messages_path.exists():
            messages_content = messages_path.read_text()
        else:
            raise FileNotFoundError(f"Messages file not found for session: {session_id}")

        messages = []
        for line in messages_content.strip().split("\n"):
            if not line:
                continue
            msg_data = json.loads(line)
            messages.append(Message(**msg_data))

        # Load config
        config_path = session_dir / "config.json"
        config_data = json.loads(config_path.read_text())
        config = AgentConfig(**config_data)

        # Load meta
        meta_path = session_dir / "meta.json"
        meta = json.loads(meta_path.read_text())

        return messages, config, meta

    async def list(self) -> list[dict[str, Any]]:
        """List all sessions with metadata.

        Returns:
            List of session info dicts with keys:
            - id: session identifier
            - created_at: ISO timestamp
            - updated_at: ISO timestamp
            - model: model name from config
            - tokens: total token count from meta (if available)
        """
        sessions: list[dict[str, Any]] = []

        if not self.sessions_dir.exists():
            return sessions

        for session_dir in self.sessions_dir.iterdir():
            if not session_dir.is_dir():
                continue

            meta_path = session_dir / "meta.json"
            config_path = session_dir / "config.json"

            # Skip if required files don't exist
            if not meta_path.exists() or not config_path.exists():
                continue

            try:
                meta = json.loads(meta_path.read_text())
                config_data = json.loads(config_path.read_text())

                session_info = {
                    "id": session_dir.name,
                    "created_at": meta.get("created_at", ""),
                    "updated_at": meta.get("updated_at", ""),
                    "model": config_data.get("model", ""),
                    "tokens": meta.get("total_tokens", meta.get("tokens")),
                }
                sessions.append(session_info)
            except (json.JSONDecodeError, KeyError):
                # Skip corrupted sessions
                continue

        # Sort by updated_at descending (most recent first)
        sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return sessions

    async def delete(self, session_id: str) -> None:
        """Delete a session and all its files.

        Args:
            session_id: Unique session identifier

        Raises:
            FileNotFoundError: If session doesn't exist
        """
        session_dir = self._get_session_dir(session_id)
        if not session_dir.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")

        # Delete all files in session directory
        for file_path in session_dir.iterdir():
            file_path.unlink()

        # Delete directory
        session_dir.rmdir()
