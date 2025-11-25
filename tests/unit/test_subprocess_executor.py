"""Unit tests for SubprocessExecutor."""

import os
import tempfile
from pathlib import Path

import pytest

from agentrunner.core.executors.subprocess_executor import SubprocessExecutor


class TestSubprocessExecutor:
    """Test subprocess command executor."""

    @pytest.mark.asyncio
    async def test_simple_command(self):
        """Test executing a simple command."""
        executor = SubprocessExecutor()

        result = await executor.execute_command(
            command="echo 'Hello World'",
            cwd=str(Path.cwd()),
            env=os.environ.copy(),
            timeout=5,
        )

        assert result.exit_code == 0
        assert "Hello World" in result.stdout
        assert result.stderr == ""
        assert result.duration_ms > 0
        assert result.metadata is not None
        assert result.metadata["platform"] == os.sys.platform

    @pytest.mark.asyncio
    async def test_command_with_stderr(self):
        """Test command that writes to stderr."""
        executor = SubprocessExecutor()

        result = await executor.execute_command(
            command="echo 'error message' >&2",
            cwd=str(Path.cwd()),
            env=os.environ.copy(),
            timeout=5,
        )

        assert result.exit_code == 0
        assert result.stdout == ""
        assert "error message" in result.stderr

    @pytest.mark.asyncio
    async def test_command_failure(self):
        """Test command that fails."""
        executor = SubprocessExecutor()

        result = await executor.execute_command(
            command="exit 42",
            cwd=str(Path.cwd()),
            env=os.environ.copy(),
            timeout=5,
        )

        assert result.exit_code == 42
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_command_timeout(self):
        """Test command that times out."""
        executor = SubprocessExecutor()

        with pytest.raises(TimeoutError, match="timed out after 1 seconds"):
            await executor.execute_command(
                command="sleep 10",
                cwd=str(Path.cwd()),
                env=os.environ.copy(),
                timeout=1,
            )

    @pytest.mark.asyncio
    async def test_command_with_cwd(self):
        """Test command with custom working directory."""
        executor = SubprocessExecutor()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = await executor.execute_command(
                command="pwd",
                cwd=tmpdir,
                env=os.environ.copy(),
                timeout=5,
            )

            assert result.exit_code == 0
            assert tmpdir in result.stdout

    @pytest.mark.asyncio
    async def test_command_with_env(self):
        """Test command with custom environment variables."""
        executor = SubprocessExecutor()

        custom_env = os.environ.copy()
        custom_env["TEST_VAR"] = "test_value_123"

        result = await executor.execute_command(
            command="echo $TEST_VAR",
            cwd=str(Path.cwd()),
            env=custom_env,
            timeout=5,
        )

        assert result.exit_code == 0
        assert "test_value_123" in result.stdout

    @pytest.mark.asyncio
    async def test_output_truncation(self):
        """Test that large output is truncated."""
        # Create executor with small max output bytes
        executor = SubprocessExecutor(max_output_bytes=100)

        # Generate output larger than max_output_bytes
        result = await executor.execute_command(
            command="python3 -c 'print(\"x\" * 1000)'",
            cwd=str(Path.cwd()),
            env=os.environ.copy(),
            timeout=5,
        )

        assert result.exit_code == 0
        assert len(result.stdout) < 200  # Should be truncated
        assert "[Output truncated" in result.stdout

    @pytest.mark.asyncio
    async def test_get_name(self):
        """Test get_name returns correct executor name."""
        executor = SubprocessExecutor()
        assert executor.get_name() == "subprocess"

    @pytest.mark.asyncio
    async def test_uid_gid_metadata(self):
        """Test that UID/GID are stored in metadata."""
        executor = SubprocessExecutor(uid=1000, gid=1000)

        result = await executor.execute_command(
            command="echo test",
            cwd=str(Path.cwd()),
            env=os.environ.copy(),
            timeout=5,
        )

        assert result.metadata is not None
        assert result.metadata["uid"] == 1000
        assert result.metadata["gid"] == 1000
        assert result.metadata["privileges_dropped"] is True

    @pytest.mark.asyncio
    async def test_no_uid_gid(self):
        """Test executor without UID/GID isolation."""
        executor = SubprocessExecutor()

        result = await executor.execute_command(
            command="echo test",
            cwd=str(Path.cwd()),
            env=os.environ.copy(),
            timeout=5,
        )

        assert result.metadata is not None
        assert result.metadata["uid"] is None
        assert result.metadata["gid"] is None
        assert result.metadata["privileges_dropped"] is False
