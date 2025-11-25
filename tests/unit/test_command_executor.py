"""Unit tests for CommandExecutor abstraction."""

from unittest.mock import AsyncMock, patch

import pytest

from agentrunner.core.command_executor import CommandResult
from agentrunner.core.executors.subprocess_executor import SubprocessExecutor


class TestCommandResult:
    """Test CommandResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating command result."""
        result = CommandResult(
            stdout="output",
            stderr="error",
            exit_code=0,
            duration_ms=100,
        )
        assert result.stdout == "output"
        assert result.stderr == "error"
        assert result.exit_code == 0
        assert result.duration_ms == 100
        assert result.metadata is None

    def test_create_result_with_metadata(self) -> None:
        """Test creating command result with metadata."""
        result = CommandResult(
            stdout="output",
            stderr="",
            exit_code=0,
            duration_ms=50,
            metadata={"uid": 1000, "platform": "linux"},
        )
        assert result.metadata == {"uid": 1000, "platform": "linux"}


class TestSubprocessExecutor:
    """Test SubprocessExecutor implementation."""

    @pytest.mark.asyncio
    async def test_execute_simple_command(self) -> None:
        """Test executing simple command."""
        executor = SubprocessExecutor()
        result = await executor.execute_command(
            command="echo 'hello world'",
            cwd="/tmp",
            env={"PATH": "/usr/bin"},
            timeout=5,
        )

        assert result.exit_code == 0
        assert "hello world" in result.stdout
        assert result.stderr == ""
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_execute_command_with_stderr(self) -> None:
        """Test executing command that writes to stderr."""
        executor = SubprocessExecutor()
        result = await executor.execute_command(
            command="echo 'error message' >&2",
            cwd="/tmp",
            env={"PATH": "/usr/bin"},
            timeout=5,
        )

        assert result.exit_code == 0
        assert "error message" in result.stderr

    @pytest.mark.asyncio
    async def test_execute_command_nonzero_exit(self) -> None:
        """Test executing command with non-zero exit code."""
        executor = SubprocessExecutor()
        result = await executor.execute_command(
            command="exit 42",
            cwd="/tmp",
            env={"PATH": "/usr/bin"},
            timeout=5,
        )

        assert result.exit_code == 42

    @pytest.mark.asyncio
    async def test_execute_command_invalid_cwd(self) -> None:
        """Test command execution with invalid working directory."""
        executor = SubprocessExecutor()

        with pytest.raises(OSError, match="Failed to execute command"):
            await executor.execute_command(
                command="echo test",
                cwd="/nonexistent/directory",
                env={"PATH": "/usr/bin"},
                timeout=5,
            )

    @pytest.mark.asyncio
    async def test_output_truncation_by_lines(self) -> None:
        """Test output truncation when exceeding line limit."""
        executor = SubprocessExecutor(max_output_lines=10)

        # Generate 30 lines of output (15 stdout + 15 stderr) to exceed limit of 10
        # Use bash explicitly to ensure brace expansion works
        command = "bash -c 'for i in {1..15}; do echo line_$i; done; for i in {1..15}; do echo err_$i >&2; done'"
        result = await executor.execute_command(
            command=command,
            cwd="/tmp",
            env={"PATH": "/usr/bin:/bin"},
            timeout=5,
        )

        assert result.exit_code == 0
        # Should be truncated
        assert "truncated" in result.stdout.lower() or "truncated" in result.stderr.lower()

    @pytest.mark.asyncio
    async def test_executor_name(self) -> None:
        """Test executor name."""
        executor = SubprocessExecutor()
        assert executor.get_name() == "subprocess"

    @pytest.mark.asyncio
    async def test_metadata_includes_uid_gid(self) -> None:
        """Test metadata includes UID/GID information."""
        executor = SubprocessExecutor(uid=1000, gid=1000)
        result = await executor.execute_command(
            command="echo test",
            cwd="/tmp",
            env={"PATH": "/usr/bin"},
            timeout=5,
        )

        assert result.metadata is not None
        assert result.metadata["uid"] == 1000
        assert result.metadata["gid"] == 1000
        assert "platform" in result.metadata

    @pytest.mark.asyncio
    @patch("sys.platform", "linux")
    async def test_privilege_dropping_on_unix(self) -> None:
        """Test privilege dropping is attempted on Unix systems."""
        executor = SubprocessExecutor(uid=1000, gid=1000)

        # Mock the subprocess creation to avoid actual privilege dropping
        with patch("asyncio.create_subprocess_shell") as mock_create:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"output", b"")
            mock_process.returncode = 0
            mock_create.return_value = mock_process

            await executor.execute_command(
                command="echo test",
                cwd="/tmp",
                env={"PATH": "/usr/bin"},
                timeout=5,
            )

            # Verify preexec_fn was provided
            assert mock_create.call_args.kwargs["preexec_fn"] is not None

    @pytest.mark.asyncio
    @patch("sys.platform", "win32")
    async def test_no_privilege_dropping_on_windows(self) -> None:
        """Test privilege dropping is skipped on Windows."""
        executor = SubprocessExecutor(uid=1000, gid=1000)

        with patch("asyncio.create_subprocess_shell") as mock_create:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"output", b"")
            mock_process.returncode = 0
            mock_create.return_value = mock_process

            await executor.execute_command(
                command="echo test",
                cwd="/tmp",
                env={"PATH": "/usr/bin"},
                timeout=5,
            )

            # Verify preexec_fn was NOT provided on Windows
            assert mock_create.call_args.kwargs["preexec_fn"] is None

    @pytest.mark.asyncio
    async def test_unicode_output_handling(self) -> None:
        """Test handling of unicode characters in output."""
        executor = SubprocessExecutor()
        result = await executor.execute_command(
            command="echo '你好世界 🌍'",
            cwd="/tmp",
            env={"PATH": "/usr/bin"},
            timeout=5,
        )

        assert result.exit_code == 0
        # Should handle unicode gracefully
        assert len(result.stdout) > 0

    @pytest.mark.asyncio
    async def test_environment_variables(self) -> None:
        """Test environment variables are passed correctly."""
        executor = SubprocessExecutor()
        result = await executor.execute_command(
            command="echo $TEST_VAR",
            cwd="/tmp",
            env={"PATH": "/usr/bin", "TEST_VAR": "custom_value"},
            timeout=5,
        )

        assert result.exit_code == 0
        assert "custom_value" in result.stdout

    @pytest.mark.asyncio
    async def test_working_directory(self) -> None:
        """Test working directory is set correctly."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            executor = SubprocessExecutor()
            result = await executor.execute_command(
                command="pwd",
                cwd=tmpdir,
                env={"PATH": "/usr/bin"},
                timeout=5,
            )

            assert result.exit_code == 0
            # Output should contain the temp directory path
            assert tmpdir in result.stdout

    @pytest.mark.asyncio
    async def test_multiline_output(self) -> None:
        """Test handling of multiline output."""
        executor = SubprocessExecutor()
        result = await executor.execute_command(
            command="echo 'line1'; echo 'line2'; echo 'line3'",
            cwd="/tmp",
            env={"PATH": "/usr/bin"},
            timeout=5,
        )

        assert result.exit_code == 0
        assert "line1" in result.stdout
        assert "line2" in result.stdout
        assert "line3" in result.stdout

    @pytest.mark.asyncio
    async def test_duration_tracking(self) -> None:
        """Test execution duration is tracked."""
        executor = SubprocessExecutor()
        result = await executor.execute_command(
            command="python3 -c 'import time; time.sleep(0.1)'",
            cwd="/tmp",
            env={"PATH": "/usr/bin:/usr/local/bin:/opt/homebrew/bin"},
            timeout=5,
        )

        assert result.exit_code == 0
        # Should take at least 100ms
        assert result.duration_ms >= 100
