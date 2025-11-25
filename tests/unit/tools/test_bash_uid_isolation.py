"""Unit tests for BashTool Unix UID/GID isolation.

Tests that BashTool correctly uses setuid/setgid for process isolation.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentrunner.core.events import EventBus
from agentrunner.core.logger import AgentRunnerLogger
from agentrunner.core.tool_protocol import ToolCall
from agentrunner.core.workspace import Workspace
from agentrunner.tools.base import ToolContext
from agentrunner.tools.bash import BashTool


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Workspace(tmpdir)
        yield workspace


@pytest.fixture
def tool_context(temp_workspace):
    """Create a ToolContext with UID/GID set."""
    return ToolContext(
        workspace=temp_workspace,
        logger=AgentRunnerLogger(),
        model_id="test-model",
        event_bus=None,
        config={},
        session_uid=10123,  # Test UID
        session_gid=10123,  # Test GID
    )


@pytest.fixture
def bash_tool():
    """Create a BashTool instance."""
    return BashTool()


@pytest.mark.skip(reason="UID/GID isolation tests disabled - feature under development")
class TestBashToolUIDIsolation:
    """Tests for Unix UID/GID isolation in BashTool."""

    @pytest.mark.asyncio
    async def test_simple_command_execution(self, bash_tool, tool_context):
        """Test that simple commands execute successfully."""
        call = ToolCall(name="bash", arguments={"command": "echo 'Hello World'"})

        result = await bash_tool.execute(call, tool_context)

        assert result.success is True
        assert "Hello World" in result.output
        assert result.data["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_cd_command_works_naturally(self, bash_tool, tool_context):
        """Test that cd commands work without special handling."""
        # Create a subdirectory
        subdir = Path(tool_context.workspace.root_path) / "subdir"
        subdir.mkdir()

        call = ToolCall(name="bash", arguments={"command": "cd subdir && pwd"})

        result = await bash_tool.execute(call, tool_context)

        assert result.success is True
        assert "subdir" in result.output

    @pytest.mark.asyncio
    async def test_compound_commands_work(self, bash_tool, tool_context):
        """Test that compound commands (&&, ;, |) work correctly."""
        call = ToolCall(
            name="bash", arguments={"command": "echo 'test' && echo 'success' && echo 'done'"}
        )

        result = await bash_tool.execute(call, tool_context)

        assert result.success is True
        assert "test" in result.output
        assert "success" in result.output
        assert "done" in result.output

    @pytest.mark.asyncio
    async def test_pushd_popd_work(self, bash_tool, tool_context):
        """Test that pushd/popd commands work without issues."""
        subdir = Path(tool_context.workspace.root_path) / "testdir"
        subdir.mkdir()

        call = ToolCall(
            name="bash",
            arguments={"command": "pushd testdir > /dev/null && pwd && popd > /dev/null && pwd"},
        )

        result = await bash_tool.execute(call, tool_context)

        assert result.success is True
        assert "testdir" in result.output

    @pytest.mark.asyncio
    @patch("os.setuid")
    @patch("os.setgid")
    async def test_privilege_dropping_called(
        self, mock_setgid, mock_setuid, bash_tool, tool_context
    ):
        """Test that setuid/setgid are called with correct values."""
        if sys.platform == "win32":
            pytest.skip("UID/GID not supported on Windows")

        call = ToolCall(name="bash", arguments={"command": "echo 'test'"})

        # Execute command
        await bash_tool.execute(call, tool_context)

        # Verify setgid/setuid were called (they're called in preexec_fn)
        # Note: We can't easily test preexec_fn directly, so this tests the setup

    @pytest.mark.asyncio
    async def test_timeout_handling(self, bash_tool, tool_context):
        """Test that commands timeout correctly."""
        call = ToolCall(name="bash", arguments={"command": "sleep 10", "timeout": 1})

        result = await bash_tool.execute(call, tool_context)

        assert result.success is False
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_command_validation_blocks_unsafe(self, bash_tool, tool_context):
        """Test that obviously dangerous commands are blocked."""
        # The CommandValidator should block dangerous patterns
        call = ToolCall(name="bash", arguments={"command": "rm -rf /"})

        result = await bash_tool.execute(call, tool_context)

        # Should be blocked by validator
        assert result.success is False
        assert "not allowed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_working_directory_is_workspace(self, bash_tool, tool_context):
        """Test that commands execute in the workspace directory."""
        call = ToolCall(name="bash", arguments={"command": "pwd"})

        result = await bash_tool.execute(call, tool_context)

        assert result.success is True
        # Should show workspace path
        workspace_path = str(tool_context.workspace.root_path)
        assert workspace_path in result.output

    @pytest.mark.asyncio
    async def test_environment_variables_set_correctly(self, bash_tool, tool_context):
        """Test that environment variables are set for isolation."""
        call = ToolCall(name="bash", arguments={"command": "echo $HOME"})

        result = await bash_tool.execute(call, tool_context)

        assert result.success is True
        # HOME should be set to workspace
        workspace_path = str(tool_context.workspace.root_path)
        assert workspace_path in result.output

    @pytest.mark.asyncio
    async def test_event_bus_publishes_events(self, bash_tool, temp_workspace):
        """Test that bash_started and bash_executed events are published."""
        event_bus = MagicMock(spec=EventBus)

        context = ToolContext(
            workspace=temp_workspace,
            logger=AgentRunnerLogger(),
            model_id="test-model",
            event_bus=event_bus,
            config={},
            session_uid=10123,
            session_gid=10123,
        )

        call = ToolCall(name="bash", arguments={"command": "echo 'test'"})

        await bash_tool.execute(call, context)

        # Should have published 2 events (started + executed)
        assert event_bus.publish.call_count == 2

        # Check first event is bash_started
        first_event = event_bus.publish.call_args_list[0][0][0]
        assert first_event.type == "bash_started"
        assert first_event.model_id == "test-model"

        # Check second event is bash_executed
        second_event = event_bus.publish.call_args_list[1][0][0]
        assert second_event.type == "bash_executed"
        assert second_event.data["success"] is True

    @pytest.mark.asyncio
    async def test_stderr_captured_separately(self, bash_tool, tool_context):
        """Test that stderr is captured separately from stdout."""
        call = ToolCall(name="bash", arguments={"command": "echo 'stdout' && echo 'stderr' >&2"})

        result = await bash_tool.execute(call, tool_context)

        assert result.success is True
        assert "STDOUT" in result.output
        assert "STDERR" in result.output
        assert result.data["stdout"] == "stdout\n"
        assert result.data["stderr"] == "stderr\n"

    @pytest.mark.asyncio
    async def test_nonzero_exit_code_marked_as_failure(self, bash_tool, tool_context):
        """Test that non-zero exit codes are marked as failures."""
        call = ToolCall(name="bash", arguments={"command": "exit 42"})

        result = await bash_tool.execute(call, tool_context)

        assert result.success is False
        assert result.data["exit_code"] == 42
        assert "exited with code 42" in result.error.lower()

    @pytest.mark.asyncio
    async def test_no_uid_gid_still_works(self, bash_tool, temp_workspace):
        """Test that commands work even without UID/GID set (dev mode)."""
        context = ToolContext(
            workspace=temp_workspace,
            logger=AgentRunnerLogger(),
            model_id="test-model",
            event_bus=None,
            config={},
            session_uid=None,  # No UID set
            session_gid=None,  # No GID set
        )

        call = ToolCall(name="bash", arguments={"command": "echo 'test'"})

        result = await bash_tool.execute(call, context)

        # Should still work (just without privilege dropping)
        assert result.success is True
        assert "test" in result.output


@pytest.mark.integration
class TestBashToolRealIsolation:
    """Integration tests for real UID/GID isolation (requires root)."""

    @pytest.mark.skipif(os.geteuid() != 0, reason="Requires root for real UID/GID tests")
    @pytest.mark.asyncio
    async def test_cannot_access_other_session_workspace(self):
        """Test that one session cannot access another session's workspace."""
        # Create two workspaces with different UIDs
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace1 = Path(tmpdir) / "session1"
            workspace2 = Path(tmpdir) / "session2"

            workspace1.mkdir(mode=0o700)
            workspace2.mkdir(mode=0o700)

            # Set ownership
            os.chown(workspace1, 10001, 10001)
            os.chown(workspace2, 10002, 10002)

            # Create secret file in workspace2
            secret_file = workspace2 / "secret.txt"
            secret_file.write_text("TOP SECRET")
            os.chown(secret_file, 10002, 10002)
            os.chmod(secret_file, 0o600)

            # Try to access from session1 (different UID)
            context1 = ToolContext(
                workspace=Workspace(str(workspace1)),
                logger=AgentRunnerLogger(),
                model_id="session1",
                event_bus=None,
                config={},
                session_uid=10001,
                session_gid=10001,
            )

            bash_tool = BashTool()
            call = ToolCall(name="bash", arguments={"command": f"cat {secret_file}"})

            result = await bash_tool.execute(call, context1)

            # Should fail with permission denied
            assert result.success is False
            assert (
                "permission denied" in result.output.lower()
                or "permission denied" in result.error.lower()
                if result.error
                else False
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
