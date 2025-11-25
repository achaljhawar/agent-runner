"""Tests for file tool event publishing.

Validates that file tools correctly publish file_created and file_modified events
when they modify files.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agentrunner.core.logger import AgentRunnerLogger
from agentrunner.core.tool_protocol import ToolCall
from agentrunner.core.workspace import Workspace
from agentrunner.tools.base import ToolContext
from agentrunner.tools.edit import EditFileTool, InsertLinesTool, MultiEditTool
from agentrunner.tools.file_io import CreateFileTool, WriteFileTool


@pytest.fixture
def temp_workspace(tmp_path):
    """Create temporary workspace."""
    return Workspace(str(tmp_path))


@pytest.fixture
def logger():
    """Create logger."""
    return AgentRunnerLogger()


@pytest.fixture
def mock_event_bus():
    """Create mock EventBus."""
    event_bus = MagicMock()
    event_bus.publish = MagicMock()
    return event_bus


@pytest.fixture
def context_with_eventbus(temp_workspace, logger, mock_event_bus):
    """Create ToolContext with EventBus."""
    return ToolContext(
        workspace=temp_workspace,
        logger=logger,
        model_id="test-model",
        event_bus=mock_event_bus,
    )


@pytest.fixture
def context_without_eventbus(temp_workspace, logger):
    """Create ToolContext without EventBus."""
    return ToolContext(
        workspace=temp_workspace,
        logger=logger,
        model_id="test-model",
        event_bus=None,
    )


class TestCreateFileToolEvents:
    """Test CreateFileTool event publishing."""

    @pytest.mark.asyncio
    async def test_publishes_file_created_event(self, context_with_eventbus):
        """CreateFileTool should publish file_created event when creating file."""
        tool = CreateFileTool()
        call = ToolCall(
            id="test-1",
            name="create_file",
            arguments={"file_path": "test.txt", "content": "Hello\nWorld"},
        )

        result = await tool.execute(call, context_with_eventbus)

        assert result.success
        assert context_with_eventbus.event_bus.publish.called

        # Verify event was published
        published_event = context_with_eventbus.event_bus.publish.call_args[0][0]
        assert published_event.type == "file_created"
        assert published_event.data["path"] == "test.txt"
        assert published_event.data["size"] > 0
        assert published_event.data["line_count"] == 2
        assert published_event.id  # Has unique event ID
        assert published_event.ts  # Has timestamp

    @pytest.mark.asyncio
    async def test_does_not_crash_without_eventbus(self, context_without_eventbus):
        """CreateFileTool should work when EventBus is None."""
        tool = CreateFileTool()
        call = ToolCall(
            id="test-2", name="create_file", arguments={"file_path": "test.txt", "content": "Hello"}
        )

        result = await tool.execute(call, context_without_eventbus)

        assert result.success
        assert result.output == "Created test.txt"


class TestWriteFileToolEvents:
    """Test WriteFileTool event publishing."""

    @pytest.mark.asyncio
    async def test_publishes_file_created_for_new_file(self, context_with_eventbus):
        """WriteFileTool should publish file_created when file doesn't exist."""
        tool = WriteFileTool()
        call = ToolCall(
            id="test-3",
            name="write_file",
            arguments={"file_path": "new.txt", "content": "New file", "overwrite": True},
        )

        result = await tool.execute(call, context_with_eventbus)

        assert result.success
        assert context_with_eventbus.event_bus.publish.called

        published_event = context_with_eventbus.event_bus.publish.call_args[0][0]
        assert published_event.type == "file_created"
        assert published_event.data["path"] == "new.txt"

    @pytest.mark.asyncio
    async def test_publishes_file_modified_for_existing_file(self, context_with_eventbus):
        """WriteFileTool should publish file_modified when file exists."""
        # Create existing file
        workspace_path = Path(context_with_eventbus.workspace.root_path)
        existing_file = workspace_path / "existing.txt"
        existing_file.write_text("Old content")

        tool = WriteFileTool()
        call = ToolCall(
            id="test-4",
            name="write_file",
            arguments={"file_path": "existing.txt", "content": "New content", "overwrite": True},
        )

        result = await tool.execute(call, context_with_eventbus)

        assert result.success
        assert context_with_eventbus.event_bus.publish.called

        published_event = context_with_eventbus.event_bus.publish.call_args[0][0]
        assert published_event.type == "file_modified"
        assert published_event.data["path"] == "existing.txt"
        assert "old_size" in published_event.data
        assert "new_size" in published_event.data
        assert published_event.data["line_count"] == 1


class TestEditFileToolEvents:
    """Test EditFileTool event publishing."""

    @pytest.mark.asyncio
    async def test_publishes_file_modified_event(self, context_with_eventbus):
        """EditFileTool should publish file_modified event."""
        # Create file
        workspace_path = Path(context_with_eventbus.workspace.root_path)
        test_file = workspace_path / "test.py"
        test_file.write_text("def foo():\n    return 1\n")

        tool = EditFileTool()
        call = ToolCall(
            id="test-5",
            name="edit_file",
            arguments={"file_path": "test.py", "old_string": "return 1", "new_string": "return 42"},
        )

        result = await tool.execute(call, context_with_eventbus)

        assert result.success
        assert context_with_eventbus.event_bus.publish.called

        published_event = context_with_eventbus.event_bus.publish.call_args[0][0]
        assert published_event.type == "file_modified"
        assert published_event.data["path"] == "test.py"
        assert published_event.data["old_size"] > 0
        assert published_event.data["new_size"] > 0


class TestMultiEditToolEvents:
    """Test MultiEditTool event publishing."""

    @pytest.mark.asyncio
    async def test_publishes_file_modified_event(self, context_with_eventbus):
        """MultiEditTool should publish file_modified event."""
        # Create file
        workspace_path = Path(context_with_eventbus.workspace.root_path)
        test_file = workspace_path / "multi.txt"
        test_file.write_text("line1\nline2\nline3\n")

        tool = MultiEditTool()
        call = ToolCall(
            id="test-6",
            name="multi_edit",
            arguments={
                "file_path": "multi.txt",
                "edits": [
                    {"old_string": "line1", "new_string": "LINE1"},
                    {"old_string": "line2", "new_string": "LINE2"},
                ],
            },
        )

        result = await tool.execute(call, context_with_eventbus)

        assert result.success
        assert context_with_eventbus.event_bus.publish.called

        published_event = context_with_eventbus.event_bus.publish.call_args[0][0]
        assert published_event.type == "file_modified"
        assert published_event.data["path"] == "multi.txt"


class TestInsertLinesToolEvents:
    """Test InsertLinesTool event publishing."""

    @pytest.mark.asyncio
    async def test_publishes_file_modified_event(self, context_with_eventbus):
        """InsertLinesTool should publish file_modified event."""
        # Create file
        workspace_path = Path(context_with_eventbus.workspace.root_path)
        test_file = workspace_path / "insert.txt"
        test_file.write_text("line1\nline2\n")

        tool = InsertLinesTool()
        call = ToolCall(
            id="test-7",
            name="insert_lines",
            arguments={"file_path": "insert.txt", "line": 2, "content": "INSERTED\n"},
        )

        result = await tool.execute(call, context_with_eventbus)

        assert result.success
        assert context_with_eventbus.event_bus.publish.called

        published_event = context_with_eventbus.event_bus.publish.call_args[0][0]
        assert published_event.type == "file_modified"
        assert published_event.data["path"] == "insert.txt"


class TestEventStructure:
    """Test event structure matches TypeScript interface."""

    @pytest.mark.asyncio
    async def test_file_created_event_structure(self, context_with_eventbus):
        """file_created event should match TypeScript FileCreatedEvent interface."""
        tool = CreateFileTool()
        call = ToolCall(
            id="test-8",
            name="create_file",
            arguments={"file_path": "structure.txt", "content": "test"},
        )

        await tool.execute(call, context_with_eventbus)

        published_event = context_with_eventbus.event_bus.publish.call_args[0][0]

        # Validate structure matches TypeScript interface
        assert published_event.type == "file_created"
        assert "id" in dir(published_event)  # Has event_id
        assert "ts" in dir(published_event)  # Has timestamp
        assert "data" in dir(published_event)

        # Validate data fields
        data = published_event.data
        assert "path" in data
        assert "size" in data
        assert "line_count" in data

        # Validate types
        assert isinstance(data["path"], str)
        assert isinstance(data["size"], int)
        assert isinstance(data["line_count"], int)

    @pytest.mark.asyncio
    async def test_file_modified_event_structure(self, context_with_eventbus):
        """file_modified event should match TypeScript FileModifiedEvent interface."""
        # Create existing file
        workspace_path = Path(context_with_eventbus.workspace.root_path)
        test_file = workspace_path / "modify.txt"
        test_file.write_text("old")

        tool = WriteFileTool()
        call = ToolCall(
            id="test-9",
            name="write_file",
            arguments={"file_path": "modify.txt", "content": "new", "overwrite": True},
        )

        await tool.execute(call, context_with_eventbus)

        published_event = context_with_eventbus.event_bus.publish.call_args[0][0]

        # Validate structure matches TypeScript interface
        assert published_event.type == "file_modified"
        assert "id" in dir(published_event)
        assert "ts" in dir(published_event)

        # Validate data fields
        data = published_event.data
        assert "path" in data
        assert "old_size" in data
        assert "new_size" in data
        assert "line_count" in data

        # Validate types
        assert isinstance(data["path"], str)
        assert isinstance(data["old_size"], int)
        assert isinstance(data["new_size"], int)
        assert isinstance(data["line_count"], int)


class TestEventOptional:
    """Test that tools work correctly when EventBus is None."""

    @pytest.mark.asyncio
    async def test_create_file_works_without_eventbus(self, context_without_eventbus):
        """Tools should work when event_bus is None (backward compatibility)."""
        tool = CreateFileTool()
        call = ToolCall(
            id="test-10",
            name="create_file",
            arguments={"file_path": "no_event.txt", "content": "test"},
        )

        result = await tool.execute(call, context_without_eventbus)

        assert result.success
        assert result.files_changed == [
            str(Path(context_without_eventbus.workspace.root_path) / "no_event.txt")
        ]

    @pytest.mark.asyncio
    async def test_edit_file_works_without_eventbus(self, context_without_eventbus):
        """EditFileTool should work when event_bus is None."""
        # Create file
        workspace_path = Path(context_without_eventbus.workspace.root_path)
        test_file = workspace_path / "edit_no_event.txt"
        test_file.write_text("old")

        tool = EditFileTool()
        call = ToolCall(
            id="test-11",
            name="edit_file",
            arguments={"file_path": "edit_no_event.txt", "old_string": "old", "new_string": "new"},
        )

        result = await tool.execute(call, context_without_eventbus)

        assert result.success
        assert test_file.read_text() == "new"
