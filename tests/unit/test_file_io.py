"""Unit tests for file I/O tools."""

import pytest

from agentrunner.core.logger import AgentRunnerLogger
from agentrunner.core.tool_protocol import ToolCall
from agentrunner.core.workspace import Workspace
from agentrunner.tools.base import ToolContext
from agentrunner.tools.file_io import CreateFileTool, DeleteFileTool, WriteFileTool
from agentrunner.tools.read_file import ReadFileTool


@pytest.fixture
def workspace(tmp_path):
    """Create a temporary workspace."""
    return Workspace(str(tmp_path))


@pytest.fixture
def logger():
    """Create a test logger."""
    return AgentRunnerLogger(log_dir=None)


@pytest.fixture
def context(workspace, logger):
    """Create a tool context."""
    return ToolContext(workspace=workspace, logger=logger, model_id="test-model")


@pytest.fixture
def read_tool():
    """Create ReadFileTool instance."""
    return ReadFileTool()


@pytest.fixture
def create_tool():
    """Create CreateFileTool instance."""
    return CreateFileTool()


@pytest.fixture
def write_tool():
    """Create WriteFileTool instance."""
    return WriteFileTool()


@pytest.fixture
def delete_tool():
    """Create DeleteFileTool instance."""
    return DeleteFileTool()


# ReadFileTool tests


@pytest.mark.asyncio
async def test_read_file_basic(read_tool, context, tmp_path):
    """Test basic file reading."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("line 1\nline 2\nline 3")

    call = ToolCall(
        id="1",
        name="read_file",
        arguments={"target_file": "test.txt"},
    )

    result = await read_tool.execute(call, context)

    assert result.success
    assert "     1|line 1" in result.output
    assert "     2|line 2" in result.output
    assert "     3|line 3" in result.output
    assert result.data["total_lines"] == 3


@pytest.mark.asyncio
async def test_read_file_with_offset(read_tool, context, tmp_path):
    """Test reading file with offset."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("\n".join([f"line {i}" for i in range(1, 11)]))

    call = ToolCall(
        id="1",
        name="read_file",
        arguments={"target_file": "test.txt", "offset": 5},
    )

    result = await read_tool.execute(call, context)

    assert result.success
    assert "     5|line 5" in result.output
    assert "     1|line 1" not in result.output
    assert result.data["offset"] == 5


@pytest.mark.asyncio
async def test_read_file_with_limit(read_tool, context, tmp_path):
    """Test reading file with limit."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("\n".join([f"line {i}" for i in range(1, 101)]))

    call = ToolCall(
        id="1",
        name="read_file",
        arguments={"target_file": "test.txt", "offset": 1, "limit": 5},
    )

    result = await read_tool.execute(call, context)

    assert result.success
    assert result.data["lines_shown"] == 5
    assert "     1|line 1" in result.output
    assert "     5|line 5" in result.output
    assert "line 6" not in result.output


@pytest.mark.asyncio
async def test_read_file_with_offset_and_limit(read_tool, context, tmp_path):
    """Test reading file with both offset and limit."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("\n".join([f"line {i}" for i in range(1, 101)]))

    call = ToolCall(
        id="1",
        name="read_file",
        arguments={"target_file": "test.txt", "offset": 50, "limit": 3},
    )

    result = await read_tool.execute(call, context)

    assert result.success
    assert result.data["lines_shown"] == 3
    assert "    50|line 50" in result.output
    assert "    51|line 51" in result.output
    assert "    52|line 52" in result.output
    assert "line 49" not in result.output
    assert "line 53" not in result.output


@pytest.mark.asyncio
async def test_read_file_empty(read_tool, context, tmp_path):
    """Test reading empty file."""
    test_file = tmp_path / "empty.txt"
    test_file.write_text("")

    call = ToolCall(
        id="1",
        name="read_file",
        arguments={"target_file": "empty.txt"},
    )

    result = await read_tool.execute(call, context)

    assert result.success
    assert result.output == "File is empty."
    assert result.data["total_lines"] == 0


@pytest.mark.asyncio
async def test_read_file_not_found(read_tool, context):
    """Test reading non-existent file."""
    call = ToolCall(
        id="1",
        name="read_file",
        arguments={"target_file": "nonexistent.txt"},
    )

    result = await read_tool.execute(call, context)

    assert not result.success
    assert result.error_code == "E_NOT_FOUND"
    assert "Could not find file" in result.error


@pytest.mark.asyncio
async def test_read_file_binary(read_tool, context, tmp_path):
    """Test reading binary file."""
    test_file = tmp_path / "binary.bin"
    test_file.write_bytes(b"\x00\x01\x02\x03\xff")

    call = ToolCall(
        id="1",
        name="read_file",
        arguments={"target_file": "binary.bin"},
    )

    result = await read_tool.execute(call, context)

    assert result.success
    assert "Binary file" in result.output
    assert result.data["binary"] is True
    assert result.data["size"] == 5


@pytest.mark.asyncio
async def test_read_file_outside_workspace(read_tool, context):
    """Test reading file outside workspace."""
    call = ToolCall(
        id="1",
        name="read_file",
        arguments={"target_file": "/etc/passwd"},
    )

    result = await read_tool.execute(call, context)

    assert not result.success
    assert result.error_code == "E_PERMISSIONS"


@pytest.mark.asyncio
async def test_read_file_missing_path(read_tool, context):
    """Test reading without file_path."""
    call = ToolCall(
        id="1",
        name="read_file",
        arguments={},
    )

    result = await read_tool.execute(call, context)

    assert not result.success
    assert result.error_code == "E_VALIDATION"


@pytest.mark.asyncio
async def test_read_file_invalid_offset(read_tool, context, tmp_path):
    """Test reading with invalid offset."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("line 1\nline 2")

    call = ToolCall(
        id="1",
        name="read_file",
        arguments={"target_file": "test.txt", "offset": 0},
    )

    result = await read_tool.execute(call, context)

    assert not result.success
    assert result.error_code == "E_VALIDATION"
    assert "offset must be >= 1" in result.error


@pytest.mark.asyncio
async def test_read_file_is_directory(read_tool, context, tmp_path):
    """Test reading a directory."""
    test_dir = tmp_path / "testdir"
    test_dir.mkdir()

    call = ToolCall(
        id="1",
        name="read_file",
        arguments={"target_file": "testdir"},
    )

    result = await read_tool.execute(call, context)

    assert not result.success
    assert result.error_code in ("E_VALIDATION", "E_NOT_FOUND")
    assert "Could not find file" in result.error


@pytest.mark.asyncio
async def test_read_file_long_lines(read_tool, context, tmp_path):
    """Test reading file with very long lines (truncation)."""
    test_file = tmp_path / "long.txt"
    # Create a line longer than 2000 characters
    long_line = "x" * 2500
    test_file.write_text(f"{long_line}\nshort line")

    call = ToolCall(
        id="1",
        name="read_file",
        arguments={"target_file": "long.txt"},
    )

    result = await read_tool.execute(call, context)

    # xAI version handles long lines differently - it includes them but limits total output
    # The line won't be truncated per-line, but total content may be limited by token count
    assert result.success
    assert "xxxx" in result.output  # Content is present


# CreateFileTool tests


@pytest.mark.asyncio
async def test_create_file_success(create_tool, context, tmp_path):
    """Test creating a new file."""
    call = ToolCall(
        id="1",
        name="create_file",
        arguments={"file_path": "newfile.txt", "content": "Test content"},
    )

    result = await create_tool.execute(call, context)

    assert result.success
    assert "Created" in result.output
    assert result.files_changed == [str(tmp_path / "newfile.txt")]

    # Verify file was created
    test_file = tmp_path / "newfile.txt"
    assert test_file.exists()
    assert test_file.read_text() == "Test content"


@pytest.mark.asyncio
async def test_create_file_with_parent_dirs(create_tool, context, tmp_path):
    """Test creating file with nested parent directories."""
    call = ToolCall(
        id="1",
        name="create_file",
        arguments={"file_path": "nested/dir/file.txt", "content": "Content"},
    )

    result = await create_tool.execute(call, context)

    assert result.success
    test_file = tmp_path / "nested" / "dir" / "file.txt"
    assert test_file.exists()
    assert test_file.read_text() == "Content"


@pytest.mark.asyncio
async def test_create_file_with_diff(create_tool, context, tmp_path):
    """Test that create_file generates diff."""
    call = ToolCall(
        id="1",
        name="create_file",
        arguments={"file_path": "file.txt", "content": "Line 1\nLine 2\n"},
    )

    result = await create_tool.execute(call, context)

    assert result.success
    assert result.diffs is not None
    assert len(result.diffs) == 1
    assert "format" in result.diffs[0]
    assert result.diffs[0]["format"] == "unified"


@pytest.mark.asyncio
async def test_create_file_empty_content(create_tool, context, tmp_path):
    """Test creating file with empty content."""
    call = ToolCall(
        id="1",
        name="create_file",
        arguments={"file_path": "empty.txt", "content": ""},
    )

    result = await create_tool.execute(call, context)

    assert result.success
    test_file = tmp_path / "empty.txt"
    assert test_file.exists()
    assert test_file.read_text() == ""


@pytest.mark.asyncio
async def test_create_file_missing_path(create_tool, context):
    """Test creating file without file_path."""
    call = ToolCall(
        id="1",
        name="create_file",
        arguments={"content": "content"},
    )

    result = await create_tool.execute(call, context)

    assert not result.success
    assert result.error_code == "E_VALIDATION"


@pytest.mark.asyncio
async def test_create_file_missing_content(create_tool, context):
    """Test creating file without content."""
    call = ToolCall(
        id="1",
        name="create_file",
        arguments={"file_path": "test.txt"},
    )

    result = await create_tool.execute(call, context)

    assert not result.success
    assert result.error_code == "E_VALIDATION"


@pytest.mark.asyncio
async def test_create_file_outside_workspace(create_tool, context):
    """Test creating file outside workspace."""
    call = ToolCall(
        id="1",
        name="create_file",
        arguments={"file_path": "/etc/test.txt", "content": "content"},
    )

    result = await create_tool.execute(call, context)

    assert not result.success
    assert result.error_code == "E_PERMISSIONS"


def test_create_tool_definition(create_tool):
    """Test CreateFileTool definition."""
    definition = create_tool.get_definition()

    assert definition.name == "create_file"
    assert definition.description
    assert "file_path" in definition.parameters["properties"]
    assert "content" in definition.parameters["properties"]
    assert definition.parameters["required"] == ["file_path", "content"]
    # create_file should NOT require read-first
    # Note: requires_read_first removed - LLM learns from natural feedback


@pytest.mark.asyncio
async def test_create_file_invalid_parent_path(create_tool, context, tmp_path):
    """Test creating file with invalid parent directory path."""
    # Create a file where we want to create a directory
    blocking_file = tmp_path / "blocker.txt"
    blocking_file.write_text("blocking")

    call = ToolCall(
        id="1",
        name="create_file",
        arguments={"file_path": "blocker.txt/subfile.txt", "content": "test"},
    )

    result = await create_tool.execute(call, context)

    # Should fail because can't create directory through a file
    assert not result.success
    assert "parent directory" in result.error.lower() or "permission" in result.error.lower()


# WriteFileTool tests


@pytest.mark.asyncio
async def test_write_file_create(write_tool, context, tmp_path):
    """Test creating a new file."""
    call = ToolCall(
        id="1",
        name="write_file",
        arguments={"file_path": "new.txt", "content": "Hello, World!"},
    )

    result = await write_tool.execute(call, context)

    assert result.success
    assert "File created" in result.output
    assert result.files_changed == ["new.txt"]

    test_file = tmp_path / "new.txt"
    assert test_file.exists()
    assert test_file.read_text() == "Hello, World!"


@pytest.mark.asyncio
async def test_write_file_overwrite(write_tool, context, tmp_path):
    """Test overwriting an existing file."""
    test_file = tmp_path / "existing.txt"
    test_file.write_text("old content")

    call = ToolCall(
        id="1",
        name="write_file",
        arguments={"file_path": "existing.txt", "content": "new content"},
    )

    result = await write_tool.execute(call, context)

    assert result.success
    assert "File updated" in result.output
    assert result.files_changed == ["existing.txt"]
    assert result.diffs is not None
    assert len(result.diffs) > 0

    assert test_file.read_text() == "new content"


@pytest.mark.asyncio
async def test_write_file_diff_generation(write_tool, context, tmp_path):
    """Test diff generation when overwriting."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("line 1\nline 2\nline 3")

    call = ToolCall(
        id="1",
        name="write_file",
        arguments={"file_path": "test.txt", "content": "line 1\nmodified line 2\nline 3"},
    )

    result = await write_tool.execute(call, context)

    assert result.success
    assert result.diffs is not None
    assert len(result.diffs) > 0
    diff_dict = result.diffs[0]
    assert "path" in diff_dict
    assert "diff" in diff_dict
    diff_text = diff_dict["diff"]
    assert "@@" in diff_text
    assert "-line 2" in diff_text
    assert "+modified line 2" in diff_text


@pytest.mark.asyncio
async def test_write_file_no_overwrite(write_tool, context, tmp_path):
    """Test writing with overwrite=false to existing file."""
    test_file = tmp_path / "existing.txt"
    test_file.write_text("content")

    call = ToolCall(
        id="1",
        name="write_file",
        arguments={
            "file_path": "existing.txt",
            "content": "new content",
            "overwrite": False,
        },
    )

    result = await write_tool.execute(call, context)

    assert not result.success
    assert result.error_code == "E_VALIDATION"
    assert "overwrite=false" in result.error


@pytest.mark.asyncio
async def test_write_file_create_parent_dirs(write_tool, context, tmp_path):
    """Test creating parent directories."""
    call = ToolCall(
        id="1",
        name="write_file",
        arguments={"file_path": "subdir/nested/file.txt", "content": "content"},
    )

    result = await write_tool.execute(call, context)

    assert result.success
    test_file = tmp_path / "subdir" / "nested" / "file.txt"
    assert test_file.exists()
    assert test_file.read_text() == "content"


@pytest.mark.asyncio
async def test_write_file_outside_workspace(write_tool, context):
    """Test writing file outside workspace."""
    call = ToolCall(
        id="1",
        name="write_file",
        arguments={"file_path": "/tmp/test.txt", "content": "content"},
    )

    result = await write_tool.execute(call, context)

    assert not result.success
    assert result.error_code == "E_PERMISSIONS"


@pytest.mark.asyncio
async def test_write_file_missing_path(write_tool, context):
    """Test writing without file_path."""
    call = ToolCall(
        id="1",
        name="write_file",
        arguments={"content": "content"},
    )

    result = await write_tool.execute(call, context)

    assert not result.success
    assert result.error_code == "E_VALIDATION"


@pytest.mark.asyncio
async def test_write_file_missing_content(write_tool, context):
    """Test writing without content."""
    call = ToolCall(
        id="1",
        name="write_file",
        arguments={"file_path": "test.txt"},
    )

    result = await write_tool.execute(call, context)

    assert not result.success
    assert result.error_code == "E_VALIDATION"


@pytest.mark.asyncio
async def test_write_file_empty_content(write_tool, context, tmp_path):
    """Test writing empty content."""
    call = ToolCall(
        id="1",
        name="write_file",
        arguments={"file_path": "empty.txt", "content": ""},
    )

    result = await write_tool.execute(call, context)

    assert result.success
    test_file = tmp_path / "empty.txt"
    assert test_file.exists()
    assert test_file.read_text() == ""


@pytest.mark.asyncio
async def test_write_file_invalid_parent_path(write_tool, context, tmp_path):
    """Test writing file with invalid parent directory path."""
    # Create a file where we want to create a directory
    blocking_file = tmp_path / "blocker.txt"
    blocking_file.write_text("blocking")

    call = ToolCall(
        id="1",
        name="write_file",
        arguments={"file_path": "blocker.txt/subfile.txt", "content": "test"},
    )

    result = await write_tool.execute(call, context)

    # Should fail
    assert not result.success
    assert result.error_code == "E_PERMISSIONS"


@pytest.mark.asyncio
async def test_write_file_read_existing_for_diff(write_tool, context, tmp_path):
    """Test that write_file reads existing file for diff generation."""
    test_file = tmp_path / "existing.txt"
    test_file.write_text("old content")

    call = ToolCall(
        id="1",
        name="write_file",
        arguments={"file_path": "existing.txt", "content": "new content"},
    )

    result = await write_tool.execute(call, context)

    assert result.success
    # Should have generated a diff
    assert result.diffs is not None
    assert len(result.diffs) > 0


# DeleteFileTool tests


@pytest.mark.asyncio
async def test_delete_file_success(delete_tool, context, tmp_path):
    """Test deleting an existing file."""
    test_file = tmp_path / "delete_me.txt"
    test_file.write_text("content")

    call = ToolCall(
        id="1",
        name="delete_file",
        arguments={"file_path": "delete_me.txt"},
    )

    result = await delete_tool.execute(call, context)

    assert result.success
    assert "File deleted" in result.output
    assert result.files_changed == ["delete_me.txt"]
    assert not test_file.exists()


@pytest.mark.asyncio
async def test_delete_file_not_found(delete_tool, context):
    """Test deleting non-existent file."""
    call = ToolCall(
        id="1",
        name="delete_file",
        arguments={"file_path": "nonexistent.txt"},
    )

    result = await delete_tool.execute(call, context)

    assert not result.success
    assert result.error_code == "E_NOT_FOUND"


@pytest.mark.asyncio
async def test_delete_file_is_directory(delete_tool, context, tmp_path):
    """Test deleting a directory."""
    test_dir = tmp_path / "testdir"
    test_dir.mkdir()

    call = ToolCall(
        id="1",
        name="delete_file",
        arguments={"file_path": "testdir"},
    )

    result = await delete_tool.execute(call, context)

    assert not result.success
    assert result.error_code == "E_VALIDATION"
    assert "Not a file" in result.error


@pytest.mark.asyncio
async def test_delete_file_outside_workspace(delete_tool, context):
    """Test deleting file outside workspace."""
    call = ToolCall(
        id="1",
        name="delete_file",
        arguments={"file_path": "/etc/passwd"},
    )

    result = await delete_tool.execute(call, context)

    assert not result.success
    assert result.error_code == "E_PERMISSIONS"


@pytest.mark.asyncio
async def test_delete_file_missing_path(delete_tool, context):
    """Test deleting without file_path."""
    call = ToolCall(
        id="1",
        name="delete_file",
        arguments={},
    )

    result = await delete_tool.execute(call, context)

    assert not result.success
    assert result.error_code == "E_VALIDATION"


# Tool definition tests


def test_read_tool_definition(read_tool):
    """Test ReadFileTool definition."""
    definition = read_tool.get_definition()

    assert definition.name == "read_file"
    assert definition.description
    assert "target_file" in definition.parameters["properties"]
    assert "offset" in definition.parameters["properties"]
    assert "limit" in definition.parameters["properties"]
    assert definition.parameters["required"] == ["target_file"]


def test_write_tool_definition(write_tool):
    """Test WriteFileTool definition."""
    definition = write_tool.get_definition()

    assert definition.name == "write_file"
    assert definition.description
    assert "file_path" in definition.parameters["properties"]
    assert "content" in definition.parameters["properties"]
    assert "overwrite" in definition.parameters["properties"]
    assert definition.parameters["required"] == ["file_path", "content"]
    # Note: requires_read_first removed - LLM learns from natural feedback


def test_delete_tool_definition(delete_tool):
    """Test DeleteFileTool definition."""
    definition = delete_tool.get_definition()

    assert definition.name == "delete_file"
    assert definition.description
    assert "file_path" in definition.parameters["properties"]
    assert definition.parameters["required"] == ["file_path"]
    assert definition.safety["requires_confirmation"] is True
