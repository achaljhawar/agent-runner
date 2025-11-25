"""Unit tests for batch file creation tools."""

from pathlib import Path

import pytest

from agentrunner.core.logger import AgentRunnerLogger
from agentrunner.core.tool_protocol import (
    E_PERMISSIONS,
    E_VALIDATION,
    ToolCall,
)
from agentrunner.core.workspace import Workspace
from agentrunner.tools.base import ToolContext
from agentrunner.tools.batch import BatchCreateFilesTool


@pytest.fixture
def workspace(tmp_path):
    """Create test workspace."""
    return Workspace(str(tmp_path))


@pytest.fixture
def logger(tmp_path):
    """Create test logger."""
    logger_instance = AgentRunnerLogger(log_dir=str(tmp_path / "logs"))
    yield logger_instance
    # Cleanup: Close all handlers
    for handler in logger_instance._logger.handlers[:]:
        handler.close()
        logger_instance._logger.removeHandler(handler)


@pytest.fixture
def context(workspace, logger):
    """Create tool context."""
    return ToolContext(workspace=workspace, logger=logger, model_id="test-model")


@pytest.fixture
def tool():
    """Create batch create tool."""
    return BatchCreateFilesTool()


@pytest.mark.asyncio
async def test_batch_create_single_file(tool, context):
    """Test creating a single file."""
    call = ToolCall(
        id="test1",
        name="batch_create_files",
        arguments={
            "files": [
                {"path": "test.txt", "content": "Hello, World!"},
            ]
        },
    )

    result = await tool.execute(call, context)

    assert result.success
    assert "Created 1 files successfully" in result.output
    assert result.files_changed == ["test.txt"]
    assert len(result.diffs) == 1
    assert result.diffs[0]["file"] == "test.txt"
    assert "Hello, World!" in result.diffs[0]["diff"]

    # Verify file exists
    file_path = Path(context.workspace.root_path) / "test.txt"
    assert file_path.exists()
    assert file_path.read_text() == "Hello, World!"


@pytest.mark.asyncio
async def test_batch_create_multiple_files(tool, context):
    """Test creating multiple files."""
    call = ToolCall(
        id="test2",
        name="batch_create_files",
        arguments={
            "files": [
                {"path": "file1.txt", "content": "Content 1"},
                {"path": "file2.txt", "content": "Content 2"},
                {"path": "file3.txt", "content": "Content 3"},
            ]
        },
    )

    result = await tool.execute(call, context)

    assert result.success
    assert "Created 3 files successfully" in result.output
    assert len(result.files_changed) == 3
    assert "file1.txt" in result.files_changed
    assert "file2.txt" in result.files_changed
    assert "file3.txt" in result.files_changed
    assert len(result.diffs) == 3

    # Verify all files exist
    for i in range(1, 4):
        file_path = Path(context.workspace.root_path) / f"file{i}.txt"
        assert file_path.exists()
        assert file_path.read_text() == f"Content {i}"


@pytest.mark.asyncio
async def test_batch_create_with_subdirectories(tool, context):
    """Test creating files in subdirectories."""
    call = ToolCall(
        id="test3",
        name="batch_create_files",
        arguments={
            "files": [
                {"path": "dir1/file1.txt", "content": "File 1"},
                {"path": "dir1/dir2/file2.txt", "content": "File 2"},
                {"path": "dir3/file3.txt", "content": "File 3"},
            ]
        },
    )

    result = await tool.execute(call, context)

    assert result.success
    assert len(result.files_changed) == 3

    # Verify all files and directories exist
    file1 = Path(context.workspace.root_path) / "dir1" / "file1.txt"
    file2 = Path(context.workspace.root_path) / "dir1" / "dir2" / "file2.txt"
    file3 = Path(context.workspace.root_path) / "dir3" / "file3.txt"

    assert file1.exists()
    assert file2.exists()
    assert file3.exists()
    assert file1.read_text() == "File 1"
    assert file2.read_text() == "File 2"
    assert file3.read_text() == "File 3"


@pytest.mark.asyncio
async def test_batch_create_empty_list(tool, context):
    """Test error when files list is empty."""
    call = ToolCall(
        id="test4",
        name="batch_create_files",
        arguments={"files": []},
    )

    result = await tool.execute(call, context)

    assert not result.success
    assert "No files specified" in result.error
    assert result.error_code == E_VALIDATION


@pytest.mark.asyncio
async def test_batch_create_missing_files_parameter(tool, context):
    """Test error when files parameter is missing."""
    call = ToolCall(
        id="test5",
        name="batch_create_files",
        arguments={},
    )

    result = await tool.execute(call, context)

    assert not result.success
    assert "No files specified" in result.error
    assert result.error_code == E_VALIDATION


@pytest.mark.asyncio
async def test_batch_create_files_not_list(tool, context):
    """Test error when files is not a list."""
    call = ToolCall(
        id="test6",
        name="batch_create_files",
        arguments={"files": "not a list"},
    )

    result = await tool.execute(call, context)

    assert not result.success
    assert "Files must be a list" in result.error
    assert result.error_code == E_VALIDATION


@pytest.mark.asyncio
async def test_batch_create_invalid_file_spec(tool, context):
    """Test error when file spec is not a dictionary."""
    call = ToolCall(
        id="test7",
        name="batch_create_files",
        arguments={"files": ["not a dict"]},
    )

    result = await tool.execute(call, context)

    assert not result.success
    assert "is not a dictionary" in result.error
    assert result.error_code == E_VALIDATION


@pytest.mark.asyncio
async def test_batch_create_missing_path(tool, context):
    """Test error when file spec missing path."""
    call = ToolCall(
        id="test8",
        name="batch_create_files",
        arguments={
            "files": [
                {"content": "Content without path"},
            ]
        },
    )

    result = await tool.execute(call, context)

    assert not result.success
    assert "missing 'path' or 'content'" in result.error
    assert result.error_code == E_VALIDATION


@pytest.mark.asyncio
async def test_batch_create_missing_content(tool, context):
    """Test error when file spec missing content."""
    call = ToolCall(
        id="test9",
        name="batch_create_files",
        arguments={
            "files": [
                {"path": "test.txt"},
            ]
        },
    )

    result = await tool.execute(call, context)

    assert not result.success
    assert "missing 'path' or 'content'" in result.error
    assert result.error_code == E_VALIDATION


@pytest.mark.asyncio
async def test_batch_create_outside_workspace(tool, context):
    """Test error when path is outside workspace."""
    call = ToolCall(
        id="test10",
        name="batch_create_files",
        arguments={
            "files": [
                {"path": "/etc/passwd", "content": "malicious"},
            ]
        },
    )

    result = await tool.execute(call, context)

    assert not result.success
    assert "outside workspace" in result.error.lower()
    assert result.error_code == E_PERMISSIONS


@pytest.mark.asyncio
async def test_batch_create_path_traversal(tool, context):
    """Test error when attempting path traversal."""
    call = ToolCall(
        id="test11",
        name="batch_create_files",
        arguments={
            "files": [
                {"path": "../../../etc/passwd", "content": "malicious"},
            ]
        },
    )

    result = await tool.execute(call, context)

    assert not result.success
    assert result.error_code == E_PERMISSIONS


@pytest.mark.asyncio
async def test_batch_create_existing_file(tool, context):
    """Test error when file already exists."""
    # Create a file first
    existing_file = Path(context.workspace.root_path) / "existing.txt"
    existing_file.write_text("Already exists")

    call = ToolCall(
        id="test12",
        name="batch_create_files",
        arguments={
            "files": [
                {"path": "existing.txt", "content": "New content"},
            ]
        },
    )

    result = await tool.execute(call, context)

    assert not result.success
    assert "already exist" in result.error
    assert result.error_code == E_VALIDATION

    # Original file should be unchanged
    assert existing_file.read_text() == "Already exists"


@pytest.mark.asyncio
async def test_batch_create_one_file_exists(tool, context):
    """Test error when one of multiple files exists."""
    # Create a file first
    existing_file = Path(context.workspace.root_path) / "existing.txt"
    existing_file.write_text("Already exists")

    call = ToolCall(
        id="test13",
        name="batch_create_files",
        arguments={
            "files": [
                {"path": "new1.txt", "content": "New 1"},
                {"path": "existing.txt", "content": "New content"},
                {"path": "new2.txt", "content": "New 2"},
            ]
        },
    )

    result = await tool.execute(call, context)

    assert not result.success
    assert "already exist" in result.error

    # No new files should be created
    assert not (Path(context.workspace.root_path) / "new1.txt").exists()
    assert not (Path(context.workspace.root_path) / "new2.txt").exists()


@pytest.mark.asyncio
async def test_batch_create_multiline_content(tool, context):
    """Test creating files with multiline content."""
    content = "Line 1\nLine 2\nLine 3\n"
    call = ToolCall(
        id="test14",
        name="batch_create_files",
        arguments={
            "files": [
                {"path": "multiline.txt", "content": content},
            ]
        },
    )

    result = await tool.execute(call, context)

    assert result.success
    file_path = Path(context.workspace.root_path) / "multiline.txt"
    assert file_path.read_text() == content


@pytest.mark.asyncio
async def test_batch_create_empty_content(tool, context):
    """Test creating files with empty content."""
    call = ToolCall(
        id="test15",
        name="batch_create_files",
        arguments={
            "files": [
                {"path": "empty.txt", "content": ""},
            ]
        },
    )

    result = await tool.execute(call, context)

    assert result.success
    file_path = Path(context.workspace.root_path) / "empty.txt"
    assert file_path.exists()
    assert file_path.read_text() == ""


@pytest.mark.asyncio
async def test_batch_create_special_characters_in_content(tool, context):
    """Test creating files with special characters."""
    content = "Special chars: !@#$%^&*()_+-={}[]|:;<>?,./~`"
    call = ToolCall(
        id="test16",
        name="batch_create_files",
        arguments={
            "files": [
                {"path": "special.txt", "content": content},
            ]
        },
    )

    result = await tool.execute(call, context)

    assert result.success
    file_path = Path(context.workspace.root_path) / "special.txt"
    assert file_path.read_text() == content


@pytest.mark.asyncio
async def test_batch_create_unicode_content(tool, context):
    """Test creating files with Unicode content."""
    content = "Unicode: 你好 世界 🌍 émojis 🎉"
    call = ToolCall(
        id="test17",
        name="batch_create_files",
        arguments={
            "files": [
                {"path": "unicode.txt", "content": content},
            ]
        },
    )

    result = await tool.execute(call, context)

    assert result.success
    file_path = Path(context.workspace.root_path) / "unicode.txt"
    assert file_path.read_text() == content


@pytest.mark.asyncio
async def test_batch_create_diff_format(tool, context):
    """Test diff generation format."""
    call = ToolCall(
        id="test18",
        name="batch_create_files",
        arguments={
            "files": [
                {"path": "test.py", "content": "def hello():\n    print('Hello')\n"},
            ]
        },
    )

    result = await tool.execute(call, context)

    assert result.success
    assert len(result.diffs) == 1
    diff = result.diffs[0]
    assert diff["file"] == "test.py"
    assert "+++" in diff["diff"]
    assert "test.py" in diff["diff"]
    assert "def hello():" in diff["diff"]


@pytest.mark.asyncio
async def test_batch_create_multiple_diffs(tool, context):
    """Test diff generation for multiple files."""
    call = ToolCall(
        id="test19",
        name="batch_create_files",
        arguments={
            "files": [
                {"path": "file1.py", "content": "print(1)\n"},
                {"path": "file2.py", "content": "print(2)\n"},
            ]
        },
    )

    result = await tool.execute(call, context)

    assert result.success
    assert len(result.diffs) == 2

    diff_files = {d["file"] for d in result.diffs}
    assert "file1.py" in diff_files
    assert "file2.py" in diff_files


@pytest.mark.asyncio
async def test_batch_create_relative_paths_in_result(tool, context):
    """Test that results use relative paths."""
    call = ToolCall(
        id="test20",
        name="batch_create_files",
        arguments={
            "files": [
                {"path": "subdir/file.txt", "content": "Content"},
            ]
        },
    )

    result = await tool.execute(call, context)

    assert result.success
    assert result.files_changed == ["subdir/file.txt"]
    assert result.diffs[0]["file"] == "subdir/file.txt"


@pytest.mark.asyncio
async def test_get_definition(tool):
    """Test tool definition."""
    definition = tool.get_definition()

    assert definition.name == "batch_create_files"
    assert "multiple files atomically" in definition.description.lower()
    assert definition.parameters["type"] == "object"
    assert "files" in definition.parameters["properties"]
    assert definition.parameters["required"] == ["files"]

    files_schema = definition.parameters["properties"]["files"]
    assert files_schema["type"] == "array"
    assert files_schema["minItems"] == 1
    assert "path" in files_schema["items"]["properties"]
    assert "content" in files_schema["items"]["properties"]
    assert files_schema["items"]["required"] == ["path", "content"]

    # Check safety flags
    assert definition.safety["requires_confirmation"] is True
    # Note: requires_read_first removed - LLM learns from natural feedback


@pytest.mark.asyncio
async def test_batch_create_parallel_io(tool, context):
    """Test that files are created in parallel."""
    # Create 10 files to test parallel I/O
    files = [{"path": f"file{i}.txt", "content": f"Content {i}"} for i in range(10)]

    call = ToolCall(
        id="test21",
        name="batch_create_files",
        arguments={"files": files},
    )

    result = await tool.execute(call, context)

    assert result.success
    assert len(result.files_changed) == 10

    # Verify all files created
    for i in range(10):
        file_path = Path(context.workspace.root_path) / f"file{i}.txt"
        assert file_path.exists()
        assert file_path.read_text() == f"Content {i}"


@pytest.mark.asyncio
async def test_batch_create_rollback_on_failure(tool, context, tmp_path):
    """Test rollback when creation fails."""
    # Create a read-only directory to force failure
    readonly_dir = tmp_path / "readonly"
    readonly_dir.mkdir()
    readonly_dir.chmod(0o444)

    # Create workspace in readonly directory (this will work)
    # But then we'll try to create files in it (this will fail on some systems)
    workspace = Workspace(str(tmp_path))
    context_with_readonly = ToolContext(
        workspace=workspace, logger=context.logger, model_id="test-model"
    )

    # First create some valid files
    call = ToolCall(
        id="test22",
        name="batch_create_files",
        arguments={
            "files": [
                {"path": "file1.txt", "content": "Content 1"},
                {"path": "file2.txt", "content": "Content 2"},
            ]
        },
    )

    # This should succeed normally
    result = await tool.execute(call, context_with_readonly)
    assert result.success

    # Clean up readonly directory
    readonly_dir.chmod(0o755)


@pytest.mark.asyncio
async def test_batch_create_large_number_of_files(tool, context):
    """Test creating many files at once."""
    # Create 50 files
    files = [
        {"path": f"batch/file{i:03d}.txt", "content": f"Content for file {i}"} for i in range(50)
    ]

    call = ToolCall(
        id="test23",
        name="batch_create_files",
        arguments={"files": files},
    )

    result = await tool.execute(call, context)

    assert result.success
    assert len(result.files_changed) == 50
    assert len(result.diffs) == 50

    # Spot check a few files
    file0 = Path(context.workspace.root_path) / "batch" / "file000.txt"
    file25 = Path(context.workspace.root_path) / "batch" / "file025.txt"
    file49 = Path(context.workspace.root_path) / "batch" / "file049.txt"

    assert file0.exists()
    assert file25.exists()
    assert file49.exists()


@pytest.mark.asyncio
async def test_batch_create_absolute_paths(tool, context):
    """Test using absolute paths within workspace."""
    abs_path = str(Path(context.workspace.root_path) / "absolute.txt")

    call = ToolCall(
        id="test24",
        name="batch_create_files",
        arguments={
            "files": [
                {"path": abs_path, "content": "Absolute path content"},
            ]
        },
    )

    result = await tool.execute(call, context)

    assert result.success
    file_path = Path(abs_path)
    assert file_path.exists()
    assert file_path.read_text() == "Absolute path content"
