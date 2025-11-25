"""Tests for CleanWorkspaceTool."""

import pytest

from agentrunner.core.logger import AgentRunnerLogger
from agentrunner.core.tool_protocol import ToolCall
from agentrunner.core.workspace import Workspace
from agentrunner.tools.base import ToolContext
from agentrunner.tools.clean_directory import CleanWorkspaceTool


@pytest.fixture
def workspace(tmp_path):
    """Create temporary workspace."""
    return Workspace(str(tmp_path))


@pytest.fixture
def context(workspace):
    """Create tool context."""
    return ToolContext(
        workspace=workspace,
        logger=AgentRunnerLogger("test"),
        model_id="test-model",
    )


@pytest.fixture
def tool():
    """Create CleanWorkspaceTool instance."""
    return CleanWorkspaceTool()


@pytest.mark.asyncio
async def test_get_definition(tool):
    """Test tool definition."""
    definition = tool.get_definition()
    assert definition.name == "clean_workspace"
    assert "DANGEROUS" in definition.description
    # No parameters - tool always cleans current workspace


@pytest.mark.asyncio
async def test_clean_empty_workspace(tool, context, tmp_path):
    """Test cleaning already empty workspace."""
    # Workspace (tmp_path) is already empty
    call = ToolCall(
        id="test",
        name="clean_workspace",
        arguments={},
    )

    result = await tool.execute(call, context)
    assert result.success is True
    assert "already clean" in result.output.lower()
    assert result.data["items_removed"] == 0


@pytest.mark.asyncio
async def test_clean_workspace_with_files(tool, context, tmp_path):
    """Test cleaning workspace with files."""
    # Create files in workspace root
    (tmp_path / "file1.txt").write_text("content1")
    (tmp_path / "file2.txt").write_text("content2")
    (tmp_path / ".hidden").write_text("hidden")

    call = ToolCall(
        id="test",
        name="clean_workspace",
        arguments={},
    )

    result = await tool.execute(call, context)
    assert result.success is True
    assert result.data["items_removed"] == 3
    assert tmp_path.exists()  # Workspace itself should remain
    assert len(list(tmp_path.iterdir())) == 0  # But contents should be gone


@pytest.mark.asyncio
async def test_clean_workspace_with_subdirs(tool, context, tmp_path):
    """Test cleaning workspace with subdirectories."""
    # Create nested structure in workspace
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "file.txt").write_text("content")
    (tmp_path / "root_file.txt").write_text("root content")

    call = ToolCall(
        id="test",
        name="clean_workspace",
        arguments={},
    )

    result = await tool.execute(call, context)
    assert result.success is True
    assert result.data["items_removed"] == 2  # subdir and root_file.txt
    assert tmp_path.exists()  # Workspace remains
    assert not subdir.exists()  # But subdir is gone
    assert len(list(tmp_path.iterdir())) == 0


@pytest.mark.asyncio
async def test_clean_workspace_preserves_workspace_itself(tool, context, tmp_path):
    """Test that the workspace itself is not deleted, only contents."""
    # Create file in workspace
    (tmp_path / "delete_me.txt").write_text("content")

    call = ToolCall(
        id="test",
        name="clean_workspace",
        arguments={},
    )

    result = await tool.execute(call, context)
    assert result.success is True
    assert tmp_path.exists()
    assert tmp_path.is_dir()
    assert len(list(tmp_path.iterdir())) == 0


@pytest.mark.asyncio
async def test_clean_cache_directories_scenario(tool, context, tmp_path):
    """Test real-world scenario: cleaning npm cache dirs before scaffolding."""
    # Simulate failed create-next-app leaving cache dirs in workspace root
    (tmp_path / ".npm").mkdir()
    (tmp_path / ".tmp").mkdir()
    (tmp_path / "Library").mkdir()
    (tmp_path / ".npm" / "cache.json").write_text("{}")

    call = ToolCall(
        id="test",
        name="clean_workspace",
        arguments={},
    )

    result = await tool.execute(call, context)
    assert result.success is True
    assert result.data["items_removed"] == 3
    assert tmp_path.exists()  # Workspace remains
    assert not (tmp_path / ".npm").exists()
    assert not (tmp_path / ".tmp").exists()
    assert not (tmp_path / "Library").exists()


@pytest.mark.asyncio
async def test_clean_workspace_with_permission_error(tool, context, tmp_path, monkeypatch):
    """Test handling of permission errors during cleanup."""
    # Create subdirectory in workspace
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    # Mock shutil.rmtree to raise PermissionError
    import shutil

    def mock_rmtree(path):
        raise PermissionError("Permission denied")

    monkeypatch.setattr(shutil, "rmtree", mock_rmtree)

    call = ToolCall(
        id="test",
        name="clean_workspace",
        arguments={},
    )

    result = await tool.execute(call, context)
    # Should fail gracefully
    assert result.success is False
    assert result.error_code == "E_VALIDATION"
    assert "Failed to clean workspace" in result.error
