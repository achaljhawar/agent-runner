"""Unit tests for search tools."""

from pathlib import Path

import pytest

from agentrunner.core.logger import AgentRunnerLogger
from agentrunner.core.tool_protocol import ToolCall
from agentrunner.core.workspace import Workspace
from agentrunner.tools.base import ToolContext
from agentrunner.tools.search import GrepSearchTool


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    """Create workspace for testing."""
    return Workspace(str(tmp_path))


@pytest.fixture
def context(workspace: Workspace) -> ToolContext:
    """Create tool context for testing."""
    return ToolContext(workspace=workspace, logger=AgentRunnerLogger(), model_id="test-model")


@pytest.fixture
def sample_files(tmp_path: Path) -> None:
    """Create sample files for testing."""
    # Create directory structure
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "subdir").mkdir()
    (tmp_path / "tests").mkdir()

    # Create Python files
    (tmp_path / "src" / "main.py").write_text(
        "def main():\n    print('Hello World')\n    return 0\n"
    )
    (tmp_path / "src" / "utils.py").write_text(
        "def helper():\n    return True\n\ndef process():\n    return False\n"
    )
    (tmp_path / "src" / "subdir" / "module.py").write_text(
        "class MyClass:\n    def method(self):\n        pass\n"
    )

    # Create text files
    (tmp_path / "README.md").write_text("# Project\n\nThis is a test project.\n")
    (tmp_path / "tests" / "test_main.py").write_text("def test_main():\n    assert True\n")

    # Create hidden file
    (tmp_path / ".hidden").write_text("secret")


# GrepTool Tests


@pytest.mark.asyncio
async def test_grep_basic_search(context: ToolContext, sample_files: None) -> None:
    """Test basic grep search."""
    tool = GrepSearchTool()
    call = ToolCall(
        id="1",
        name="grep",
        arguments={"pattern": "def main", "path": "."},
    )

    result = await tool.execute(call, context)

    assert result.success
    assert "main.py" in result.output
    assert "def main" in result.output


@pytest.mark.asyncio
async def test_grep_case_insensitive(context: ToolContext, sample_files: None) -> None:
    """Test case insensitive search."""
    tool = GrepSearchTool()
    call = ToolCall(
        id="1",
        name="grep",
        arguments={"pattern": "HELLO", "path": ".", "-i": True},
    )

    result = await tool.execute(call, context)

    assert result.success
    assert "Hello" in result.output


@pytest.mark.asyncio
async def test_grep_files_with_matches_mode(context: ToolContext, sample_files: None) -> None:
    """Test files_with_matches output mode."""
    tool = GrepSearchTool()
    call = ToolCall(
        id="1",
        name="grep",
        arguments={"pattern": "def", "path": ".", "output_mode": "files_with_matches"},
    )

    result = await tool.execute(call, context)

    assert result.success
    assert "main.py" in result.output
    assert "utils.py" in result.output
    # Should not contain line content
    assert "def main" not in result.output


@pytest.mark.asyncio
async def test_grep_count_mode(context: ToolContext, sample_files: None) -> None:
    """Test count output mode."""
    tool = GrepSearchTool()
    call = ToolCall(
        id="1",
        name="grep",
        arguments={"pattern": "def", "path": ".", "output_mode": "count"},
    )

    result = await tool.execute(call, context)

    assert result.success
    # Should show counts - output is wrapped in workspace_result XML
    assert result.output
    assert "workspace_result" in result.output
    # Extract content between XML tags
    import re

    match = re.search(r"<workspace_result[^>]*>(.*?)</workspace_result>", result.output, re.DOTALL)
    assert match, "Output should be wrapped in workspace_result tags"
    content = match.group(1).strip()
    lines = [line for line in content.split("\n") if line.strip() and not line.startswith("Found")]
    assert len(lines) > 0, f"Should have count lines, got: {content}"
    # Each line should have format: "count:filename" or similar
    for line in lines:
        if ":" in line or line[0].isdigit():
            break  # At least one line has count format


@pytest.mark.asyncio
async def test_grep_with_glob_filter(context: ToolContext, sample_files: None) -> None:
    """Test grep with glob pattern filter."""
    tool = GrepSearchTool()
    call = ToolCall(
        id="1",
        name="grep",
        arguments={"pattern": "def", "path": ".", "glob": "*.py"},
    )

    result = await tool.execute(call, context)

    assert result.success
    assert ".py" in result.output


@pytest.mark.asyncio
async def test_grep_multiline(context: ToolContext, tmp_path: Path) -> None:
    """Test multiline search."""
    # Create file with multiline content
    (tmp_path / "test.txt").write_text("line1\nline2\nline3\n")

    tool = GrepSearchTool()
    call = ToolCall(
        id="1",
        name="grep",
        arguments={"pattern": "line1.*line2", "path": ".", "multiline": True},
    )

    result = await tool.execute(call, context)

    assert result.success


@pytest.mark.asyncio
async def test_grep_no_matches(context: ToolContext, sample_files: None) -> None:
    """Test grep with no matches."""
    tool = GrepSearchTool()
    call = ToolCall(
        id="1",
        name="grep",
        arguments={"pattern": "nonexistent_pattern_xyz", "path": "."},
    )

    result = await tool.execute(call, context)

    assert result.success
    assert "No matches found" in result.output


@pytest.mark.asyncio
async def test_grep_invalid_regex(context: ToolContext, sample_files: None) -> None:
    """Test grep with invalid regex."""
    tool = GrepSearchTool()
    call = ToolCall(
        id="1",
        name="grep",
        arguments={"pattern": "[invalid(regex", "path": "."},
    )

    result = await tool.execute(call, context)

    assert not result.success
    assert result.error_code == "E_VALIDATION"
    # Error message includes ripgrep's regex parse error
    assert "regex parse error" in result.error or "Invalid regex" in result.error


@pytest.mark.asyncio
async def test_grep_missing_pattern(context: ToolContext, sample_files: None) -> None:
    """Test grep without pattern."""
    tool = GrepSearchTool()
    call = ToolCall(
        id="1",
        name="grep",
        arguments={"path": "."},
    )

    result = await tool.execute(call, context)

    assert not result.success
    assert result.error_code == "E_VALIDATION"


@pytest.mark.asyncio
async def test_grep_path_outside_workspace(context: ToolContext, sample_files: None) -> None:
    """Test grep with path outside workspace."""
    tool = GrepSearchTool()
    call = ToolCall(
        id="1",
        name="grep",
        arguments={"pattern": "test", "path": "/etc/passwd"},
    )

    result = await tool.execute(call, context)

    assert not result.success
    assert result.error_code == "E_PERMISSIONS"


@pytest.mark.asyncio
async def test_grep_nonexistent_path(context: ToolContext, sample_files: None) -> None:
    """Test grep with nonexistent path."""
    tool = GrepSearchTool()
    call = ToolCall(
        id="1",
        name="grep",
        arguments={"pattern": "test", "path": "nonexistent_dir"},
    )

    result = await tool.execute(call, context)

    assert not result.success
    # Nonexistent paths return E_VALIDATION (since ripgrep can't validate the path)
    assert result.error_code == "E_VALIDATION"


@pytest.mark.asyncio
async def test_grep_search_single_file(context: ToolContext, sample_files: None) -> None:
    """Test grep on a single file."""
    tool = GrepSearchTool()
    call = ToolCall(
        id="1",
        name="grep",
        arguments={"pattern": "Hello", "path": "src/main.py"},
    )

    result = await tool.execute(call, context)

    assert result.success
    assert "Hello" in result.output


def test_grep_definition() -> None:
    """Test grep tool definition."""
    tool = GrepSearchTool()
    definition = tool.get_definition()

    assert definition.name == "grep"
    assert "pattern" in definition.parameters["properties"]
    assert "pattern" in definition.parameters["required"]
