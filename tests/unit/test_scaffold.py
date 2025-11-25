"""Tests for ScaffoldProjectTool.

Per .coding_agent_guide: Test REAL modules, mock only external boundaries.
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agentrunner.core.exceptions import E_VALIDATION
from agentrunner.core.logger import AgentRunnerLogger
from agentrunner.core.tool_protocol import ToolCall, ToolDefinition, ToolResult
from agentrunner.core.workspace import Workspace
from agentrunner.tools.base import ToolContext
from agentrunner.tools.scaffold import ProjectType, ScaffoldProjectTool


@pytest.fixture
def temp_workspace(tmp_path):
    """Create temporary workspace."""
    return Workspace(str(tmp_path))


@pytest.fixture
def logger():
    """Create logger."""
    return AgentRunnerLogger()


@pytest.fixture
def tool_context(temp_workspace, logger):
    """Create tool context."""
    return ToolContext(
        workspace=temp_workspace,
        logger=logger,
        model_id="test-model",
    )


@pytest.fixture
def scaffold_tool(tmp_path):
    """Create ScaffoldProjectTool instance with temp cache."""
    cache_dir = tmp_path / "test-cache"
    return ScaffoldProjectTool(enable_cache=True, cache_dir=cache_dir)


@pytest.fixture
def scaffold_tool_no_cache():
    """Create ScaffoldProjectTool instance with caching disabled."""
    return ScaffoldProjectTool(enable_cache=False)


class TestScaffoldProjectTool:
    """Test ScaffoldProjectTool class."""

    def test_get_definition(self, scaffold_tool):
        """Test tool definition."""
        definition = scaffold_tool.get_definition()

        assert isinstance(definition, ToolDefinition)
        assert definition.name == "scaffold_project"
        assert "Scaffold a new project" in definition.description
        assert "project_type" in definition.parameters["properties"]
        assert "typescript" in definition.parameters["properties"]
        assert "tailwind" in definition.parameters["properties"]
        assert "use_src_dir" in definition.parameters["properties"]
        assert definition.parameters["required"] == ["project_type", "use_src_dir"]

    def test_project_types_enum(self):
        """Test ProjectType enum values."""
        assert ProjectType.NEXT.value == "next"
        assert ProjectType.REACT.value == "react"
        assert ProjectType.VITE_REACT.value == "vite-react"
        assert ProjectType.PYTHON_POETRY.value == "python-poetry"
        assert ProjectType.DJANGO.value == "django"

    def test_build_command_next_typescript_tailwind(self, scaffold_tool):
        """Test command building for Next.js with TypeScript and Tailwind."""
        command = scaffold_tool._build_command(
            project_type="next",
            typescript=True,
            tailwind=True,
            template=None,
        )

        assert "npx --yes create-next-app@latest ." in command
        assert "--typescript" in command
        assert "--tailwind" in command
        assert "--app" in command

    def test_build_command_next_javascript_no_tailwind(self, scaffold_tool):
        """Test command building for Next.js with JavaScript, no Tailwind."""
        command = scaffold_tool._build_command(
            project_type="next",
            typescript=False,
            tailwind=False,
            template=None,
        )

        assert "npx --yes create-next-app@latest ." in command
        assert "--javascript" in command
        assert "--no-tailwind" in command

    def test_build_command_react_typescript(self, scaffold_tool):
        """Test command building for React with TypeScript."""
        command = scaffold_tool._build_command(
            project_type="react",
            typescript=True,
            tailwind=False,
            template=None,
        )

        assert "npx --yes create-react-app ." in command
        assert "--template typescript" in command

    def test_build_command_vite_react_typescript(self, scaffold_tool):
        """Test command building for Vite + React with TypeScript."""
        command = scaffold_tool._build_command(
            project_type="vite-react",
            typescript=True,
            tailwind=False,
            template=None,
        )

        assert "npm create vite@latest ." in command
        assert "--template react-ts" in command

    def test_build_command_vite_vue_javascript(self, scaffold_tool):
        """Test command building for Vite + Vue with JavaScript."""
        command = scaffold_tool._build_command(
            project_type="vite-vue",
            typescript=False,
            tailwind=False,
            template=None,
        )

        assert "npm create vite@latest ." in command
        assert "--template vue" in command

    def test_build_command_python_poetry(self, scaffold_tool):
        """Test command building for Python Poetry."""
        command = scaffold_tool._build_command(
            project_type="python-poetry",
            typescript=False,
            tailwind=False,
            template=None,
        )

        assert command == "poetry new ."

    def test_build_command_django(self, scaffold_tool):
        """Test command building for Django."""
        command = scaffold_tool._build_command(
            project_type="django",
            typescript=False,
            tailwind=False,
            template=None,
        )

        assert command == "django-admin startproject ."

    def test_build_command_astro_custom_template(self, scaffold_tool):
        """Test command building for Astro with custom template."""
        command = scaffold_tool._build_command(
            project_type="astro",
            typescript=True,
            tailwind=False,
            template="blog",
        )

        assert "npm create astro@latest ." in command
        assert "--template blog" in command

    @pytest.mark.asyncio
    async def test_execute_missing_project_type(self, scaffold_tool, tool_context):
        """Test execution with missing project_type."""
        call = ToolCall(
            id="test",
            name="scaffold_project",
            arguments={"project_name": "my-app"},
        )

        result = await scaffold_tool.execute(call, tool_context)

        assert not result.success
        assert "project_type is required" in result.error
        assert result.error_code == E_VALIDATION

    # test_execute_missing_project_name removed - project_name no longer required (always uses ".")

    @pytest.mark.asyncio
    async def test_execute_invalid_project_type(self, scaffold_tool, tool_context):
        """Test execution with invalid project_type."""
        call = ToolCall(
            id="test",
            name="scaffold_project",
            arguments={"project_type": "invalid-type", "project_name": "my-app"},
        )

        result = await scaffold_tool.execute(call, tool_context)

        assert not result.success
        assert "Unsupported project type" in result.error
        assert result.error_code == E_VALIDATION

    @pytest.mark.asyncio
    async def test_execute_success_next_project(self, scaffold_tool, tool_context):
        """Test successful scaffolding of Next.js project."""
        # Mock BashTool execution (imported inside execute method)
        with patch("agentrunner.tools.bash.BashTool") as MockBashTool:
            mock_bash_instance = AsyncMock()
            mock_bash_instance.execute.return_value = ToolResult(
                success=True,
                output="Successfully created my-next-app",
            )
            MockBashTool.return_value = mock_bash_instance

            call = ToolCall(
                id="test",
                name="scaffold_project",
                arguments={
                    "project_type": "next",
                    "use_src_dir": False,
                    "typescript": True,
                    "tailwind": True,
                },
            )

            result = await scaffold_tool.execute(call, tool_context)

            assert result.success
            assert "scaffolded successfully" in result.output
            assert "CRITICAL" in result.output or "npm install" in result.output

            # Verify bash tool was called with correct command
            bash_call = mock_bash_instance.execute.call_args[0][0]
            assert "npx --yes create-next-app@latest ." in bash_call.arguments["command"]

    @pytest.mark.asyncio
    async def test_execute_success_python_poetry(self, scaffold_tool, tool_context):
        """Test successful scaffolding of Python Poetry project."""
        with patch("agentrunner.tools.bash.BashTool") as MockBashTool:
            mock_bash_instance = AsyncMock()
            mock_bash_instance.execute.return_value = ToolResult(
                success=True,
                output="Created package py-project",
            )
            MockBashTool.return_value = mock_bash_instance

            call = ToolCall(
                id="test",
                name="scaffold_project",
                arguments={
                    "project_type": "python-poetry",
                },
            )

            result = await scaffold_tool.execute(call, tool_context)

            assert result.success
            assert "scaffolded successfully" in result.output
            assert "scaffolded successfully" in result.output
            assert "poetry install" in result.output

    @pytest.mark.asyncio
    async def test_execute_bash_failure(self, scaffold_tool, tool_context):
        """Test handling of bash command failure."""
        with patch("agentrunner.tools.bash.BashTool") as MockBashTool:
            mock_bash_instance = AsyncMock()
            mock_bash_instance.execute.return_value = ToolResult(
                success=False,
                error="Command failed: npm not found",
                error_code="E_TIMEOUT",
            )
            MockBashTool.return_value = mock_bash_instance

            call = ToolCall(
                id="test",
                name="scaffold_project",
                arguments={
                    "project_type": "next",
                },
            )

            result = await scaffold_tool.execute(call, tool_context)

            assert not result.success
            assert "Scaffolding failed" in result.error
            assert "npm not found" in result.error

    @pytest.mark.asyncio
    async def test_execute_defaults_to_typescript(self, scaffold_tool, tool_context):
        """Test that typescript defaults to True."""
        with patch("agentrunner.tools.bash.BashTool") as MockBashTool:
            mock_bash_instance = AsyncMock()
            mock_bash_instance.execute.return_value = ToolResult(success=True, output="Done")
            MockBashTool.return_value = mock_bash_instance

            call = ToolCall(
                id="test",
                name="scaffold_project",
                arguments={"project_type": "vite-react", "project_name": "app"},
            )

            await scaffold_tool.execute(call, tool_context)

            bash_call = mock_bash_instance.execute.call_args[0][0]
            # Should use typescript template by default
            assert "react-ts" in bash_call.arguments["command"]

    @pytest.mark.asyncio
    async def test_execute_tailwind_false_by_default(self, scaffold_tool, tool_context):
        """Test that tailwind defaults to False."""
        with patch("agentrunner.tools.bash.BashTool") as MockBashTool:
            mock_bash_instance = AsyncMock()
            mock_bash_instance.execute.return_value = ToolResult(success=True, output="Done")
            MockBashTool.return_value = mock_bash_instance

            call = ToolCall(
                id="test",
                name="scaffold_project",
                arguments={"project_type": "next", "project_name": "app"},
            )

            await scaffold_tool.execute(call, tool_context)

            bash_call = mock_bash_instance.execute.call_args[0][0]
            # Should not include tailwind by default
            assert "--no-tailwind" in bash_call.arguments["command"]


class TestProjectTypeEnum:
    """Test ProjectType enum."""

    def test_all_project_types_have_commands(self, scaffold_tool):
        """Test that all project types have associated commands."""
        for project_type in ProjectType:
            command = scaffold_tool._build_command(
                project_type=project_type.value,
                typescript=True,
                tailwind=False,
                template=None,
            )
            assert command != "", f"No command for {project_type.value}"
            assert len(command) > 0, f"Empty command for {project_type.value}"


class TestScaffoldCaching:
    """Test scaffold project caching functionality."""

    def test_cache_key_generation(self, scaffold_tool):
        """Test cache key generation is consistent and unique."""
        key1 = scaffold_tool._get_cache_key("next", True, True, None)
        key2 = scaffold_tool._get_cache_key("next", True, True, None)
        key3 = scaffold_tool._get_cache_key("next", False, True, None)

        # Same config should generate same key
        assert key1 == key2
        # Different config should generate different key
        assert key1 != key3
        # Key should be 8 chars (MD5 hash truncated)
        assert len(key1) == 8

    def test_cache_initialization(self, tmp_path):
        """Test that cache directory is created on initialization."""
        cache_dir = tmp_path / "test-cache"
        assert not cache_dir.exists()

        ScaffoldProjectTool(enable_cache=True, cache_dir=cache_dir)

        assert cache_dir.exists()
        assert cache_dir.is_dir()

    def test_cache_disabled_no_directory(self, tmp_path):
        """Test that cache directory is not created when caching is disabled."""
        cache_dir = tmp_path / "test-cache"
        ScaffoldProjectTool(enable_cache=False, cache_dir=cache_dir)

        assert not cache_dir.exists()

    @pytest.mark.asyncio
    async def test_cache_miss_scaffolds_and_caches(self, scaffold_tool, tool_context, tmp_path):
        """Test that first scaffold (cache miss) runs bash and caches result."""
        with patch("agentrunner.tools.bash.BashTool") as MockBashTool:
            mock_bash_instance = AsyncMock()
            mock_bash_instance.execute.return_value = ToolResult(
                success=True,
                output="Project created",
            )
            MockBashTool.return_value = mock_bash_instance

            # Create files in workspace root (scaffolding happens into ".")
            (tool_context.workspace.root_path / "package.json").write_text('{"name": "test"}')
            (tool_context.workspace.root_path / "app").mkdir()

            call = ToolCall(
                id="test",
                name="scaffold_project",
                arguments={
                    "project_type": "next",
                    "typescript": True,
                    "tailwind": True,
                },
            )

            result = await scaffold_tool.execute(call, tool_context)

            assert result.success
            assert mock_bash_instance.execute.called
            assert "Cached for future use" in result.output

            cache_key = scaffold_tool._get_cache_key("next", True, True, None)
            cache_path = scaffold_tool.cache_dir / cache_key
            assert cache_path.exists()
            assert (cache_path / "package.json").exists()
            assert (cache_path / "app").exists()

    @pytest.mark.asyncio
    async def test_cache_hit_skips_bash(self, scaffold_tool, tool_context, tmp_path):
        """Test that second scaffold (cache hit) uses cache, skips bash."""
        cache_key = scaffold_tool._get_cache_key("next", True, False, None)
        cache_path = scaffold_tool.cache_dir / cache_key
        cache_path.mkdir(parents=True)
        (cache_path / "cached-file.txt").write_text("from cache")
        (cache_path / "src").mkdir()
        (cache_path / "src" / "index.ts").write_text("// cached")

        with patch("agentrunner.tools.bash.BashTool") as MockBashTool:
            mock_bash_instance = AsyncMock()
            MockBashTool.return_value = mock_bash_instance

            call = ToolCall(
                id="test",
                name="scaffold_project",
                arguments={
                    "project_type": "next",
                    "typescript": True,
                    "tailwind": False,
                },
            )

            result = await scaffold_tool.execute(call, tool_context)

            assert result.success
            assert not mock_bash_instance.execute.called
            assert "from cache" in result.output

            # Verify files were copied from cache into workspace root
            workspace_root = tool_context.workspace.root_path
            assert (workspace_root / "cached-file.txt").exists()
            assert (workspace_root / "cached-file.txt").read_text() == "from cache"
            assert (workspace_root / "src" / "index.ts").exists()
            assert (workspace_root / "src" / "index.ts").read_text() == "// cached"

    @pytest.mark.asyncio
    async def test_cache_disabled_always_uses_bash(self, scaffold_tool_no_cache, tool_context):
        """Test that when caching is disabled, bash is always used."""
        with patch("agentrunner.tools.bash.BashTool") as MockBashTool:
            mock_bash_instance = AsyncMock()
            mock_bash_instance.execute.return_value = ToolResult(
                success=True,
                output="Project created",
            )
            MockBashTool.return_value = mock_bash_instance

            call = ToolCall(
                id="test",
                name="scaffold_project",
                arguments={
                    "project_type": "next",
                    "typescript": True,
                },
            )

            result = await scaffold_tool_no_cache.execute(call, tool_context)

            # Verify bash was called
            assert result.success
            assert mock_bash_instance.execute.called
            # Should not mention caching
            assert (
                "cache" not in result.output.lower() or "scaffolded successfully" in result.output
            )

    @pytest.mark.asyncio
    async def test_cache_read_failure_falls_back_to_bash(self, scaffold_tool, tool_context):
        """Test that if cached template copy fails, it falls back to bash."""
        cache_key = scaffold_tool._get_cache_key("next", True, False, None)
        cache_path = scaffold_tool.cache_dir / cache_key
        cache_path.mkdir(parents=True)
        (cache_path / "dummy.txt").write_text("cached")

        with patch("agentrunner.tools.bash.BashTool") as MockBashTool:
            mock_bash_instance = AsyncMock()
            mock_bash_instance.execute.return_value = ToolResult(
                success=True,
                output="Project created via bash",
            )
            MockBashTool.return_value = mock_bash_instance

            # Mock Path.iterdir in the cache read path to fail
            original_iterdir = Path.iterdir

            def mock_iterdir(self):
                if self == cache_path:
                    raise OSError("Permission denied")
                return original_iterdir(self)

            with patch.object(Path, "iterdir", mock_iterdir):
                call = ToolCall(
                    id="test",
                    name="scaffold_project",
                    arguments={
                        "project_type": "next",
                        "typescript": True,
                        "tailwind": False,
                    },
                )

                result = await scaffold_tool.execute(call, tool_context)

                assert result.success
                assert mock_bash_instance.execute.called
