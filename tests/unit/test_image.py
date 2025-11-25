"""Unit tests for image tools."""

import os
from unittest.mock import patch

import pytest

from agentrunner.core.logger import AgentRunnerLogger
from agentrunner.core.tool_protocol import ToolCall
from agentrunner.core.workspace import Workspace
from agentrunner.tools.base import ToolContext
from agentrunner.tools.image import ImageFetchTool, ImageGenerationTool


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
def fetch_tool():
    """Create ImageFetchTool instance."""
    return ImageFetchTool()


@pytest.fixture
def generation_tool():
    """Create ImageGenerationTool instance."""
    return ImageGenerationTool()


class TestImageFetchTool:
    """Tests for ImageFetchTool."""

    def test_get_definition(self, fetch_tool):
        """Test tool definition."""
        definition = fetch_tool.get_definition()
        assert definition.name == "fetch_image"
        assert "Fetches high-quality images" in definition.description
        assert "query" in definition.parameters["properties"]
        assert "save_path" in definition.parameters["properties"]
        assert "source" in definition.parameters["properties"]

    @pytest.mark.asyncio
    async def test_missing_query(self, fetch_tool, context):
        """Test execution with missing query."""
        call = ToolCall(
            id="1",
            name="fetch_image",
            arguments={"save_path": "test.jpg"},
        )

        result = await fetch_tool.execute(call, context)

        assert not result.success
        assert "query is required" in result.error
        assert result.error_code == "E_VALIDATION"

    @pytest.mark.asyncio
    async def test_missing_save_path(self, fetch_tool, context):
        """Test execution with missing save_path."""
        call = ToolCall(
            id="1",
            name="fetch_image",
            arguments={"query": "mountain"},
        )

        result = await fetch_tool.execute(call, context)

        assert not result.success
        assert "save_path is required" in result.error
        assert result.error_code == "E_VALIDATION"

    @pytest.mark.asyncio
    async def test_unsupported_source(self, fetch_tool, context):
        """Test execution with unsupported source."""
        call = ToolCall(
            id="1",
            name="fetch_image",
            arguments={
                "query": "mountain",
                "save_path": "test.jpg",
                "source": "invalid_source",
            },
        )

        result = await fetch_tool.execute(call, context)

        assert not result.success
        assert "Unsupported source" in result.error
        assert result.error_code == "E_VALIDATION"

    @pytest.mark.asyncio
    async def test_missing_httpx(self, fetch_tool, context):
        """Test execution without httpx library."""
        call = ToolCall(
            id="1",
            name="fetch_image",
            arguments={
                "query": "mountain",
                "save_path": "test.jpg",
                "source": "pexels",  # Use pexels since unsplash is no longer supported
            },
        )

        with patch.dict("sys.modules", {"httpx": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'httpx'")):
                result = await fetch_tool.execute(call, context)

        assert not result.success
        assert "httpx library" in result.error
        assert result.error_code == "E_VALIDATION"

    @pytest.mark.asyncio
    async def test_pexels_missing_api_key(self, fetch_tool, context):
        """Test Pexels fetch without API key."""
        call = ToolCall(
            id="1",
            name="fetch_image",
            arguments={
                "query": "mountain",
                "save_path": "test.jpg",
                "source": "pexels",
            },
        )

        with patch.dict(os.environ, {}, clear=True):
            result = await fetch_tool.execute(call, context)

        assert not result.success
        assert "PEXELS_API_KEY" in result.error
        assert result.error_code == "E_VALIDATION"

    @pytest.mark.asyncio
    async def test_creates_parent_directory(self, fetch_tool, context, tmp_path):
        """Test that parent directories are created when save_path contains nested dirs."""
        # Test that the tool creates parent directories - we'll test this
        # by checking directory creation logic, not by mocking the full API call
        call = ToolCall(
            id="1",
            name="fetch_image",
            arguments={
                "query": "test",
                "save_path": "nested/dir/image.jpg",
                "source": "pexels",
            },
        )

        # Without API key, it should still create the directory structure before failing
        with patch.dict(os.environ, {}, clear=True):
            result = await fetch_tool.execute(call, context)

        # Test should fail due to missing API key, but directories should be created
        assert not result.success
        assert "PEXELS_API_KEY" in result.error
        # The directory creation happens before API key check, so it should exist
        assert (tmp_path / "nested" / "dir").exists()


class TestImageGenerationTool:
    """Tests for ImageGenerationTool."""

    def test_get_definition(self, generation_tool):
        """Test tool definition."""
        definition = generation_tool.get_definition()
        assert definition.name == "generate_image"
        assert "Generates images using" in definition.description  # Updated to be more flexible
        assert "prompt" in definition.parameters["properties"]
        assert "save_path" in definition.parameters["properties"]
        # provider parameter was removed - now uses Google Gemini by default
        assert "aspect_ratio" in definition.parameters["properties"]

    @pytest.mark.asyncio
    async def test_missing_prompt(self, generation_tool, context):
        """Test execution with missing prompt."""
        call = ToolCall(
            id="1",
            name="generate_image",
            arguments={"save_path": "test.png"},
        )

        result = await generation_tool.execute(call, context)

        assert not result.success
        assert "prompt is required" in result.error
        assert result.error_code == "E_VALIDATION"

    @pytest.mark.asyncio
    async def test_missing_save_path(self, generation_tool, context):
        """Test execution with missing save_path."""
        call = ToolCall(
            id="1",
            name="generate_image",
            arguments={"prompt": "a mountain"},
        )

        result = await generation_tool.execute(call, context)

        assert not result.success
        assert "save_path is required" in result.error
        assert result.error_code == "E_VALIDATION"
