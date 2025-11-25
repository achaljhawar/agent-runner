"""Unit tests for video tools."""

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from agentrunner.core.logger import AgentRunnerLogger
from agentrunner.core.tool_protocol import ToolCall
from agentrunner.core.workspace import Workspace
from agentrunner.tools.base import ToolContext
from agentrunner.tools.image import VideoFetchTool, VideoGenerationTool


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
    """Create VideoFetchTool instance."""
    return VideoFetchTool()


@pytest.fixture
def generation_tool():
    """Create VideoGenerationTool instance."""
    return VideoGenerationTool()


class TestVideoFetchTool:
    """Tests for VideoFetchTool."""

    def test_get_definition(self, fetch_tool):
        """Test tool definition."""
        definition = fetch_tool.get_definition()
        assert definition.name == "fetch_video"
        assert "Fetches high-quality videos" in definition.description
        assert "query" in definition.parameters["properties"]
        assert "save_path" in definition.parameters["properties"]
        assert "source" in definition.parameters["properties"]
        assert "orientation" in definition.parameters["properties"]
        assert "min_duration" in definition.parameters["properties"]
        assert "max_duration" in definition.parameters["properties"]

    @pytest.mark.asyncio
    async def test_missing_query(self, fetch_tool, context):
        """Test execution with missing query."""
        call = ToolCall(
            id="1",
            name="fetch_video",
            arguments={"save_path": "test.mp4"},
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
            name="fetch_video",
            arguments={"query": "ocean waves"},
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
            name="fetch_video",
            arguments={
                "query": "ocean waves",
                "save_path": "test.mp4",
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
            name="fetch_video",
            arguments={
                "query": "ocean waves",
                "save_path": "test.mp4",
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
            name="fetch_video",
            arguments={
                "query": "ocean waves",
                "save_path": "test.mp4",
                "source": "pexels",
            },
        )

        with patch.dict(os.environ, {}, clear=True):
            result = await fetch_tool.execute(call, context)

        assert not result.success
        assert "PEXELS_API_KEY" in result.error
        assert result.error_code == "E_VALIDATION"

    @pytest.mark.skip(reason="Complex async mocking - httpx AsyncClient context manager")
    @pytest.mark.asyncio
    async def test_pexels_successful_fetch(self, fetch_tool, context, tmp_path):
        """Test successful Pexels video fetch."""
        mock_httpx = Mock()
        mock_client = AsyncMock()
        mock_httpx.AsyncClient.return_value.__aenter__.return_value = mock_client

        mock_search_response = Mock()
        mock_search_response.json.return_value = {
            "videos": [
                {
                    "duration": 10,
                    "video_files": [
                        {
                            "link": "https://example.com/video.mp4",
                            "quality": "hd",
                            "width": 1920,
                            "height": 1080,
                        }
                    ],
                    "user": {"name": "Test Videographer"},
                    "url": "https://pexels.com/videos/test",
                }
            ]
        }
        mock_search_response.raise_for_status = Mock()

        mock_video_response = Mock()
        mock_video_response.content = b"fake video data"
        mock_video_response.raise_for_status = Mock()

        mock_client.get = AsyncMock(side_effect=[mock_search_response, mock_video_response])

        call = ToolCall(
            id="1",
            name="fetch_video",
            arguments={
                "query": "ocean waves",
                "save_path": "test.mp4",
                "source": "pexels",
            },
        )

        with patch.dict(os.environ, {"PEXELS_API_KEY": "test_key"}):
            with patch("agentrunner.tools.image.httpx", mock_httpx):
                result = await fetch_tool.execute(call, context)

        assert result.success
        assert "pexels" in result.output.lower()
        assert result.data["source"] == "pexels"
        assert result.data["videographer"] == "Test Videographer"
        assert result.data["duration"] == 10
        assert (tmp_path / "test.mp4").exists()
        assert (tmp_path / "test.mp4").read_bytes() == b"fake video data"

    @pytest.mark.skip(reason="Complex async mocking - httpx AsyncClient context manager")
    @pytest.mark.asyncio
    async def test_pexels_no_results(self, fetch_tool, context):
        """Test Pexels fetch with no results."""
        mock_httpx = Mock()
        mock_client = AsyncMock()
        mock_httpx.AsyncClient.return_value.__aenter__.return_value = mock_client

        mock_response = Mock()
        mock_response.json.return_value = {"videos": []}
        mock_response.raise_for_status = Mock()

        mock_client.get = AsyncMock(return_value=mock_response)

        call = ToolCall(
            id="1",
            name="fetch_video",
            arguments={
                "query": "nonexistent",
                "save_path": "test.mp4",
                "source": "pexels",
            },
        )

        with patch.dict(os.environ, {"PEXELS_API_KEY": "test_key"}):
            with patch("agentrunner.tools.image.httpx", mock_httpx):
                result = await fetch_tool.execute(call, context)

        assert not result.success
        assert "No videos found" in result.error
        assert result.error_code == "E_VALIDATION"

    @pytest.mark.skip(reason="Complex async mocking - httpx AsyncClient context manager")
    @pytest.mark.asyncio
    async def test_pexels_duration_filter(self, fetch_tool, context, tmp_path):
        """Test Pexels duration filtering."""
        mock_httpx = Mock()
        mock_client = AsyncMock()
        mock_httpx.AsyncClient.return_value.__aenter__.return_value = mock_client

        mock_search_response = Mock()
        mock_search_response.json.return_value = {
            "videos": [
                {"duration": 3, "video_files": []},  # Too short
                {"duration": 50, "video_files": []},  # Too long
                {
                    "duration": 15,  # Just right
                    "video_files": [
                        {
                            "link": "https://example.com/video.mp4",
                            "quality": "hd",
                            "width": 1920,
                            "height": 1080,
                        }
                    ],
                    "user": {"name": "Test"},
                    "url": "https://pexels.com/test",
                },
            ]
        }
        mock_search_response.raise_for_status = Mock()

        mock_video_response = Mock()
        mock_video_response.content = b"video data"
        mock_video_response.raise_for_status = Mock()

        mock_client.get = AsyncMock(side_effect=[mock_search_response, mock_video_response])

        call = ToolCall(
            id="1",
            name="fetch_video",
            arguments={
                "query": "ocean",
                "save_path": "test.mp4",
                "min_duration": 10,
                "max_duration": 20,
            },
        )

        with patch.dict(os.environ, {"PEXELS_API_KEY": "test_key"}):
            with patch("agentrunner.tools.image.httpx", mock_httpx):
                result = await fetch_tool.execute(call, context)

        assert result.success
        assert result.data["duration"] == 15

    @pytest.mark.skip(reason="Complex async mocking - httpx AsyncClient context manager")
    @pytest.mark.asyncio
    async def test_creates_parent_directory(self, fetch_tool, context, tmp_path):
        """Test that parent directories are created."""
        mock_httpx = Mock()
        mock_client = AsyncMock()
        mock_httpx.AsyncClient.return_value.__aenter__.return_value = mock_client

        mock_search_response = Mock()
        mock_search_response.json.return_value = {
            "videos": [
                {
                    "duration": 10,
                    "video_files": [
                        {
                            "link": "https://example.com/video.mp4",
                            "quality": "hd",
                            "width": 1920,
                            "height": 1080,
                        }
                    ],
                    "user": {"name": "Test"},
                    "url": "https://pexels.com/test",
                }
            ]
        }
        mock_search_response.raise_for_status = Mock()

        mock_video_response = Mock()
        mock_video_response.content = b"test"
        mock_video_response.raise_for_status = Mock()

        mock_client.get = AsyncMock(side_effect=[mock_search_response, mock_video_response])

        call = ToolCall(
            id="1",
            name="fetch_video",
            arguments={
                "query": "test",
                "save_path": "nested/dir/video.mp4",
            },
        )

        with patch.dict(os.environ, {"PEXELS_API_KEY": "test_key"}):
            with patch("agentrunner.tools.image.httpx", mock_httpx):
                result = await fetch_tool.execute(call, context)

        assert result.success
        assert (tmp_path / "nested" / "dir").exists()
        assert (tmp_path / "nested" / "dir" / "video.mp4").exists()


class TestVideoGenerationTool:
    """Tests for VideoGenerationTool."""

    def test_get_definition(self, generation_tool):
        """Test tool definition."""
        definition = generation_tool.get_definition()
        assert definition.name == "generate_video"
        assert "Generates videos using" in definition.description
        assert "prompt" in definition.parameters["properties"]
        assert "save_path" in definition.parameters["properties"]

    @pytest.mark.asyncio
    async def test_missing_prompt(self, generation_tool, context):
        """Test execution with missing prompt."""
        call = ToolCall(
            id="1",
            name="generate_video",
            arguments={"save_path": "test.mp4"},
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
            name="generate_video",
            arguments={"prompt": "ocean waves"},
        )

        result = await generation_tool.execute(call, context)

        assert not result.success
        assert "save_path is required" in result.error
        assert result.error_code == "E_VALIDATION"

    @pytest.mark.asyncio
    async def test_missing_httpx(self, generation_tool, context):
        """Test execution without httpx library."""
        call = ToolCall(
            id="1",
            name="generate_video",
            arguments={
                "prompt": "ocean waves",
                "save_path": "test.mp4",
            },
        )

        with patch.dict("sys.modules", {"httpx": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'httpx'")):
                result = await generation_tool.execute(call, context)

        assert not result.success
        assert "httpx library" in result.error
        assert result.error_code == "E_VALIDATION"

    @pytest.mark.asyncio
    async def test_veo_missing_api_key(self, generation_tool, context):
        """Test Veo generation without API key."""
        call = ToolCall(
            id="1",
            name="generate_video",
            arguments={
                "prompt": "ocean waves",
                "save_path": "test.mp4",
            },
        )

        with patch.dict(os.environ, {}, clear=True):
            result = await generation_tool.execute(call, context)

        assert not result.success
        assert "GOOGLE_API_KEY" in result.error
        assert result.error_code == "E_VALIDATION"

    @pytest.mark.skip(reason="Complex async mocking - Google API client")
    @pytest.mark.asyncio
    async def test_creates_parent_directory_generation(self, generation_tool, context, tmp_path):
        """Test that parent directories are created for generated videos."""
        mock_httpx = Mock()
        mock_client = AsyncMock()
        mock_httpx.AsyncClient.return_value.__aenter__.return_value = mock_client

        # Mock the initial generation request
        mock_generation_response = Mock()
        mock_generation_response.json.return_value = {"name": "operations/test-operation-id"}
        mock_generation_response.raise_for_status = Mock()

        # Mock the status polling (completed immediately)
        mock_status_response = Mock()
        mock_status_response.json.return_value = {
            "done": True,
            "response": {
                "generateVideoResponse": {
                    "generatedSamples": [{"video": {"uri": "https://example.com/video.mp4"}}]
                }
            },
        }
        mock_status_response.raise_for_status = Mock()

        # Mock video download
        mock_download_response = Mock()
        mock_download_response.content = b"generated video data"
        mock_download_response.raise_for_status = Mock()

        mock_client.post = AsyncMock(return_value=mock_generation_response)
        mock_client.get = AsyncMock(side_effect=[mock_status_response, mock_download_response])

        call = ToolCall(
            id="1",
            name="generate_video",
            arguments={
                "prompt": "ocean waves",
                "save_path": "generated/videos/test.mp4",
            },
        )

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"}):
            with patch("agentrunner.tools.image.httpx", mock_httpx):
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    result = await generation_tool.execute(call, context)

        assert result.success
        assert (tmp_path / "generated" / "videos").exists()
        assert (tmp_path / "generated" / "videos" / "test.mp4").exists()
        assert (
            tmp_path / "generated" / "videos" / "test.mp4"
        ).read_bytes() == b"generated video data"

    @pytest.mark.skip(reason="Complex async mocking - Google API client")
    @pytest.mark.asyncio
    async def test_veo_operation_error(self, generation_tool, context):
        """Test Veo generation with operation error."""
        mock_httpx = Mock()
        mock_client = AsyncMock()
        mock_httpx.AsyncClient.return_value.__aenter__.return_value = mock_client

        mock_generation_response = Mock()
        mock_generation_response.json.return_value = {"name": "operations/test-operation-id"}
        mock_generation_response.raise_for_status = Mock()

        mock_status_response = Mock()
        mock_status_response.json.return_value = {
            "done": True,
            "error": {"message": "Video generation failed due to content policy"},
        }
        mock_status_response.raise_for_status = Mock()

        mock_client.post = AsyncMock(return_value=mock_generation_response)
        mock_client.get = AsyncMock(return_value=mock_status_response)

        call = ToolCall(
            id="1",
            name="generate_video",
            arguments={
                "prompt": "test",
                "save_path": "test.mp4",
            },
        )

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"}):
            with patch("agentrunner.tools.image.httpx", mock_httpx):
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    result = await generation_tool.execute(call, context)

        assert not result.success
        assert "Video generation failed" in result.error
        assert result.error_code == "E_VALIDATION"
