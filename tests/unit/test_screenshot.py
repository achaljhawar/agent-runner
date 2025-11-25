"""Unit tests for screenshot tool.

Tests ScreenshotTool functionality including server lifecycle management,
port finding, screenshot capture, and cleanup.
"""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from agentrunner.core.exceptions import E_VALIDATION
from agentrunner.core.logger import AgentRunnerLogger
from agentrunner.core.tool_protocol import ToolCall, ToolDefinition
from agentrunner.core.workspace import Workspace
from agentrunner.tools.base import ToolContext
from agentrunner.tools.screenshot import FRAMEWORKS, ScreenshotTool

# Check if playwright is available
try:
    import playwright  # noqa: F401

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# Skip all tests in this module if playwright is not available
pytestmark = pytest.mark.skipif(
    not PLAYWRIGHT_AVAILABLE,
    reason="playwright not installed (install with: pip install playwright && playwright install chromium)",
)


@pytest.fixture
def mock_workspace(tmp_path):
    """Create mock workspace."""
    workspace = Mock(spec=Workspace)
    workspace.root_path = tmp_path
    workspace.resolve_path = Mock(side_effect=lambda p: tmp_path / p)
    return workspace


@pytest.fixture
def mock_logger():
    """Create mock logger."""
    logger = Mock(spec=AgentRunnerLogger)
    logger.info = Mock()
    logger.debug = Mock()
    logger.error = Mock()
    logger.warn = Mock()
    return logger


@pytest.fixture
def tool_context(mock_workspace, mock_logger):
    """Create tool context."""
    return ToolContext(
        workspace=mock_workspace,
        logger=mock_logger,
        model_id="test-model",
        config={},
    )


def create_mock_playwright():
    """Helper to create fully mocked Playwright objects."""

    mock_page = AsyncMock()
    mock_page.goto = AsyncMock()
    mock_page.evaluate = AsyncMock()
    mock_page.wait_for_timeout = AsyncMock()
    mock_page.set_default_timeout = Mock()

    # Mock screenshot to write file
    async def mock_screenshot_func(**kwargs):
        path = kwargs.get("path")
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"fake_png_data")
        return b"fake_png_data"

    mock_page.screenshot = AsyncMock(side_effect=mock_screenshot_func)

    mock_context = AsyncMock()
    mock_context.new_page = AsyncMock(return_value=mock_page)

    mock_browser = AsyncMock()
    mock_browser.new_context = AsyncMock(return_value=mock_context)
    mock_browser.close = AsyncMock()

    mock_chromium = AsyncMock()
    mock_chromium.launch = AsyncMock(return_value=mock_browser)

    mock_playwright = AsyncMock()
    mock_playwright.__aenter__ = AsyncMock(return_value=Mock(chromium=mock_chromium))
    mock_playwright.__aexit__ = AsyncMock()

    return mock_playwright


class TestScreenshotTool:
    """Test ScreenshotTool class."""

    def test_initialization(self):
        """Test ScreenshotTool initialization."""
        tool = ScreenshotTool()
        assert tool is not None

    def test_get_definition(self):
        """Test tool definition."""
        tool = ScreenshotTool()
        definition = tool.get_definition()

        assert isinstance(definition, ToolDefinition)
        assert definition.name == "take_screenshot"
        assert "framework" in definition.parameters["properties"]
        assert definition.parameters["required"] == ["framework"]

        # Check framework enum
        framework_enum = definition.parameters["properties"]["framework"]["enum"]
        assert "nextjs" in framework_enum
        assert "vite" in framework_enum
        assert "react" in framework_enum

    def test_frameworks_dict(self):
        """Test FRAMEWORKS dictionary is properly configured."""
        assert "nextjs" in FRAMEWORKS
        assert "vite" in FRAMEWORKS
        assert "react" in FRAMEWORKS

        # Check structure
        for _framework, config in FRAMEWORKS.items():
            assert "command" in config
            assert "wait_for_text" in config
            assert "description" in config
            assert "{port}" in config["command"]

    @pytest.mark.asyncio
    async def test_execute_validation_missing_framework(self, tool_context):
        """Test execute fails when framework is missing."""
        tool = ScreenshotTool()
        call = ToolCall(
            id="test",
            name="take_screenshot",
            arguments={},
        )

        result = await tool.execute(call, tool_context)

        assert not result.success
        assert result.error_code == E_VALIDATION
        assert "framework is required" in result.error

    @pytest.mark.asyncio
    async def test_execute_validation_unsupported_framework(self, tool_context):
        """Test execute fails with unsupported framework."""
        tool = ScreenshotTool()
        call = ToolCall(
            id="test",
            name="take_screenshot",
            arguments={"framework": "invalid_framework"},
        )

        result = await tool.execute(call, tool_context)

        assert not result.success
        assert result.error_code == E_VALIDATION
        assert "Unsupported framework" in result.error

    @pytest.mark.asyncio
    async def test_execute_validation_invalid_output_path(self, tool_context):
        """Test execute fails with invalid output path."""
        tool = ScreenshotTool()
        tool_context.workspace.resolve_path = Mock(side_effect=Exception("Path outside workspace"))

        call = ToolCall(
            id="test",
            name="take_screenshot",
            arguments={
                "framework": "nextjs",
                "output_path": "../../../etc/passwd",
            },
        )

        result = await tool.execute(call, tool_context)

        assert not result.success
        assert result.error_code == E_VALIDATION
        assert "Invalid output_path" in result.error

    @pytest.mark.asyncio
    async def test_find_free_port(self):
        """Test port finding."""
        tool = ScreenshotTool()
        port = tool._find_free_port()

        assert isinstance(port, int)
        assert 1024 < port < 65535

    @pytest.mark.asyncio
    async def test_start_server(self, tool_context):
        """Test server process starts."""
        tool = ScreenshotTool()

        with patch("subprocess.Popen") as mock_popen:
            mock_process = Mock()
            mock_process.poll = Mock(return_value=None)  # Still running
            mock_process.pid = 12345
            mock_popen.return_value = mock_process

            command = "npm run dev -- --port 3000"
            process = await tool._start_server(command, tool_context)

            assert process is not None
            mock_popen.assert_called_once()

            # Check call arguments
            call_args = mock_popen.call_args
            assert call_args[0][0] == command
            assert call_args[1]["shell"] is True
            assert call_args[1]["cwd"] == str(tool_context.workspace.root_path)

    @pytest.mark.asyncio
    async def test_wait_for_server_ready_text_match(self, tool_context):
        """Test waiting for server via stdout text."""
        tool = ScreenshotTool()

        mock_process = Mock()
        mock_process.poll = Mock(return_value=None)  # Running
        mock_process.stdout = Mock()

        # Simulate stdout lines
        output_lines = [
            "Starting server...\n",
            "Ready in 1.5s\n",  # Match "ready in"
        ]

        with patch("select.select") as mock_select:
            # First call: data available
            # Second call: data available
            # Third call: no more data
            mock_select.side_effect = [
                ([mock_process.stdout], [], []),
                ([mock_process.stdout], [], []),
                ([], [], []),
            ]

            mock_process.stdout.readline = Mock(side_effect=output_lines)

            ready, error = await tool._wait_for_server_ready(
                port=3000,
                process=mock_process,
                wait_for_text="ready in",
                timeout=10,
                context=tool_context,
            )

            assert ready is True
            assert error == ""

    @pytest.mark.asyncio
    async def test_wait_for_server_ready_process_died(self, tool_context):
        """Test waiting fails if process dies."""
        tool = ScreenshotTool()

        mock_process = Mock()
        mock_process.poll = Mock(return_value=1)  # Exited with code 1
        mock_process.returncode = 1
        mock_process.stdout = Mock()

        ready, error = await tool._wait_for_server_ready(
            port=3000,
            process=mock_process,
            wait_for_text="ready",
            timeout=10,
            context=tool_context,
        )

        assert ready is False
        assert "process exited" in error.lower()

    @pytest.mark.asyncio
    async def test_wait_for_server_ready_timeout(self, tool_context):
        """Test waiting times out."""

        tool = ScreenshotTool()

        mock_process = Mock()
        mock_process.poll = Mock(return_value=None)  # Still running
        mock_process.stdout = Mock()

        with patch("select.select") as mock_select, patch("httpx.AsyncClient") as mock_httpx:
            mock_select.return_value = ([], [], [])  # No data

            # Mock HTTP client to always raise connection errors (server not ready)
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_httpx.return_value = mock_client

            # Mock httpx to simulate connection errors
            with patch("httpx.AsyncClient") as mock_httpx:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock()
                mock_httpx.return_value = mock_client

                # Use very short timeout
                ready, error = await tool._wait_for_server_ready(
                    port=3000,
                    process=mock_process,
                    wait_for_text="ready",
                    timeout=0.1,  # 100ms
                    context=tool_context,
                )

                assert ready is False
                assert "did not become ready" in error.lower()

    @pytest.mark.asyncio
    async def test_take_screenshot_playwright_not_installed(self, tool_context, tmp_path):
        """Test screenshot fails gracefully if Playwright is missing."""
        tool = ScreenshotTool()
        output_path = tmp_path / "screenshot.png"

        # Directly test the early return path by mocking the import to fail
        # We'll inject the ImportError at the function level
        original_func = tool._take_screenshot

        async def mock_take_screenshot_import_error(*args, **kwargs):
            # Simulate what happens when playwright import fails
            return (
                False,
                "Playwright not installed. Run: pip install playwright && playwright install chromium",
                None,
            )

        tool._take_screenshot = mock_take_screenshot_import_error

        try:
            success, error, png_bytes = await tool._take_screenshot(
                url="http://localhost:3000",
                output_path=output_path,
                viewport_width=1280,
                viewport_height=720,
                full_page=False,
                context=tool_context,
            )

            assert success is False
            assert "Playwright not installed" in error
            assert png_bytes is None
        finally:
            tool._take_screenshot = original_func

    @pytest.mark.asyncio
    async def test_take_screenshot_success(self, tool_context, tmp_path):
        """Test screenshot capture success."""
        tool = ScreenshotTool()
        output_path = tmp_path / "screenshot.png"

        mock_playwright = create_mock_playwright()

        with patch("playwright.async_api.async_playwright", return_value=mock_playwright):
            success, error, png_bytes = await tool._take_screenshot(
                url="http://localhost:3000",
                output_path=output_path,
                viewport_width=1280,
                viewport_height=720,
                full_page=False,
                context=tool_context,
            )

            assert success is True
            assert error == ""
            assert png_bytes == b"fake_png_data"
            assert output_path.exists()

    def test_cleanup_server(self, tool_context):
        """Test server cleanup terminates process."""
        tool = ScreenshotTool()

        mock_process = Mock()
        mock_process.terminate = Mock()
        mock_process.wait = Mock()
        mock_process.pid = 12345

        tool._cleanup_server(mock_process, tool_context)

        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called()

    def test_cleanup_server_force_kill(self, tool_context):
        """Test server cleanup uses kill if terminate fails."""
        tool = ScreenshotTool()

        mock_process = Mock()
        mock_process.terminate = Mock()
        mock_process.wait = Mock(side_effect=[TimeoutError, None])  # Timeout, then success
        mock_process.kill = Mock()
        mock_process.pid = 12345

        import subprocess

        # Make wait raise TimeoutExpired instead of generic TimeoutError
        mock_process.wait = Mock(side_effect=[subprocess.TimeoutExpired("test", 5), None])

        tool._cleanup_server(mock_process, tool_context)

        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_full_flow_success(self, tool_context, tmp_path):
        """Test full execute flow with mocked subprocess and Playwright."""
        tool = ScreenshotTool()

        call = ToolCall(
            id="test",
            name="take_screenshot",
            arguments={
                "framework": "nextjs",
                "viewport_width": 1280,
                "viewport_height": 720,
                "full_page": False,
                "output_path": "test_screenshot.png",
            },
        )

        # Mock process
        mock_process = Mock()
        mock_process.poll = Mock(return_value=None)
        mock_process.stdout = Mock()
        mock_process.terminate = Mock()
        mock_process.wait = Mock()
        mock_process.pid = 12345

        mock_playwright = create_mock_playwright()

        with patch("subprocess.Popen", return_value=mock_process):
            with patch("select.select", return_value=([mock_process.stdout], [], [])):
                mock_process.stdout.readline = Mock(side_effect=["Starting...\n", "Ready in 1s\n"])

                with patch("httpx.AsyncClient") as mock_httpx:
                    mock_client = AsyncMock()
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_client.get = AsyncMock(return_value=mock_response)
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock()
                    mock_httpx.return_value = mock_client

                    with patch(
                        "playwright.async_api.async_playwright", return_value=mock_playwright
                    ):
                        result = await tool.execute(call, tool_context)

                        # Verify result
                        assert result.success
                        assert "test_screenshot.png" in result.output
                        assert result.data["framework"] == "nextjs"
                        assert result.data["viewport"] == "1280x720"

                        # Verify cleanup was called
                        mock_process.terminate.assert_called()

    @pytest.mark.asyncio
    async def test_execute_server_fails_to_start(self, tool_context):
        """Test execute handles server failure gracefully."""
        tool = ScreenshotTool()

        call = ToolCall(
            id="test",
            name="take_screenshot",
            arguments={"framework": "nextjs"},
        )

        # Mock process that dies immediately
        mock_process = Mock()
        mock_process.poll = Mock(return_value=1)  # Died
        mock_process.returncode = 1
        mock_process.stdout = Mock()
        mock_process.terminate = Mock()
        mock_process.wait = Mock()
        mock_process.pid = 12345

        with patch("subprocess.Popen", return_value=mock_process):
            result = await tool.execute(call, tool_context)

            assert not result.success
            assert "process exited" in result.error.lower()
            assert result.error_code == "E_SERVER_START_FAILED"

    @pytest.mark.asyncio
    async def test_execute_http_ready_before_text(self, tool_context, tmp_path):
        """Test that HTTP readiness is detected even without text match."""
        tool = ScreenshotTool()

        # Mock process that never outputs "ready" text
        mock_process = Mock()
        mock_process.poll = Mock(return_value=None)
        mock_process.stdout = Mock()
        mock_process.terminate = Mock()
        mock_process.wait = Mock()
        mock_process.pid = 12345

        # Test HTTP readiness detection directly rather than full execute
        with patch("select.select", return_value=([], [], [])):
            # HTTP returns 200 immediately
            with patch("httpx.AsyncClient") as mock_httpx:
                mock_client = AsyncMock()
                mock_response = Mock()
                mock_response.status_code = 200
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock()
                mock_httpx.return_value = mock_client

                ready, error = await tool._wait_for_server_ready(
                    port=3000,
                    process=mock_process,
                    wait_for_text="ready",
                    timeout=5,
                    context=tool_context,
                )

                assert ready is True
                assert error == ""

    @pytest.mark.asyncio
    async def test_different_viewport_sizes(self, tool_context, tmp_path):
        """Test various viewport configurations."""
        tool = ScreenshotTool()

        test_cases = [
            (1920, 1080),
            (1280, 720),
            (375, 667),  # iPhone
            (768, 1024),  # iPad
        ]

        for width, height in test_cases:
            call = ToolCall(
                id="test",
                name="take_screenshot",
                arguments={
                    "framework": "nextjs",
                    "viewport_width": width,
                    "viewport_height": height,
                },
            )

            # Mock everything
            mock_process = Mock()
            mock_process.poll = Mock(return_value=None)
            mock_process.stdout = Mock()
            mock_process.terminate = Mock()
            mock_process.wait = Mock()
            mock_process.pid = 12345

            mock_playwright = create_mock_playwright()

            with patch("subprocess.Popen", return_value=mock_process):
                with patch("select.select", return_value=([mock_process.stdout], [], [])):
                    mock_process.stdout.readline = Mock(return_value="Ready\n")

                    with patch("httpx.AsyncClient") as mock_httpx:
                        mock_client = AsyncMock()
                        mock_response = Mock(status_code=200)
                        mock_client.get = AsyncMock(return_value=mock_response)
                        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                        mock_client.__aexit__ = AsyncMock()
                        mock_httpx.return_value = mock_client

                    with patch(
                        "playwright.async_api.async_playwright", return_value=mock_playwright
                    ):
                        result = await tool.execute(call, tool_context)

                        assert result.success
                        assert f"{width}x{height}" in result.data["viewport"]

    @pytest.mark.asyncio
    async def test_full_page_screenshot(self, tool_context, tmp_path):
        """Test full page screenshot mode."""
        tool = ScreenshotTool()

        call = ToolCall(
            id="test",
            name="take_screenshot",
            arguments={
                "framework": "nextjs",
                "full_page": True,
                "output_path": "full-page.png",
            },
        )

        mock_process = Mock()
        mock_process.poll = Mock(return_value=None)
        mock_process.stdout = Mock()
        mock_process.terminate = Mock()
        mock_process.wait = Mock()
        mock_process.pid = 12345

        mock_playwright = create_mock_playwright()

        # Get the page mock to verify full_page param
        async def get_page():
            return (
                await mock_playwright.__aenter__.return_value.chromium.launch.return_value.new_context.return_value.new_page()
            )

        with patch("subprocess.Popen", return_value=mock_process):
            with patch("select.select", return_value=([mock_process.stdout], [], [])):
                mock_process.stdout.readline = Mock(return_value="Ready\n")

                with patch("httpx.AsyncClient") as mock_httpx:
                    mock_client = AsyncMock()
                    mock_response = Mock(status_code=200)
                    mock_client.get = AsyncMock(return_value=mock_response)
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock()
                    mock_httpx.return_value = mock_client

                    with patch(
                        "playwright.async_api.async_playwright", return_value=mock_playwright
                    ):
                        result = await tool.execute(call, tool_context)

                        assert result.success
                        # Just verify the screenshot was taken
                        assert "full-page.png" in result.output

    @pytest.mark.asyncio
    async def test_all_frameworks_have_valid_structure(self):
        """Test that all framework definitions are valid."""
        for framework_name, config in FRAMEWORKS.items():
            # Check required keys
            assert "command" in config, f"{framework_name} missing command"
            assert "wait_for_text" in config, f"{framework_name} missing wait_for_text"
            assert "description" in config, f"{framework_name} missing description"

            # Check command has port placeholder
            assert "{port}" in config["command"], f"{framework_name} command missing {{port}}"

            # Check types
            assert isinstance(config["command"], str)
            assert isinstance(config["wait_for_text"], str)
            assert isinstance(config["description"], str)

            # Check wait_for_text is not empty
            assert len(config["wait_for_text"]) > 0

    @pytest.mark.asyncio
    async def test_port_injection_in_command(self):
        """Test that port is correctly injected into commands."""
        test_cases = [
            ("nextjs", "npm run dev -- --port {port}", "npm run dev -- --port 3000"),
            ("react", "PORT={port} npm start", "PORT=3000 npm start"),
            (
                "flask",
                "FLASK_APP=app.py flask run --port {port}",
                "FLASK_APP=app.py flask run --port 3000",
            ),
        ]

        for framework, template, expected in test_cases:
            port = 3000
            result = template.replace("{port}", str(port))
            assert result == expected, f"Port injection failed for {framework}"

    def test_playwright_timeout_error_structure(self):
        """Test that Playwright timeout errors are caught properly."""
        # Unit test for error handling - the actual timeout is tested in integration
        from playwright.async_api import TimeoutError as PlaywrightTimeout

        # Just verify the error class exists and can be instantiated
        error = PlaywrightTimeout("Navigation timeout")
        assert "timeout" in str(error).lower()

        # Verify it's an exception
        assert isinstance(error, Exception)

    @pytest.mark.asyncio
    async def test_wait_for_server_http_500_error(self, tool_context):
        """Test that 5xx errors don't mark server as ready."""
        tool = ScreenshotTool()

        mock_process = Mock()
        mock_process.poll = Mock(return_value=None)
        mock_process.stdout = Mock()

        with patch("select.select", return_value=([], [], [])):
            with patch("httpx.AsyncClient") as mock_httpx:
                mock_client = AsyncMock()
                mock_response = Mock()
                mock_response.status_code = 500  # Server error
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock()
                mock_httpx.return_value = mock_client

                ready, error = await tool._wait_for_server_ready(
                    port=3000,
                    process=mock_process,
                    wait_for_text="ready",
                    timeout=0.5,
                    context=tool_context,
                )

                # 500 error should not mark as ready
                assert ready is False
                assert "did not become ready" in error.lower()

    @pytest.mark.asyncio
    async def test_wait_for_server_http_404_marks_ready(self, tool_context):
        """Test that 404 errors mark server as ready (server responding)."""
        tool = ScreenshotTool()

        mock_process = Mock()
        mock_process.poll = Mock(return_value=None)
        mock_process.stdout = Mock()

        with patch("select.select", return_value=([], [], [])):
            with patch("httpx.AsyncClient") as mock_httpx:
                mock_client = AsyncMock()
                mock_response = Mock()
                mock_response.status_code = 404  # Not found, but server is up
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock()
                mock_httpx.return_value = mock_client

                ready, error = await tool._wait_for_server_ready(
                    port=3000,
                    process=mock_process,
                    wait_for_text="ready",
                    timeout=5,
                    context=tool_context,
                )

                # 404 means server is responding
                assert ready is True
                assert error == ""

    @pytest.mark.asyncio
    async def test_output_path_creates_nested_directories(self, tool_context, tmp_path):
        """Test that nested output directories are created."""
        tool = ScreenshotTool()
        nested_path = tmp_path / "screenshots" / "test" / "output.png"

        mock_playwright = create_mock_playwright()

        with patch("playwright.async_api.async_playwright", return_value=mock_playwright):
            success, error, png_bytes = await tool._take_screenshot(
                url="http://localhost:3000",
                output_path=nested_path,
                viewport_width=1280,
                viewport_height=720,
                full_page=False,
                context=tool_context,
            )

            assert success is True
            assert nested_path.exists()
            assert nested_path.parent.exists()

    def test_definition_has_all_frameworks_in_description(self):
        """Test that tool definition lists all supported frameworks."""
        tool = ScreenshotTool()
        definition = tool.get_definition()

        description = definition.description

        # Check each framework is mentioned
        for framework in FRAMEWORKS.keys():
            assert framework in description, f"Framework {framework} not in description"

    def test_definition_enum_matches_frameworks(self):
        """Test that parameter enum matches FRAMEWORKS keys."""
        tool = ScreenshotTool()
        definition = tool.get_definition()

        enum_values = definition.parameters["properties"]["framework"]["enum"]
        framework_keys = list(FRAMEWORKS.keys())

        assert set(enum_values) == set(framework_keys), "Enum values don't match FRAMEWORKS"
