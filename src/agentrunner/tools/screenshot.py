"""Screenshot tool for web applications.

Takes screenshots of web development servers by automatically managing
server lifecycle: start → wait → screenshot → cleanup.
"""

import asyncio
import os
import signal
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from agentrunner.core.exceptions import E_VALIDATION
from agentrunner.core.tool_protocol import ToolCall, ToolDefinition, ToolResult
from agentrunner.tools.base import BaseTool, ToolContext

# Framework presets with command templates and wait signals
FRAMEWORKS = {
    "nextjs": {
        "command": "npm run dev -- --port {port}",
        "wait_for_text": "ready",
        "description": "Next.js development server",
    },
    "vite": {
        "command": "npm run dev -- --port {port}",
        "wait_for_text": "ready in",
        "description": "Vite (React, Vue, Svelte)",
    },
    "react": {
        "command": "PORT={port} npm start",
        "wait_for_text": "webpack compiled",
        "description": "Create React App",
    },
    "svelte": {
        "command": "npm run dev -- --port {port}",
        "wait_for_text": "ready in",
        "description": "SvelteKit development server",
    },
    "flask": {
        "command": "FLASK_APP=app.py flask run --port {port}",
        "wait_for_text": "running on",
        "description": "Flask web application",
    },
    "fastapi": {
        "command": "uvicorn main:app --port {port} --reload",
        "wait_for_text": "uvicorn running",
        "description": "FastAPI application",
    },
    "django": {
        "command": "python manage.py runserver {port}",
        "wait_for_text": "starting development server",
        "description": "Django web framework",
    },
    "express": {
        "command": "PORT={port} npm start",
        "wait_for_text": "listening",
        "description": "Express.js server",
    },
    "streamlit": {
        "command": "streamlit run app.py --server.port {port} --server.headless true",
        "wait_for_text": "you can now view",
        "description": "Streamlit data app",
    },
    "static": {
        "command": "python -m http.server {port}",
        "wait_for_text": "serving http",
        "description": "Python static HTTP server",
    },
}


class ScreenshotTool(BaseTool):
    """Take screenshots of web applications.

    Automatically starts dev server, waits for ready, takes screenshot,
    and cleans up. Supports Next.js, Vite, Create React App.
    """

    def get_definition(self) -> ToolDefinition:
        """Get tool definition."""
        framework_list = ", ".join(FRAMEWORKS.keys())

        return ToolDefinition(
            name="take_screenshot",
            description=(
                "Start a web development server and take a screenshot of the running application. "
                "Automatically finds a free port, starts the server, waits for it to be ready, "
                "captures a screenshot, and cleans up the server process. "
                "Perfect for visually inspecting your web applications. "
                f"\n\nSupported frameworks: {framework_list}"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "framework": {
                        "type": "string",
                        "enum": list(FRAMEWORKS.keys()),
                        "description": f"Framework type. Options: {framework_list}",
                    },
                    "viewport_width": {
                        "type": "integer",
                        "description": "Browser viewport width in pixels (default: 1280)",
                        "default": 1280,
                    },
                    "viewport_height": {
                        "type": "integer",
                        "description": "Browser viewport height in pixels (default: 720)",
                        "default": 720,
                    },
                    "full_page": {
                        "type": "boolean",
                        "description": "Capture entire scrollable page (default: false - viewport only)",
                        "default": False,
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Where to save screenshot PNG (default: 'screenshot.png')",
                        "default": "screenshot.png",
                    },
                    "wait_timeout": {
                        "type": "integer",
                        "description": "Max seconds to wait for server to be ready (default: 60)",
                        "default": 60,
                    },
                },
                "required": ["framework"],
            },
            safety={
                "requires_confirmation": False,
                "requires_read_first": False,
            },
        )

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """Execute screenshot tool.

        Args:
            call: Tool call with framework and screenshot parameters
            context: Tool execution context

        Returns:
            ToolResult with screenshot path and metadata
        """
        # Extract arguments
        framework = call.arguments.get("framework")
        viewport_width = call.arguments.get("viewport_width", 1280)
        viewport_height = call.arguments.get("viewport_height", 720)
        full_page = call.arguments.get("full_page", False)
        output_path_str = call.arguments.get("output_path", "screenshot.png")
        wait_timeout = call.arguments.get("wait_timeout", 60)

        # Validate framework
        if not framework:
            return ToolResult(
                success=False,
                error="framework is required",
                error_code=E_VALIDATION,
            )

        if framework not in FRAMEWORKS:
            return ToolResult(
                success=False,
                error=f"Unsupported framework: {framework}. Supported: {', '.join(FRAMEWORKS.keys())}",
                error_code=E_VALIDATION,
            )

        # Resolve output path
        try:
            abs_output_path = Path(context.workspace.resolve_path(output_path_str))
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Invalid output_path: {e}",
                error_code=E_VALIDATION,
            )

        # Get framework config
        framework_config = FRAMEWORKS[framework]
        command_template = framework_config["command"]
        wait_for_text = framework_config["wait_for_text"]

        process = None

        try:
            # 1. Find free port
            port = self._find_free_port()
            context.logger.info(
                "Starting server for screenshot",
                framework=framework,
                port=port,
            )

            # 2. Replace {port} in command
            command = command_template.replace("{port}", str(port))

            # Emit server starting event
            if context.event_bus:
                from agentrunner.core.events import StreamEvent

                context.event_bus.publish(
                    StreamEvent(
                        type="server_starting",
                        data={
                            "framework": framework,
                            "command": command,
                            "port": port,
                        },
                        model_id=context.model_id,
                        ts=datetime.now(UTC).isoformat(),
                    )
                )

            # 3. Start server process
            process = await self._start_server(command, context)

            # 4. Wait for server to be ready
            ready, error_msg = await self._wait_for_server_ready(
                port, process, wait_for_text, wait_timeout, context
            )

            if not ready:
                return ToolResult(
                    success=False,
                    error=f"Server failed to start: {error_msg}",
                    error_code="E_SERVER_START_FAILED",
                )

            # Emit server ready event
            if context.event_bus:
                from agentrunner.core.events import StreamEvent

                context.event_bus.publish(
                    StreamEvent(
                        type="server_ready",
                        data={"port": port, "url": f"http://localhost:{port}"},
                        model_id=context.model_id,
                        ts=datetime.now(UTC).isoformat(),
                    )
                )

            # 5. Take screenshot
            url = f"http://localhost:{port}"
            success, error_msg, png_bytes = await self._take_screenshot(
                url,
                abs_output_path,
                viewport_width,
                viewport_height,
                full_page,
                context,
            )

            if not success:
                return ToolResult(
                    success=False,
                    error=error_msg,
                    error_code="E_SCREENSHOT_FAILED",
                )

            # Get file info
            file_size = abs_output_path.stat().st_size
            relative_path = abs_output_path.relative_to(context.workspace.root_path)

            # Emit file_created event (for UI file tree update)
            if context.event_bus:
                from agentrunner.core.events import StreamEvent

                context.event_bus.publish(
                    StreamEvent(
                        type="file_created",
                        data={
                            "path": str(relative_path),
                            "size": file_size,
                            "line_count": 0,  # PNG file, no lines
                        },
                        model_id=context.model_id,
                        ts=datetime.now(UTC).isoformat(),
                    )
                )

            # Emit screenshot taken event (for additional metadata)
            if context.event_bus:
                from agentrunner.core.events import StreamEvent

                context.event_bus.publish(
                    StreamEvent(
                        type="screenshot_taken",
                        data={
                            "output_path": str(relative_path),
                            "absolute_path": str(abs_output_path),
                            "size_bytes": file_size,
                            "viewport": f"{viewport_width}x{viewport_height}",
                            "url": url,
                            "framework": framework,
                        },
                        model_id=context.model_id,
                        ts=datetime.now(UTC).isoformat(),
                    )
                )

            # Return success
            return ToolResult(
                success=True,
                output=(
                    f"Screenshot saved to {relative_path} ({file_size} bytes)\n"
                    f"Full path: {abs_output_path}\n"
                    f"Captured: {url}\n"
                    f"Viewport: {viewport_width}x{viewport_height}"
                ),
                data={
                    "output_path": str(relative_path),
                    "size_bytes": file_size,
                    "viewport": f"{viewport_width}x{viewport_height}",
                    "url": url,
                    "framework": framework,
                },
                files_changed=[str(relative_path)],
            )

        finally:
            # 6. Always clean up server
            if process:
                self._cleanup_server(process, context)

                if context.event_bus:
                    from agentrunner.core.events import StreamEvent

                    context.event_bus.publish(
                        StreamEvent(
                            type="server_stopped",
                            data={"port": port, "framework": framework},
                            model_id=context.model_id,
                            ts=datetime.now(UTC).isoformat(),
                        )
                    )

    def _find_free_port(self) -> int:
        """Find an available port for the server.

        Returns:
            Available port number
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port: int = s.getsockname()[1]
        return port

    async def _start_server(self, command: str, context: ToolContext) -> "subprocess.Popen[str]":
        """Start dev server subprocess.

        Args:
            command: Command to run (with port already injected)
            context: Tool context

        Returns:
            Running subprocess
        """
        context.logger.info(
            "Starting dev server",
            command=command,
            cwd=str(context.workspace.root_path),
        )

        # Start process
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=str(context.workspace.root_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            # Create new process group for cleanup
            preexec_fn=os.setpgrp if hasattr(os, "setpgrp") else None,
        )

        return process

    async def _wait_for_server_ready(
        self,
        port: int,
        process: "subprocess.Popen[str]",
        wait_for_text: str,
        timeout: int,
        context: ToolContext,
    ) -> tuple[bool, str]:
        """Wait until server is ready.

        Args:
            port: Port server should be running on
            process: Server subprocess
            wait_for_text: Text to wait for in stdout
            timeout: Max seconds to wait
            context: Tool context

        Returns:
            (success, error_message)
        """
        import select
        from datetime import timedelta

        import httpx

        url = f"http://localhost:{port}"
        deadline = datetime.now(UTC) + timedelta(seconds=timeout)
        output_buffer: list[str] = []

        context.logger.info(
            "Waiting for server ready",
            url=url,
            wait_for_text=wait_for_text,
            timeout=timeout,
        )

        while datetime.now(UTC) < deadline:
            # Check if process died
            if process.poll() is not None:
                output = "".join(output_buffer[-50:])  # Last 50 lines
                return (
                    False,
                    f"Server process exited with code {process.returncode}. " f"Output: {output}",
                )

            # Read stdout (non-blocking)
            if process.stdout:
                # Check if there's data to read
                if select.select([process.stdout], [], [], 0)[0]:
                    line = process.stdout.readline()
                    if line:
                        output_buffer.append(line)
                        context.logger.debug("Server output", line=line.strip())

                        # Check for wait_for_text
                        if wait_for_text.lower() in line.lower():
                            context.logger.info("Server ready (text match)", text=wait_for_text)
                            # Give it an extra second to fully settle
                            await asyncio.sleep(1)
                            return True, ""

            # Also try HTTP request
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    response = await client.get(url)
                    if response.status_code < 500:
                        context.logger.info(
                            "Server ready (HTTP response)",
                            status=response.status_code,
                        )
                        return True, ""
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout):
                pass  # Server not ready yet
            except Exception as e:
                context.logger.debug("HTTP check error", error=str(e))

            # Wait before retry
            await asyncio.sleep(0.5)

        # Timeout
        output = "".join(output_buffer[-50:])
        return (
            False,
            f"Server did not become ready within {timeout}s. " f"Recent output: {output}",
        )

    async def _take_screenshot(
        self,
        url: str,
        output_path: Path,
        viewport_width: int,
        viewport_height: int,
        full_page: bool,
        context: ToolContext,
    ) -> tuple[bool, str, bytes | None]:
        """Take screenshot using Playwright.

        Args:
            url: URL to screenshot
            output_path: Where to save PNG
            viewport_width: Browser width
            viewport_height: Browser height
            full_page: Capture full scrollable page
            context: Tool context

        Returns:
            (success, error_message, png_bytes)
        """
        try:
            from playwright.async_api import (
                TimeoutError as PlaywrightTimeout,
            )
            from playwright.async_api import (
                async_playwright,
            )
        except ImportError as e:
            return (
                False,
                f"Playwright not available: {str(e)}. "
                "In Docker: Should already be installed. "
                "Locally: Run 'pip install playwright && playwright install chromium'",
                None,
            )

        browser = None
        try:
            async with async_playwright() as p:
                # Launch headless Chromium
                try:
                    browser = await p.chromium.launch(
                        headless=True,
                        args=["--no-sandbox", "--disable-dev-shm-usage"],  # For Docker/ECS
                    )
                except Exception as e:
                    return (
                        False,
                        f"Failed to launch browser: {e}. "
                        "Make sure chromium is installed: playwright install chromium",
                        None,
                    )

                browser_context = await browser.new_context(
                    viewport={"width": viewport_width, "height": viewport_height},
                    device_scale_factor=2,  # High-DPI for crisp screenshots
                )

                page = await browser_context.new_page()
                page.set_default_timeout(30000)  # 30s timeout

                context.logger.info("Navigating to URL", url=url)

                # Navigate and wait for network idle
                try:
                    await page.goto(url, wait_until="networkidle", timeout=30000)
                except PlaywrightTimeout:
                    # Try without waiting for network idle
                    context.logger.warn("Network idle timeout, trying with load event only")
                    await page.goto(url, wait_until="load", timeout=30000)

                # Wait for fonts to load (with error handling)
                try:
                    await page.evaluate("() => document.fonts.ready")
                except Exception as e:
                    context.logger.debug("Font loading check failed", error=str(e))

                # Additional wait for React/Next.js hydration
                await page.wait_for_timeout(2000)

                # Take screenshot
                output_path.parent.mkdir(parents=True, exist_ok=True)
                png_bytes = await page.screenshot(
                    path=str(output_path), full_page=full_page, timeout=30000
                )

                await browser.close()
                browser = None  # Mark as closed

                context.logger.info(
                    "Screenshot captured",
                    path=str(output_path),
                    size_bytes=len(png_bytes),
                )

                return True, "", png_bytes

        except PlaywrightTimeout as e:
            context.logger.error("Screenshot timeout", error=str(e), url=url)
            return False, f"Screenshot timeout after 30s: {e}", None
        except Exception as e:
            context.logger.error("Screenshot failed", error=str(e), url=url)
            return False, f"Screenshot failed: {e}", None
        finally:
            # Ensure browser is closed
            if browser:
                try:
                    await browser.close()
                except Exception as e:
                    context.logger.debug("Error closing browser", error=str(e))

    def _cleanup_server(self, process: "subprocess.Popen[str]", context: ToolContext) -> None:
        """Terminate server and all child processes.

        Args:
            process: Server subprocess to terminate
            context: Tool context
        """
        try:
            # Kill process group first (catches npm/node children)
            if hasattr(os, "killpg"):
                try:
                    pgid = os.getpgid(process.pid)
                    os.killpg(pgid, signal.SIGTERM)
                    context.logger.debug("Terminated process group", pgid=pgid)
                except (ProcessLookupError, OSError) as e:
                    context.logger.debug("Process group kill failed", error=str(e))

            # Try graceful shutdown of main process
            try:
                process.terminate()
                process.wait(timeout=5)
                context.logger.info("Server process cleaned up gracefully", pid=process.pid)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't stop
                context.logger.warn("Server didn't stop gracefully, force killing", pid=process.pid)
                process.kill()
                process.wait()
                context.logger.info("Server process force killed", pid=process.pid)

        except Exception as e:
            context.logger.warn(
                "Error cleaning up server", error=str(e), pid=getattr(process, "pid", "unknown")
            )
