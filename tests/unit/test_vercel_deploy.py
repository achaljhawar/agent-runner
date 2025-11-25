"""Unit tests for Vercel deployment tool."""

import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from agentrunner.core.events import EventBus
from agentrunner.core.exceptions import E_UNSAFE, E_VALIDATION
from agentrunner.core.logger import AgentRunnerLogger
from agentrunner.core.tool_protocol import ToolCall
from agentrunner.core.workspace import Workspace
from agentrunner.tools.base import ToolContext
from agentrunner.tools.vercel_deploy import VercelDeployTool


@pytest.fixture
def workspace(tmp_path):
    """Create test workspace."""
    return Workspace(str(tmp_path))


@pytest.fixture
def logger():
    """Create test logger."""
    return AgentRunnerLogger()


@pytest.fixture
def event_bus():
    """Create test event bus."""
    return EventBus()


@pytest.fixture
def context(workspace, logger, event_bus):
    """Create tool context."""
    return ToolContext(
        workspace=workspace,
        logger=logger,
        model_id="test-model",
        event_bus=event_bus,
    )


@pytest.fixture
def tool():
    """Create Vercel deploy tool."""
    return VercelDeployTool()


def test_get_definition(tool):
    """Test tool definition."""
    definition = tool.get_definition()

    assert definition.name == "deploy_to_vercel"
    assert "Deploy" in definition.description
    assert "Vercel" in definition.description
    assert "VERCEL_TOKEN" in definition.description
    assert "random" in definition.description.lower()

    assert definition.parameters["required"] == []


@pytest.mark.asyncio
async def test_missing_vercel_token(tool, context):
    """Test error when VERCEL_TOKEN is missing."""
    with patch.dict(os.environ, {}, clear=True):
        call = ToolCall(
            id="test",
            name="deploy_to_vercel",
            arguments={},
        )

        result = await tool.execute(call, context)

        assert result.success is False
        assert "VERCEL_TOKEN" in result.error
        assert result.error_code == E_VALIDATION


def test_generate_project_name(tool):
    """Test project name generation."""
    name = tool._generate_project_name()

    assert len(name) == 6
    assert name.isalnum()
    assert name.islower()

    # Test uniqueness (generate 10 names, should all be different)
    names = {tool._generate_project_name() for _ in range(10)}
    assert len(names) >= 8  # At least 8 unique names out of 10


@pytest.mark.asyncio
async def test_successful_deployment(tool, context):
    """Test successful deployment with auto-generated project name."""
    with patch.dict(os.environ, {"VERCEL_TOKEN": "test_token"}):
        with patch("subprocess.run") as mock_run, patch("time.sleep"):
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Deployed successfully",
                stderr="",
            )

            call = ToolCall(
                id="test",
                name="deploy_to_vercel",
                arguments={},
            )

            result = await tool.execute(call, context)

            assert result.success is True
            assert ".vercel.app" in result.output
            assert "Successfully deployed" in result.output

            # Verify command has all required flags
            deploy_call = [c for c in mock_run.call_args_list if "vercel" in str(c)][0]
            args = deploy_call[0][0]
            assert "vercel" in args
            assert "deploy" in args
            assert "--prod" in args
            assert "--yes" in args
            assert "--name" in args
            assert "--token" in args
            assert "test_token" in args

            # Verify project name was generated (6 chars)
            name_idx = args.index("--name") + 1
            project_name = args[name_idx]
            assert len(project_name) == 6
            assert project_name.isalnum()
            assert project_name.islower()


@pytest.mark.asyncio
async def test_deployment_cli_failure(tool, context):
    """Test Vercel CLI failure (exit code != 0)."""
    with patch.dict(os.environ, {"VERCEL_TOKEN": "test_token"}):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr="Error: Invalid token",
            )

            call = ToolCall(
                id="test",
                name="deploy_to_vercel",
                arguments={},
            )

            result = await tool.execute(call, context)

            assert result.success is False
            assert "Vercel CLI failed" in result.error
            assert result.error_code == E_UNSAFE


@pytest.mark.asyncio
async def test_deployment_build_failure(tool, context):
    """Test Vercel build failure on server (exit code 0 but build fails)."""
    with patch.dict(os.environ, {"VERCEL_TOKEN": "test_token"}):
        with patch("subprocess.run") as mock_run, patch("time.sleep"):
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Deploying...\nBuild error occurred\nError: Module not found",
                stderr="",
            )

            call = ToolCall(
                id="test",
                name="deploy_to_vercel",
                arguments={},
            )

            result = await tool.execute(call, context)

            assert result.success is False
            assert "Vercel build failed" in result.error
            assert "Build error occurred" in result.error
            assert result.error_code == E_UNSAFE


@pytest.mark.asyncio
async def test_deployment_timeout(tool, context):
    """Test deployment timeout."""
    with patch.dict(os.environ, {"VERCEL_TOKEN": "test_token"}):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("vercel", 300)

            call = ToolCall(
                id="test",
                name="deploy_to_vercel",
                arguments={},
            )

            result = await tool.execute(call, context)

            assert result.success is False
            assert "timed out" in result.error.lower()
            assert result.error_code == E_UNSAFE


@pytest.mark.asyncio
async def test_event_emission(tool, context):
    """Test deployment event emission."""
    events = []

    def capture_event(event):
        events.append(event)

    context.event_bus.subscribe(capture_event)

    with patch.dict(os.environ, {"VERCEL_TOKEN": "test_token"}):
        with patch("subprocess.run") as mock_run, patch("time.sleep"):
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Deployed",
                stderr="",
            )

            call = ToolCall(
                id="test",
                name="deploy_to_vercel",
                arguments={},
            )

            result = await tool.execute(call, context)

            assert result.success is True
            assert len(events) == 1
            assert events[0].type == "deployment_ready"
            assert ".vercel.app" in events[0].data["url"]
            assert events[0].data["production"] is True
            assert len(events[0].data["project_name"]) == 6


@pytest.mark.asyncio
async def test_deployment_tracking(tool, context):
    """Test deployment tracking per model."""
    with patch.dict(os.environ, {"VERCEL_TOKEN": "test_token"}):
        with patch("subprocess.run") as mock_run, patch("time.sleep"):
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Deployed",
                stderr="",
            )

            call = ToolCall(
                id="test",
                name="deploy_to_vercel",
                arguments={},
            )

            result = await tool.execute(call, context)

            assert result.success is True
            assert "test-model" in tool._deployments
            assert ".vercel.app" in tool._deployments["test-model"]["url"]
            assert len(tool._deployments["test-model"]["project_name"]) == 6
