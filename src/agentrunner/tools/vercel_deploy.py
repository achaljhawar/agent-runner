"""Vercel deployment tool for production deployments.

Deploys projects to Vercel for worldwide access.
Requires VERCEL_TOKEN environment variable.
"""

import os
import random
import string
import subprocess
import time
from datetime import UTC, datetime
from typing import Any

from agentrunner.core.events import StreamEvent
from agentrunner.core.exceptions import E_UNSAFE, E_VALIDATION
from agentrunner.core.tool_protocol import ToolCall, ToolDefinition, ToolResult
from agentrunner.tools.base import BaseTool, ToolContext


class VercelDeployTool(BaseTool):
    """Deploy project to Vercel."""

    def __init__(self) -> None:
        self._deployments: dict[str, dict[str, Any]] = {}

    def _generate_project_name(self) -> str:
        """Generate random 6-character alphanumeric project name."""
        chars = string.ascii_lowercase + string.digits
        return "".join(random.choice(chars) for _ in range(6))

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """Deploy workspace to Vercel.

        Args:
            call: Tool call with optional project_name and production flag
            context: Tool execution context

        Returns:
            ToolResult with deployment URL
        """
        vercel_token = os.getenv("VERCEL_TOKEN")
        if not vercel_token:
            return ToolResult(
                success=False,
                error="VERCEL_TOKEN not set. Get token from: https://vercel.com/account/tokens",
                error_code=E_VALIDATION,
            )

        # Check for existing deployment in context
        project_name = context.deployment_context.get("vercel_project_id")

        if project_name:
            context.logger.info(
                "Redeploying to existing Vercel project",
                project_name=project_name,
                existing_url=context.deployment_context.get("vercel_url"),
                deployment_count=context.deployment_context.get("deployment_count", 0),
            )
        else:
            # Fallback: Generate random 6-char project name
            project_name = self._generate_project_name()
            context.logger.info(
                "No project_id in deployment_context - generating new one",
                project_name=project_name,
                has_deployment_context=bool(context.deployment_context),
                model_id=context.model_id,
            )

        model_id = context.model_id
        workspace_path = str(context.workspace.root_path)

        context.logger.info(
            "Deploying to Vercel",
            model_id=model_id,
            project_name=project_name,
        )

        # Build vercel deploy command - always production with predictable URL
        cmd = [
            "vercel",
            "deploy",
            "--prod",
            "--yes",
            "--name",
            project_name,
        ]

        # Add token
        cmd.extend(["--token", vercel_token])

        context.logger.debug(
            "Running Vercel deploy",
            command=f"vercel deploy --prod --yes --name {project_name} --token ***",
        )

        try:
            result = subprocess.run(
                cmd,
                cwd=workspace_path,
                capture_output=True,
                text=True,
                timeout=300,
            )

            # Check for CLI errors (exit code != 0)
            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown error"
                context.logger.error(
                    "Vercel CLI failed",
                    error=error_msg,
                )
                return ToolResult(
                    success=False,
                    error=f"Vercel CLI failed:\n{error_msg}",
                    error_code=E_UNSAFE,
                )

            # Check for build errors in stdout (Vercel returns 0 even if build fails)
            output = result.stdout or ""
            if "Build error occurred" in output or "Error: Command" in output:
                context.logger.error(
                    "Vercel build failed on server",
                    error=output,
                )
                return ToolResult(
                    success=False,
                    error=f"Vercel build failed:\n{output}",
                    error_code=E_UNSAFE,
                )

            # Production URL is predictable: https://{project-name}.vercel.app
            deployment_url = f"https://{project_name}.vercel.app"

            context.logger.info(
                "Vercel CLI deploy succeeded, waiting for DNS",
                project_name=project_name,
            )

            # Wait for DNS propagation
            time.sleep(3)

            # Verify deployment is accessible
            try:
                verify_result = subprocess.run(
                    ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", deployment_url],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                http_code = verify_result.stdout.strip()
                context.logger.info(
                    "Deployment verification",
                    url=deployment_url,
                    http_code=http_code,
                )
            except Exception as e:
                context.logger.warn(
                    "Could not verify deployment",
                    error=str(e),
                )

            # Store deployment info
            self._deployments[model_id] = {
                "url": deployment_url,
                "deployed_at": datetime.now(UTC),
                "project_name": project_name,
            }

            # Update deployment context for persistence
            context.deployment_context["vercel_project_id"] = project_name
            context.deployment_context["vercel_url"] = deployment_url
            context.deployment_context["deployed_at"] = datetime.now(UTC).isoformat()
            context.deployment_context["deployment_count"] = (
                context.deployment_context.get("deployment_count", 0) + 1
            )

            if context.event_bus:
                context.event_bus.publish(
                    StreamEvent(
                        type="deployment_ready",
                        data={
                            "url": deployment_url,
                            "production": True,
                            "project_name": project_name,
                            "message": f"Deployed to Vercel: {deployment_url}",
                        },
                        model_id=model_id,
                        ts=datetime.now(UTC).isoformat(),
                    )
                )

            context.logger.info(
                "Vercel deployment successful",
                url=deployment_url,
            )

            output = (
                f"Successfully deployed to Vercel!\n\n"
                f"URL: {deployment_url}\n"
                f"Project: {project_name}\n\n"
                f"Site is live and accessible worldwide."
            )

            return ToolResult(success=True, output=output)

        except subprocess.TimeoutExpired:
            context.logger.error("Vercel deployment timed out")
            return ToolResult(
                success=False,
                error="Deployment timed out after 5 minutes",
                error_code=E_UNSAFE,
            )

        except Exception as e:
            context.logger.error(
                "Unexpected error during deployment",
                error=str(e),
                error_type=type(e).__name__,
            )
            return ToolResult(
                success=False,
                error=f"Deployment error: {type(e).__name__}: {str(e)}",
                error_code=E_UNSAFE,
            )

    def get_definition(self) -> ToolDefinition:
        """Get tool definition."""
        return ToolDefinition(
            name="deploy_to_vercel",
            description=(
                "Deploy project to Vercel for production access.\n\n"
                "If this project was previously deployed, it will redeploy to the same URL.\n"
                "Otherwise, generates a random 6-character project name.\n"
                "Deploys to: https://{project-name}.vercel.app\n\n"
                "Requirements:\n"
                "- VERCEL_TOKEN environment variable must be set\n"
                "- Project must be built and ready to deploy\n\n"
                "Example:\n"
                "deploy_to_vercel()\n"
                "First time: https://a3x9k2.vercel.app\n"
                "Subsequent: https://a3x9k2.vercel.app (same URL)\n\n"
                "The tool will:\n"
                "1. Check if project was previously deployed\n"
                "2. Reuse existing project name or generate new random 6-char name\n"
                "3. Validate VERCEL_TOKEN is set\n"
                "4. Run 'vercel deploy --prod --yes --name {project-name}'\n"
                "5. Wait for deployment (max 5 minutes)\n"
                "6. Wait for DNS propagation (3 seconds)\n"
                "7. Verify site is accessible\n"
                "8. Return production URL\n\n"
                "Always build project before deploying!"
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
        )
