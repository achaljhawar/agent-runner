"""Project scaffolding tool using official CLI tools.

Provides automated project generation using framework-specific scaffolding tools.
Eliminates manual file creation for supported project types.

Features hybrid caching: First scaffold is slow (~20s), subsequent uses cache (<2s).
"""

import hashlib
import shutil
from enum import Enum
from pathlib import Path

from agentrunner.core.exceptions import E_VALIDATION
from agentrunner.core.tool_protocol import ToolCall, ToolDefinition, ToolResult
from agentrunner.tools.base import BaseTool, ToolContext


class ProjectType(str, Enum):
    """Supported project types for scaffolding."""

    NEXT = "next"
    REACT = "react"
    VITE_REACT = "vite-react"
    VITE_VUE = "vite-vue"
    VITE_SVELTE = "vite-svelte"
    VUE = "vue"
    ASTRO = "astro"
    REMIX = "remix"
    PYTHON_POETRY = "python-poetry"
    PYTHON_PDM = "python-pdm"
    FASTAPI = "fastapi"
    DJANGO = "django"


class ScaffoldProjectTool(BaseTool):
    """Scaffold projects using official CLI tools.

    Automatically runs the correct scaffolding command based on project type.
    Uses bash tool internally (120s timeout) to execute scaffolding commands.

    Features hybrid caching for performance:
    - First scaffold: Runs live command (~10-30s), caches result
    - Subsequent: Copies from cache (<2s)
    - Cache location: ~/.agentrunner/scaffold-cache/
    """

    def __init__(self, enable_cache: bool = True, cache_dir: Path | None = None):
        """Initialize scaffold tool with caching enabled by default."""
        self.enable_cache = enable_cache
        self.cache_dir = cache_dir or Path.home() / ".agentrunner" / "scaffold-cache"

        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _snapshot_files(self, root: Path) -> set[str]:
        """Create a set of relative file paths under root for diffing."""
        files: set[str] = set()
        # Exclude internal env/cache directories
        exclude_dirs = {".agentrunner-env", self.cache_dir.name if self.cache_dir else ""}
        for path in root.rglob("*"):
            try:
                if path.is_file():
                    # Exclude anything inside exclude_dirs
                    parts = {p.name for p in path.relative_to(root).parents}
                    if parts.intersection(exclude_dirs):
                        continue
                    files.add(str(path.relative_to(root)))
            except Exception:
                # Skip problematic paths
                continue
        return files

    def _get_cache_key(
        self,
        project_type: str,
        typescript: bool,
        tailwind: bool,
        template: str | None,
        use_src_dir: bool = False,
    ) -> str:
        """Generate cache key for template configuration."""
        config = (
            f"{project_type}-ts{typescript}-tw{tailwind}-src{use_src_dir}-tpl{template or 'none'}"
        )
        return hashlib.md5(config.encode()).hexdigest()[:8]

    def _build_command(
        self,
        project_type: str,
        typescript: bool,
        tailwind: bool,
        template: str | None,
        use_src_dir: bool = False,
    ) -> str:
        """Build scaffolding command. Always scaffolds into current directory."""
        project_name = "."

        commands = {
            ProjectType.NEXT.value: (
                f"npx --yes create-next-app@latest {project_name} "
                f"{'--typescript' if typescript else '--javascript'} "
                f"{'--tailwind' if tailwind else '--no-tailwind'} "
                f"{'--src-dir' if use_src_dir else '--no-src-dir'} "
                f"--app --import-alias '@/*' --turbopack --no-git --yes "
                f"--eslint --skip-install"
            ),
            ProjectType.REACT.value: (
                f"npx --yes create-react-app {project_name} "
                f"{'--template typescript' if typescript else ''} "
                f"--use-npm"
            ),
            ProjectType.VITE_REACT.value: (
                f"npm create vite@latest {project_name} -- "
                f"--template {'react-ts' if typescript else 'react'} --yes"
            ),
            ProjectType.VITE_VUE.value: (
                f"npm create vite@latest {project_name} -- "
                f"--template {'vue-ts' if typescript else 'vue'} --yes"
            ),
            ProjectType.VITE_SVELTE.value: (
                f"npm create vite@latest {project_name} -- "
                f"--template {'svelte-ts' if typescript else 'svelte'} --yes"
            ),
            ProjectType.VUE.value: (
                f"npm create vue@latest {project_name} -- "
                f"{'--typescript' if typescript else ''} --yes"
            ),
            ProjectType.ASTRO.value: (
                f"npm create astro@latest {project_name} -- "
                f"--template {template or 'basics'} "
                f"--typescript strict --yes"
            ),
            ProjectType.REMIX.value: f"npx --yes create-remix@latest {project_name} --yes --no-git-init --install",
            ProjectType.PYTHON_POETRY.value: f"poetry new {project_name}",
            ProjectType.PYTHON_PDM.value: f"pdm init {project_name} --non-interactive",
            ProjectType.FASTAPI.value: (
                f"cookiecutter gh:tiangolo/full-stack-fastapi-template "
                f"--no-input project_name={project_name}"
            ),
            ProjectType.DJANGO.value: f"django-admin startproject {project_name}",
        }

        return commands.get(project_type, "")

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """Execute project scaffolding into current workspace."""
        project_type = call.arguments.get("project_type")
        typescript = call.arguments.get("typescript", True)
        tailwind = call.arguments.get("tailwind", False)
        template = call.arguments.get("template")
        use_src_dir = call.arguments.get("use_src_dir", False)  # Default to no src dir (App Router)

        if not project_type:
            return ToolResult(
                success=False,
                error="project_type is required",
                error_code=E_VALIDATION,
            )

        # Validate project type
        try:
            ProjectType(project_type)
        except ValueError:
            valid_types = [t.value for t in ProjectType]
            return ToolResult(
                success=False,
                error=f"Unsupported project type: {project_type}. Supported types: {', '.join(valid_types)}",
                error_code=E_VALIDATION,
            )

        # Build command
        command = self._build_command(project_type, typescript, tailwind, template, use_src_dir)

        if not command:
            return ToolResult(
                success=False,
                error=f"No scaffolding command available for {project_type}",
                error_code=E_VALIDATION,
            )

        target_path = context.workspace.root_path

        if self.enable_cache:
            cache_key = self._get_cache_key(
                project_type, typescript, tailwind, template, use_src_dir
            )
            cache_path = self.cache_dir / cache_key

            if cache_path.exists():
                context.logger.info(
                    "Using cached template",
                    cache_key=cache_key,
                    project_type=project_type,
                )

                try:
                    for item in cache_path.iterdir():
                        if item.is_dir():
                            shutil.copytree(
                                item, target_path / item.name, symlinks=False, dirs_exist_ok=True
                            )
                        else:
                            shutil.copy2(item, target_path / item.name)

                    # Chown copied files to session UID so subprocess can write
                    if context.session_uid is not None and context.session_gid is not None:
                        try:
                            import os

                            for root, dirs, files in os.walk(target_path):
                                os.chown(root, context.session_uid, context.session_gid)
                                for d in dirs:
                                    try:
                                        os.chown(
                                            os.path.join(root, d),
                                            context.session_uid,
                                            context.session_gid,
                                        )
                                    except (OSError, FileNotFoundError):
                                        pass
                                for f in files:
                                    try:
                                        os.chown(
                                            os.path.join(root, f),
                                            context.session_uid,
                                            context.session_gid,
                                        )
                                    except (OSError, FileNotFoundError):
                                        pass
                        except (OSError, PermissionError):
                            pass  # Not running as root, can't chown

                    # Emit scaffold complete event
                    if context.event_bus:
                        from datetime import UTC, datetime

                        from agentrunner.core.events import StreamEvent

                        context.event_bus.publish(
                            StreamEvent(
                                type="scaffold_complete",
                                data={},
                                model_id=context.model_id,
                                ts=datetime.now(UTC).isoformat(),
                            )
                        )

                    output = (
                        f"Project scaffolded from cache (<2s)!\n"
                        f"Type: {project_type} (TypeScript: {typescript}, Tailwind: {tailwind})\n"
                        f"Cache key: {cache_key}\n\n"
                        f"OPTIMIZATION TIP: Create files and write code FIRST, then run 'npm install' as the last step.\n"
                        f"This lets the user see immediate progress while npm install runs (2-5min)."
                    )

                    return ToolResult(success=True, output=output)

                except Exception as e:
                    context.logger.warn(
                        "Cache copy failed, falling back to live scaffold",
                        error=str(e),
                    )

        context.logger.info("Scaffolding live", project_type=project_type)

        from agentrunner.core.tool_protocol import ToolCall as BashCall
        from agentrunner.tools.bash import BashTool

        bash_call = BashCall(
            id=f"{call.id}_bash",
            name="bash",
            arguments={"command": command},
        )

        bash_tool = BashTool()
        result = await bash_tool.execute(bash_call, context)

        if result.success:
            if self.enable_cache:
                cache_key = self._get_cache_key(project_type, typescript, tailwind, template)
                cache_path = self.cache_dir / cache_key

                if not cache_path.exists():
                    try:
                        cache_path.mkdir(parents=True)
                        for item in target_path.iterdir():
                            if item.name == ".agentrunner-env" or str(item).startswith(
                                str(self.cache_dir)
                            ):
                                continue
                            # Skip package-lock.json - it's incomplete from --skip-install
                            if item.name == "package-lock.json":
                                continue
                            if item.is_dir():
                                shutil.copytree(item, cache_path / item.name, symlinks=False)
                            else:
                                shutil.copy2(item, cache_path / item.name)
                        context.logger.info(
                            "Cached scaffolded project",
                            cache_key=cache_key,
                            project_type=project_type,
                        )
                    except Exception as e:
                        context.logger.warn(
                            "Failed to cache project",
                            error=str(e),
                        )
            # Publish scaffold complete event (UI will query backend for actual files)
            if context.event_bus:
                from datetime import UTC, datetime

                from agentrunner.core.events import StreamEvent

                context.event_bus.publish(
                    StreamEvent(
                        type="scaffold_complete",
                        data={},
                        model_id=context.model_id,
                        ts=datetime.now(UTC).isoformat(),
                    )
                )

            output = result.output or ""
            output += f"\n\nProject scaffolded successfully using {project_type}!"

            # Add Next.js-specific structure info
            if project_type == ProjectType.NEXT.value:
                if use_src_dir:
                    output += "\n📁 Structure: /src/app (Pages Router style)"
                else:
                    output += "\n📁 Structure: /app in root (App Router - recommended)"

            if self.enable_cache:
                cache_key = self._get_cache_key(
                    project_type, typescript, tailwind, template, use_src_dir
                )
                output += f"\n📦 Cached for future use (cache key: {cache_key})"

            if project_type in [
                ProjectType.NEXT.value,
                ProjectType.REACT.value,
                ProjectType.VITE_REACT.value,
                ProjectType.VITE_VUE.value,
                ProjectType.VITE_SVELTE.value,
                ProjectType.VUE.value,
                ProjectType.ASTRO.value,
                ProjectType.REMIX.value,
            ]:
                output += "\n\nOPTIMIZATION TIP: Create files and write code FIRST, then run 'npm install' last.\n"
                output += "After 'npm install' completes (2-5min), you can run:\n"
                output += "  • npm run dev (start dev server)\n"
                output += "  • npm run build (production build)\n"
                output += "  • npm run lint (check code)\n"
            elif project_type in [
                ProjectType.PYTHON_POETRY.value,
                ProjectType.PYTHON_PDM.value,
            ]:
                output += "\n\nNext steps:\n"
                output += "  1. Install dependencies (poetry install or pdm install)\n"

            context.logger.info(
                "Project scaffolded",
                project_type=project_type,
                typescript=typescript,
                tailwind=tailwind,
            )

            return ToolResult(success=True, output=output)

        return ToolResult(
            success=False,
            error=f"Scaffolding failed: {result.error}",
            error_code=result.error_code,
            output=result.output,
        )

    def get_definition(self) -> ToolDefinition:
        """Get tool definition for LLM."""
        return ToolDefinition(
            name="scaffold_project",
            description=(
                "Scaffold a new project using official CLI tools into the current workspace. "
                "ALWAYS use this instead of manually creating files for frameworks. "
                "Automatically runs create-next-app, create-react-app, npm create vite, poetry new, etc. "
                "Scaffolds directly into current directory (no subdirectory created).\n\n"
                "CRITICAL WORKFLOW (Node.js projects):\n"
                "1. scaffold_project() - Creates package.json + project structure (fast, <2s from cache)\n"
                "2. bash('npm install', timeout=300) - REQUIRED! Installs ~300 packages (may take 2-3 min)\n"
                "3. Now you can run: npm run dev, npm run build, npm run lint\n\n"
                "IMPORTANT:\n"
                "   • Use timeout=300 for npm install (default 120s will timeout!)\n"
                "   • If you skip 'npm install', you'll get errors like:\n"
                "     - 'typescript not found' when running npm run build\n"
                "     - 'eslint not found' when running npm run lint\n"
                "     - Failed to transpile next.config.ts\n\n"
                "ALWAYS run 'npm install' with adequate timeout immediately after scaffolding!"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "project_type": {
                        "type": "string",
                        "enum": [t.value for t in ProjectType],
                        "description": (
                            "Type of project to scaffold. Supported: "
                            "next (Next.js), react (Create React App), vite-react, vite-vue, vite-svelte, "
                            "vue, astro, remix, python-poetry, python-pdm, fastapi, django"
                        ),
                    },
                    "typescript": {
                        "type": "boolean",
                        "description": "Use TypeScript (default: true for JS frameworks)",
                        "default": True,
                    },
                    "tailwind": {
                        "type": "boolean",
                        "description": "Include Tailwind CSS (Next.js, Vite, Astro)",
                        "default": False,
                    },
                    "use_src_dir": {
                        "type": "boolean",
                        "description": (
                            "REQUIRED for Next.js projects. Choose project structure:\n"
                            "• false = App Router with /app in root (RECOMMENDED for new projects)\n"
                            "• true = Pages Router style with /src/app or /src/pages\n"
                            "Most new Next.js projects should use false (App Router)."
                        ),
                    },
                    "template": {
                        "type": "string",
                        "description": "Specific template to use (framework-dependent, e.g., 'blog' for Astro)",
                    },
                },
                "required": ["project_type", "use_src_dir"],
            },
        )
