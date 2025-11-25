"""Configuration system for Agent.

Provides configuration loading, merging, and validation with precedence:
1. Environment variables (highest)
2. Project config (.agentrunner/config.json)
3. User profile (~/.agentrunner/profiles/<name>.json)
4. Defaults (lowest)
"""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agentrunner.core.exceptions import ConfigurationError


@dataclass
class AgentConfig:
    """Configuration for agentrunner agent orchestration.

    This contains ONLY agent-level settings (max rounds, timeouts, etc.).
    Provider-specific settings (model, temperature, max_tokens) are in ProviderConfig.

    Separation of concerns:
    - AgentConfig: Orchestration (how the agent loop works)
    - ProviderConfig: LLM behavior (which model, how it generates)
    """

    max_rounds: int = 50
    response_buffer_tokens: int = 1000
    allow_streaming: bool = True
    tool_timeout_s: int = (
        120  # Increased from 30 to 120 for scaffolding tools (npm, create-next-app, etc.)
    )
    safety: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_rounds < 1:
            raise ConfigurationError(f"max_rounds must be >= 1, got {self.max_rounds}")
        if self.response_buffer_tokens < 0:
            raise ConfigurationError(
                f"response_buffer_tokens must be >= 0, got {self.response_buffer_tokens}"
            )
        if self.tool_timeout_s < 1:
            raise ConfigurationError(f"tool_timeout_s must be >= 1, got {self.tool_timeout_s}")

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentConfig":
        """Create config from dictionary, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


def load_user_config(profile_name: str = "default") -> AgentConfig:
    """Load user configuration from ~/.agentrunner/profiles/<name>.json.

    Args:
        profile_name: Name of profile to load (default: "default")

    Returns:
        AgentConfig loaded from profile, or default config if not found

    Raises:
        ConfigurationError: If profile file is invalid JSON
    """
    profile_path = Path.home() / ".agentrunner" / "profiles" / f"{profile_name}.json"

    if not profile_path.exists():
        return AgentConfig()

    try:
        with profile_path.open() as f:
            data = json.load(f)
        return AgentConfig.from_dict(data)
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in profile {profile_name}: {e}") from e
    except Exception as e:
        raise ConfigurationError(f"Failed to load profile {profile_name}: {e}") from e


def load_project_config(project_root: Path | None = None) -> AgentConfig | None:
    """Load project-specific configuration from .agentrunner/config.json.

    Args:
        project_root: Root directory to search for .agentrunner/config.json
                     (default: current directory)

    Returns:
        AgentConfig if config file exists, None otherwise

    Raises:
        ConfigurationError: If config file is invalid JSON
    """
    if project_root is None:
        project_root = Path.cwd()

    config_path = project_root / ".agentrunner" / "config.json"

    if not config_path.exists():
        return None

    try:
        with config_path.open() as f:
            data = json.load(f)
        return AgentConfig.from_dict(data)
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in project config: {e}") from e
    except Exception as e:
        raise ConfigurationError(f"Failed to load project config: {e}") from e


def load_env_overrides() -> dict[str, Any]:
    """Load configuration overrides from environment variables.

    Supported environment variables (agent-level only):
    - AGENTRUNNER_MAX_ROUNDS: Maximum agentic loop iterations
    - AGENTRUNNER_TOOL_TIMEOUT: Tool execution timeout in seconds

    Note: Provider-specific env vars (AGENTRUNNER_MODEL, AGENTRUNNER_TEMPERATURE, AGENTRUNNER_MAX_TOKENS)
    are handled in CLI. This function only handles agent orchestration settings.

    Returns:
        Dictionary of configuration overrides
    """
    overrides: dict[str, Any] = {}

    if max_rounds_str := os.getenv("AGENTRUNNER_MAX_ROUNDS"):
        try:
            overrides["max_rounds"] = int(max_rounds_str)
        except ValueError as e:
            raise ConfigurationError(f"Invalid AGENTRUNNER_MAX_ROUNDS: {max_rounds_str}") from e

    if timeout_str := os.getenv("AGENTRUNNER_TOOL_TIMEOUT"):
        try:
            overrides["tool_timeout_s"] = int(timeout_str)
        except ValueError as e:
            raise ConfigurationError(f"Invalid AGENTRUNNER_TOOL_TIMEOUT: {timeout_str}") from e

    return overrides


def merge_configs(
    base: AgentConfig,
    project: AgentConfig | None = None,
    env_overrides: dict[str, Any] | None = None,
) -> AgentConfig:
    """Merge configurations with precedence: env > project > base.

    Args:
        base: Base configuration (typically from user profile)
        project: Project-specific configuration (optional)
        env_overrides: Environment variable overrides (optional)

    Returns:
        Merged configuration
    """
    # Start with base config as dict
    merged = base.to_dict()

    # Get default values to determine what was explicitly set
    defaults = AgentConfig().to_dict()

    # Apply project config (only override non-default values)
    if project:
        project_dict = project.to_dict()
        for key, value in project_dict.items():
            # For safety dict, merge rather than replace
            if key == "safety" and isinstance(value, dict):
                merged.setdefault("safety", {}).update(value)
            # Only override if the project value differs from default
            elif value != defaults.get(key):
                merged[key] = value

    # Apply environment overrides (highest precedence)
    if env_overrides:
        for key, value in env_overrides.items():
            merged[key] = value

    return AgentConfig.from_dict(merged)


def load_config(profile_name: str = "default", project_root: Path | None = None) -> AgentConfig:
    """Load and merge all configuration sources.

    Precedence (highest to lowest):
    1. Environment variables
    2. Project config (.agentrunner/config.json)
    3. User profile (~/.agentrunner/profiles/<name>.json)
    4. Defaults

    Args:
        profile_name: User profile to load (default: "default")
        project_root: Project root directory (default: current directory)

    Returns:
        Merged configuration

    Raises:
        ConfigurationError: If any config source is invalid
    """
    # Load base config from user profile
    base_config = load_user_config(profile_name)

    # Load project-specific config
    project_config = load_project_config(project_root)

    # Load environment overrides
    env_overrides = load_env_overrides()

    # Merge with precedence
    return merge_configs(base_config, project_config, env_overrides)
