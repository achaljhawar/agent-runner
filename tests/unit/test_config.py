"""Tests for configuration system."""

import json
from pathlib import Path

import pytest

from agentrunner.core.config import (
    AgentConfig,
    load_config,
    load_env_overrides,
    load_project_config,
    load_user_config,
    merge_configs,
)
from agentrunner.core.exceptions import ConfigurationError


class TestAgentConfig:
    def test_default_config(self):
        config = AgentConfig()
        assert config.max_rounds == 50
        assert config.response_buffer_tokens == 1000
        assert config.allow_streaming is True
        assert config.tool_timeout_s == 120  # Default increased to 120 for scaffolding tools
        assert config.safety == {}
        # Note: model, temperature, max_tokens moved to ProviderConfig

    def test_custom_config(self):
        config = AgentConfig(
            max_rounds=200,
            response_buffer_tokens=2000,
            allow_streaming=False,
            tool_timeout_s=60,
            safety={"bash_whitelist": ["npm", "pip"]},
        )
        assert config.max_rounds == 200
        assert config.response_buffer_tokens == 2000
        assert config.allow_streaming is False
        assert config.tool_timeout_s == 60
        assert config.safety == {"bash_whitelist": ["npm", "pip"]}
        # Note: model, temperature, max_tokens moved to ProviderConfig

    def test_validation_max_rounds_negative(self):
        with pytest.raises(ConfigurationError, match="max_rounds must be >= 1"):
            AgentConfig(max_rounds=0)

    def test_validation_temperature_too_low(self):
        # Temperature validation moved to ProviderConfig
        # This test no longer applies to AgentConfig
        pass

    def test_validation_temperature_too_high(self):
        # Temperature validation moved to ProviderConfig
        # This test no longer applies to AgentConfig
        pass

    def test_validation_buffer_negative(self):
        with pytest.raises(ConfigurationError, match="response_buffer_tokens must be >= 0"):
            AgentConfig(response_buffer_tokens=-1)

    def test_validation_timeout_negative(self):
        with pytest.raises(ConfigurationError, match="tool_timeout_s must be >= 1"):
            AgentConfig(tool_timeout_s=0)

    def test_to_dict(self):
        config = AgentConfig(max_rounds=100)
        data = config.to_dict()
        assert data["max_rounds"] == 100
        assert "response_buffer_tokens" in data
        assert "allow_streaming" in data

    def test_from_dict(self):
        data = {"max_rounds": 150, "tool_timeout_s": 45, "unknown_key": "should_be_ignored"}
        config = AgentConfig.from_dict(data)
        assert config.max_rounds == 150
        assert config.tool_timeout_s == 45
        assert not hasattr(config, "unknown_key")


class TestLoadUserConfig:
    def test_load_nonexistent_profile(self):
        config = load_user_config("nonexistent_profile_xyz")
        assert config.max_rounds == 50  # Returns defaults

    def test_load_valid_profile(self, tmp_path):
        profile_dir = tmp_path / ".agentrunner" / "profiles"
        profile_dir.mkdir(parents=True)

        profile_path = profile_dir / "test.json"
        profile_data = {"max_rounds": 200, "temperature": 0.2}
        profile_path.write_text(json.dumps(profile_data))

        # Mock home directory
        original_home = Path.home
        Path.home = lambda: tmp_path

        try:
            config = load_user_config("test")
            assert config.max_rounds == 200
        finally:
            Path.home = original_home

    def test_load_invalid_json(self, tmp_path):
        profile_dir = tmp_path / ".agentrunner" / "profiles"
        profile_dir.mkdir(parents=True)

        profile_path = profile_dir / "bad.json"
        profile_path.write_text("{invalid json")

        original_home = Path.home
        Path.home = lambda: tmp_path

        try:
            with pytest.raises(ConfigurationError, match="Invalid JSON"):
                load_user_config("bad")
        finally:
            Path.home = original_home


class TestLoadProjectConfig:
    def test_load_nonexistent_project_config(self, tmp_path):
        config = load_project_config(tmp_path)
        assert config is None

    def test_load_valid_project_config(self, tmp_path):
        config_dir = tmp_path / ".agentrunner"
        config_dir.mkdir()

        config_path = config_dir / "config.json"
        config_data = {"max_rounds": 300, "safety": {"bash_whitelist": ["pytest", "ruff"]}}
        config_path.write_text(json.dumps(config_data))

        config = load_project_config(tmp_path)
        assert config is not None
        assert config.max_rounds == 300
        assert config.safety == {"bash_whitelist": ["pytest", "ruff"]}

    def test_load_project_config_current_dir(self, tmp_path, monkeypatch):
        config_dir = tmp_path / ".agentrunner"
        config_dir.mkdir()

        config_path = config_dir / "config.json"
        config_path.write_text(json.dumps({"max_rounds": 75}))

        monkeypatch.chdir(tmp_path)

        config = load_project_config()
        assert config is not None
        assert config.max_rounds == 75

    def test_load_invalid_project_json(self, tmp_path):
        config_dir = tmp_path / ".agentrunner"
        config_dir.mkdir()

        config_path = config_dir / "config.json"
        config_path.write_text("not json")

        with pytest.raises(ConfigurationError, match="Invalid JSON"):
            load_project_config(tmp_path)


class TestLoadEnvOverrides:
    def test_no_env_vars(self, monkeypatch):
        for key in [
            "AGENTRUNNER_MAX_ROUNDS",
            "AGENTRUNNER_TOOL_TIMEOUT",
        ]:
            monkeypatch.delenv(key, raising=False)

        overrides = load_env_overrides()
        assert overrides == {}

    def test_model_override(self, monkeypatch):
        # Model override moved to ProviderConfig
        # This test no longer applies to AgentConfig
        pass

    def test_max_rounds_override(self, monkeypatch):
        monkeypatch.setenv("AGENTRUNNER_MAX_ROUNDS", "500")
        overrides = load_env_overrides()
        assert overrides["max_rounds"] == 500

    def test_temperature_override(self, monkeypatch):
        # Temperature override moved to ProviderConfig
        # This test no longer applies to AgentConfig
        pass

    def test_max_tokens_override(self, monkeypatch):
        # Max tokens override moved to ProviderConfig
        # This test no longer applies to AgentConfig
        pass

    def test_timeout_override(self, monkeypatch):
        monkeypatch.setenv("AGENTRUNNER_TOOL_TIMEOUT", "120")
        overrides = load_env_overrides()
        assert overrides["tool_timeout_s"] == 120

    def test_all_overrides(self, monkeypatch):
        monkeypatch.setenv("AGENTRUNNER_MAX_ROUNDS", "1000")
        monkeypatch.setenv("AGENTRUNNER_TOOL_TIMEOUT", "60")

        overrides = load_env_overrides()
        assert overrides["max_rounds"] == 1000
        assert overrides["tool_timeout_s"] == 60

    def test_invalid_max_rounds(self, monkeypatch):
        monkeypatch.setenv("AGENTRUNNER_MAX_ROUNDS", "not_a_number")
        with pytest.raises(ConfigurationError, match="Invalid AGENTRUNNER_MAX_ROUNDS"):
            load_env_overrides()

    def test_invalid_temperature(self, monkeypatch):
        # Temperature validation moved to ProviderConfig
        # This test no longer applies to AgentConfig
        pass

    def test_invalid_max_tokens(self, monkeypatch):
        # Max tokens validation moved to ProviderConfig
        # This test no longer applies to AgentConfig
        pass

    def test_invalid_timeout(self, monkeypatch):
        monkeypatch.setenv("AGENTRUNNER_TOOL_TIMEOUT", "xyz")
        with pytest.raises(ConfigurationError, match="Invalid AGENTRUNNER_TOOL_TIMEOUT"):
            load_env_overrides()


class TestMergeConfigs:
    def test_merge_base_only(self):
        base = AgentConfig(max_rounds=100, tool_timeout_s=45)
        merged = merge_configs(base)
        assert merged.max_rounds == 100
        assert merged.tool_timeout_s == 45

    def test_merge_base_and_project(self):
        base = AgentConfig(max_rounds=100, tool_timeout_s=30)
        project = AgentConfig(max_rounds=150, response_buffer_tokens=2000)

        merged = merge_configs(base, project)
        assert merged.max_rounds == 150  # Overridden
        assert merged.response_buffer_tokens == 2000  # From project
        assert merged.tool_timeout_s == 30  # From base

    def test_merge_safety_dicts(self):
        base = AgentConfig(safety={"bash_whitelist": ["npm"]})
        project = AgentConfig(safety={"bash_blacklist": ["rm"]})

        merged = merge_configs(base, project)
        assert "bash_whitelist" in merged.safety
        assert "bash_blacklist" in merged.safety
        assert merged.safety["bash_whitelist"] == ["npm"]
        assert merged.safety["bash_blacklist"] == ["rm"]

    def test_merge_with_env_overrides(self):
        base = AgentConfig(max_rounds=100, tool_timeout_s=30)
        project = AgentConfig(max_rounds=150)
        env = {"max_rounds": 500, "tool_timeout_s": 60}

        merged = merge_configs(base, project, env)
        assert merged.max_rounds == 500  # Env has highest precedence
        assert merged.tool_timeout_s == 60  # Env overrides base

    def test_merge_none_project(self):
        base = AgentConfig(tool_timeout_s=30)
        merged = merge_configs(base, None, {"max_rounds": 200})
        assert merged.tool_timeout_s == 30
        assert merged.max_rounds == 200


class TestLoadConfig:
    def test_load_config_defaults_only(self, tmp_path, monkeypatch):
        # Mock home to non-existent directory
        Path.home = lambda: tmp_path / "nonexistent"
        monkeypatch.chdir(tmp_path)

        for key in ["AGENTRUNNER_MAX_ROUNDS", "AGENTRUNNER_TOOL_TIMEOUT"]:
            monkeypatch.delenv(key, raising=False)

        config = load_config()
        assert config.max_rounds == 50  # Default
        assert config.tool_timeout_s == 120  # Default increased to 120 for scaffolding tools

    def test_load_config_with_profile(self, tmp_path, monkeypatch):
        profile_dir = tmp_path / ".agentrunner" / "profiles"
        profile_dir.mkdir(parents=True)

        profile_path = profile_dir / "custom.json"
        profile_path.write_text(json.dumps({"max_rounds": 150, "tool_timeout_s": 45}))

        Path.home = lambda: tmp_path
        monkeypatch.chdir(tmp_path)

        for key in ["AGENTRUNNER_MAX_ROUNDS", "AGENTRUNNER_TOOL_TIMEOUT"]:
            monkeypatch.delenv(key, raising=False)

        config = load_config("custom")
        assert config.max_rounds == 150
        assert config.tool_timeout_s == 45

    def test_load_config_with_project(self, tmp_path, monkeypatch):
        profile_dir = tmp_path / ".agentrunner" / "profiles"
        profile_dir.mkdir(parents=True)
        profile_path = profile_dir / "default.json"
        profile_path.write_text(json.dumps({"max_rounds": 60}))

        project_dir = tmp_path / "project" / ".agentrunner"
        project_dir.mkdir(parents=True)
        project_path = project_dir / "config.json"
        project_path.write_text(json.dumps({"max_rounds": 80, "tool_timeout_s": 45}))

        Path.home = lambda: tmp_path

        for key in ["AGENTRUNNER_MAX_ROUNDS", "AGENTRUNNER_TOOL_TIMEOUT"]:
            monkeypatch.delenv(key, raising=False)

        config = load_config(project_root=tmp_path / "project")
        assert config.max_rounds == 80  # Project overrides profile
        assert config.tool_timeout_s == 45

    def test_load_config_full_precedence(self, tmp_path, monkeypatch):
        profile_dir = tmp_path / ".agentrunner" / "profiles"
        profile_dir.mkdir(parents=True)
        profile_path = profile_dir / "default.json"
        profile_path.write_text(json.dumps({"max_rounds": 100, "tool_timeout_s": 30}))

        project_dir = tmp_path / "project" / ".agentrunner"
        project_dir.mkdir(parents=True)
        project_path = project_dir / "config.json"
        project_path.write_text(json.dumps({"max_rounds": 150, "response_buffer_tokens": 2000}))

        Path.home = lambda: tmp_path
        monkeypatch.setenv("AGENTRUNNER_MAX_ROUNDS", "200")

        config = load_config(project_root=tmp_path / "project")
        assert config.max_rounds == 200  # Env has highest precedence
        assert config.response_buffer_tokens == 2000  # From project
        assert config.tool_timeout_s == 30  # From profile
