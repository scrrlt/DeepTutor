from pathlib import Path

import pytest
import yaml

from src.utils.config_manager import ConfigManager


from collections.abc import Generator

@pytest.fixture(autouse=True)
def reset_config_manager_singleton() -> Generator[None, None, None]:
    """Ensure ConfigManager singleton is reset for each test."""
    ConfigManager._instance = None  # type: ignore[attr-defined]
    yield
    ConfigManager._instance = None  # type: ignore[attr-defined]


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_atomic_save_and_deep_merge(tmp_path: Path):
    project = tmp_path
    cfg_path = project / "config" / "main.yaml"
    base_cfg = {
        "llm": {"model": "Pro/Flash", "provider": "openai"},
        "paths": {
            "user_data_dir": "./data/user",
            "knowledge_bases_dir": "./data/knowledge_bases",
            "user_log_dir": "./data/user/logs",
        },
    }
    write_yaml(cfg_path, base_cfg)

    cm = ConfigManager(project_root=project)

    loaded = cm.load_config(force_reload=True)
    assert loaded["llm"]["model"] == "Pro/Flash"

    # Deep merge update
    assert cm.save_config({"llm": {"model": "Other"}, "features": {"enable_solve": True}})

    # Backup exists
    assert (project / "config" / "main.yaml.bak").exists()

    updated = cm.load_config(force_reload=True)
    assert updated["llm"]["model"] == "Other"
    assert updated["llm"]["provider"] == "openai"
    assert updated["features"]["enable_solve"] is True


def test_env_layering(tmp_path: Path):
    project = tmp_path
    (project / ".env").write_text("LLM_MODEL=Base\n", encoding="utf-8")
    (project / ".env.local").write_text("LLM_MODEL=Override\n", encoding="utf-8")

    # Minimal valid config for schema
    cfg_path = project / "config" / "main.yaml"
    base_cfg = {
        "llm": {"model": "Pro/Flash", "provider": "openai"},
        "paths": {
            "user_data_dir": "./data/user",
            "knowledge_bases_dir": "./data/knowledge_bases",
            "user_log_dir": "./data/user/logs",
        },
    }
    write_yaml(cfg_path, base_cfg)

    cm = ConfigManager(project_root=project)
    env = cm.get_env_info()
    assert env["model"] == "Override"


def test_load_missing_config_returns_empty(tmp_path: Path):
    project = tmp_path

    cm = ConfigManager(project_root=project)
    assert cm.load_config(force_reload=True) == {}


def test_validate_required_env_reports_missing(tmp_path: Path, monkeypatch):
    project = tmp_path

    (project / ".env").write_text("LLM_MODEL=Value\n", encoding="utf-8")
    cm = ConfigManager(project_root=project)

    # Ensure environment variable is absent to assert missing detection
    monkeypatch.delenv("MISSING_KEY", raising=False)

    missing = cm.validate_required_env(["MISSING_KEY", "LLM_MODEL"])
    assert "missing" in missing
    assert missing["missing"] == ["MISSING_KEY"]
