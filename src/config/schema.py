from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.logging import get_logger

logger = get_logger(__name__)

from src.logging import get_logger



logger = get_logger(__name__)

class LLMConfig(BaseModel):
    model: str
    provider: str = "openai"
    max_retries: int = Field(default=3, ge=0)
    timeout: float = Field(default=60.0, gt=0)
    base_url: str | None = None
    model_config = ConfigDict(extra="allow")


class PathsConfig(BaseModel):
    user_data_dir: str
    knowledge_bases_dir: str
    user_log_dir: str
    model_config = ConfigDict(extra="allow")


class AgentConfig(BaseModel):
    """Configuration for a single agent."""

    enabled: bool = True
    model: str | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_retries: int = Field(default=3, ge=0)
    model_config = ConfigDict(extra="ignore")


class SystemConfig(BaseModel):
    language: str = "en"
    output_base_dir: str = "data/output"
    save_intermediate_results: bool = True
    auto_solve: bool = True
    model_config = ConfigDict(extra="allow")


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    save_to_file: bool = True
    verbose: bool = False
    model_config = ConfigDict(extra="ignore")


class MonitoringConfig(BaseModel):
    """Monitoring and telemetry settings."""

    enabled: bool = True
    track_token_usage: bool = True
    track_time: bool = True
    model_config = ConfigDict(extra="ignore")


class AppConfig(BaseModel):
    llm: LLMConfig
    paths: PathsConfig
    system: SystemConfig = SystemConfig()
    agents: dict[str, AgentConfig] = Field(default_factory=dict)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    model_config = ConfigDict(extra="allow")

    @field_validator("llm", mode="before")
    @classmethod
    def ensure_llm(cls, v: Any) -> dict[str, Any]:
        if not isinstance(v, dict):
            raise ValueError("llm section must be a mapping")
        if "model" not in v:
            # Fallback for now if not provided, but ideally required
            v["model"] = "gpt-4o"
        return v


CURRENT_SCHEMA_VERSION = 1


def migrate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    No-op migration for now; placeholder for future versioned changes.
    """
    return cfg
