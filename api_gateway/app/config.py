"""Configuration management for API Gateway."""

import os
from pathlib import Path
from typing import Optional


def _find_env_file() -> Path:
    """Find .env file in project root."""
    # Start from this file's location and go up to find project root
    current = Path(__file__).resolve()
    # api_gateway/app/config.py -> go up 1 levels to project root
    project_root = current.parent.parent
    return project_root / ".env"


def get_env_var(key: str, required: bool = False) -> Optional[str]:
    """Get environment variable, with fallback to .env file.

    Args:
        key: Environment variable name
        required: If True, raise ValueError if not found

    Returns:
        Environment variable value or None if not found and not required

    Raises:
        ValueError: If required=True and variable not found
    """
    value = os.getenv(key)
    if value:
        return value

    env_path = _find_env_file()
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == key:
                parsed_value = v.strip().strip('"').strip("'")
                if parsed_value:
                    return parsed_value

    if required:
        raise ValueError(
            f"Required environment variable '{key}' not found in environment or .env file"
        )
    return None


class Config:
    """NYC Taxi Duration API Gateway configuration."""

    # API Gateway
    API_PORT: int = int(get_env_var("API_PORT", required=True))
    API_HOST: str = get_env_var("API_HOST", required=True)

    MLFLOW_TRACKING_URI: str = get_env_var("MLFLOW_TRACKING_URI", required=True)
    MLFLOW_EXPERIMENT_NAME: str = get_env_var("MLFLOW_EXPERIMENT_NAME", required=True)
    MLFLOW_MODEL_REFRESH_INTERVAL: str = get_env_var("MLFLOW_MODEL_REFRESH_INTERVAL", required=True)


config = Config()
