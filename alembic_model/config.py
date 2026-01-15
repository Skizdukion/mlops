"""Configuration management for API Gateway and DB."""

import os
from pathlib import Path
from typing import Optional


def _find_env_file() -> Path:
    """Find .env file in project root."""
    # Start from this file's location and go up to find project root
    current = Path(__file__).resolve()
    # alembic/config.py -> go up 1 level to project root (week_1)
    # Original was api_gateway/app/config.py -> 2 levels.
    project_root = current.parent.parent
    return project_root / ".env"


def get_env_var(key: str, required: bool = False) -> Optional[str]:
    """Get environment variable, with fallback to .env file."""
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
    """NYC Taxi Duration Configuration."""

    POSTGRES_USER: str = get_env_var("POSTGRES_USER", required=True)
    POSTGRES_PASSWORD: str = get_env_var("POSTGRES_PASSWORD", required=True)
    POSTGRES_DB: str = get_env_var("POSTGRES_DB", required=True)
    POSTGRES_HOST: str = get_env_var("POSTGRES_HOST", required=True)
    POSTGRES_PORT: str = get_env_var("POSTGRES_PORT", required=True)

    @property
    def DATABASE_URL(self) -> str:
        """Get database URL."""
        return f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"


config = Config()
