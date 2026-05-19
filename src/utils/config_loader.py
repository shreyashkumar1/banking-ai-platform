"""YAML Configuration Loader."""

import yaml
from pathlib import Path
from typing import Any


def load_config(env: str = "dev") -> dict[str, Any]:
    """Load environment-specific configuration."""
    config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get(env, config.get("default", {}))


def load_schemas() -> dict[str, Any]:
    """Load BigQuery schema definitions."""
    schema_path = Path(__file__).parent.parent.parent / "config" / "schemas.yaml"
    with open(schema_path) as f:
        return yaml.safe_load(f)
