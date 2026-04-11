"""Shared test fixtures for KnowMe integration tests.

All tests use real SQLite databases — no mocking.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import yaml


@pytest.fixture()
def tmp_project(tmp_path: Path) -> Path:
    """Create a minimal KnowMe project directory with schema, config, and roles."""
    # Create database from schema
    schema_path = Path(__file__).parent.parent / "schema.sql"
    schema_sql = schema_path.read_text(encoding="utf-8")

    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(schema_sql)
    conn.close()

    # Create config
    config = {
        "database": {
            "path": str(db_path),
            "schema_path": str(schema_path),
        },
        "llm": {
            "generative_model": "test-model",
            "embedding_model": "test-embedding",
        },
        "roles": {
            "path": str(Path(__file__).parent.parent / "roles"),
            "schema_file": "Prompt-Schema.md",
            "Observers": [
                {"name": "Psychologist", "file": "Role-Psychologist.md"},
            ],
            "Helpers": [],
        },
        "content": {
            "default_author": "TestAuthor",
        },
        "logging": {
            "level": "WARNING",
            "file": str(tmp_path / "test.log"),
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config, default_flow_style=False), encoding="utf-8")

    # Create a sample text file for loading
    sample = tmp_path / "sample.txt"
    sample.write_text("This is a test document about Matt's childhood.", encoding="utf-8")

    return tmp_path


@pytest.fixture()
def config_path(tmp_project: Path) -> str:
    """Return path to the test config file."""
    return str(tmp_project / "config.yaml")


@pytest.fixture()
def db_path(tmp_project: Path) -> str:
    """Return path to the test database."""
    return str(tmp_project / "test.db")
