"""Integration tests for KnowMe CLI.

Tests use real SQLite databases and Click's CliRunner — no mocking.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from click.testing import CliRunner

from knowme import VERSION, cli, get_config


class TestGetConfig:
    """Config loading and discovery."""

    def test_explicit_path(self, config_path: str) -> None:
        cfg = get_config(config_path)
        assert cfg["llm"]["generative_model"] == "test-model"

    def test_cwd_fallback(self, tmp_project: Path, monkeypatch: object) -> None:
        monkeypatch.chdir(tmp_project)  # type: ignore[attr-defined]
        cfg = get_config()
        assert cfg["database"]["path"].endswith("test.db")

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        import pytest

        with pytest.raises(FileNotFoundError):
            get_config(str(tmp_path / "nonexistent.yaml"))


class TestInit:
    """Database initialization."""

    def test_init_creates_database(self, tmp_project: Path) -> None:
        new_db = tmp_project / "new.db"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--config", str(tmp_project / "config.yaml"), "init", str(new_db)],
        )
        assert result.exit_code == 0
        assert new_db.exists()
        assert "Created database" in result.output

    def test_init_refuses_existing(self, tmp_project: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--config", str(tmp_project / "config.yaml"), "init"],
        )
        assert result.exit_code == 0
        assert "already exists" in result.output


class TestLoad:
    """File loading and queue management."""

    def test_load_file_creates_queue_entry(self, tmp_project: Path, db_path: str) -> None:
        runner = CliRunner()
        sample = tmp_project / "sample.txt"
        result = runner.invoke(
            cli,
            [
                "--config",
                str(tmp_project / "config.yaml"),
                "load",
                str(sample),
                "--name",
                "test-doc",
                "--author",
                "Matt",
                "--lifestage",
                "CHILDHOOD",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Loaded 1" in result.output

        # Verify queue entry in DB
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) as cnt FROM analysis_queue")
        count = cur.fetchone()["cnt"]
        conn.close()
        assert count == 1

    def test_load_list_queue(self, tmp_project: Path) -> None:
        runner = CliRunner()
        config = str(tmp_project / "config.yaml")

        # Load a file first
        runner.invoke(
            cli,
            [
                "--config",
                config,
                "load",
                str(tmp_project / "sample.txt"),
                "--name",
                "test",
                "--author",
                "Matt",
                "--lifestage",
                "CHILDHOOD",
            ],
        )

        # List queue
        result = runner.invoke(
            cli,
            ["--config", config, "load", "--list"],
        )
        assert result.exit_code == 0
        assert "Analysis Queue" in result.output


class TestQueue:
    """Queue listing."""

    def test_queue_list_empty(self, tmp_project: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--config", str(tmp_project / "config.yaml"), "queue", "--list"],
        )
        assert result.exit_code == 0
        assert "empty" in result.output.lower()


class TestRun:
    """Run command — session-based pipeline."""

    def test_run_dryrun(self, tmp_project: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--config", str(tmp_project / "config.yaml"), "run", "--dryrun"],
        )
        assert result.exit_code == 0
        assert "Would create session" in result.output
        assert "test-model" in result.output

    def test_run_dryrun_model_override(self, tmp_project: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--config",
                str(tmp_project / "config.yaml"),
                "run",
                "--dryrun",
                "--model",
                "gpt-5.4",
            ],
        )
        assert result.exit_code == 0
        assert "gpt-5.4" in result.output

    def test_run_list_empty(self, tmp_project: Path, monkeypatch: object) -> None:
        monkeypatch.chdir(tmp_project)  # type: ignore[attr-defined]
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--config", str(tmp_project / "config.yaml"), "run", "--list"],
        )
        assert result.exit_code == 0
        assert "No results directory" in result.output


class TestVersion:
    """Version output."""

    def test_version_matches_constant(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert VERSION in result.output
