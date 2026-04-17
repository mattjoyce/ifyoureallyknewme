#!/usr/bin/env python3
"""KnowMe -- Personal knowledge management and analysis system.

Single-file CLI for ingesting personal content, running LLM-based expert
analysis, clustering observations into consensus records, and generating
biographical profiles.

Data flow:
  Files -> Load (sources + queue) -> Analyze (sessions + LLM observers)
  -> Knowledge Records -> Merge / Consensus -> Profile
"""

from __future__ import annotations

import base64
import contextlib
import glob as glob_module
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import time
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import httpx
import llm as llm_lib
import loaden
import numpy as np
import yaml
from openai import OpenAI
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table
from scipy.cluster.hierarchy import fcluster, linkage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VERSION = "0.2.0"

# Default similarity thresholds
DEFAULT_MERGE_THRESHOLD = 0.85
DEFAULT_CONSENSUS_THRESHOLD = 0.92

# Session output filenames
SESSION_DB_NAME = "session.db"
SESSION_CONFIG_NAME = "config.yaml"
SESSION_PROFILE_NAME = "profile.md"

# Sort order sentinels
UNKNOWN_LIFESTAGE_ORDER = 999
UNKNOWN_CONFIDENCE_ORDER = 5

# File encodings to try when reading source content
ENCODING_FALLBACKS: list[str] = [
    "utf-8",
    "latin-1",
    "windows-1252",
    "cp1252",
    "ISO-8859-1",
]

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

console = Console()
logger = logging.getLogger(__name__)

LIFE_STAGES: list[str] = [
    "CHILDHOOD",
    "ADOLESCENCE",
    "EARLY_ADULTHOOD",
    "EARLY_CAREER",
    "MID_CAREER",
    "LATE_CAREER",
]

DOMAINS: list[str] = [
    "Personal History",
    "Professional Evolution",
    "Psychological and Behavioral Evolution",
    "Relationships and Networks",
    "Community and Ideological Engagement",
    "Daily Routines and Health",
    "Values, Beliefs, and Goals",
    "Active Projects and Learning",
]

# ---------------------------------------------------------------------------
# Configuration  (loaden)
# ---------------------------------------------------------------------------

_cached_config_path: str | None = None


def get_config(config_path: str | None = None) -> dict[str, Any]:
    """Load and return configuration via loaden.

    Searches for a config file in this order:
    1. Explicit *config_path* argument
    2. Previously cached path
    3. ``KNOWME_CONFIG_PATH`` environment variable
    """
    global _cached_config_path
    config_path = config_path or _cached_config_path

    if config_path is None:
        config_path = os.getenv("KNOWME_CONFIG_PATH")

    if config_path is None:
        cwd_config = Path("config.yaml")
        if cwd_config.exists():
            config_path = str(cwd_config)

    if config_path is None:
        raise FileNotFoundError(
            "No config path provided. Pass --config or set KNOWME_CONFIG_PATH."
        )

    resolved = Path(config_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved}")

    _cached_config_path = str(resolved)
    cfg = loaden.load_config(str(resolved))
    # Ensure runtime-only keys exist
    cfg.setdefault("dryrun", False)
    return cfg


def configure_logging(level: str | None = None, log_file: str | None = None) -> None:
    """Configure root logger with Rich console + optional file handler."""
    level = level or "WARNING"
    log_file = log_file or "knowme.log"

    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    root.setLevel(getattr(logging, level.upper()))

    console_handler = RichHandler(rich_tracebacks=True, markup=True)
    root.addHandler(console_handler)

    if log_file != "-":
        log_dir = Path(log_file).parent
        if log_dir != Path(".") and not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root.addHandler(file_handler)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def get_connection(db_path: str) -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    """Return a (connection, cursor) pair with Row factory enabled."""
    if not db_path:
        raise ValueError("No database path provided.")
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn, conn.cursor()


@contextmanager
def db_cursor(
    db_path: str, *, commit: bool = False
) -> Generator[tuple[sqlite3.Connection, sqlite3.Cursor], None, None]:
    """Context manager for DB access.  Commits if *commit* is True, always closes."""
    conn, cursor = get_connection(db_path)
    try:
        yield conn, cursor
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def create_database(config: dict[str, Any]) -> bool:
    """Create a new database from schema.sql."""
    try:
        schema_path = Path(loaden.get(config, "database.schema_path"))
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found at {schema_path}")

        db_path = loaden.get(config, "database.path")
        if not db_path:
            raise ValueError("No database path in configuration.")

        conn = sqlite3.connect(db_path)
        conn.executescript(schema_path.read_text(encoding="utf-8"))
        conn.commit()
        conn.close()
        logger.info("Created database at %s", db_path)
        return True
    except (sqlite3.Error, FileNotFoundError) as exc:
        logger.error("Error creating database: %s", exc)
        return False


def store_knowledge_record(
    config: dict[str, Any],
    note_id: str,
    record_type: str,
    author: str,
    content: dict[str, Any],
    ts: str,
    embedding_bytes: bytes,
    session_id: str | None = None,
    qa_id: str | None = None,
    source_id: str | None = None,
    keywords: list[str] | None = None,
) -> None:
    """Insert a single knowledge record into the database."""
    db_path = loaden.get(config, "database.path")
    conn, cursor = get_connection(db_path)
    cursor.execute(
        """
        INSERT INTO knowledge_records
        (id, type, author, content, created_at,
         embedding, session_id, qa_id, source_id, keywords)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            note_id,
            record_type,
            author,
            json.dumps(content),
            ts,
            embedding_bytes,
            session_id,
            qa_id,
            source_id,
            ",".join(keywords) if keywords else None,
        ),
    )
    conn.commit()
    conn.close()
    logger.info("Stored knowledge record: %s", note_id)


@dataclass
class KnowledgeRecord:
    """Lightweight data-transfer object for knowledge records."""

    id: str
    type: str
    domain: str | None = None
    author: str | None = None
    content: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    version: str | None = None
    consensus_id: str | None = None
    session_id: str | None = None
    qa_id: str | None = None
    source_id: str | None = None
    keywords: str | None = None
    embedding: str | None = None  # base64


def get_filtered_knowledge_records(
    config: dict[str, Any],
    session_id: str | None = None,
    domain: list[str] | None = None,
    confidence: list[str] | None = None,
    lifestage: list[str] | None = None,
    observation_text: list[str] | None = None,
    record_type: str | None = None,
    author: str | None = None,
    consensus_id: str | None = None,
    qa_id: str | None = None,
    source_id: str | None = None,
    include_embedding: bool = False,
) -> list[KnowledgeRecord]:
    """Query knowledge_records with optional filters."""
    db_path = loaden.get(config, "database.path")
    conn, cursor = get_connection(db_path)

    cols = (
        "id, type, domain, author, content, created_at, version, "
        "consensus_id, session_id, qa_id, source_id, keywords"
    )
    if include_embedding:
        cols += ", embedding"

    query = f"SELECT {cols} FROM knowledge_records WHERE 1=1"
    params: list[Any] = []

    for col, val in [
        ("session_id", session_id),
        ("type", record_type),
        ("author", author),
        ("consensus_id", consensus_id),
        ("qa_id", qa_id),
        ("source_id", source_id),
    ]:
        if val is not None:
            query += f" AND {col} = ?"
            params.append(val)

    cursor.execute(query, params)
    records: list[KnowledgeRecord] = []

    for row in cursor.fetchall():
        try:
            content = json.loads(row[4])
        except json.JSONDecodeError:
            logger.warning("Bad JSON in record %s", row[0])
            continue

        emb_b64: str | None = None
        if include_embedding and row[12] is not None:
            emb_b64 = base64.b64encode(row[12]).decode("utf-8")

        rec = KnowledgeRecord(
            id=row[0],
            type=row[1],
            domain=row[2],
            author=row[3],
            content=content,
            created_at=row[5],
            version=row[6],
            consensus_id=row[7],
            session_id=row[8],
            qa_id=row[9],
            source_id=row[10],
            keywords=row[11],
            embedding=emb_b64,
        )

        # Client-side filters on JSON content
        if domain and not any(
            d.lower() in rec.content.get("domain", "").lower() for d in domain
        ):
            continue
        if confidence and not any(
            c.lower() in rec.content.get("confidence", "").lower() for c in confidence
        ):
            continue
        if lifestage and not any(
            ls.lower() in rec.content.get("life_stage", "").lower() for ls in lifestage
        ):
            continue
        if observation_text and not any(
            ot.lower() in rec.content.get("observation", "").lower() for ot in observation_text
        ):
            continue

        records.append(rec)

    conn.close()
    return records


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


_openai_client: OpenAI | None = None


def _get_openai_client() -> OpenAI:
    """Lazy singleton for OpenAI client."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


def get_embedding(config: dict[str, Any], text: str, model: str | None = None) -> np.ndarray:
    """Get an embedding vector from OpenAI."""
    model = model or loaden.get(config, "llm.embedding_model")
    client = _get_openai_client()
    response = client.embeddings.create(model=model, input=text)
    return np.array(response.data[0].embedding, dtype=np.float32)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    v1, v2 = v1.flatten(), v2.flatten()
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def cosine_similarity_base64(a: str, b: str, dtype: type = np.float32) -> float:
    """Cosine similarity between two base64-encoded vectors."""
    try:
        return cosine_similarity(
            np.frombuffer(base64.b64decode(a), dtype=dtype),
            np.frombuffer(base64.b64decode(b), dtype=dtype),
        )
    except (ValueError, TypeError):
        return 0.0


def compute_pairwise_similarities(
    embeddings: list[np.ndarray],
) -> np.ndarray:
    """Pairwise cosine similarity matrix."""
    if not embeddings:
        return np.array([])
    stacked = np.vstack(embeddings)
    norms = np.linalg.norm(stacked, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = stacked / norms
    return np.dot(normed, normed.T)


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def call_llm(
    input_content: str | dict[str, Any] | list[Any],
    prompt: str,
    model_name: str,
    json_output: bool = True,
    options: dict[str, Any] | None = None,
) -> Any:
    """Call an LLM via Simon Willison's ``llm`` library."""
    if isinstance(input_content, (dict, list)):
        input_str = json.dumps(input_content)
    else:
        input_str = str(input_content)

    model = llm_lib.get_model(model_name or "gpt-4o-mini")
    prompt_kwargs: dict[str, Any] = dict(options or {})

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.prompt(prompt=f"{prompt}\n\n{input_str}", **prompt_kwargs)
            content_str: str = response.text()
            break
        except (ConnectionError, OSError, httpx.NetworkError) as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1,
                    max_retries,
                    e,
                    wait,
                )
                time.sleep(wait)
            else:
                raise

    if json_output:
        try:
            return json.loads(content_str)
        except json.JSONDecodeError:
            return extract_json(content_str)
    return content_str


def extract_json(text: str) -> Any:
    """Extract JSON from potentially markdown-wrapped or messy text.

    Tries ``json.loads`` first, then falls back to regex extraction.
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    for pattern in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
        for match in re.finditer(pattern, text):
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue
    raise ValueError("No valid JSON found in text")


def llm_process(
    config: dict[str, Any],
    input_content: str | dict[str, Any] | list[Any],
    prompt: str,
    model: str | None = None,
    expect_json: bool = True,
) -> Any:
    """High-level: call LLM, return parsed result."""
    model = model or loaden.get(config, "llm.generative_model")
    options = loaden.get(config, "llm.options", {}) or {}
    return call_llm(input_content, prompt, model, json_output=expect_json, options=options)


def get_role_file(config: dict[str, Any], role_name: str, role_type: str) -> Path:
    """Resolve a role markdown file path."""
    if role_type not in ("Helper", "Observer"):
        raise ValueError(f"Invalid role type: {role_type}")

    key = "roles.Observers" if role_type == "Observer" else "roles.Helpers"
    roles: list[dict[str, str]] = loaden.get(config, key, [])
    target = next((r for r in roles if r["name"] == role_name), None)
    if target is None:
        raise ValueError(f"Role '{role_name}' not found in {role_type}s")

    roles_dir = loaden.get(config, "roles.path", "./roles")
    role_path = Path(roles_dir) / target["file"]
    resolved = role_path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Role file not found: {resolved}")
    return resolved


def get_role_prompt(config: dict[str, Any], role_name: str, role_type: str) -> str:
    """Load a role prompt, injecting the schema for Observers."""
    role_file = get_role_file(config, role_name, role_type)
    prompt = role_file.read_text(encoding="utf-8")

    if role_type == "Observer":
        roles_dir = loaden.get(config, "roles.path", "./roles")
        schema_file = loaden.get(config, "roles.schema_file", "")
        if schema_file:
            schema_path = (Path(roles_dir) / schema_file).resolve()
            if schema_path.exists():
                schema_text = schema_path.read_text(encoding="utf-8")
                prompt = prompt.replace("{{schema}}", schema_text)
    return prompt


def extract_keywords(config: dict[str, Any], content: str) -> list[str]:
    """Extract keywords from text using the KeywordExtractor role."""
    if not isinstance(content, str) or not content:
        return []

    model = loaden.get(config, "llm.generative_model")
    prompt = get_role_prompt(config, "KeywordExtractor", "Helper")

    try:
        result = llm_process(
            config,
            input_content=content,
            prompt=prompt,
            model=model,
            expect_json=True,
        )
        kw_raw: Any = None
        if isinstance(result, dict):
            kw_raw = result.get("keywords", "")
        else:
            parsed = extract_json(str(result))
            kw_raw = parsed.get("keywords", "") if isinstance(parsed, dict) else ""

        if isinstance(kw_raw, str):
            return [k.strip() for k in kw_raw.split(",") if k.strip()]
        return []
    except Exception:
        logger.exception("Keyword extraction failed")
        return []


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def utcnow_iso() -> str:
    """ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


def generate_id(prefix: str, *args: object) -> str:
    """Deterministic hash-based ID with prefix."""
    raw = "-".join(str(a) for a in args) + "-" + utcnow_iso()
    return f"{prefix}_{hashlib.sha256(raw.encode()).hexdigest()[:12]}"


def resolve_file_patterns(
    patterns: tuple[str, ...] | list[str], recursive: bool = True
) -> list[str]:
    """Expand glob patterns into sorted, unique, absolute paths."""
    matched: set[str] = set()
    for pat in patterns:
        p = Path(pat)
        abs_pat = str(p if p.is_absolute() else (Path.cwd() / p).resolve())
        matched.update(glob_module.glob(abs_pat, recursive=recursive))
    return sorted(matched)


def read_file_fallback(file_path: str | Path) -> str:
    """Read text file trying multiple encodings."""
    for enc in ENCODING_FALLBACKS:
        try:
            return Path(file_path).read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return Path(file_path).read_bytes().decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Ingest  (QA transcript processing)
# ---------------------------------------------------------------------------


def extract_qa_pairs(qa_content: str) -> list[tuple[str, str]]:
    """Extract (question, answer) tuples from ``Q. / A.`` formatted text."""
    parts = re.split(r"(?:^|\n)Q\.\s*", qa_content, flags=re.MULTILINE)
    pairs: list[tuple[str, str]] = []
    for part in parts[1:]:
        if "A." in part:
            q, a = part.split("A.", 1)
            pairs.append((q.strip(), a.strip()))
        else:
            pairs.append((part.strip(), ""))
    return pairs


def ingest_qa_content(
    db_path: str,
    content: str,
    name: str,
    author: str,
    lifestage: str,
    filename: str,
) -> str | None:
    """Process QA content into a session + qa_pairs rows."""
    qa_pairs = extract_qa_pairs(content)
    if not qa_pairs:
        return None

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    ts = utcnow_iso()
    session_id = generate_id("session", name)

    try:
        cursor.execute(
            """INSERT INTO sessions
               (id, title, description, file_path, created_at, metadata,
                author, lifestage, content_type)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                name,
                f"Processed from {filename}",
                filename,
                ts,
                json.dumps({"source": "qa", "qa_count": len(qa_pairs)}),
                author,
                lifestage,
                "qa",
            ),
        )
        for idx, (question, answer) in enumerate(qa_pairs):
            qa_id = generate_id("qa", session_id, idx)
            cursor.execute(
                """INSERT INTO qa_pairs
                   (id, session_id, question, answer, sequence, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (qa_id, session_id, question, answer, idx + 1, ts),
            )
        conn.commit()
        logger.info("Ingested %d QA pairs into session %s", len(qa_pairs), session_id)
        return session_id
    except Exception:
        conn.rollback()
        logger.exception("Error ingesting QA content")
        return None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class Loader:
    """Load files into the sources table and enqueue for analysis."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.db_path: str = loaden.get(config, "database.path")

    # -- single file --------------------------------------------------------

    def load_file(
        self,
        file_path: str,
        name: str,
        author: str,
        description: str,
        lifestage: str,
        priority: int = 0,
    ) -> str | None:
        """Load one file -> source record + queue entry.  Returns source_id."""
        existing = self._check_duplicate_source(file_path, name, description)
        if existing:
            self._create_queue_entry(existing, author, lifestage, priority)
            return existing

        ts = utcnow_iso()
        source_id = generate_id("source", name, ts)
        conn, cursor = get_connection(self.db_path)
        try:
            cursor.execute(
                """INSERT INTO sources (id, created_at, description, content_path, title)
                   VALUES (?, ?, ?, ?, ?)""",
                (source_id, ts, description, str(Path(file_path).resolve()), name),
            )
            conn.commit()
            self._create_queue_entry(source_id, author, lifestage, priority)
            return source_id
        except Exception:
            conn.rollback()
            logger.exception("Error loading %s", file_path)
            return None
        finally:
            conn.close()

    # -- batch --------------------------------------------------------------

    def batch_load(
        self,
        file_patterns: tuple[str, ...] | list[str],
        name_template: str | None = None,
        author: str | None = None,
        description: str | None = None,
        lifestage: str = "AUTO",
        priority: int = 0,
    ) -> dict[str, Any]:
        matched = resolve_file_patterns(file_patterns)
        loaded: list[str] = []
        source_ids: list[str] = []
        for idx, fp in enumerate(matched, 1):
            item_name = (
                name_template.format(index=idx)
                if name_template and "{index}" in name_template
                else name_template or Path(fp).stem
            )
            sid = self.load_file(
                fp,
                item_name,
                author or "",
                description or f"Content from {Path(fp).name}",
                lifestage,
                priority,
            )
            if sid:
                loaded.append(fp)
                source_ids.append(sid)
        return {
            "matched_files": matched,
            "loaded_files": loaded,
            "source_ids": source_ids,
        }

    # -- queue management ---------------------------------------------------

    def get_queue_items(
        self, status: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        conn, cursor = get_connection(self.db_path)
        try:
            query = """
                SELECT aq.id, aq.author, aq.lifestage, aq.type,
                       aq.created_at, aq.status, aq.priority,
                       s.id AS source_id, s.description, s.content_path, s.title
                FROM analysis_queue aq
                JOIN sources s ON aq.source_id = s.id
            """
            params: list[Any] = []
            if status:
                query += " WHERE aq.status = ?"
                params.append(status)
            query += " ORDER BY aq.priority DESC, aq.created_at ASC"
            if limit > 0:
                query += f" LIMIT {limit}"
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_source_and_queue_item(
        self,
        source_id: str | None = None,
        queue_id: str | None = None,
    ) -> dict[str, Any] | None:
        if not (source_id or queue_id) or (source_id and queue_id):
            raise ValueError("Provide exactly one of source_id / queue_id.")
        conn, cursor = get_connection(self.db_path)
        col = "aq.source_id" if source_id else "aq.id"
        param = source_id or queue_id
        cursor.execute(
            f"""SELECT aq.id, aq.author, aq.lifestage, aq.type,
                       aq.created_at, aq.status, aq.priority,
                       s.id AS source_id, s.description, s.content_path, s.title
                FROM analysis_queue aq
                JOIN sources s ON aq.source_id = s.id
                WHERE {col} = ?""",
            (param,),
        )
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    # -- private helpers ----------------------------------------------------

    def _create_queue_entry(
        self,
        source_id: str,
        author: str,
        lifestage: str,
        priority: int,
        status: str = "pending",
    ) -> str | None:
        conn, cursor = get_connection(self.db_path)
        try:
            ts = utcnow_iso()
            queue_id = generate_id("queue", source_id, ts)
            cursor.execute(
                """INSERT INTO analysis_queue
                   (id, author, lifestage, type, priority, status, created_at, source_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (queue_id, author, lifestage, "document", priority, status, ts, source_id),
            )
            conn.commit()
            return queue_id
        except Exception:
            conn.rollback()
            logger.exception("Error creating queue entry")
            return None
        finally:
            conn.close()

    def _check_duplicate_source(
        self, file_path: str, title: str, description: str
    ) -> str | None:
        norm_path = str(Path(file_path).resolve())
        conn, cursor = get_connection(self.db_path)
        try:
            cursor.execute("SELECT id FROM sources WHERE content_path = ?", (norm_path,))
            row = cursor.fetchone()
            if not row:
                return None
            source_id: str = row[0]
            cursor.execute(
                "UPDATE sources SET title = ?, description = ? WHERE id = ?",
                (title, description, source_id),
            )
            cursor.execute(
                "UPDATE analysis_queue SET status = 'cancelled' "
                "WHERE source_id = ? AND status = 'pending'",
                (source_id,),
            )
            conn.commit()
            return source_id
        except Exception:
            conn.rollback()
            return None
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# AnalysisManager
# ---------------------------------------------------------------------------


def _find_cached_records(
    results_dir: Path,
    source_id: str,
    role_name: str,
    model_name: str,
    current_session_db: str,
) -> list[dict[str, Any]]:
    """Search previous session DBs for cached analysis records.

    Returns matching knowledge_records if a previous run with the same
    model already analyzed this source with this role.
    """
    safe_model = re.sub(r"[^\w\-.]", "_", model_name)
    for session_dir in sorted(results_dir.iterdir(), reverse=True):
        db_file = session_dir / SESSION_DB_NAME
        if not db_file.exists() or str(db_file) == current_session_db:
            continue
        # Only check sessions from the same model
        if safe_model not in session_dir.name:
            continue
        try:
            conn = sqlite3.connect(str(db_file))
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                """SELECT kr.id, kr.type, kr.author, kr.content, kr.created_at,
                          kr.version, kr.embedding, kr.session_id, kr.keywords
                   FROM knowledge_records kr
                   JOIN sessions s ON kr.session_id = s.id
                   WHERE s.source_id = ? AND kr.author = ?""",
                (source_id, role_name),
            )
            rows = [dict(r) for r in cur.fetchall()]
            conn.close()
            if rows:
                return rows
        except Exception:
            logger.debug("Cache read failed for %s", db_file)
            continue
    return []


class AnalysisManager:
    """Run LLM expert analyses on sessions."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.cache_enabled: bool = False
        self.results_dir: Path = Path("results")

    @property
    def _db(self) -> str:
        return loaden.get(self.config, "database.path")

    @property
    def _model(self) -> str:
        return loaden.get(self.config, "llm.generative_model")

    # -- session creation ---------------------------------------------------

    def create_session(self, item: dict[str, Any]) -> str | None:
        """Create a session record from a queue item.  Skips if exists."""
        conn, cursor = get_connection(self._db)
        cursor.execute(
            "SELECT id FROM sessions WHERE source_id = ?",
            (item["source_id"],),
        )
        if cursor.fetchone():
            conn.close()
            return None

        session_id = generate_id("session", item["source_id"], item["id"])
        cursor.execute(
            """INSERT INTO sessions
               (id, title, description, author, lifestage, file_path,
                content_type, source_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                item["title"],
                item.get("description", ""),
                item["author"],
                item["lifestage"],
                item["content_path"],
                "document",
                item["source_id"],
                utcnow_iso(),
            ),
        )
        conn.commit()
        conn.close()
        return session_id

    # -- analysis -----------------------------------------------------------

    def run_role_analysis(self, session_id: str, role_name: str) -> list[str] | None:
        """Run one expert role on a session, store observations."""
        input_content = self._create_session_context(session_id)
        prompt = get_role_prompt(self.config, role_name, "Observer")
        response = llm_process(
            self.config,
            input_content=input_content,
            prompt=prompt,
            model=self._model,
            expect_json=True,
        )

        observations = response if isinstance(response, list) else [response]
        conn, cursor = get_connection(self._db)
        note_ids: list[str] = []

        for obs in observations:
            text = obs.get("observation", "")
            if not text:
                continue

            emb = get_embedding(self.config, text)
            note_id = generate_id("note", session_id, role_name)
            keywords = extract_keywords(self.config, text)

            if not self.config.get("dryrun"):
                try:
                    cursor.execute(
                        """INSERT INTO knowledge_records
                           (id, type, author, content, created_at, version,
                            embedding, session_id, keywords)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            note_id,
                            "note",
                            role_name,
                            json.dumps(obs),
                            utcnow_iso(),
                            "1.0",
                            emb.tobytes(),
                            session_id,
                            ",".join(keywords) if keywords else None,
                        ),
                    )
                    conn.commit()
                    note_ids.append(note_id)
                except Exception:
                    logger.exception("Error storing record %s", note_id)

        conn.close()
        return note_ids or None

    def _try_cache(self, session_id: str, role: str) -> int:
        """Try to load cached records for this session+role. Returns count loaded."""
        conn, cursor = get_connection(self._db)
        cursor.execute("SELECT source_id FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return 0

        cached = _find_cached_records(
            self.results_dir,
            row["source_id"],
            role,
            self._model,
            self._db,
        )
        if not cached:
            return 0

        conn, cursor = get_connection(self._db)
        count = 0
        for rec in cached:
            new_id = generate_id("note", session_id, role)
            try:
                cursor.execute(
                    """INSERT INTO knowledge_records
                       (id, type, author, content, created_at, version,
                        embedding, session_id, keywords)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        new_id,
                        rec["type"],
                        rec["author"],
                        rec["content"],
                        rec["created_at"],
                        rec["version"],
                        rec["embedding"],
                        session_id,
                        rec["keywords"],
                    ),
                )
                count += 1
            except Exception:
                logger.exception("Error copying cached record")
        conn.commit()
        conn.close()
        return count

    def run_multiple_analyses(self, session_id: str, role_names: list[str]) -> dict[str, int]:
        """Run all observer roles on a session, skipping already-done."""
        conn, cursor = get_connection(self._db)
        cursor.execute("SELECT id FROM sessions WHERE id = ?", (session_id,))
        if not cursor.fetchone():
            conn.close()
            return {}
        conn.close()

        results: dict[str, int] = {}
        for role in role_names:
            conn, cursor = get_connection(self._db)
            cursor.execute(
                "SELECT COUNT(*) FROM knowledge_records WHERE session_id = ? AND author = ?",
                (session_id, role),
            )
            if cursor.fetchone()[0] > 0:
                conn.close()
                results[role] = 0
                continue
            conn.close()

            if self.cache_enabled:
                cached_count = self._try_cache(session_id, role)
                if cached_count > 0:
                    console.print(f"[blue]  Cached {role}: {cached_count} records[/blue]")
                    results[role] = cached_count
                    continue

            ids = self.run_role_analysis(session_id, role)
            results[role] = len(ids) if ids else 0
        return results

    def get_unanalyzed_sessions(self, observer_names: list[str]) -> list[dict[str, Any]]:
        """Return sessions missing one or more observer analyses."""
        conn, cursor = get_connection(self._db)
        cursor.execute("SELECT id, title, created_at FROM sessions ORDER BY created_at DESC")
        sessions = [dict(r) for r in cursor.fetchall()]
        incomplete: list[dict[str, Any]] = []

        for sess in sessions:
            missing: list[str] = []
            for name in observer_names:
                cursor.execute(
                    "SELECT COUNT(*) FROM knowledge_records "
                    "WHERE session_id = ? AND author = ?",
                    (sess["id"], name),
                )
                if cursor.fetchone()[0] == 0:
                    missing.append(name)
            if missing:
                sess["missing_observers"] = missing
                incomplete.append(sess)

        conn.close()
        return incomplete

    def update_queue_status(self, queue_id: str, status: str) -> bool:
        try:
            conn, cursor = get_connection(self._db)
            cursor.execute(
                "UPDATE analysis_queue SET status = ? WHERE id = ?",
                (status, queue_id),
            )
            conn.commit()
            conn.close()
            return True
        except Exception:
            logger.exception("Error updating queue %s", queue_id)
            return False

    # -- private ------------------------------------------------------------

    def _create_session_context(self, session_id: str) -> str:
        conn, cursor = get_connection(self._db)
        try:
            cursor.execute(
                """SELECT id, title, description, author, lifestage, file_path
                   FROM sessions WHERE id = ?""",
                (session_id,),
            )
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Session {session_id} not found")
            sess = dict(row)

            ctx = f"Title: {sess['title']}\n"
            if sess.get("description"):
                ctx += f"Description: {sess['description']}\n"
            ctx += f"Author: {sess['author']}\nLife Stage: {sess['lifestage']}\n---\n"

            fp = sess.get("file_path")
            if fp and Path(fp).exists():
                ctx += read_file_fallback(fp)
            else:
                raise FileNotFoundError(f"Content file not found: {fp}")
            return ctx
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# ConsensusManager
# ---------------------------------------------------------------------------


class ConsensusManager:
    """Cluster similar observations and generate consensus records."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @property
    def _db(self) -> str:
        return loaden.get(self.config, "database.path")

    def load_records(self, record_type: str = "note") -> list[dict[str, Any]]:
        """Load records with embeddings not yet in a consensus."""
        conn, cursor = get_connection(self._db)
        try:
            query = (
                "SELECT id, content, embedding, author, type, session_id "
                "FROM knowledge_records "
                "WHERE embedding IS NOT NULL AND consensus_id IS NULL"
            )
            params: list[Any] = []
            if record_type:
                query += " AND type = ?"
                params.append(record_type)
            cursor.execute(query, params)

            records: list[dict[str, Any]] = []
            for row in cursor.fetchall():
                if row[2]:
                    content = json.loads(row[1])
                    records.append(
                        {
                            "id": row[0],
                            "observation": content["observation"],
                            "domain": content.get("domain"),
                            "life_stage": content.get("life_stage"),
                            "confidence": content.get("confidence"),
                            "author": row[3],
                            "type": row[4],
                            "session_id": row[5],
                            "embedding": np.frombuffer(row[2], dtype=np.float32),
                        }
                    )
            return records
        finally:
            conn.close()

    def find_clusters(
        self, threshold: float = 0.85, record_type: str = "note"
    ) -> dict[str, Any]:
        records = self.load_records(record_type)
        if not records:
            return {}

        groups: dict[str, list[dict[str, Any]]] = {}
        for r in records:
            groups.setdefault(r["life_stage"], []).append(r)

        all_clusters: dict[str, Any] = {}
        for life_stage, stage_records in groups.items():
            if len(stage_records) < 2:
                continue
            embedding_list = [r["embedding"] for r in stage_records]
            embeddings = np.vstack(embedding_list)
            sims = compute_pairwise_similarities(embedding_list)
            link = linkage(embeddings, method="complete", metric="cosine")
            labels = fcluster(link, t=1 - threshold, criterion="distance")

            buckets: dict[int, dict[str, list[Any]]] = {}
            for idx, cid in enumerate(labels):
                bucket = buckets.setdefault(cid, {"records": [], "indices": []})
                rec_copy = {k: v for k, v in stage_records[idx].items() if k != "embedding"}
                bucket["records"].append(rec_copy)
                bucket["indices"].append(idx)

            for cid, data in buckets.items():
                if len(data["records"]) < 2:
                    continue
                idxs = data["indices"]
                csim = sims[np.ix_(idxs, idxs)]
                avg = (np.sum(csim) - len(idxs)) / (len(idxs) * (len(idxs) - 1))
                key = f"{life_stage}_cluster_{cid}"
                all_clusters[key] = {
                    "life_stage": life_stage,
                    "notes": data["records"],
                    "average_similarity": float(avg),
                    "unique_domains": list(
                        {r["domain"] for r in data["records"] if r["domain"]}
                    ),
                    "unique_authors": list({r["author"] for r in data["records"]}),
                }
        return all_clusters

    def find_similar_consensus(self, threshold: float = 0.92) -> dict[str, Any]:
        clusters = self.find_clusters(threshold, record_type="consensus")
        result: dict[str, Any] = {}
        for cid, cluster in clusters.items():
            if len(cluster["notes"]) > 1:
                result[cid] = {
                    "consensus_ids": [n["id"] for n in cluster["notes"]],
                    "observation_count": len(cluster["notes"]),
                    "average_similarity": cluster["average_similarity"],
                    "observations": [n["observation"] for n in cluster["notes"]],
                }
        return result

    def make_consensus(
        self, cluster: dict[str, Any], author: str = "ConsensusMaker"
    ) -> dict[str, Any] | list[Any]:
        if len(cluster.get("notes", [])) < 2:
            return {}
        model = loaden.get(self.config, "llm.generative_model")
        prompt = get_role_prompt(self.config, "ConsensusMaker", "Helper")
        return llm_process(self.config, cluster, prompt, model)

    def save_consensus(
        self,
        consensus: dict[str, Any],
        author: str,
        ts: str | None = None,
    ) -> str:
        ts = ts or utcnow_iso()
        emb = get_embedding(self.config, consensus["observation"])
        kw = extract_keywords(self.config, consensus["observation"])
        note_id = generate_id("consensus", author, ts)

        store_knowledge_record(
            self.config,
            note_id,
            "consensus",
            author,
            consensus,
            ts,
            emb.tobytes(),
            keywords=kw,
        )

        source_records = consensus.get("source_records", [])
        if source_records:
            conn, cursor = get_connection(self._db)
            ph = ",".join("?" * len(source_records))
            cursor.execute(
                f"UPDATE knowledge_records SET consensus_id = ? WHERE id IN ({ph})",
                [note_id, *source_records],
            )
            conn.commit()
            conn.close()
        return note_id

    def process_clusters(
        self,
        threshold: float = 0.85,
        kr_type: str = "note",
        author: str = "ConsensusMaker",
    ) -> dict[str, Any]:
        clusters = self.find_clusters(threshold, kr_type)
        if self.config.get("dryrun"):
            return {"clusters": clusters}

        ids: list[str] = []
        for _cid, cluster in clusters.items():
            consensus = self.make_consensus(cluster, author)
            if not consensus:
                continue
            ts = utcnow_iso()
            rid = self.save_consensus(consensus, author, ts)
            ids.append(rid)
            cluster["consensus_ids"] = rid

        return {
            "clusters": clusters,
            "consensus_count": len(ids),
            "cluster_count": len(clusters),
        }

    def reset_consensus(self) -> int:
        """Delete all consensus records and unlink knowledge records."""
        conn, cursor = get_connection(self._db)
        try:
            cursor.execute(
                "UPDATE knowledge_records SET consensus_id = NULL "
                "WHERE consensus_id IS NOT NULL"
            )
            cursor.execute(
                "DELETE FROM knowledge_records "
                "WHERE author = 'ConsensusMaker' AND type = 'consensus'"
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def reset_consensus_clusters(self, clusters: dict[str, Any]) -> int:
        """Reset specific consensus clusters."""
        conn, cursor = get_connection(self._db)
        count = 0
        try:
            for cluster in clusters.values():
                for cid in cluster.get("consensus_ids", []):
                    cursor.execute(
                        "UPDATE knowledge_records SET consensus_id = NULL "
                        "WHERE consensus_id = ?",
                        (cid,),
                    )
                    cursor.execute("DELETE FROM knowledge_records WHERE id = ?", (cid,))
                    count += 1
            conn.commit()
        finally:
            conn.close()
        return count


# ---------------------------------------------------------------------------
# ProfileGenerator
# ---------------------------------------------------------------------------


class ProfileGenerator:
    """Generate biographical profiles from knowledge records."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.db_path: str = loaden.get(config, "database.path")

    def dump_observations(self) -> str:
        conn, cursor = get_connection(self.db_path)
        cursor.execute(
            "SELECT content FROM knowledge_records "
            "WHERE consensus_id IS NULL ORDER BY created_at"
        )
        lines: list[str] = []
        for row in cursor.fetchall():
            with contextlib.suppress(json.JSONDecodeError):
                lines.append(json.loads(row[0]).get("observation", ""))
        conn.close()
        return "\n".join(lines)

    def generate_profile(self, format_type: str = "md", mode: str = "short") -> str:
        consensus = self._get_records("consensus")
        notes = self._get_records("note")
        all_records = consensus + notes

        if format_type == "raw":
            return self._format_raw(all_records)

        organized = self._organize(all_records)
        content = self._synthesize(organized, mode)

        if format_type == "json":
            return json.dumps(content, indent=2)
        return self._format_markdown(content)

    # -- private ------------------------------------------------------------

    def _get_records(self, record_type: str) -> list[dict[str, Any]]:
        conn, cursor = get_connection(self.db_path)
        try:
            if record_type == "consensus":
                cursor.execute(
                    "SELECT id, content, created_at FROM knowledge_records "
                    "WHERE type = 'consensus' ORDER BY created_at"
                )
            else:
                cursor.execute(
                    "SELECT id, content, author, created_at FROM knowledge_records "
                    "WHERE type = ? AND consensus_id IS NULL ORDER BY created_at",
                    (record_type,),
                )
            out: list[dict[str, Any]] = []
            for row in cursor.fetchall():
                rec: dict[str, Any] = {
                    "id": row[0],
                    "content": json.loads(row[1]),
                    "created_at": row[-1],
                }
                if record_type != "consensus":
                    rec["author"] = row[2]
                out.append(rec)
            return out
        finally:
            conn.close()

    @staticmethod
    def _organize(
        records: list[dict[str, Any]],
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        org: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for r in records:
            c = r["content"]
            ls = c.get("life_stage", "NOT_KNOWN")
            if "domain" in c:
                org[ls][c["domain"]].append(r)
            elif "domains" in c:
                for d in c["domains"]:
                    org[ls][d].append(r)
            else:
                org[ls]["Unclassified"].append(r)
        return org

    @staticmethod
    def _load_role_prompt(role_file: str) -> str:
        role_path = Path(__file__).parent / "roles" / role_file
        return role_path.read_text()

    def _synthesize(
        self,
        organized: dict[str, dict[str, list[dict[str, Any]]]],
        mode: str,
    ) -> dict[str, Any]:
        model = loaden.get(self.config, "llm.generative_model")
        if mode == "collaborator":
            prompt = self._load_role_prompt("Role-CollaboratorProfile.md")
        elif mode == "agent":
            prompt = self._load_role_prompt("Role-AgentProfile.md")
        else:
            prompt = (
                "You are an expert biographer. Synthesize the observations into a "
                "cohesive biographical narrative. For each life stage create a section. "
                "Include a brief overall summary. Respond as JSON: "
                '{"summary": "...", "life_stages": [{"stage": "...", "narrative": "..."}]}'
            )
        input_data: dict[str, Any] = {"mode": mode, "life_stages": {}}
        for ls, domains in organized.items():
            input_data["life_stages"][ls] = {}
            for dom, recs in domains.items():
                input_data["life_stages"][ls][dom] = [
                    {
                        "observation": r["content"].get("observation", ""),
                        "confidence": r["content"].get("confidence", "MODERATE"),
                    }
                    for r in recs
                ]
        try:
            return llm_process(
                self.config,
                input_content=input_data,
                prompt=prompt,
                model=model,
                expect_json=True,
            )
        except Exception:
            logger.exception("Profile synthesis failed")
            return {"summary": "Error generating profile.", "life_stages": []}

    @staticmethod
    def _format_markdown(content: dict[str, Any]) -> str:
        # Section-based format (collaborator / agent profiles)
        if "sections" in content:
            purpose = content.get("purpose", "Profile")
            md = f"# {purpose}\n\n"
            for section in content["sections"]:
                md += f"## {section['heading']}\n\n{section['content']}\n\n"
            return md
        # Life-stage format (short / long biographical profiles)
        md = f"# Personal Profile\n\n## Summary\n\n{content.get('summary', '')}\n\n"
        for stage in content.get("life_stages", []):
            title = stage["stage"].replace("_", " ").title()
            md += f"## {title}\n\n{stage['narrative']}\n\n"
        return md

    @staticmethod
    def _format_raw(records: list[dict[str, Any]]) -> str:
        conf_order = {"VERY_HIGH": 0, "HIGH": 1, "MODERATE": 2, "LOW": 3, "VERY_LOW": 4}

        items: list[tuple[tuple[int, str, int], dict[str, Any]]] = []
        for r in records:
            c = r["content"]
            ls = c.get("life_stage", "NOT_KNOWN")
            ls_i = LIFE_STAGES.index(ls) if ls in LIFE_STAGES else UNKNOWN_LIFESTAGE_ORDER
            author = r.get("author", "Consensus")
            conf = c.get("confidence", "MODERATE")
            items.append(
                (
                    (ls_i, author, conf_order.get(conf, UNKNOWN_CONFIDENCE_ORDER)),
                    {
                        "life_stage": ls,
                        "author": author,
                        "confidence": conf,
                        "observation": c.get("observation", ""),
                        "id": r.get("id", ""),
                    },
                )
            )
        items.sort(key=lambda x: x[0])

        out = "# Raw Knowledge Records\n\n"
        current_ls = current_author = current_conf = ""
        for _, d in items:
            if d["life_stage"] != current_ls:
                current_ls = d["life_stage"]
                out += f"## {current_ls}\n"
                current_author = current_conf = ""
            if d["author"] != current_author:
                current_author = d["author"]
                out += f"### {current_author}\n"
                current_conf = ""
            if d["confidence"] != current_conf:
                current_conf = d["confidence"]
                out += f"#### {current_conf}\n"
            out += f"- {d['observation']} [{d['id']}]\n"
        return out


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


class Query:
    """Query knowledge records with filtering and similarity search."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def get_kr(self, **kwargs: Any) -> list[KnowledgeRecord]:
        return get_filtered_knowledge_records(self.config, **kwargs)

    def get_observations_by_similarity(
        self,
        topic: str,
        threshold: float,
        domain: list[str] | None = None,
        lifestage: list[str] | None = None,
        confidence: list[str] | None = None,
    ) -> list[KnowledgeRecord]:
        records = self.get_kr(
            domain=domain,
            lifestage=lifestage,
            confidence=confidence,
            include_embedding=True,
        )
        emb = get_embedding(self.config, topic)
        emb_b64 = base64.b64encode(emb.tobytes()).decode("utf-8")
        return [
            r
            for r in records
            if r.embedding and cosine_similarity_base64(emb_b64, r.embedding) >= threshold
        ]


# ===========================================================================
# Shared pipeline helpers
# ===========================================================================


def _process_queue(
    cfg: dict[str, Any],
    observer_names: list[str],
    *,
    limit: int | None = None,
    dryrun: bool = False,
    cache: bool = False,
) -> int:
    """Process pending queue items through analysis.  Returns count processed."""
    mgr = AnalysisManager(cfg)
    mgr.cache_enabled = cache
    loader = Loader(cfg)
    items = loader.get_queue_items(status="pending", limit=limit or 0)
    processed = 0
    for item in items:
        if dryrun:
            console.print(f"[yellow]Would process: {item['title']}[/yellow]")
            continue
        sid = mgr.create_session(item)
        if sid:
            mgr.update_queue_status(item["id"], "completed")
        else:
            mgr.update_queue_status(item["id"], "failed")
            continue
        try:
            for sess in mgr.get_unanalyzed_sessions(observer_names):
                mgr.run_multiple_analyses(sess["id"], observer_names)
        except FileNotFoundError as e:
            console.print(f"[red]Skipping {item['title']}: {e}[/red]")
            mgr.update_queue_status(item["id"], "failed")
            continue
        processed += 1
    return processed


# ===========================================================================
# CLI
# ===========================================================================


@click.group()
@click.version_option(version=VERSION)
@click.option("--config", "config_path", type=click.Path(exists=True), help="Config file")
@click.option("--db-path", type=click.Path(), help="Override database path")
@click.option("--model", help="Override LLM model")
@click.pass_context
def cli(
    ctx: click.Context,
    config_path: str | None,
    db_path: str | None,
    model: str | None,
) -> None:
    """KnowMe -- Personal knowledge management and analysis system."""
    cfg = get_config(config_path)

    if db_path:
        cfg["database"]["path"] = db_path
    if model:
        cfg["llm"]["generative_model"] = model

    ctx.obj = cfg

    configure_logging(
        level=loaden.get(cfg, "logging.level"),
        log_file=loaden.get(cfg, "logging.file"),
    )


# -- init -------------------------------------------------------------------


@cli.command()
@click.argument("db_path", type=click.Path(), required=False)
@click.pass_context
def init(ctx: click.Context, db_path: str | None) -> None:
    """Initialize a new knowledge database."""
    cfg = ctx.obj
    target = db_path or loaden.get(cfg, "database.path")
    if Path(target).exists():
        console.print(f"[yellow]Database {target} already exists[/yellow]")
        return
    if db_path:
        cfg["database"]["path"] = db_path
    if create_database(cfg):
        console.print(f"[green]Created database at {target}[/green]")
    else:
        console.print(f"[red]Error creating database at {target}[/red]")


# -- load -------------------------------------------------------------------


@cli.command()
@click.argument("file_patterns", nargs=-1, type=click.Path())
@click.option("--name", "-n", help="Name template ({index} for batch)")
@click.option("--author", "-a", help="Content author")
@click.option("--description", "-d", help="Content description")
@click.option(
    "--lifestage",
    "-l",
    type=click.Choice([*LIFE_STAGES, "AUTO"], case_sensitive=False),
    default="AUTO",
    help="Life stage of content",
)
@click.option("--priority", "-p", type=int, default=0, help="Queue priority (higher = first)")
@click.option("--list", "list_queue", is_flag=True, help="List queue contents")
@click.option("--dryrun", is_flag=True, help="Show what would be loaded without loading")
@click.pass_context
def load(
    ctx: click.Context,
    file_patterns: tuple[str, ...],
    name: str | None,
    author: str | None,
    description: str | None,
    lifestage: str,
    priority: int,
    list_queue: bool,
    dryrun: bool,
) -> None:
    """Load documents as sources and enqueue for analysis."""
    cfg = ctx.obj
    author_name = author or loaden.get(cfg, "content.default_author", "")
    if not author_name and not list_queue:
        console.print("[red]Author required (--author or config default_author)[/red]")
        return

    loader = Loader(cfg)

    if list_queue:
        items = loader.get_queue_items(status="pending")
        table = Table(title="Analysis Queue")
        for col in ("ID", "Title", "Author", "Life Stage", "Status", "Priority"):
            table.add_column(col)
        for it in items:
            table.add_row(
                it["id"],
                it["title"],
                it["author"],
                it["lifestage"],
                it["status"],
                str(it["priority"]),
            )
        console.print(table)
        return

    if not file_patterns:
        console.print("[red]FILE_PATTERNS required when not using --list[/red]")
        return

    matched = resolve_file_patterns(file_patterns)
    if not matched:
        console.print("[yellow]No files matched[/yellow]")
        return

    if dryrun:
        table = Table(title="Matched Files")
        table.add_column("Index")
        table.add_column("Path")
        table.add_column("Size")
        for i, fp in enumerate(matched, 1):
            size = Path(fp).stat().st_size
            table.add_row(str(i), fp, f"{size / 1024:.1f}KB")
        console.print(table)
        return

    result = loader.batch_load(
        file_patterns=matched,
        name_template=name,
        author=author_name,
        description=description or "",
        lifestage=lifestage,
        priority=priority,
    )
    console.print(
        f"[green]Loaded {len(result['loaded_files'])} of "
        f"{len(result['matched_files'])} files[/green]"
    )


# -- analyze ----------------------------------------------------------------


@cli.command()
@click.option("--queue", "-q", is_flag=True, help="Process queue items")
@click.option("--queue-id", help="Process specific queue item")
@click.option("--limit", "-l", type=int, default=1, help="Max queue items to process")
@click.option("--model", "-m", help="Override LLM model")
@click.option("--dryrun", is_flag=True, help="Show what would be analyzed without running")
@click.pass_context
def analyze(
    ctx: click.Context,
    queue: bool,
    queue_id: str | None,
    limit: int,
    model: str | None,
    dryrun: bool,
) -> None:
    """Run expert analysis on queued content."""
    cfg = ctx.obj
    if model:
        cfg["llm"]["generative_model"] = model
    if dryrun:
        cfg["dryrun"] = True

    observer_names = [o["name"] for o in loaden.get(cfg, "roles.Observers", [])]
    mgr = AnalysisManager(cfg)

    if not (queue or queue_id):
        console.print("[yellow]Use --queue or --queue-id[/yellow]")
        return

    loader = Loader(cfg)

    if queue_id:
        item = loader.get_source_and_queue_item(queue_id=queue_id)
        if not item:
            console.print(f"[red]Queue item {queue_id} not found[/red]")
            return
        sid = mgr.create_session(item)
        if sid:
            mgr.update_queue_status(item["id"], "completed")
            result = mgr.run_multiple_analyses(sid, observer_names)
            console.print(f"[green]Analyzed session {sid}: {result}[/green]")
        else:
            console.print("[yellow]Session already exists[/yellow]")
        return

    processed = _process_queue(cfg, observer_names, limit=limit, dryrun=dryrun)
    if processed == 0 and not dryrun:
        console.print("[yellow]Queue is empty[/yellow]")


# -- merge ------------------------------------------------------------------


@cli.command()
@click.option(
    "--threshold",
    "-t",
    default=DEFAULT_MERGE_THRESHOLD,
    help="Similarity threshold for clustering",
)
@click.option("--dryrun", is_flag=True, help="Show clusters without creating consensus")
@click.option("--reset", is_flag=True, help="Reset existing consensus records")
@click.option(
    "--type",
    "kr_type",
    type=click.Choice(["note", "fact"]),
    default="note",
    help="Record type to cluster",
)
@click.option("--list", "list_clusters", is_flag=True, help="List clusters without merging")
@click.option("--model", "-m", help="Override generative model")
@click.pass_context
def merge(
    ctx: click.Context,
    threshold: float,
    dryrun: bool,
    reset: bool,
    kr_type: str,
    list_clusters: bool,
    model: str | None,
) -> None:
    """Cluster similar observations and create consensus records."""
    cfg = ctx.obj
    if model:
        cfg["llm"]["generative_model"] = model
    if dryrun:
        cfg["dryrun"] = True

    cm = ConsensusManager(cfg)

    if reset:
        n = cm.reset_consensus()
        console.print(f"[green]Reset {n} consensus records[/green]")
        return

    result = cm.process_clusters(threshold=threshold, kr_type=kr_type)

    if dryrun:
        for cid, info in result["clusters"].items():
            console.print(f"[cyan]{cid}[/cyan]")
            for note in info.get("notes", []):
                console.print(f"  {note.get('observation', '')}")
            console.print()
        return

    if list_clusters:
        table = Table(title=f"Clusters (threshold={threshold})")
        table.add_column("Cluster")
        table.add_column("Count", justify="right")
        table.add_column("Avg Similarity", justify="right")
        for cid, cluster in result["clusters"].items():
            table.add_row(
                cid,
                str(len(cluster["notes"])),
                f"{cluster['average_similarity']:.2f}",
            )
        console.print(table)
        return

    console.print(
        f"[green]Created {result['consensus_count']} consensus records "
        f"from {result['cluster_count']} clusters[/green]"
    )


# -- consensus --------------------------------------------------------------


@cli.command()
@click.option(
    "--threshold",
    "-t",
    default=DEFAULT_CONSENSUS_THRESHOLD,
    help="Similarity threshold for consensus matching",
)
@click.option("--reset", "-r", is_flag=True, help="Reset similar consensus records")
@click.option("--dryrun", is_flag=True, help="Show what would be reset without acting")
@click.pass_context
def consensus(
    ctx: click.Context,
    threshold: float,
    reset: bool,
    dryrun: bool,
) -> None:
    """Find and optionally reset similar consensus records."""
    cfg = ctx.obj
    cm = ConsensusManager(cfg)
    clusters = cm.find_similar_consensus(threshold)

    if not clusters:
        console.print("[yellow]No similar consensus records found[/yellow]")
        return

    table = Table(title=f"Similar Consensus (threshold={threshold})")
    table.add_column("Cluster")
    table.add_column("Count", justify="right")
    table.add_column("Avg Similarity", justify="right")
    for cid, c in clusters.items():
        table.add_row(cid, str(c["observation_count"]), f"{c['average_similarity']:.2f}")
    console.print(table)

    if reset:
        if dryrun:
            total = sum(c["observation_count"] for c in clusters.values())
            console.print(f"[yellow]Would reset {total} records[/yellow]")
        else:
            n = cm.reset_consensus_clusters(clusters)
            console.print(f"[green]Reset {n} consensus records[/green]")


# -- queue ------------------------------------------------------------------


@cli.command()
@click.argument("file_patterns", nargs=-1, type=click.Path())
@click.option("--name", "-n", help="Name for the queued content")
@click.option("--author", "-a", help="Content author")
@click.option(
    "--lifestage",
    "-l",
    type=click.Choice([*LIFE_STAGES, "AUTO"], case_sensitive=False),
    default="AUTO",
    help="Life stage of content",
)
@click.option(
    "--type",
    "-t",
    "content_type",
    type=click.Choice(["document", "qa"]),
    help="Content type",
)
@click.option("--qa", is_flag=True, help="Process as QA transcript")
@click.option("--list", "list_queue", is_flag=True, help="List current queue")
@click.option("--dryrun", is_flag=True, help="Show what would be queued without acting")
@click.pass_context
def queue(
    ctx: click.Context,
    file_patterns: tuple[str, ...],
    name: str | None,
    author: str | None,
    lifestage: str,
    content_type: str | None,
    qa: bool,
    list_queue: bool,
    dryrun: bool,
) -> None:
    """Add content to the analysis queue or ingest QA transcripts."""
    cfg = ctx.obj
    db_path = loaden.get(cfg, "database.path")

    if list_queue:
        conn, cursor = get_connection(db_path)
        cursor.execute(
            "SELECT id, author, lifestage, type, status, created_at "
            "FROM analysis_queue ORDER BY priority DESC, created_at DESC"
        )
        rows = cursor.fetchall()
        conn.close()
        if not rows:
            console.print("[yellow]Queue is empty[/yellow]")
            return
        table = Table(title="Analysis Queue")
        for col in ("ID", "Author", "Life Stage", "Type", "Status"):
            table.add_column(col)
        for r in rows:
            table.add_row(r[0], r[1], r[2], r[3], r[4])
        console.print(table)
        return

    if not file_patterns:
        console.print("[red]FILE_PATTERNS required[/red]")
        return

    effective_author = author or loaden.get(cfg, "content.default_author", "")
    if not effective_author:
        console.print("[red]Author required[/red]")
        return

    eff_type = "qa" if qa else content_type
    if not eff_type:
        console.print("[red]--type or --qa required[/red]")
        return

    matched = resolve_file_patterns(file_patterns)
    if not matched:
        console.print("[yellow]No files matched[/yellow]")
        return

    if dryrun:
        for i, fp in enumerate(matched, 1):
            console.print(f"  {i}. {fp}")
        return

    if qa:
        for fp in matched:
            content = read_file_fallback(fp)
            sid = ingest_qa_content(
                db_path,
                content,
                name or Path(fp).stem,
                effective_author,
                lifestage,
                Path(fp).name,
            )
            if sid:
                console.print(f"[green]Ingested {fp} -> session {sid}[/green]")
            else:
                console.print(f"[yellow]No QA pairs found in {fp}[/yellow]")


# -- profile ----------------------------------------------------------------


@cli.command()
@click.option("--output", "-o", type=click.Path(), help="Write profile to file")
@click.option("--dump", "-d", is_flag=True, help="Dump raw observations")
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["md", "json", "raw"]),
    default="md",
    help="Output format",
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["short", "long", "collaborator", "agent"]),
    default="short",
    help="Profile type: short/long biography, collaborator guide, or agent decision spec",
)
@click.pass_context
def profile(
    ctx: click.Context,
    output: str | None,
    dump: bool,
    fmt: str,
    mode: str,
) -> None:
    """Generate a profile from the knowledge base."""
    cfg = ctx.obj
    gen = ProfileGenerator(cfg)

    if dump:
        console.print(gen.dump_observations())
        return

    with Progress() as progress:
        task = progress.add_task("[cyan]Generating profile...", total=1)
        content = gen.generate_profile(fmt, mode)
        progress.update(task, advance=1)

    if output:
        Path(output).write_text(content, encoding="utf-8")
        console.print(f"[green]Profile written to {output}[/green]")
    else:
        console.print(Panel(content))


# -- regen-embeddings -------------------------------------------------------


@cli.command("regen-embeddings")
@click.option("--batch-size", type=int, default=100, help="Records per batch")
@click.option("--dryrun", is_flag=True, help="Show what would be regenerated without acting")
@click.pass_context
def regen_embeddings(ctx: click.Context, batch_size: int, dryrun: bool) -> None:
    """Regenerate embeddings for all knowledge records."""
    cfg = ctx.obj
    db_path = loaden.get(cfg, "database.path")
    emb_model = loaden.get(cfg, "llm.embedding_model")
    console.print(f"[blue]Using model: {emb_model}[/blue]")

    conn, cursor = get_connection(db_path)
    cursor.execute("SELECT COUNT(*) FROM knowledge_records")
    total = cursor.fetchone()[0]
    conn.close()
    console.print(f"[blue]Total records: {total}[/blue]")

    offset = 0
    processed = success = failed = 0

    while True:
        conn, cursor = get_connection(db_path)
        cursor.execute(
            "SELECT id, content FROM knowledge_records LIMIT ? OFFSET ?",
            (batch_size, offset),
        )
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            break

        for row in rows:
            try:
                content = json.loads(row[1])
                text = content.get("observation") or content.get("content", "")
                if not text:
                    continue

                emb = get_embedding(cfg, text)

                if not dryrun:
                    conn2, cur2 = get_connection(db_path)
                    cur2.execute(
                        "UPDATE knowledge_records SET embedding = ? WHERE id = ?",
                        (emb.tobytes(), row[0]),
                    )
                    conn2.commit()
                    conn2.close()

                success += 1
            except Exception:
                logger.exception("Failed for record %s", row[0])
                failed += 1
            processed += 1

        offset += batch_size

    console.print(f"[green]Done: {processed} processed, {success} ok, {failed} failed[/green]")


# -- run --------------------------------------------------------------------


def _prepare_session_db(source_db: str, session_db: str) -> None:
    """Copy source DB to session path and clear analysis output tables."""
    shutil.copy2(source_db, session_db)
    with db_cursor(session_db, commit=True) as (_conn, cur):
        cur.execute("DELETE FROM knowledge_records")
        cur.execute("DELETE FROM processing_log")
        cur.execute("DELETE FROM sessions")
        cur.execute("UPDATE analysis_queue SET status = 'pending'")


@cli.command()
@click.option("--model", "-m", help="Override generative model for this run")
@click.option("--limit", "-l", type=int, default=0, help="Limit queue items (0=all)")
@click.option("--list", "list_runs", is_flag=True, help="List previous runs")
@click.option("--dryrun", is_flag=True, help="Show what would happen")
@click.option("--cache", is_flag=True, help="Reuse analysis from previous same-model runs")
@click.pass_context
def run(
    ctx: click.Context,
    model: str | None,
    limit: int,
    list_runs: bool,
    dryrun: bool,
    cache: bool,
) -> None:
    """Run full analysis pipeline (analyze + merge + profile) into a session folder."""
    cfg = ctx.obj
    results_dir = Path("results")

    if list_runs:
        if not results_dir.exists():
            console.print("[yellow]No results directory[/yellow]")
            return
        dirs = sorted(
            [
                d
                for d in results_dir.iterdir()
                if d.is_dir() and (d / SESSION_DB_NAME).exists()
            ],
            key=lambda d: d.name,
            reverse=True,
        )
        if not dirs:
            console.print("[yellow]No previous runs found[/yellow]")
            return
        table = Table(title="Previous Runs")
        table.add_column("Session")
        table.add_column("Model")
        table.add_column("Records", justify="right")
        table.add_column("Consensus", justify="right")
        for d in dirs:
            db_path = str(d / SESSION_DB_NAME)
            with db_cursor(db_path) as (_conn, cur):
                cur.execute("SELECT COUNT(*) FROM knowledge_records")
                kr_count = cur.fetchone()[0]
                cur.execute(
                    "SELECT COUNT(*) FROM knowledge_records WHERE consensus_id IS NOT NULL"
                )
                cons_count = cur.fetchone()[0]
            # Parse model from dir name: YYYY-MM-DD_HHMM_modelname
            parts = d.name.split("_", 2)
            model_name = parts[2] if len(parts) > 2 else "unknown"
            table.add_row(d.name, model_name, str(kr_count), str(cons_count))
        console.print(table)
        return

    # Determine model
    gen_model = model or loaden.get(cfg, "llm.generative_model")
    source_db = loaden.get(cfg, "database.path")

    if not Path(source_db).exists():
        console.print(f"[red]Source database not found: {source_db}[/red]")
        return

    # Create session directory
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    safe_model = re.sub(r"[^\w\-.]", "_", gen_model)
    session_dir = results_dir / f"{ts}_{safe_model}"

    if dryrun:
        console.print(f"[yellow]Would create session: {session_dir}[/yellow]")
        console.print(f"[yellow]Model: {gen_model}[/yellow]")
        console.print(f"[yellow]Source DB: {source_db}[/yellow]")
        return

    session_dir.mkdir(parents=True, exist_ok=True)
    session_db = str(session_dir / SESSION_DB_NAME)

    # Copy and prepare session DB
    console.print(f"[blue]Preparing session DB from {source_db}...[/blue]")
    _prepare_session_db(source_db, session_db)

    # Snapshot config
    config_snapshot = dict(cfg)
    config_snapshot["_session"] = {
        "model": gen_model,
        "source_db": str(Path(source_db).resolve()),
        "session_db": session_db,
        "created_at": datetime.now(UTC).isoformat(),
    }
    (session_dir / SESSION_CONFIG_NAME).write_text(
        yaml.dump(config_snapshot, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )

    # Point config at session DB and model
    cfg["database"]["path"] = session_db
    cfg["llm"]["generative_model"] = gen_model

    # --- Analyze ---
    console.print(f"\n[bold cyan]═══ Analyzing with {gen_model} ═══[/bold cyan]")
    observer_names = [o["name"] for o in loaden.get(cfg, "roles.Observers", [])]
    processed = _process_queue(
        cfg,
        observer_names,
        limit=limit if limit > 0 else None,
        cache=cache,
    )
    if processed == 0:
        console.print("[yellow]No queue items to analyze[/yellow]")

    # --- Merge ---
    console.print("\n[bold cyan]═══ Merging clusters ═══[/bold cyan]")
    cm = ConsensusManager(cfg)
    merge_result = cm.process_clusters(threshold=DEFAULT_MERGE_THRESHOLD, kr_type="note")
    console.print(
        f"[green]Created {merge_result['consensus_count']} consensus records "
        f"from {merge_result['cluster_count']} clusters[/green]"
    )

    # --- Profile ---
    console.print("\n[bold cyan]═══ Generating profile ═══[/bold cyan]")
    gen = ProfileGenerator(cfg)
    profile_content = gen.generate_profile("md", "long")
    profile_path = session_dir / SESSION_PROFILE_NAME
    profile_path.write_text(profile_content, encoding="utf-8")
    console.print(f"[green]Profile written to {profile_path}[/green]")

    # --- Summary ---
    with db_cursor(session_db) as (_conn, cur):
        cur.execute("SELECT COUNT(*) FROM knowledge_records")
        kr_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM knowledge_records WHERE consensus_id IS NOT NULL")
        cons_count = cur.fetchone()[0]

    console.print("\n[bold green]═══ Run Complete ═══[/bold green]")
    console.print(f"  Model:     {gen_model}")
    console.print(f"  Session:   {session_dir}")
    console.print(f"  Records:   {kr_count}")
    console.print(f"  Consensus: {cons_count}")
    console.print(f"  Profile:   {profile_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
