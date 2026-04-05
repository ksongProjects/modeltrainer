from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from .config import CONTROL_DB_PATH, ensure_directories

SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS data_sources (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        kind TEXT NOT NULL,
        description TEXT,
        config_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS dataset_versions (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        source_id TEXT NOT NULL,
        status TEXT NOT NULL,
        tags_json TEXT NOT NULL DEFAULT '[]',
        summary_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS saved_dataset_tags (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        normalized_name TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS feature_set_versions (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        dataset_version_id TEXT NOT NULL,
        status TEXT NOT NULL,
        summary_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS research_layers (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        stage TEXT NOT NULL,
        status TEXT NOT NULL,
        description TEXT NOT NULL,
        config_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS research_layer_controls (
        layer_id TEXT PRIMARY KEY,
        overrides_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS factor_definitions (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        formula TEXT NOT NULL,
        config_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS universe_definitions (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT NOT NULL,
        config_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_specs (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        kind TEXT NOT NULL,
        description TEXT NOT NULL,
        config_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_versions (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        model_spec_id TEXT NOT NULL,
        feature_set_version_id TEXT NOT NULL,
        status TEXT NOT NULL,
        artifact_uri TEXT NOT NULL,
        metrics_json TEXT NOT NULL,
        summary_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS training_runs (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        state TEXT NOT NULL,
        phase TEXT NOT NULL,
        current_stage TEXT NOT NULL,
        config_json TEXT NOT NULL,
        dataset_version_id TEXT,
        feature_set_version_id TEXT,
        model_spec_id TEXT,
        model_version_id TEXT,
        pending_overrides_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS testing_runs (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        state TEXT NOT NULL,
        phase TEXT NOT NULL,
        current_stage TEXT NOT NULL,
        config_json TEXT NOT NULL,
        model_version_id TEXT,
        feature_set_version_id TEXT,
        baseline_model_version_id TEXT,
        pending_overrides_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS run_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        run_kind TEXT NOT NULL,
        phase TEXT NOT NULL,
        stage TEXT NOT NULL,
        event_type TEXT NOT NULL,
        severity TEXT NOT NULL,
        message TEXT NOT NULL,
        progress_pct REAL NOT NULL,
        payload_json TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS metric_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        run_kind TEXT NOT NULL,
        phase TEXT NOT NULL,
        stage TEXT NOT NULL,
        group_name TEXT NOT NULL,
        name TEXT NOT NULL,
        value REAL NOT NULL,
        step INTEGER NOT NULL,
        metadata_json TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS calculation_traces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        run_kind TEXT NOT NULL,
        phase TEXT NOT NULL,
        stage TEXT NOT NULL,
        formula_id TEXT NOT NULL,
        label TEXT NOT NULL,
        inputs_json TEXT NOT NULL,
        transformed_inputs_json TEXT NOT NULL,
        output_json TEXT NOT NULL,
        units TEXT NOT NULL,
        provenance_json TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS artifact_manifests (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        run_kind TEXT NOT NULL,
        artifact_type TEXT NOT NULL,
        path TEXT NOT NULL,
        metadata_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS acceptance_policies (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT NOT NULL,
        config_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
]

COLUMN_MIGRATIONS = {
    "dataset_versions": {
        "tags_json": "TEXT NOT NULL DEFAULT '[]'",
    },
}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect() -> sqlite3.Connection:
    ensure_directories()
    connection = sqlite3.connect(CONTROL_DB_PATH, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with connect() as connection:
        for statement in SCHEMA_STATEMENTS:
            connection.execute(statement)
        for table, columns in COLUMN_MIGRATIONS.items():
            _ensure_columns(connection, table, columns)
        connection.commit()


def _ensure_columns(connection: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing_columns = {
        row["name"]
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }
    for column_name, definition in columns.items():
        if column_name not in existing_columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column_name} {definition}")


@contextmanager
def db_cursor() -> Iterator[sqlite3.Cursor]:
    connection = connect()
    try:
        yield connection.cursor()
        connection.commit()
    finally:
        connection.close()
