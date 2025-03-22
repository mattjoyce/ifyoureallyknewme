"""
Core database operations that are commonly used across multiple modules.
Only includes the most essential shared database functionality.
Module-specific database operations should remain in their respective modules.
"""
import logging
import os
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union


from .config import ConfigSchema

# Get logger
logger = logging.getLogger(__name__)


def create_database(config:ConfigSchema, db_path: Optional[str] = None) -> bool:
    """
    Create a new database with the schema from schema.sql.
    
    Args:
        db_path: Path where the database file should be created.
                If None, uses the path from config.
    
    Returns:
        bool: True if database creation was successful, False otherwise
    """
    try:
        schema_path = Path(config.database.schema_path)
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found at {schema_path}")
        
        db_path = db_path or config.db_path
        if not db_path:
            raise ValueError("No database path provided and not found in configuration")

        # Connect to database which creates it if it doesn't exist
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Read schema from file
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
            
        # Execute schema statements
        cursor.executescript(schema_sql)
        
        conn.commit()
        conn.close()
        logger.info(f"Successfully created database at {db_path}")
        return True
    except (sqlite3.Error, FileNotFoundError) as e:
        logger.error(f"Error creating database: {str(e)}")
        return False

def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Get a connection to the database with proper row factory set.
    
    Args:
        db_path: Path to the database. If None, uses path from config.
        
    Returns:
        sqlite3.Connection with row_factory set to sqlite3.Row
    
    Raises:
        ValueError: If no database path provided or found in config
        sqlite3.Error: If connection fails
    """

    if not db_path:
        raise ValueError("No database path provided and not found in configuration")
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")
        
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor=conn.cursor()
        return conn, cursor
    except sqlite3.Error as e:    
        raise sqlite3.Error(f"Error connecting to database: {str(e)}")

def store_knowledge_record(config : ConfigSchema,
    note_id: str,
    record_type: str,
    author: str,
    content: Dict[str, Any],
    timestamp: str,
    embedding_bytes: bytes,
    session_id: Optional[str] = None,
    qa_id: Optional[str] = None,
    source_id: Optional[str] = None,
    keywords: Optional[List[str]] = None,
) -> None:
    """
    Store a knowledge record with keywords.
    """
    DB_PATH=config.database.path

    MODEL_NAME=config.llm.generative_model

        # CREATE TABLE knowledge_records (
        #     id TEXT PRIMARY KEY,
        #     type TEXT,
        #     domain TEXT,
        #     author TEXT,
        #     content JSON,
        #     created_at TEXT,
        #     version TEXT,
        #     embedding BLOB,
        #     consensus_id TEXT,
        #     session_id TEXT NULL,
        #     qa_id TEXT NULL,
        #     source_id TEXT NULL,
        #     keywords TEXT,
        #     FOREIGN KEY (qa_id) REFERENCES qa_pairs(id),
        #     FOREIGN KEY (session_id) REFERENCES sessions(id),
        #     FOREIGN KEY (source_id) REFERENCES sources(id)
        # );



    # Get database connection
    conn, cursor = get_connection(DB_PATH)

    # Store as knowledge record
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
            timestamp,
            embedding_bytes,
            session_id,
            qa_id,
            source_id,
            ",".join(keywords) if keywords else None,
        ),
    )

    conn.commit()
    logger.info(f"Successfully stored knowledge record: {note_id}")
    conn.close()