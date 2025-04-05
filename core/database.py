"""
Core database operations that are commonly used across multiple modules.
Only includes the most essential shared database functionality.
Module-specific database operations should remain in their respective modules.
"""
import logging
import os
import json
import sqlite3
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from pydantic import BaseModel, Field



from .config import ConfigSchema

# Get logger
logger = logging.getLogger(__name__)

class KnowledgeRecord(BaseModel):
    id: str
    type: str
    domain: Optional[str] = Field(default=None)
    author: Optional[str] = Field(default=None)
    content: dict
    created_at: Optional[str] = Field(default=None)
    version: Optional[str] = Field(default=None)
    consensus_id: Optional[str] = Field(default=None)
    session_id: Optional[str] = Field(default=None)
    qa_id: Optional[str] = Field(default=None)
    source_id: Optional[str] = Field(default=None)
    keywords: Optional[str] = Field(default=None)
    embedding: Optional[str] = Field(default=None) # blob #Base64 string

def get_filtered_knowledge_records(
    config: ConfigSchema,
    session_id: Optional[str] = None,
    domain: Optional[List[str]] = None,
    confidence: Optional[List[str]] = None,
    lifestage: Optional[List[str]] = None,
    observation_text: Optional[List[str]] = None,
    type: Optional[str] = None,
    author: Optional[str] = None,
    consensus_id: Optional[str] = None,
    qa_id: Optional[str] = None,
    source_id: Optional[str] = None,
    embedding: bool = False
) -> List[KnowledgeRecord]:
    conn, cursor = get_connection(config.database.path)

    if embedding:
        query = "SELECT id, type, domain, author, content, created_at, version, consensus_id, session_id, qa_id, source_id, keywords, embedding FROM knowledge_records WHERE 1=1"
    else:
        query = "SELECT id, type, domain, author, content, created_at, version, consensus_id, session_id, qa_id, source_id, keywords FROM knowledge_records WHERE 1=1"

    params = []
    if session_id:
        query += " AND session_id = ?"
        params.append(session_id)
    if type:
        query += " AND type = ?"
        params.append(type)
    if author:
        query += " AND author = ?"
        params.append(author)
    if consensus_id:
        query += " AND consensus_id = ?"
        params.append(consensus_id)
    if qa_id:
        query += " AND qa_id = ?"
        params.append(qa_id)
    if source_id:
        query += " AND source_id = ?"
        params.append(source_id)
    cursor.execute(query, params)
    results = cursor.fetchall()
    records = []
    for row in results:
        try:
            content = json.loads(row[4])

            embedding_base64 = base64.b64encode(row[12]).decode("utf-8") if embedding else None

            record = KnowledgeRecord(
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
                embedding=embedding_base64,
            )
            domain_filter = not domain or any(d.lower() in record.content.get("domain", "").lower() for d in domain)
            confidence_filter = not confidence or any(c.lower() in record.content.get("confidence", "").lower() for c in confidence)
            lifestage_filter = not lifestage or any(ls.lower() in record.content.get("life_stage", "").lower() for ls in lifestage)
            observation_filter = not observation_text or any(ot.lower() in record.content.get("observation", "").lower() for ot in observation_text)
            if domain_filter and confidence_filter and lifestage_filter and observation_filter:
                records.append(record)
        except json.JSONDecodeError:
            print(f"Error decoding JSON: {row[4]}")
    conn.close()
    return records

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


