"""
Core database operations that are commonly used across multiple modules.
Only includes the most essential shared database functionality.
Module-specific database operations should remain in their respective modules.
"""
import logging
import sqlite3
from pathlib import Path
from typing import Optional

from .config import get_config, configure_logging

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


def create_database(db_path: Optional[str] = None) -> bool:
    """
    Create a new database with the schema from schema.sql.
    
    Args:
        db_path: Path where the database file should be created.
                If None, uses the path from config.
    
    Returns:
        bool: True if database creation was successful, False otherwise
    """
    try:
        config = get_config()
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
    config = get_config()
    db_path = db_path or config.db_path
    if not db_path:
        raise ValueError("No database path provided and not found in configuration")
        
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn
