#!/usr/bin/env python3
"""
Embedding Regeneration Script

This script regenerates all embeddings in the knowledge database using a specified
embedding model. It's useful when switching to a new embedding model or when
embeddings need to be refreshed.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import sqlite3
import json
import numpy as np

# Add the parent directory to path for local imports
sys.path.append(str(Path(__file__).parent))

from core.config import get_config, ConfigSchema
from core.database import get_connection
from core.embedding import get_embedding

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Regenerate embeddings for the knowledge database")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--db-path", type=str, help="Override database path from config")
    parser.add_argument("--model", type=str, help="Override embedding model from config")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing records")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without saving")
    return parser.parse_args()

def get_records_needing_embeddings(config: ConfigSchema, batch_size: int, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Get records from the database that need embeddings regenerated.
    
    Args:
        config: Application configuration
        batch_size: Number of records to retrieve
        offset: Offset for pagination
        
    Returns:
        List of records with ID and content
    """
    conn, cursor = get_connection(config.database.path)
    
    try:
        cursor.execute("""
            SELECT id, content 
            FROM knowledge_records
            LIMIT ? OFFSET ?
        """, (batch_size, offset))
        
        records = []
        for row in cursor.fetchall():
            try:
                content = json.loads(row['content'])
                # Extract text for embedding - primarily the observation field
                text = content.get('observation', '')
                if not text and isinstance(content, dict):
                    # Try to find any text content if observation is not present
                    for key in ['content', 'text', 'description']:
                        if key in content and isinstance(content[key], str):
                            text = content[key]
                            break
                
                if text:
                    records.append({
                        'id': row['id'],
                        'text': text
                    })
                else:
                    logger.warning(f"Couldn't find embeddable text for record {row['id']}")
            except json.JSONDecodeError:
                logger.error(f"Couldn't parse JSON for record {row['id']}")
        
        return records
    finally:
        conn.close()

def get_total_record_count(config: ConfigSchema) -> int:
    """Get the total number of knowledge records in the database."""
    conn, cursor = get_connection(config.database.path)
    try:
        cursor.execute("SELECT COUNT(*) FROM knowledge_records")
        return cursor.fetchone()[0]
    finally:
        conn.close()

def update_record_embedding(config: ConfigSchema, record_id: str, embedding: np.ndarray, dry_run: bool = False) -> bool:
    """
    Update a record's embedding in the database.
    
    Args:
        config: Application configuration
        record_id: ID of the record to update
        embedding: New embedding to store
        dry_run: If True, don't actually update the database
        
    Returns:
        True if successful, False otherwise
    """
    if dry_run:
        logger.info(f"[DRY RUN] Would update embedding for record {record_id}")
        return True
    
    conn, cursor = get_connection(config.database.path)
    
    try:
        embedding_bytes = embedding.tobytes()
        cursor.execute("""
            UPDATE knowledge_records
            SET embedding = ?
            WHERE id = ?
        """, (embedding_bytes, record_id))
        
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        logger.error(f"Error updating embedding for record {record_id}: {str(e)}")
        conn.rollback()
        return False
    finally:
        conn.close()

def regenerate_embeddings(config: ConfigSchema, batch_size: int = 100, dry_run: bool = False) -> Dict[str, int]:
    """
    Regenerate embeddings for all knowledge records.
    
    Args:
        config: Application configuration
        batch_size: Number of records to process at once
        dry_run: If True, don't actually update the database
        
    Returns:
        Statistics about the regeneration process
    """
    model = config.llm.embedding_model
    logger.info(f"Using embedding model: {model}")
    
    total_records = get_total_record_count(config)
    logger.info(f"Found {total_records} total knowledge records")
    
    offset = 0
    processed = 0
    success = 0
    failed = 0
    
    while True:
        records = get_records_needing_embeddings(config, batch_size, offset)
        if not records:
            break
            
        logger.info(f"Processing batch of {len(records)} records (offset {offset})")
        
        for record in records:
            try:
                # Generate new embedding
                embedding = get_embedding(config, record['text'])
                
                # Update record with new embedding
                if update_record_embedding(config, record['id'], embedding, dry_run):
                    success += 1
                else:
                    failed += 1
                    
                processed += 1
                if processed % 10 == 0:
                    logger.info(f"Processed {processed}/{total_records} records")
                    
            except Exception as e:
                logger.error(f"Error processing record {record['id']}: {str(e)}")
                failed += 1
                
        offset += batch_size
        
    return {
        "total": total_records,
        "processed": processed,
        "success": success,
        "failed": failed
    }

def main():
    args = parse_arguments()
    
    # Load configuration
    config = get_config(args.config)
    
    # Apply overrides if provided
    if args.db_path:
        config.database.path = args.db_path
        logger.info(f"Overriding database path: {args.db_path}")
    
    if args.model:
        config.llm.embedding_model = args.model
        logger.info(f"Overriding embedding model: {args.model}")
    
    # Regenerate embeddings
    logger.info(f"Starting embedding regeneration{'(DRY RUN)' if args.dry_run else ''}")
    stats = regenerate_embeddings(config, args.batch_size, args.dry_run)
    
    # Print summary
    logger.info("Embedding regeneration complete")
    logger.info(f"Total records: {stats['total']}")
    logger.info(f"Processed: {stats['processed']}")
    logger.info(f"Success: {stats['success']}")
    logger.info(f"Failed: {stats['failed']}")
    
if __name__ == "__main__":
    main()