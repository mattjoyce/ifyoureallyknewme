"""
Analysis module for TheInterview project.

This module provides functionality for performing expert analysis on 
interview sessions and documents, including running LLM-based expert analyzers.
"""
import json
import logging
import os
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
from .config import get_config
from .utils import generate_id
from .ingest import get_session_data, create_session_context
from .embedding import get_embedding
from .llm_calling import llm_process_with_prompt, get_role_prompt
from .database import get_connection

# Set up logging
logger = logging.getLogger(__name__)

def extract_json(text: str) -> Any:
    """
    Extract JSON from text that may contain other content.
    
    Args:
        text: Text that may contain JSON
        
    Returns:
        Parsed JSON object
        
    Raises:
        ValueError: If no valid JSON is found
    """
    import re
    import json
    
    # Look for JSON patterns
    json_patterns = [
        r"\[\s*\{.*\}\s*\]",  # Array of objects
        r"\{.*\}",            # Object
        r"\[.*\]"             # Array
    ]
    
    for pattern in json_patterns:
        matches = re.finditer(pattern, text, re.DOTALL)
        for match in matches:
            potential_json = match.group()
            try:
                result = json.loads(potential_json)
                return result
            except json.JSONDecodeError:
                pass
    
    raise ValueError("No valid JSON found in text")


def extract_keywords(content: str, model: Optional[str] = None) -> List[str]:
    """
    Extract keywords from text using the KeywordExtractor role.
    
    Args:
        text: Text to extract keywords from
        model: Optional model override
        
    Returns:
        List of extracted keywords
    """
    config=get_config()
    model = model or config.llm.generative_model
    
    # Get role path
    role_prompt = get_role_prompt("KeywordExtractor","Helper")
    
    try:
        # Use centralized LLM calling
        try:
            result = llm_process_with_prompt(
                input_content=content,
                prompt=role_prompt,
                model=model,
                expect_json=True
            )
            
            # extract element 'keywords' from the json result
            result = extract_json(result)['keywords']

            # Process the comma-separated output
            keywords = [kw.strip() for kw in result.split(",")]
            return [kw for kw in keywords if kw]  # Filter out empty strings
            
        except Exception as e:
            logger.error(f"Error in keyword extraction: {str(e)}")
            return []
            
    except Exception as e:
        logger.error(f"Error in keyword extraction setup: {str(e)}")
        return []






def store_knowledge_record(
    cursor: sqlite3.Cursor,
    note_id: str,
    record_type: str,
    author: str,
    content: Dict[str, Any],
    timestamp: str,
    embedding_bytes: bytes,
    session_id: Optional[str] = None,
    qa_id: Optional[str] = None,
    source_id: Optional[str] = None,
    model: Optional[str] = None
) -> None:
    """
    Store a knowledge record with keywords.
    """
    # Extract keywords from the observation text
    observation_text = content.get('observation', content.get('content', ''))
    keywords = extract_keywords(observation_text, model) if observation_text else []
    
    # Store as knowledge record
    cursor.execute(
        """
        INSERT INTO knowledge_records
        (id, type, author, content, created_at, version, 
         embedding, session_id, qa_id, source_id, keywords)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            note_id,
            record_type,
            author,
            json.dumps(content),
            timestamp,
            "1.0",
            embedding_bytes,
            session_id,
            qa_id,
            source_id,
            ",".join(keywords) if keywords else None
        ),
    )


def run_role_analysis(
    db_path: str, 
    session_id: str, 
    role_name: str, 
    model: Optional[str] = None
) -> Optional[List[str]]:
    """
    Run expert analysis on a session and store observations.
    
    Args:
        db_path: Path to the SQLite database
        session_id: ID of the session to analyze
        expert_role: Name of the expert role to use
        model: Name of the LLM model to use (default: from config)
        
    Returns:
        List of note IDs if successful, None otherwise
    """

    # Get session data
    session, qa_pairs = get_session_data(db_path, session_id)
    
    # Create context document
    context = create_session_context(session, qa_pairs)

    # Get role prompt
    role_prompt = get_role_prompt(role_name, "Observer")
        
    logger.info(f"Running {role_name} analysis on session {session_id}")
    
    # Use centralized LLM calling
    response = llm_process_with_prompt(
        input_content=context,
        prompt=role_prompt,
        model=model,
        expect_json=True
    )
    
    # Handle response format
    observations = response if isinstance(response, list) else [response]
    print(observations)
    # Store notes in the database
    conn, cursor = get_connection(db_path)
    
    timestamp = datetime.utcnow().isoformat()
    note_ids = []
      
    # Process each observation individually and link to questions
    for obs in observations:
     
        observation_text = obs.get('observation')
        embedding = get_embedding(observation_text)
        embedding_bytes = embedding.tobytes()

        # Generate unique note ID for each observation
        note_id = generate_id("note", session_id, role_name)
        
        # Store the knowledge record
        store_knowledge_record(
            cursor=cursor,
            note_id=note_id,
            record_type="note",
            author=role_name,
            content=obs,
            timestamp=timestamp,
            embedding_bytes=embedding_bytes,
            session_id=session_id,
            model=model
        )
        note_ids.append(note_id)
    
    conn.commit()
    conn.close()
    
    logger.info(f"Created {len(note_ids)} {role_name} notes for session {session_id}")
    return note_ids
        

        


def run_multiple_role_analyses(
    db_path: str,
    session_id: str,
    role_names: List[str],
    model: Optional[str] = None
) -> Dict[str, int]:
    """
    Run multiple expert analyses on a session.
    
    Args:
        db_path: Path to the SQLite database
        session_id: ID of the session to analyze
        expert_roles: List of expert roles to run
        model: Name of the LLM model to use
        
    Returns:
        Dictionary mapping expert names to counts of notes created
    """
    results = {}
    for role_name in role_names:
        note_ids = run_role_analysis(db_path, session_id, role_name, model)
        if note_ids:
            results[role_name] = len(note_ids)
    
    return results


def get_unanalyzed_sessions(db_path: str, observer_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Find sessions that haven't been analyzed by specified experts.
    
    Args:
        db_path: Path to the SQLite database
        expert_roles: List of expert roles to check. If None, uses default experts.
        config: Configuration dictionary containing expert roles.
        
    Returns:
        List of session dictionaries with id, title, created_at, and missing_experts fields
    """
    config=get_config()
    db_path = db_path or config.database.path

    # Use default observer names if not provided
    observer_names = observer_names or [observer.name for observer in config.roles.Observers]

    try:
        conn, cursor = get_connection(db_path)
        
        # Get all sessions
        cursor.execute("""
            SELECT id, title, created_at
            FROM sessions
            ORDER BY created_at DESC
        """)
        
        sessions = [dict(row) for row in cursor.fetchall()]

        # For each session, check which experts have analyzed it
        for session in sessions:
            session_id = session['id']
            session['missing_observers'] = []
            
            for role_name in observer_names:
                # Check if this expert has analyzed this session
                cursor.execute("""
                    SELECT COUNT(*) FROM knowledge_records
                    WHERE session_id = ? AND author = ?
                """, (session_id, role_name))
                
                count = cursor.fetchone()[0]
                if count == 0:
                    session['missing_observers'].append(role_name)
                    
            # Add a complete flag, if a session has no missing observers
            session['is_complete'] = len(session['missing_observers']) == 0
        
        conn.close()
        return sessions
        
    except sqlite3.Error as e:
        logger.error(f"Error finding unanalyzed sessions: {str(e)}")
        return []


def analyze_session_by_title(
    db_path: str,
    title_pattern: str,
    expert_roles: List[str],
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Find a session by title pattern and run analysis on it.
    
    Args:
        db_path: Path to the SQLite database
        title_pattern: Search pattern for session title (% is wildcard)
        expert_roles: List of expert roles to run
        model: Name of the LLM model to use
        
    Returns:
        Dictionary with session info and analysis results
    """
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Find sessions matching title pattern
        cursor.execute("""
            SELECT id, title, created_at
            FROM sessions
            WHERE title LIKE ?
            ORDER BY created_at DESC
        """, (f"%{title_pattern}%",))
        
        session_rows = cursor.fetchall()
        conn.close()
        
        if not session_rows:
            logger.warning(f"No sessions found matching '{title_pattern}'")
            return {"status": "error", "message": f"No sessions found matching '{title_pattern}'"}
        
        # Use the most recent matching session
        session = dict(session_rows[0])
        session_id = session['id']
        
        # Run analyses
        results = run_multiple_role_analyses(db_path, session_id, expert_roles, model)
        
        return {
            "status": "success",
            "session": session,
            "results": results
        }
        
    except sqlite3.Error as e:
        logger.error(f"Error analyzing session by title: {str(e)}")
        return {"status": "error", "message": str(e)}

def process_document(
    db_path: str,
    content: str,
    metadata: Dict[str, str],
    expert_roles: List[str],
    model: Optional[str] = None
) -> Dict[str, int]:
    """Process a document with multiple expert roles."""
    model = model or MODEL_NAME
    results = {}
    
    try:
        # Create session for document
        session_id = generate_id("session", metadata["name"])
        timestamp = datetime.utcnow().isoformat()
        
        # Store document as source
        source_id = generate_id("source", metadata["name"])
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Store source
        cursor.execute(
            "INSERT INTO sources (id, name, type, created_at) VALUES (?, ?, ?, ?)",
            (source_id, metadata["name"], metadata.get("type", "document"), timestamp)
        )
        
        # Store session
        cursor.execute(
            "INSERT INTO sessions (id, title, description, created_at, metadata) VALUES (?, ?, ?, ?, ?)",
            (
                session_id,
                metadata.get("title", metadata["name"]),
                metadata.get("description", ""),
                timestamp,
                json.dumps(metadata)
            )
        )
        
        # Process with each expert role
        for expert in expert_roles:
            try:
                role_path = config.get_role_file_path(expert)
                if not role_path or not role_path.exists():
                    logger.error(f"Role file for {expert} not found")
                    continue
                
                logger.info(f"Running {expert} analysis on document {metadata['name']}")
                
                # Use centralized LLM calling
                response = llm_process_with_prompt(
                    input_content=content,
                    role_file=str(role_path),
                    model=model,
                    expect_json=True
                )
                
                # Handle response format
                observations = response if isinstance(response, list) else [response]
                
                # Store observations as knowledge records
                note_count = 0
                for obs in observations:
                    observation_text = obs.get('observation', obs.get('content', ''))
                    if not observation_text:
                        continue
                        
                    embedding = get_embedding(observation_text)
                    embedding_bytes = embedding.tobytes()
                    
                    # Generate unique note ID
                    note_id = generate_id("note", session_id, expert, str(note_count))
                    
                    # Store the knowledge record
                    store_knowledge_record(
                        cursor=cursor,
                        note_id=note_id,
                        record_type="note",
                        domain=expert.lower(),
                        author=expert,
                        content=obs,
                        timestamp=timestamp,
                        embedding_bytes=embedding_bytes,
                        session_id=session_id,
                        source_id=source_id,
                        model=model
                    )
                    note_count += 1
                
                conn.commit()
                results[expert] = note_count
                
            except Exception as e:
                logger.error(f"Error processing with {expert}: {str(e)}")
                results[expert] = 0
                continue
        
        return results
        
    except Exception as e:
        logger.error(f"Error in document processing: {str(e)}")
        return {}
    finally:
        if 'conn' in locals():
            conn.close()

def process_queued_document(
    db_path: str,
    queue_id: str,
    expert_roles: Optional[List[str]] = None,
    model: Optional[str] = None
) -> Dict[str, int]:
    """
    Process a document from the analysis queue.
    
    Args:
        db_path: Path to the SQLite database
        queue_id: ID of the queue item to process
        expert_roles: List of expert roles to run (optional)
        model: Name of the LLM model to use (optional)
        
    Returns:
        Dictionary mapping expert names to counts of notes created
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get queue item
        cursor.execute(
            "SELECT name, author, lifestage, filename FROM analysis_queue WHERE id = ?",
            (queue_id,)
        )
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Queue item {queue_id} not found")
            
        name, author, lifestage, filename = row
        
        # Read the document
        with open(filename, 'r') as f:
            content = f.read()
        
        # Process the document
        results = process_document(
            db_path,
            content,
            {
                "name": name,
                "author": author,
                "lifestage": lifestage,
                "filename": filename
            },
            expert_roles or [
                "Psychologist",
                "Demographer",
                "BehavioralEconomist",
                "PoliticalScientist",
            ],
            model
        )
        
        # Mark as completed
        if results:
            cursor.execute(
                "UPDATE analysis_queue SET completed = 1 WHERE id = ?",
                (queue_id,)
            )
            conn.commit()
        
        conn.close()
        return results
        
    except Exception as e:
        logger.error(f"Error processing queue item {queue_id}: {str(e)}")
        return {}



def find_and_analyze_missing_expert_sessions(db_path: str, expert_roles: List[str], model: Optional[str] = None) -> None:
    """
    Find sessions missing analysis for specific experts and run the analysis.
    
    Args:
        db_path: Path to the SQLite database
        expert_roles: List of expert roles to check and analyze
        model: Optional model name to use for analysis
    """
    config=get_config()
    

def get_unanalyzed_documents(db_path: str, expert_roles: List[str]) -> List[Dict[str, Any]]:
    """
    Find documents that haven't been analyzed by specified experts.
    
    Args:
        db_path: Path to the SQLite database
        expert_roles: List of expert roles to check
        
    Returns:
        List of document dictionaries with id, name, and missing_experts fields
    """

    #TODO FIX
    config=get_config()
    db_path = db_path or config.database.path
    
    try:
        conn, cursor = get_connection(db_path)
        
        # Get all documents
        cursor.execute("SELECT id, name FROM sources")
        documents = [dict(row) for row in cursor.fetchall()]
        
        # For each document, check which experts have analyzed it
        for doc in documents:
            doc_id = doc['id']
            doc['missing_experts'] = []
            
            for role_name in expert_roles:
                # Check if this expert has analyzed this document
                cursor.execute("""
                    SELECT COUNT(*) FROM knowledge_records
                    WHERE source_id = ? AND author = ?
                """, (doc_id, role_name))
                
                count = cursor.fetchone()[0]
                if count == 0:
                    doc['missing_experts'].append(role_name)
                    
            # Add a complete flag, if a document has no missing experts
            doc['is_complete'] = len(doc['missing_experts']) == 0
        
        conn.close()
        return documents
        
    except sqlite3.Error as e:
        logger.error(f"Error finding unanalyzed documents: {str(e)}")
        return []