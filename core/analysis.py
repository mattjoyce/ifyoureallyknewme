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
from .config import ConfigSchema
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
        r"\{.*\}",  # Object
        r"\[.*\]",  # Array
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


def extract_keywords(config: ConfigSchema, content: str, model: Optional[str] = None) -> List[str]:
    """
    Extract keywords from text using the KeywordExtractor role.
    """
    # Type safety check
    if not isinstance(content, str) or not content:
        logger.warning(
            f"extract_keywords received non-string or empty content: {type(content)}"
        )
        return []

    # Get role path
    role_prompt = get_role_prompt(config,"KeywordExtractor", "Helper")

    try:
        # Use centralized LLM calling
        result = llm_process_with_prompt(config,
            input_content=content, prompt=role_prompt, model=model, expect_json=True
        )

        # Make sure result is a dictionary before trying to extract keywords
        if isinstance(result, dict) and "keywords" in result:
            keywords_str = result["keywords"]
        else:
            # Try to extract keywords from the result
            try:
                parsed_result = extract_json(result)
                keywords_str = parsed_result.get("keywords", "")
            except Exception as e:
                logger.error(f"Failed to extract keywords JSON: {str(e)}")
                return []

        # Process the comma-separated output
        if isinstance(keywords_str, str):
            keywords = [kw.strip() for kw in keywords_str.split(",")]
            return [kw for kw in keywords if kw]  # Filter out empty strings
        else:
            logger.warning(f"Expected string for keywords, got {type(keywords_str)}")
            return []

    except Exception as e:
        logger.error(f"Error in keyword extraction: {str(e)}")
        return []


def store_knowledge_record(config:ConfigSchema,
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
    observation_text = content.get("observation", content.get("content", ""))

    # # Make sure observation_text is a string
    # if isinstance(observation_text, dict):
    #     # If it's a dictionary, convert it to a string representation
    #     observation_text = json.dumps(observation_text)
    #     print(f"mid : {observation_text}")
    # elif not isinstance(observation_text, str):
    #     # If it's any other non-string type, convert to string
    #     observation_text = str(observation_text) if observation_text else ""
    #     print(f"non-str : {observation_text}")

    # Now extract keywords from the string
    keywords = extract_keywords(config, observation_text, model)
    logger.info(f"keywords : {keywords}")

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
            ",".join(keywords) if keywords else None,
        ),
    )


def run_role_analysis(config:ConfigSchema,
    db_path: str, session_id: str, role_name: str, model: Optional[str] = None
) -> Optional[List[str]]:
    """
    Run expert analysis on a session and store observations.
    """
    logger.debug(f"Starting role analysis for {role_name} on session {session_id}")

    # Check if analysis already exists
    conn, cursor = get_connection(db_path)
    cursor.execute(
        "SELECT COUNT(*) FROM knowledge_records WHERE session_id = ? AND author = ?",
        (session_id, role_name),
    )
    existing_count = cursor.fetchone()[0]
    conn.close()

    if existing_count > 0:
        logger.info(
            f"Analysis for {role_name} on session {session_id} already exists with {existing_count} records"
        )
        # Do you want to return existing records or regenerate?
        # For debugging, let's continue and regenerate

    try:
        # Get session data
        logger.debug(f"Retrieving session data for {session_id}")
        session, qa_pairs = get_session_data(db_path, session_id)
        logger.debug(
            f"Session data retrieved: {session['title']}, {len(qa_pairs)} QA pairs"
        )

        # Create input data
        input_content = create_session_context(session, qa_pairs)
        logger.debug(f"Created session context, length: {len(input_content)}")

        # Get role prompt
        role_prompt = get_role_prompt(config, role_name, "Observer")
        logger.debug(
            f"Retrieved role prompt for {role_name}, length: {len(role_prompt)}"
        )

        # Use centralized LLM calling
        logger.info(f"Calling LLM for {role_name} analysis on session {session_id}")
        response = llm_process_with_prompt(config,
            input_content=input_content,
            prompt=role_prompt,
            model=model,
            expect_json=True,
        )

        # Check response type
        logger.debug(f"LLM response type: {type(response)}")
        if isinstance(response, list):
            logger.debug(f"Response is a list with {len(response)} items")
        elif isinstance(response, dict):
            logger.debug(
                f"Response is a dictionary with keys: {', '.join(response.keys())}"
            )
        else:
            logger.debug(f"Response is of unexpected type: {type(response)}")

        # Handle response format
        observations = response if isinstance(response, list) else [response]
        logger.debug(f"Extracted {len(observations)} observations")

        # Store notes in the database
        conn, cursor = get_connection(db_path)

        timestamp = datetime.utcnow().isoformat()
        note_ids = []

        logger.info(f"Storing {len(observations)} observations for {role_name}")

        # Process each observation individually
        for idx, obs in enumerate(observations):
            logger.debug(f"Processing observation {idx+1}/{len(observations)}")

            observation_text = obs.get("observation")
            if not observation_text:
                logger.warning(f"Observation {idx+1} missing 'observation' field")
                continue

            embedding = get_embedding(config,observation_text)
            embedding_bytes = embedding.tobytes()

             # Generate unique note ID for each observation
            note_id = generate_id("note", session_id, role_name)
            logger.debug(f"Generated note ID: {note_id}")

            # Store the knowledge record
            try:
                store_knowledge_record(config,
                    cursor=cursor,
                    note_id=note_id,
                    record_type="note",
                    author=role_name,
                    content=obs,
                    timestamp=timestamp,
                    embedding_bytes=embedding_bytes,
                    session_id=session_id,
                    model=model,
                )
                note_ids.append(note_id)
                logger.debug(f"Stored knowledge record: {note_id}")

                # Commit after each record
                conn.commit()
                logger.debug(f"Committed record {note_id} to database")
            except Exception as e:
                logger.error(f"Error storing record {note_id}: {str(e)}")
                # Continue with other observations

        # Final database verification
        cursor.execute(
            "SELECT COUNT(*) FROM knowledge_records WHERE session_id = ? AND author = ?",
            (session_id, role_name),
        )
        final_count = cursor.fetchone()[0]
        logger.info(
            f"Final database count for {role_name} on session {session_id}: {final_count} records"
        )

        conn.close()

        return note_ids

    except Exception as e:
        logger.error(f"Error in run_role_analysis: {str(e)}", exc_info=True)
        return None


def run_multiple_role_analyses(config :ConfigSchema, db_path: str, session_id: str, role_names: List[str], role_schema_file: str, model: Optional[str] = None
) -> Dict[str, int]:
    """
    Run multiple expert analyses on a session.
    """
    logger.info(
        f"Starting analysis for session {session_id} with {len(role_names)} roles"
    )
    # Debug log the roles we're about to process
    logger.debug(f"Will process these roles: {', '.join(role_names)}")

    # Check if session exists before processing
    conn, cursor = get_connection(db_path)
    cursor.execute("SELECT id, title FROM sessions WHERE id = ?", (session_id,))
    session_info = cursor.fetchone()
    if not session_info:
        logger.error(f"Session {session_id} not found in database")
        conn.close()
        return {}
    logger.debug(f"Found session: {session_info['title']} ({session_id})")

    # Check which roles have already been processed
    all_processed = True
    for role_name in role_names:
        cursor.execute(
            "SELECT COUNT(*) FROM knowledge_records WHERE session_id = ? AND author = ?",
            (session_id, role_name),
        )
        count = cursor.fetchone()[0]
        if count > 0:
            logger.debug(
                f"Role {role_name} already has {count} records for session {session_id}"
            )
        else:
            all_processed = False
            logger.debug(f"Role {role_name} needs processing for session {session_id}")

    if all_processed:
        logger.info(f"All roles already processed for session {session_id}")
        conn.close()
        return {role: 0 for role in role_names}

    conn.close()

    results = {}
    for role_name in role_names:
        logger.info(f"Processing role: {role_name} for session {session_id}")

        # Check if this role already has records
        conn, cursor = get_connection(db_path)
        cursor.execute(
            "SELECT COUNT(*) FROM knowledge_records WHERE session_id = ? AND author = ?",
            (session_id, role_name),
        )
        existing_count = cursor.fetchone()[0]
        conn.close()

        if existing_count > 0:
            logger.info(f"Skipping {role_name} - already has {existing_count} records")
            results[role_name] = existing_count
            continue

        note_ids = run_role_analysis(config, db_path, session_id, role_name, model=model)

        # Verify results were stored
        conn, cursor = get_connection(db_path)
        cursor.execute(
            "SELECT COUNT(*) FROM knowledge_records WHERE session_id = ? AND author = ?",
            (session_id, role_name),
        )
        new_count = cursor.fetchone()[0]
        conn.close()

        if note_ids:
            results[role_name] = len(note_ids)
            logger.info(
                f"Created {len(note_ids)} {role_name} notes for session {session_id}"
            )
            logger.debug(f"Database now shows {new_count} records for {role_name}")

            # Double-check if missing_observers is still reporting this role
            sessions = get_unanalyzed_sessions(db_path, [role_name])
            still_missing = False
            for session in sessions:
                if (
                    session["id"] == session_id
                    and role_name in session["missing_observers"]
                ):
                    still_missing = True

            if still_missing:
                logger.warning(
                    f"Role {role_name} is still reported as missing after processing!"
                )
        else:
            logger.error(f"Failed to create {role_name} notes for session {session_id}")

    logger.info(f"Completed analysis for session {session_id}, results: {results}")
    return results


def get_unanalyzed_sessions(
    db_path: str, observer_names: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Find sessions that haven't been analyzed by specified experts.
    """

    logger.info(f"Checking sessions for observers: {', '.join(observer_names)}")

    try:
        conn, cursor = get_connection(db_path)

        # Get all sessions
        cursor.execute(
            """
            SELECT id, title, created_at
            FROM sessions
            ORDER BY created_at DESC
        """
        )

        sessions = [dict(row) for row in cursor.fetchall()]
        logger.debug(f"Found {len(sessions)} sessions in database")

        # For each session, check which experts have analyzed it
        incomplete_sessions = 0

        for session in sessions:
            session_id = session["id"]
            session["missing_observers"] = []

            logger.debug(f"Checking session {session_id}: {session['title']}")

            # First check distinct authors that have analyzed this session
            cursor.execute(
                """
                SELECT DISTINCT author 
                FROM knowledge_records
                WHERE session_id = ?
            """,
                (session_id,),
            )

            existing_authors = [row[0] for row in cursor.fetchall()]
            logger.debug(
                f"Session {session_id} has been analyzed by: {', '.join(existing_authors or ['none'])}"
            )

            for role_name in observer_names:
                # Check if this expert has analyzed this session
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM knowledge_records
                    WHERE session_id = ? AND author = ?
                """,
                    (session_id, role_name),
                )

                count = cursor.fetchone()[0]
                if count == 0:
                    session["missing_observers"].append(role_name)
                    logger.debug(
                        f"Session {session_id} is missing observer: {role_name}"
                    )
                else:
                    logger.debug(
                        f"Session {session_id} has {count} records from {role_name}"
                    )

            # Add a complete flag, if a session has no missing observers
            session["is_complete"] = len(session["missing_observers"]) == 0

            if not session["is_complete"]:
                incomplete_sessions += 1
                logger.debug(
                    f"Session {session_id} is incomplete, missing: {', '.join(session['missing_observers'])}"
                )
            else:
                logger.debug(f"Session {session_id} is complete")

        logger.info(
            f"Found {incomplete_sessions} incomplete sessions out of {len(sessions)}"
        )
        conn.close()
        return sessions

    except sqlite3.Error as e:
        logger.error(f"Error finding unanalyzed sessions: {str(e)}")
        return []





def process_document(
    db_path: str,
    content: str,
    metadata: Dict[str, str],
    expert_roles: List[str],
    model: Optional[str] = None,
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
            (source_id, metadata["name"], metadata.get("type", "document"), timestamp),
        )

        # Store session
        cursor.execute(
            "INSERT INTO sessions (id, title, description, created_at, metadata) VALUES (?, ?, ?, ?, ?)",
            (
                session_id,
                metadata.get("title", metadata["name"]),
                metadata.get("description", ""),
                timestamp,
                json.dumps(metadata),
            ),
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
                    expect_json=True,
                )

                # Handle response format
                observations = response if isinstance(response, list) else [response]

                # Store observations as knowledge records
                note_count = 0
                for obs in observations:
                    observation_text = obs.get("observation", obs.get("content", ""))
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
                        model=model,
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
        if "conn" in locals():
            conn.close()


def process_queued_document(
    db_path: str,
    queue_id: str,
    expert_roles: Optional[List[str]] = None,
    model: Optional[str] = None,
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
            (queue_id,),
        )
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Queue item {queue_id} not found")

        name, author, lifestage, filename = row

        # Read the document
        with open(filename, "r") as f:
            content = f.read()

        # Process the document
        results = process_document(
            db_path,
            content,
            {
                "name": name,
                "author": author,
                "lifestage": lifestage,
                "filename": filename,
            },
            expert_roles
            or [
                "Psychologist",
                "Demographer",
                "BehavioralEconomist",
                "PoliticalScientist",
            ],
            model,
        )

        # Mark as completed
        if results:
            cursor.execute(
                "UPDATE analysis_queue SET completed = 1 WHERE id = ?", (queue_id,)
            )
            conn.commit()

        conn.close()
        return results

    except Exception as e:
        logger.error(f"Error processing queue item {queue_id}: {str(e)}")
        return {}



def get_unanalyzed_documents(config:ConfigSchema,
    db_path: str, expert_roles: List[str]
) -> List[Dict[str, Any]]:
    """
    Find documents that haven't been analyzed by specified experts.

    Args:
        db_path: Path to the SQLite database
        expert_roles: List of expert roles to check

    Returns:
        List of document dictionaries with id, name, and missing_experts fields
    """

    db_path = db_path or config.database.path

    try:
        conn, cursor = get_connection(db_path)

        # Get all documents
        cursor.execute("SELECT id, name FROM sources")
        documents = [dict(row) for row in cursor.fetchall()]

        # For each document, check which experts have analyzed it
        for doc in documents:
            doc_id = doc["id"]
            doc["missing_experts"] = []

            for role_name in expert_roles:
                # Check if this expert has analyzed this document
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM knowledge_records
                    WHERE source_id = ? AND author = ?
                """,
                    (doc_id, role_name),
                )

                count = cursor.fetchone()[0]
                if count == 0:
                    doc["missing_experts"].append(role_name)

            # Add a complete flag, if a document has no missing experts
            doc["is_complete"] = len(doc["missing_experts"]) == 0

        conn.close()
        return documents

    except sqlite3.Error as e:
        logger.error(f"Error finding unanalyzed documents: {str(e)}")
        return []
