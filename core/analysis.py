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
from .utils import generate_id, timestamp,read_file_with_fallback_encodings
from .embedding import get_embedding
from .llm_calling import llm_process_with_prompt, get_role_prompt, extract_keywords
from .database import get_connection

# Set up logging
logger = logging.getLogger(__name__)


class AnalysisManager:
    def __init__(self, config:ConfigSchema):
        self.config = config


    def store_knowledge_record(self,
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
        keywords: Optional[List[str]] = None,
    ) -> None:
        """
        Store a knowledge record with keywords.
        """

        MODEL_NAME=self.config.llm.generative_model

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


    def run_role_analysis(self, session_id: str, role_name: str, ) -> Optional[List[str]]:
        """
        Run expert analysis on a session and store observations.
        """

        DB_PATH=self.config.database.path
        MODEL_NAME=self.config.llm.generative_model

        logger.debug(f"Starting role analysis for {role_name} on session {session_id}")

        # Check if analysis already exists
        conn, cursor = get_connection(DB_PATH)
        cursor.execute(
            "SELECT COUNT(*) FROM knowledge_records WHERE session_id = ? AND author = ?",
            (session_id, role_name),
        )
        existing_count = cursor.fetchone()[0]
        conn.close()

        try:
            # Create input data
            print(f"create_session_context: {session_id}")
            input_content = self.create_session_context(session_id)
            logger.debug(f"Created session context, length: {len(input_content)}")

            # Get role prompt
            role_prompt = get_role_prompt(self.config, role_name, "Observer")
            logger.debug(
                f"Retrieved role prompt for {role_name}, length: {len(role_prompt)}"
            )

            # Use centralized LLM calling
            logger.info(f"Calling LLM for {role_name} analysis on session {session_id}")
            response = llm_process_with_prompt(self.config,
                input_content=input_content,
                prompt=role_prompt,
                model=MODEL_NAME,
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
            conn, cursor = get_connection(DB_PATH)

            note_ids = []

            logger.info(f"Storing {len(observations)} observations for {role_name}")

            # Process each observation individually
            for idx, obs in enumerate(observations):
                logger.debug(f"Processing observation {idx+1}/{len(observations)}")
                observation_text = obs["observation"]
                if not observation_text:
                    logger.warning(f"Observation {idx+1} missing 'observation' field")
                    continue
                logger.info(observation_text)

                embedding = get_embedding(self.config,observation_text)
                embedding_bytes = embedding.tobytes()

                # Generate unique note ID for each observation
                note_id = generate_id("note", session_id, role_name)
                logger.debug(f"Generated note ID: {note_id}")

                # # Extract keywords from the observation text
                # observation_text = obs.get("observation", content.get("content", ""))

                # Now extract keywords from the string
                keywords = extract_keywords(self.config, observation_text)
                logger.info(f"keywords : {keywords}")                

                if not self.config.dryrun:
                    # Store the knowledge record
                    try:
                        self.store_knowledge_record(
                            cursor=cursor,
                            note_id=note_id,
                            record_type="note",
                            author=role_name,
                            content=obs,
                            timestamp=timestamp(),
                            embedding_bytes=embedding_bytes,
                            session_id=session_id,
                            keywords=keywords,
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

            cursor.close()
            conn.close()

            return note_ids

        except Exception as e:
            logger.error(f"Error in run_role_analysis: {str(e)}", exc_info=True)
            return None


    def run_multiple_role_analyses(self,session_id: str, role_names: List[str] ) -> Dict[str, int]:
        """
        Run multiple expert analyses on a session.
        """
        DB_PATH=self.config.database.path


        logger.info(
            f"Starting analysis for session {session_id} with {len(role_names)} roles"
        )
        # Debug log the roles we're about to process
        logger.debug(f"Will process these roles: {', '.join(role_names)}")

        # Check if session exists before processing
        conn, cursor = get_connection(DB_PATH)
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
            conn, cursor = get_connection(DB_PATH)
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

            note_ids = self.run_role_analysis(session_id, role_name)

            # Verify results were stored
            conn, cursor = get_connection(DB_PATH)
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
                sessions = self.get_unanalyzed_sessions([role_name])
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


    def get_unanalyzed_sessions(self,
        observer_names: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find sessions that haven't been analyzed by specified experts.

        """
        DB_PATH=self.config.database.path
        logger.debug(f"Checking sessions for observers: {', '.join(observer_names)}")

        try:
            conn, cursor = get_connection(DB_PATH)

            # Get all sessions
            cursor.execute(
                """
                SELECT id, title, created_at
                FROM sessions
                ORDER BY created_at DESC
            """
            )

            sessions_with_missing_observers = []
            sessions = [dict(row) for row in cursor.fetchall()]
            logger.debug(f"Found {len(sessions)} sessions in database")

            # For each session, check which experts have analyzed it

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
                        SELECT count(session_id) FROM knowledge_records
                        WHERE session_id = ? AND author = ?
                    """,
                        (session_id, role_name),
                    )

                    count = cursor.fetchone()[0]
                    if count == 0:
                        # Add to missing observers list
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
                    sessions_with_missing_observers.append(session)
                    logger.debug(
                        f"Session {session_id} is incomplete, missing: {', '.join(session['missing_observers'])}"
                    )
                else:
                    logger.debug(f"Session {session_id} is complete")

            logger.info(
                f"Found {len(sessions_with_missing_observers)} incomplete sessions out of {len(sessions)}"
            )
            conn.close()
            return sessions_with_missing_observers

        except sqlite3.Error as e:
            logger.error(f"Error finding unanalyzed sessions: {str(e)}")
            return []

    def create_session_context(self, session_id: str) -> str:
        """
        Create a context document with session information and content.
        Fetches the session data and reads content from the file path.
        
        Args:
            session_id: ID of the session to retrieve
            
        Returns:
            Formatted context document text
        """
        # Get database connection
        conn, cursor = get_connection(self.config.database.path)
        
        try:
            # Get session details
            cursor.execute(
                """
                SELECT id, title, description, created_at, source_id, 
                    content_type, author, lifestage, file_path
                FROM sessions
                WHERE id = ?
                """, 
                (session_id,)
            )
            
            session = cursor.fetchone()
            if not session:
                raise ValueError(f"Session {session_id} not found")
            
            #convert to dictionary
            session = dict(session)


            # Create the context header
            context = f"Title: {session['title']}\n"
            
            # Add description if available
            if session['description']:
                context += f"Description: {session['description']}\n"
            
            # Add metadata about author and lifestage
            context += f"Author: {session['author']}\n"
            context += f"Life Stage: {session['lifestage']}\n"
            context += f"---\n"
            
            # Read and add content from file path
            file_path = session['file_path']
            logger.info(f"Reading content from file: {file_path}")
            if file_path and Path(file_path).exists():
                try:
                    source_content = read_file_with_fallback_encodings(file_path)
                    #source_content = Path(file_path).read_text(encoding="utf-8")
                    context += source_content
                except Exception as e:
                    raise IOError(f"Error reading content file: {str(e)}")
            else:
                raise FileNotFoundError(f"Content file not found: {file_path}")
            
            return context
            
        finally:
            conn.close()

    def create_session(self,item : Dict[str, Any]) -> Optional[str]:
        """
        Process the next pending item in the queue.
        1. create a session record
        
        Returns:
            Session ID if an item was processed, None if queue is empty
        """
        logger.info(f"Processing queue item: {item}")

        # if session already exists, for this source, return
        conn, cursor = get_connection(self.config.database.path)
        cursor.execute(
            """
            SELECT id FROM sessions
            WHERE source_id = ? 
            """,
            (item['source_id'],)
        )
        existing_sessions = cursor.fetchall()
        if existing_sessions:
            logger.info(f"Session already exists for source: {item['source_id']}")
            return None
        
        logger.info(f"No session exists for source: {item['source_id']}")


        ## Create a session record
        conn, cursor = get_connection(self.config.database.path)
        session_id=generate_id("session", item['source_id'], item['id'])
        cursor.execute(
            """
            INSERT INTO sessions (id, title, description, author, lifestage, file_path, content_type, source_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                item['title'],
                item['description'],
                item['author'],
                item['lifestage'],
                item['content_path'],
                "document",
                item['source_id'],
                timestamp(),
            )
        )
        conn.commit()
        conn.close()

        return session_id
        

    def update_queue_item_status(self, queue_id: str, status: str) -> bool:
        """Update the status of a queue item to 'completed', 'failed', etc."""
        try:
            conn, cursor = get_connection(self.config.database.path)
            cursor.execute(
                """
                UPDATE analysis_queue
                SET status = ?
                WHERE id = ?
                """,
                (status, queue_id)
            )
            conn.commit()
            conn.close()
            logger.info(f"Updated queue item {queue_id} status to {status}")
            return True
        except Exception as e:
            logger.error(f"Error updating queue item status: {str(e)}")
            return False