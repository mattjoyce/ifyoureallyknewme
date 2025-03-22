"""
Transcript processing module for TheInterview project.

This module provides functionality for processing interview transcripts,
extracting Q&A pairs, and storing them in the database.
"""
import json
import logging
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from .utils import generate_id

# Set up logging
logger = logging.getLogger(__name__)





def extract_qa_pairs(qa_content: str) -> List[Tuple[str, str]]:
    """
    Extract question-answer pairs from qa content.
    
    Args:
        qa_content: Text content with Q&A format
        
    Returns:
        List of (question, answer) tuples
    """
    # Print the content to verify
    print("Content received by extract_qa_pairs (first 200 chars):", qa_content[:200])  # Print first 200 characters for brevity
    # Split the content by Q. pattern and process each part
    parts = re.split(r'(?:^|\n)Q\.\s*', qa_content, flags=re.MULTILINE)
    
    qa_pairs = []
    
    # Skip the first part if it doesn't start with a question
    for part in parts[1:]:  # Skip potential content before the first Q.
        # Split into question and answer
        if 'A.' in part:
            q_content, rest = part.split('A.', 1)
            question = q_content.strip()
            answer = rest.strip()
            
            # Add to pairs
            qa_pairs.append((question, answer))
        else:
            # Question without an answer - use empty string for answer
            question = part.strip()
            qa_pairs.append((question, ""))
    
    return qa_pairs


def process_content(db_path: str, content: str, name:str, author: str, lifestage=str, type=str, filename=str) -> Optional[str]:
    """
    Process content, 
    if type == QA treat as a QA session creating a session and QA pairs in the database.
    
    Args:
        db_path: Path to SQLite database
        content: Content to process
        author : Author of the content
        lifestage : Lifestage of the content
        type : Type of content
        
    Returns:
        Session ID if successful, None otherwise
    """
        
    if type.lower() == "qa":
        # Extract QA pairs
        qa_pairs = extract_qa_pairs(content)

        
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Current timestamp
        timestamp = datetime.utcnow().isoformat()
        
        try:
            session_id=create_session_record(db_path, 
                    name, 
                    f"Processed from {filename}", 
                    filename, 
                    timestamp, 
                    json.dumps({"source": "qa", "qa_count": len(qa_pairs)})
            )

            # Insert QA pairs
            for idx, (question, answer) in enumerate(qa_pairs):
                qa_id = generate_id("qa", session_id, idx)
                
                cursor.execute("""
                INSERT INTO qa_pairs (id, session_id, question, answer, sequence, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    qa_id,
                    session_id,
                    question,
                    answer,
                    idx + 1,  # 1-based sequence
                    timestamp
                ))
            
            # Commit changes
            conn.commit()
            logger.info(f"Successfully processed transcript into session {session_id} with {len(qa_pairs)} Q&A pairs")
        
            return session_id   
        except Exception as e:
            conn.rollback()
            logger.error(f"Error processing transcript: {str(e)}")
            return None
        finally:
            conn.close()

    if type.lower() == "document":
        # we add the path to the analyze queue
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Current timestamp
        timestamp = datetime.utcnow().isoformat()
        
        try:
            #insert record
            file_queue_id=generate_id("file", name, timestamp)
            cursor.execute("""
               INSERT INTO analysis_queue (id, name, author, lifestage, filename, created_at)
               VALUES (?, ?, ?, ?, ?, ?)
               """, (
                   file_queue_id,
                   name,
                   author,
                   lifestage,
                   filename,
                   timestamp
               ))
            conn.commit()
            logger.info(f"Successfully added {filename} to the analyse queue")
            return file
        except Exception as e:
            conn.rollback()
            logger.error(f"Error processing content: {str(e)}")
            return None
        finally:
            conn.close()


def get_session_data(db_path: str, session_id: str) -> Tuple[Dict, List[Dict]]:
    """
    Retrieve session data and QA pairs from the database.
    
    Args:
        db_path: Path to SQLite database
        session_id: Session ID to retrieve
        
    Returns:
        Tuple of (session_data, qa_pairs)
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get session details
    cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
    session_row = cursor.fetchone()

    if not session_row:
        conn.close()
        raise ValueError(f"Session {session_id} not found")

    session = dict(session_row)

    # Get QA pairs
    cursor.execute(
        """
        SELECT * FROM qa_pairs 
        WHERE session_id = ? 
        ORDER BY sequence
        """,
        (session_id,),
    )

    qa_pairs = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return session, qa_pairs


def create_session_context(session: Dict, qa_pairs: List[Dict]) -> str:
    """
    Create a context document with session information and QA pairs.
    
    Args:
        session: Session data
        qa_pairs: List of QA pair data
        
    Returns:
        Formatted context document text
    """
    context = f"# Interview Session: {session['title']}\n\n"

    if session.get("description"):
        context += f"{session['description']}\n\n"

    context += "## Question and Answer Pairs\n\n"

    for qa in qa_pairs:
        context += f"Q{qa['sequence']}. {qa['question']}\n\n"
        context += f"A{qa['sequence']}. {qa['answer']}\n\n"

    return context

def create_session_record(db_path: str, title: str, desciption:str , transcript_file:str, created_at:str, metadata:str) -> str:
    # create a session in the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    session_id = generate_id("session", title)
    try:
        cursor.execute("""
        INSERT INTO sessions (id, title, description, transcript_file, created_at, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            title,
            desciption,
            transcript_file,
            created_at,
            metadata
        ))
        conn.commit()
        return session_id
    except Exception as e:
        conn.rollback()
        logger.error(f"Error creating session: {str(e)}")
        return None
    finally:    
        conn.close()