"""
Content loading functionality for TheInterview project.
Provides tools for loading various content types into the knowledge database.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .config import ConfigSchema
from .database import get_connection
from .utils import generate_id, resolve_file_patterns

# Set up logging
logger = logging.getLogger(__name__)


class Loader:
    """Class for loading content into the knowledge database."""

    def __init__(self, config: ConfigSchema):
        """Initialize with configuration."""
        self.config = config
        self.db_path = config.database.path

    def load_file(
        self,
        file_path: str,
        name: str,
        author: str,
        description: str,
        lifestage: str,
        priority: int = 0,
    ) -> Optional[str]:
        """
        Load a single file into the database and queue for analysis.

        Args:
            file_path: Path to the content file
            name: Name for the content, title or subject
            author: Author of the content
            description: Description of the content
            lifestage: Life stage classification
            priority: Queue priority

        Returns:
            Source ID if successful, None otherwise
        """
        # Check for duplicate sources first
        existing_source_id = self.check_and_update_duplicate_source(
            file_path, name, description
        )

        if existing_source_id:
            # Source already exists and was updated
            # Now create a new queue entry for it
            queue_id = self.create_queue_entry(
                existing_source_id, author, lifestage, priority
            )
            return existing_source_id

        try:
            # Generate IDs
            timestamp = datetime.utcnow().isoformat()
            source_id = generate_id("source", name, timestamp)

            # Connect to database
            conn, cursor = get_connection(self.db_path)

            try:
                # Store source record only
                cursor.execute(
                    """
                    INSERT INTO sources 
                    (id, created_at, description, content_path, title) 
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        source_id,
                        timestamp,
                        description,
                        str(Path(file_path).absolute()),
                        name,
                    ),
                )
                logger.info(f"Successfully created source record: {source_id}")

                # Queue for analysis
                queue_id = self.create_queue_entry(
                    source_id, author, lifestage, priority, status="pending"
                )



                logger.info(
                    f"Successfully queued content into analysis queue: {queue_id}"
                )
                return source_id, queue_id

            except Exception as e:
                conn.rollback()
                logger.error(f"Database error: {str(e)}")
                return None
            finally:
                conn.close()

        except Exception as e:
            logger.error(f"Error loading content: {str(e)}")
            return None

    def create_queue_entry(
        self,
        source_id: str,
        author: str,
        lifestage: str,
        priority: int,
        status: str = "pending",
    ) -> str:
        """
        Create a new queue entry for a source.

        Args:
            source_id: Source ID
            author: Author name
            lifestage: Life stage
            priority: Queue priority
            status: Queue status

        Returns:
            Queue ID
        """
        conn, cursor = get_connection(self.db_path)

        try:
            queue_id = generate_id("queue", source_id, datetime.utcnow().isoformat())
            cursor.execute(
                """
                INSERT INTO analysis_queue
                (id, author, lifestage, type, priority, status, created_at, source_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    queue_id,
                    author,
                    lifestage,
                    "document",  # All content is treated as document now
                    priority,
                    status,
                    datetime.utcnow().isoformat(),
                    source_id,
                ),                
            )
            conn.commit()
            return queue_id
        except Exception as e:
            conn.rollback()
            logger.error(f"Error creating queue entry: {str(e)}")
            return None
        finally:
            conn.close()  



    def check_and_update_duplicate_source(
        self, file_path: str, title: str, description: str
    ) -> Optional[str]:
        """
        Check if a source with the same file path already exists.
        If it exists, update it and handle any pending queue items.

        Args:
            file_path: Path to the content file
            title: New title for the source
            description: New description for the source

        Returns:
            Existing source_id if found, None otherwise
        """
        normalized_path = str(Path(file_path).absolute())
        conn, cursor = get_connection(self.db_path)

        try:
            # Check if source with same path exists
            cursor.execute(
                "SELECT id FROM sources WHERE content_path = ?", (normalized_path,)
            )
            existing = cursor.fetchone()

            if not existing:
                return None  # No duplicate found

            source_id = existing[0]
            logger.info(f"Found duplicate source: {source_id}")
            # Update existing source
            cursor.execute(
                """
                UPDATE sources
                SET title = ?, description = ?
                WHERE id = ?
                """,
                (title, description, source_id),
            )

            # Find pending queue items for this source
            cursor.execute(
                """
                SELECT id FROM analysis_queue
                WHERE source_id = ? AND status = 'pending'
                """,
                (source_id,),
            )
            pending_items = cursor.fetchall()

            if pending_items:
                # Cancel pending queue items
                logger.info(f"Found {len(pending_items)} pending queue items")
                cursor.execute(
                    """
                    UPDATE analysis_queue
                    SET status = 'cancelled'
                    WHERE source_id = ? AND status = 'pending'
                    """,
                    (source_id,),
                )
                logger.info(
                    f"Cancelled {len(pending_items)} pending queue items for source {source_id}"
                )

            conn.commit()
            return source_id

        except Exception as e:
            conn.rollback()
            logger.error(f"Error checking duplicate source: {str(e)}")
            return None
        finally:
            conn.close()

    def batch_load(
        self,
        file_patterns: List[str],
        name_template: Optional[str] = None,
        author: str = None,
        description: Optional[str] = None,
        lifestage: str = "AUTO",
        priority: int = 0,
    ) -> Dict[str, Any]:
        """
        Load multiple files matching patterns into the knowledge database.

        Args:
            file_patterns: List of glob patterns to match files
            name_template: Template for naming loaded content (use {index} for batch numbering)
            author: Author of the content (defaults to config.content.default_author)
            description: Description to apply to all content
            lifestage: Life stage classification
            priority: Queue priority

        Returns:
            Dictionary with results (matched_files, loaded_files, source_ids)
        """
        # Resolve file patterns
        matched_files = resolve_file_patterns(file_patterns)
        if not matched_files:
            logger.warning("No files matched the provided patterns")
            return {"matched_files": [], "loaded_files": [], "source_ids": []}

        # Load each file
        loaded_files = []
        source_ids = []

        for idx, file_path in enumerate(matched_files, 1):
            # Generate name if template provided
            item_name = (
                name_template.format(index=idx)
                if name_template and "{index}" in name_template
                else name_template or Path(file_path).stem
            )

            # Generate description if not provided
            item_description = description or f"Content from {Path(file_path).name}"

            # Load the file
            source_id = self.load_file(
                file_path=file_path,
                name=item_name,
                author=author,
                description=item_description,
                lifestage=lifestage,
                priority=priority,
            )

            if source_id:
                loaded_files.append(file_path)
                source_ids.append(source_id)

        return {
            "matched_files": matched_files,
            "loaded_files": loaded_files,
            "source_ids": source_ids,
        }

    def get_queue_items(
        self, status: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get items from the analysis queue.

        Args:
            status: Filter by status (e.g., 'pending', 'processing', 'completed')
            limit: Maximum number of items to retrieve
            sort_by: Field to sort by ('priority', 'created_at')

        Returns:
            List of queue items as dictionaries
        """
        conn, cursor = get_connection(self.db_path)

        try:
            query = """
                SELECT aq.id, aq.author, aq.lifestage, aq.type, 
                       aq.created_at, aq.status, aq.priority, 
                       s.id as source_id, s.description, s.content_path, s.title
                FROM analysis_queue aq
                JOIN sources s ON source_id = s.id
            """

            params = []
            if status:
                query += " WHERE aq.status = ?"
                params.append(status)
            query += " ORDER BY aq.priority DESC, aq.created_at ASC"
            query += f" LIMIT {limit}"

            cursor.execute(query, params)

            # Convert to dictionaries
            items = []
            for row in cursor.fetchall():
                item = dict(row)
                items.append(item)

            return items

        finally:
            conn.close()

    def create_context_from_source(
        self, source_id: str, author: str, lifestage: str
    ) -> str:
        """
        Create a context document for a source, including description and content.
        This is used during the processing phase, not during loading.

        Args:
            source_id: Source ID to create context for
            author: Author to include in context
            lifestage: Life stage to include in context

        Returns:
            Formatted context document text
        """
        conn, cursor = get_connection(self.db_path)

        try:
            # Get source data
            cursor.execute(
                """
                SELECT id, name, description, content_path
                FROM sources
                WHERE id = ?
                """,
                (source_id,),
            )

            source = dict(cursor.fetchone())

            # Read the content file
            content_path = source.get("content_path")
            if not content_path or not Path(content_path).exists():
                raise FileNotFoundError(f"Content file not found: {content_path}")

            content = Path(content_path).read_text(encoding="utf-8")

            # Format the context
            context = f"# Source: {source['name']}\n\n"

            # Add description
            description = source.get("description", "")
            if description:
                context += f"{description}\n\n"

            # Include lifestage
            context += f"Life Stage: {lifestage}\n\n"

            # Include author
            context += f"Author: {author}\n\n"

            # Add content type hint if it's a QA transcript
            if "qa" in description.lower() or "interview" in description.lower():
                context += "Note: The following content is a Q&A transcript. Answers represent direct statements from the subject.\n\n"

            # Add the content
            context += content

            return context

        finally:
            conn.close()

    def process_queue_item(self, queue_id: str) -> Optional[str]:
        """
        Process a queue item - this method would create a session record.
        This is a placeholder for the actual processing logic that would
        be implemented in the AnalysisManager.

        Args:
            queue_id: ID of the queue item to process

        Returns:
            Session ID if successfully processed
        """
        # This would be implemented in the AnalysisManager
        # Here it's just a placeholder to show the flow
        pass
