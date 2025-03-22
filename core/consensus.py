"""
Consensus management functionality for TheInterview project.
Provides tools for finding clusters of similar notes and generating consensus records.
"""

import sqlite3
import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Scipy imports for hierarchical clustering
from scipy.cluster.hierarchy import fcluster, linkage

from .config import ConfigSchema
from .embedding import compute_pairwise_similarities, get_embedding
from .database import get_connection, store_knowledge_record
from .llm_calling import llm_process_with_prompt, get_role_prompt, extract_keywords
from .utils import generate_id

# Get logger
logger = logging.getLogger(__name__)


class ConsensusManager:
    """Manager class for handling clustering and consensus operations."""

    def __init__(self, config: ConfigSchema):
        """Initialize ConsensusManager.

        Args:
            config: Application configuration
            db_path: Path to the database. If None, uses the path from config.
        """
        self.config = config

    def load_records(self, record_type: str = "note") -> List[Dict[str, Any]]:
        """Load all records with embeddings that haven't been assigned to consensus yet.

        Args:
            record_type: Type of records to load (note, fact, etc.)

        Returns:
            List of record dictionaries
        """
        DB_PATH = self.config.database.path

        conn, cursor = get_connection(DB_PATH)

        try:
            query = """
            SELECT id, content, embedding, author, type, session_id
            FROM knowledge_records 
            WHERE embedding IS NOT NULL AND consensus_id IS NULL
            """

            params = []
            if record_type:
                query += " AND type = ?"
                params.append(record_type)

            cursor.execute(query, params)

            records = []
            for (
                id,
                content_json,
                embedding_blob,
                author,
                record_type,
                record_session_id,
            ) in cursor.fetchall():
                if embedding_blob:
                    embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                    content = json.loads(content_json)
                    records.append(
                        {
                            "id": id,
                            "observation": content["observation"],
                            "domain": content.get("domain"),
                            "life_stage": content.get("life_stage"),
                            "confidence": content.get("confidence"),
                            "author": author,
                            "type": record_type,
                            "session_id": record_session_id,
                            "embedding": embedding,
                        }
                    )
            return records

        finally:
            conn.close()

    def find_similar_consensus(self, threshold: float = 0.92) -> Dict[str, Any]:
        """Find clusters of similar consensus records that should be reconsidered.
        
        Args:
            threshold: Similarity threshold (0-1) - higher than normal clustering
                
        Returns:
            Dictionary of clusters with metadata
        """
        # Use the existing find_clusters method but specify consensus type
        consensus_clusters = self.find_clusters(threshold, record_type="consensus")
        
        if not consensus_clusters:
            return {}
        
        # Process each cluster to collect IDs of consensus records to reset
        records_to_reset = {}
        
        for cluster_id, cluster in consensus_clusters.items():
            if len(cluster["notes"]) > 1:  # Only consider clusters with multiple consensus records
                # Collect IDs and related details
                consensus_ids = [note["id"] for note in cluster["notes"]]
                records_to_reset[cluster_id] = {
                    "consensus_ids": consensus_ids,
                    "observation_count": len(consensus_ids),
                    "average_similarity": cluster["average_similarity"],
                    "observations": [note["observation"] for note in cluster["notes"]]
                }
        
        return records_to_reset


    def reset_consensus_clusters(self, consensus_clusters: Dict[str, Any]) -> int:
        """Reset similar consensus records by unlinking their source records.
        
        Args:
            consensus_clusters: Result of find_similar_consensus()
            
        Returns:
            Number of consensus records reset
        """
        if not consensus_clusters:
            return 0
        
        DB_PATH = self.config.database.path
        conn, cursor = get_connection(DB_PATH)
        
        try:
            reset_count = 0
            
            for cluster_id, cluster in consensus_clusters.items():
                consensus_ids = cluster["consensus_ids"]
                
                # Format for SQL query
                id_placeholders = ','.join(['?'] * len(consensus_ids))
                
                # First get all source records linked to these consensus records
                cursor.execute(f"""
                    SELECT id FROM knowledge_records
                    WHERE consensus_id IN ({id_placeholders})
                """, consensus_ids)
                
                source_ids = [row[0] for row in cursor.fetchall()]
                logger.info(f"Found {len(source_ids)} source records linked to {len(consensus_ids)} consensus records")
                
                # Reset consensus_id for all linked records
                if source_ids:
                    cursor.execute(f"""
                        UPDATE knowledge_records
                        SET consensus_id = NULL
                        WHERE id IN ({','.join(['?'] * len(source_ids))})
                    """, source_ids)
                
                # Delete the consensus records
                cursor.execute(f"""
                    DELETE FROM knowledge_records
                    WHERE id IN ({id_placeholders})
                """, consensus_ids)
                
                reset_count += len(consensus_ids)
            
            conn.commit()
            return reset_count
            
        finally:
            conn.close()

    def find_clusters(
        self, threshold: float = 0.85, record_type: str = "note"
    ) -> Dict[str, Any]:
        """Find clusters of similar records using hierarchical clustering.

        Args:
            threshold: Similarity threshold (0-1)
            record_type: Type of records to cluster

        Returns:
            Dictionary of clusters with metadata
        """
        records = self.load_records(record_type)
        if not records:
            return {}

        # Group records by life stage
        life_stage_groups: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            life_stage = record["life_stage"]
            if life_stage not in life_stage_groups:
                life_stage_groups[life_stage] = []
            life_stage_groups[life_stage].append(record)

        all_clusters = {}

        # Process each life stage separately
        for life_stage, stage_records in life_stage_groups.items():
            if len(stage_records) < 2:  # Need at least 2 records to cluster
                continue

            # Create embedding matrix for this life stage
            embeddings = np.vstack([r["embedding"] for r in stage_records])

            # Compute similarities
            similarities = compute_pairwise_similarities(
                [r["embedding"] for r in stage_records]
            )

            # Compute linkage matrix
            linkage_matrix = linkage(embeddings, method="complete", metric="cosine")

            # Form clusters
            distance_threshold = 1 - threshold
            cluster_labels = fcluster(
                linkage_matrix, t=distance_threshold, criterion="distance"
            )

            # Group records by cluster with original indices
            stage_clusters: Dict[int, Dict[str, List]] = {}
            for idx, cluster_id in enumerate(cluster_labels):
                if cluster_id not in stage_clusters:
                    stage_clusters[cluster_id] = {"records": [], "indices": []}
                record = stage_records[idx]
                # Remove embedding from record for JSON output
                record_copy = {k: v for k, v in record.items() if k != "embedding"}
                stage_clusters[cluster_id]["records"].append(record_copy)
                stage_clusters[cluster_id]["indices"].append(idx)

            # Process each cluster
            for cluster_id, cluster_data in stage_clusters.items():
                if (
                    len(cluster_data["records"]) > 1
                ):  # Only process clusters with multiple records
                    cluster_indices = cluster_data["indices"]
                    cluster_similarities = similarities[
                        np.ix_(cluster_indices, cluster_indices)
                    ]
                    avg_similarity = (
                        np.sum(cluster_similarities) - len(cluster_indices)
                    ) / (len(cluster_indices) * (len(cluster_indices) - 1))

                    cluster_key = f"{life_stage}_cluster_{cluster_id}"
                    all_clusters[cluster_key] = {
                        "life_stage": life_stage,
                        "notes": cluster_data["records"],
                        "average_similarity": float(avg_similarity),
                        "unique_domains": list(
                            set(
                                r["domain"]
                                for r in cluster_data["records"]
                                if r["domain"]
                            )
                        ),
                        "unique_authors": list(
                            set(r["author"] for r in cluster_data["records"])
                        ),
                    }

        return all_clusters

    def save_consensus_record(
        self, consensus: dict, author: str, timestamp: Optional[str] = None
    ) -> str:
        """Save a consensus record to the database and link source records.

        Args:
            consensus: Consensus dictionary with observation, life_stage,
                      confidence and source_records
            author: Author name for the consensus
            timestamp: Timestamp string (will generate if None)

        Returns:
            ID of the created consensus record
        """

        DB_PATH = self.config.database.path

        # Generate timestamp if not provided
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()

        embedding_bytes = get_embedding(self.config, consensus["observation"])
        keywords = extract_keywords(self.config, consensus["observation"])

        # Connect to database
        conn, cursor = get_connection(DB_PATH)

        try:
            # Serialize consensus and generate ID
            note_id = generate_id("consensus", author, timestamp)

            store_knowledge_record(
                self.config,
                note_id,
                "consensus",
                author,
                consensus,
                timestamp,
                embedding_bytes,
                keywords=keywords,
            )

            # Link source records to this consensus
            source_records = consensus.get("source_records", [])
            if source_records:
                placeholders = ",".join(["?"] * len(source_records))
                cursor.execute(
                    f"""
                UPDATE knowledge_records
                SET consensus_id = ?
                WHERE id IN ({placeholders})
                """,
                    [note_id] + source_records,
                )

            conn.commit()
            return note_id

        finally:
            conn.close()

    def make_consensus(
        self, cluster: dict, author: str = "ConsensusMaker"
    ) -> Dict[str, Any]:
        """Generate consensus from a cluster using LLM and save to database.

        Args:
            cluster: Cluster dictionary
            author: Author name for the consensus

        Returns:
            List of created consensus record IDs
        """

        MODEL_NAME = self.config.llm.generative_model

        # Skip clusters with fewer than 2 notes
        if len(cluster["notes"]) < 2:
            logger.warning(f"Skipping cluster: Only {len(cluster['notes'])} notes")
            return []

        # Generate consensus using LLM
        try:
            prompt = get_role_prompt(self.config, "ConsensusMaker", "Helper")

            # Process cluster with LLM
            consensus_record = llm_process_with_prompt(
                self.config, cluster, prompt, MODEL_NAME
            )

            logger.info(f"Consensus record: {consensus_record}")

            return consensus_record

        except Exception as e:
            logger.error(f"Error generating consensus: {str(e)}")
            return []


    def process_clusters(
        self,
        threshold: float = 0.85,
        kr_type: str = "note",
        author: str = "ConsensusMaker",
    ) -> Dict[str, Any]:
        """Find clusters and generate consensus in one operation."""

        # Find clusters
        clusters = self.find_clusters(threshold, kr_type)

        if not clusters:
            return {"clusters": {}, "consensus_count": 0, "cluster_count": 0}

        # Process consensus if not in dryrun mode
        consensus_ids = []  # This will store actual IDs as separate items

        for cluster_id, cluster in clusters.items():
            consensus = self.make_consensus(cluster, author)

            # Skip if no consensus was generated
            if not consensus:
                logger.warning(f"No consensus generated for cluster {cluster_id}")
                continue

            timestamp = datetime.utcnow().isoformat()

            if not self.config.dryrun:
                record_id = self.save_consensus_record(consensus, author, timestamp)
            else:
                record_id = generate_id("DRYRUN", timestamp)

            # Append the ID as a single item, not extend it as characters
            consensus_ids.append(record_id)

            # Add consensus ID to cluster data for reporting
            cluster["consensus_ids"] = record_id

        return {
            "clusters": clusters,
            "consensus_count": len(consensus_ids),  # Now correctly counts IDs
            "cluster_count": len(clusters),
        }

    def list_clusters(
        self, threshold: float = 0.85, kr_type: str = "note"
    ) -> Dict[str, Any]:
        """Find and list clusters without generating consensus.

        Args:
            threshold: Similarity threshold (0-1)
            record_type: Type of records to cluster

        Returns:
            Dictionary with cluster information for display
        """
        clusters = self.find_clusters(threshold, kr_type)

        # Format and count for display
        result = {
            "cluster_count": len(clusters),
            "record_count": sum(len(c["notes"]) for c in clusters.values()),
            "clusters": {},
        }

        # Format each cluster for easier display
        for cluster_id, cluster in clusters.items():
            result["clusters"][cluster_id] = {
                "life_stage": cluster["life_stage"],
                "observation_count": len(cluster["notes"]),
                "average_similarity": cluster["average_similarity"],
                "domains": cluster["unique_domains"],
                "authors": cluster["unique_authors"],
                "observations": [note["observation"] for note in cluster["notes"]],
            }

        return result
