import sqlite3
import json
import base64
import logging
from typing import Optional, List
from core.config import ConfigSchema
from core.database import (
    get_connection,
    get_filtered_knowledge_records,
    KnowledgeRecord,
)
from core.embedding import get_embedding, cosine_similarity_base64


# Set up logging
logger = logging.getLogger(__name__)


class Query:
    def __init__(self, config: ConfigSchema):
        self.config = config

    def get_kr(
        self,
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
        embedding: bool = False,
    ) -> List[KnowledgeRecord]:
        results = get_filtered_knowledge_records(
            self.config,
            session_id,
            domain,
            confidence,
            lifestage,
            observation_text,
            type,
            author,
            consensus_id,
            qa_id,
            source_id,
            embedding,
        )
        return results

    def get_observations_by_similarity(
        self,
        topic: str,
        threshold: float,
        domain: Optional[List[str]] = None,
        lifestage: Optional[List[str]] = None,
        confidence: Optional[List[str]] = None,
    ) -> list:

        knowledge_records = self.get_kr(
            domain=domain,
            lifestage=lifestage,
            confidence=confidence,
            embedding=True,
        )

        # get the embedding for the topic
        embedding = get_embedding(self.config, topic)
        embedding_base64 = base64.b64encode(embedding.tobytes()).decode("utf-8")    
        similar_observations = []
        # the embedding is a base64 string
        for kr in knowledge_records:
            similarity = cosine_similarity_base64(embedding_base64, kr.embedding)
            if similarity >= threshold:
                similar_observations.append(kr)
        return similar_observations
