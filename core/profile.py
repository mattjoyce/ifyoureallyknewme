"""
Profile generation module for TheInterview project.

This module provides functionality for generating personal profiles
from knowledge records and consensus statements stored in the database.
"""

import json
import logging
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .config import ConfigSchema
from .database import get_connection
from .llm_calling import llm_process_with_prompt, get_role_prompt
from .utils import generate_id

# Set up logging
logger = logging.getLogger(__name__)

LIFE_STAGES = [
    "CHILDHOOD",
    "ADOLESCENCE",
    "EARLY_ADULTHOOD",
    "EARLY_CAREER",
    "MID_CAREER",
    "LATE_CAREER"
]

DOMAINS = [
    "Personal History",
    "Professional Evolution",
    "Psychological and Behavioral Evolution",
    "Relationships and Networks",
    "Community and Ideological Engagement",
    "Daily Routines and Health",
    "Values, Beliefs, and Goals",
    "Active Projects and Learning"
]

class ProfileGenerator:
    """Class for generating personal profiles from knowledge records."""

    def __init__(self, config: ConfigSchema):
        """Initialize with configuration."""
        self.config = config
        self.db_path = config.database.path

    def dump_observations(self) -> str:
        """
        Dump all observations from the database.
        
        Returns:
            String of observations
        """
        conn, cursor = get_connection(self.db_path)
        query="SELECT content, created_at FROM knowledge_records WHERE consensus_id IS NULL ORDER BY created_at"
        cursor.execute(query)
        results = cursor.fetchall()
        records = []
        for row in results:
            try:
                content = json.loads(row[0])
                record = {
                    'observation': content.get('observation', ''),
                    'domain': content.get('domain', ''),
                    'confidence': content.get('confidence', ''),
                }
                records.append(record)
            except json.JSONDecodeError:
                print(f"Error decoding JSON: {row[1]}")
        conn.close()
        output = ""    
        for record in records:
            output+=f"{record['observation']}\n"
        return output



    def get_consensus_records(self) -> List[Dict[str, Any]]:
        """
        Retrieve all consensus records from the database.
        
        Returns:
            List of consensus records as dictionaries
        """
        conn, cursor = get_connection(self.db_path)
        
        try:
            cursor.execute("""
                SELECT id, content, created_at
                FROM knowledge_records
                WHERE type = 'consensus'
                ORDER BY created_at
            """)
            
            records = []
            for row in cursor.fetchall():
                record = {
                    'id': row[0],
                    'content': json.loads(row[1]),
                    'created_at': row[2]
                }
                records.append(record)
                
            return records
        finally:
            conn.close()

    def get_knowledge_records(self, record_type: str = 'note') -> List[Dict[str, Any]]:
        """
        Retrieve knowledge records that aren't part of consensus.
        
        Args:
            record_type: Type of records to retrieve (default: note)
            
        Returns:
            List of knowledge records as dictionaries
        """
        conn, cursor = get_connection(self.db_path)
        
        try:
            cursor.execute("""
                SELECT id, content, author, created_at
                FROM knowledge_records
                WHERE type = ? AND consensus_id IS NULL
                ORDER BY created_at
            """, (record_type,))
            
            records = []
            for row in cursor.fetchall():
                record = {
                    'id': row[0],
                    'content': json.loads(row[1]),
                    'author': row[2],
                    'created_at': row[3]
                }
                records.append(record)
                
            return records
        finally:
            conn.close()

    def organize_records_by_lifestage_and_domain(self, records: List[Dict[str, Any]]) -> Dict[str, Dict[str, List]]:
        """
        Organize records by life stage and domain.
        
        Args:
            records: List of records to organize
            
        Returns:
            Nested dictionary with life stages as keys, domains as sub-keys, and lists of records as values
        """
        organized = defaultdict(lambda: defaultdict(list))
        
        for record in records:
            content = record['content']
            
            # Extract life stage and domain from content
            life_stage = content.get('life_stage', 'NOT_KNOWN')
            
            if 'domain' in content:
                # Single domain record
                domain = content['domain']
                organized[life_stage][domain].append(record)
            elif 'domains' in content:
                # Multi-domain record
                for domain in content['domains']:
                    organized[life_stage][domain].append(record)
            else:
                # Domain not specified
                organized[life_stage]['Unclassified'].append(record)
                
        return organized

    def generate_profile(self, format_type: str = 'md', mode: str = 'short') -> str:
        """
        Generate a profile from knowledge records and consensus statements.
        
        Args:
            format_type: Output format (md, json, raw)
            mode: Profile length mode (short, long)
            
        Returns:
            Formatted profile text
        """
        # Get records
        consensus_records = self.get_consensus_records()
        knowledge_records = self.get_knowledge_records()
        
        # Combine records
        all_records = consensus_records + knowledge_records
        
        # For raw format, we'll display the records directly without synthesis regardless of mode
        if format_type == 'raw':
            logger.info(f"Using raw format, skipping LLM synthesis, formatting {len(all_records)} records.")
            return self._format_as_raw(all_records, 'text' if format_type == 'raw' else format_type)
            
        # For other formats, organize by life stage and domain
        organized = self.organize_records_by_lifestage_and_domain(all_records)
        
        # Generate profile using LLM
        profile_content = self._generate_profile_content(organized, mode)
        
        # Format the profile
        if format_type == 'json':
            return json.dumps(profile_content, indent=2)
        else:
            # Default to markdown
            return self._format_as_markdown(profile_content)

    def _generate_profile_content(self, organized_records: Dict[str, Dict[str, List]], mode: str) -> Dict[str, Any]:
        """
        Generate profile content using LLM.
        
        Args:
            organized_records: Records organized by life stage and domain
            mode: Profile length mode (short, long)
            
        Returns:
            Dictionary with profile content
        """
        MODEL_NAME = self.config.llm.generative_model
        
        # Create a prompt for the LLM
        # TODO: Create a ProfileGenerator role and use get_role_prompt
        prompt = """You are an expert biographer tasked with synthesizing information into a coherent profile.
        
You will be provided with observations and insights about a person organized by life stage and domain.
Your task is to create a cohesive biographical narrative that captures the essence of the person.

Please focus on identifying patterns, key influences, and transformative experiences.
Maintain a professional, objective tone while crafting an engaging narrative.

For each life stage, create a section that synthesizes the information across domains.
Include a brief summary of the entire profile at the beginning.

Respond with a JSON structure as follows:
{
  "summary": "A brief overview of the person",
  "life_stages": [
    {
      "stage": "CHILDHOOD",
      "narrative": "Synthesized narrative for this life stage"
    },
    ...
  ]
}
"""
        
        # Create input data for LLM
        input_data = {
            "mode": mode,
            "life_stages": {}
        }
        
        # Convert organized records to a format suitable for LLM
        for life_stage, domains in organized_records.items():
            input_data["life_stages"][life_stage] = {}
            for domain, records in domains.items():
                input_data["life_stages"][life_stage][domain] = [
                    {
                        "observation": r["content"].get("observation", ""),
                        "confidence": r["content"].get("confidence", "MODERATE")
                    }
                    for r in records
                ]
        
        # Call LLM to generate profile
        try:
            profile_content = llm_process_with_prompt(
                self.config,
                input_content=input_data,
                prompt=prompt,
                model=MODEL_NAME,
                expect_json=True
            )
            
            return profile_content
        except Exception as e:
            logger.error(f"Error generating profile: {str(e)}")
            return {
                "summary": "Error generating profile.",
                "life_stages": []
            }
    
    def _format_as_markdown(self, profile_content: Dict[str, Any]) -> str:
        """Format profile content as Markdown."""
        md = f"# Personal Profile\n\n## Summary\n\n{profile_content.get('summary', '')}\n\n"
        
        for stage in profile_content.get('life_stages', []):
            md += f"## {stage['stage'].replace('_', ' ').title()}\n\n"
            md += f"{stage['narrative']}\n\n"
            
        return md
        
    def _format_as_raw(self, records: List[Dict[str, Any]], format_type: str) -> str:
        """
        Format records in a token-efficient manner, grouped by expert and confidence.
        
        Args:
            records: List of knowledge records
            format_type: Output format (md, text, html, json)
            
        Returns:
            Formatted output
        """
        logger.info(f"Formatting {len(records)} records in raw mode.")
        
        # Extract and prepare records for sorting and grouping
        processed_records = []
        for record in records:
            content = record['content']
            
            # Extract relevant fields - ensure we have defaults for all fields
            life_stage = content.get('life_stage', 'NOT_KNOWN')
            confidence = content.get('confidence', 'MODERATE')
            observation = content.get('observation', '')
            
            # Get author (for knowledge records) or "Consensus" for consensus records
            author = record.get('author', 'Consensus') if 'author' in record else 'Consensus'
            
            # Get life stage order for sorting
            life_stage_order = LIFE_STAGES.index(life_stage) if life_stage in LIFE_STAGES else 999
            
            # Get confidence order for sorting
            confidence_order = {
                'VERY_HIGH': 0,
                'HIGH': 1,
                'MODERATE': 2,
                'LOW': 3,
                'VERY_LOW': 4
            }.get(confidence, 5)
            
            # Create a sort key and record data
            sort_key = (life_stage_order, author, confidence_order)
            record_data = {
                'life_stage': life_stage,
                'author': author,
                'confidence': confidence,
                'observation': observation,
                'id': record.get('id', 'unknown')
            }
            processed_records.append((sort_key, record_data))
        
        # Sort the records
        processed_records.sort(key=lambda x: x[0])
        
        # For JSON format, just return as is
        if format_type == 'json':
            raw_records = [item[1] for item in processed_records]
            return json.dumps(raw_records, indent=2)
        
        # Group records by life stage, expert, and confidence for other formats
        grouped_records = {}
        for _, record in processed_records:
            life_stage = record['life_stage']
            author = record['author']
            confidence = record['confidence']
            
            if life_stage not in grouped_records:
                grouped_records[life_stage] = {}
            
            if author not in grouped_records[life_stage]:
                grouped_records[life_stage][author] = {}
            
            if confidence not in grouped_records[life_stage][author]:
                grouped_records[life_stage][author][confidence] = []
            
            grouped_records[life_stage][author][confidence].append({
                'observation': record['observation'],
                'id': record['id']
            })
        
        # Format based on requested output type
        if format_type == 'text':
            output = "# Raw Knowledge Records\n\n"
            
            # Get ordered life stages
            ordered_life_stages = sorted(
                grouped_records.keys(),
                key=lambda ls: LIFE_STAGES.index(ls) if ls in LIFE_STAGES else 999
            )
            
            # Order confidences
            confidence_order = ['VERY_HIGH', 'HIGH', 'MODERATE', 'LOW', 'VERY_LOW']
            
            for life_stage in ordered_life_stages:
                output += f"## {life_stage}\n"
                
                for author in sorted(grouped_records[life_stage].keys()):
                    output += f"### {author}\n"
                    
                    for confidence in sorted(grouped_records[life_stage][author].keys(), 
                                            key=lambda c: confidence_order.index(c) if c in confidence_order else 999):
                        output += f"#### {confidence}\n"
                        
                        for item in grouped_records[life_stage][author][confidence]:
                            output += f"- {item['observation']} [{item['id']}]\n"
                        
                        output += "\n"
                
                output += "\n"
            
            return output
        
        # Default to markdown
        else:
            md = "# Raw Knowledge Records\n\n"
            
            # Get ordered life stages
            ordered_life_stages = sorted(
                grouped_records.keys(),
                key=lambda ls: LIFE_STAGES.index(ls) if ls in LIFE_STAGES else 999
            )
            
            # Order confidences
            confidence_order = ['VERY_HIGH', 'HIGH', 'MODERATE', 'LOW', 'VERY_LOW']
            
            for life_stage in ordered_life_stages:
                md += f"## {life_stage.replace('_', ' ').title()}\n\n"
                
                for author in sorted(grouped_records[life_stage].keys()):
                    md += f"### {author}\n\n"
                    
                    for confidence in sorted(grouped_records[life_stage][author].keys(), 
                                            key=lambda c: confidence_order.index(c) if c in confidence_order else 999):
                        md += f"#### {confidence}\n\n"
                        
                        for item in grouped_records[life_stage][author][confidence]:
                            md += f"- {item['observation']} [*{item['id']}*]\n"
                        
                        md += "\n"
                
                md += "\n"
            
            return md