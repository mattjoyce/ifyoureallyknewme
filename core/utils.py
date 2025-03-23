from datetime import datetime, timezone
import hashlib
import json
import re
import os
import glob
from typing import Dict, List, Union

def timestamp() -> str:
    # Get the current time in UTC
    current_time_utc = datetime.now(timezone.utc)

    # Format as ISO 8601
    iso_timestamp = current_time_utc.isoformat()
    return iso_timestamp

def resolve_file_patterns(file_patterns, recursive=True):
    """
    Resolve glob patterns to a list of unique file paths.
    Returns normalized absolute paths.
    """
    matched_files = []
    for pattern in file_patterns:
        # Convert to absolute path if it's not already
        if not os.path.isabs(pattern):
            pattern = os.path.join(os.getcwd(), pattern)
        
        # Normalize the path to remove unnecessary components like './'
        pattern = os.path.normpath(pattern)
        
        # Expand the glob pattern
        files = glob.glob(pattern, recursive=recursive)
        matched_files.extend(files)
    
    # Remove duplicates and sort
    return sorted(set(matched_files))

def generate_id(prefix: str, *args: object) -> str:
    """
    Generate a unique ID with a prefix.
    
    Args:
        prefix: Prefix for the ID
        *args: Additional arguments to include in the hash
        
    Returns:
        Unique ID string
    """
    hash_input = (
        "-".join(str(arg) for arg in args) + "-" + datetime.utcnow().isoformat()
    )
    id_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:12]
    return f"{prefix}_{id_hash}"


def calculate_file_hash(filepath: str) -> str:
    """
    Calculate SHA-256 hash of a file.
    
    Args:
        filepath: Path to the file to hash
        
    Returns:
        SHA-256 hash of the file
    """
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def extract_json(content: str) -> Union[Dict, List]:
    """
    Extract and validate JSON from potentially markdown-wrapped content.
    
    Args:
        content: String content that may contain JSON objects or arrays
        
    Returns:
        Parsed JSON object (dict or list)
        
    Raises:
        ValueError: If no valid JSON is found in the content
    """
    # Try to find JSON-like patterns
    json_patterns = [
        r"\{[\s\S]*\}",  # Dictionary pattern
        r"\[[\s\S]*\]"   # Array pattern
    ]
    
    for pattern in json_patterns:
        matches = re.finditer(pattern, content)
        for match in matches:
            try:
                # Try to parse each potential JSON match
                potential_json = match.group()
                parsed = json.loads(potential_json)
                return parsed
            except json.JSONDecodeError:
                continue
    
    raise ValueError("No valid JSON found in content")

