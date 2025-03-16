"""
LLM utility functions for TheInterview project.
Provides abstracted interface for LLM operations to make it easy to switch providers.
"""

import subprocess
import tempfile
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .config import get_config


def call_llm_with_prompt(
    input_content: Union[str, Dict, List], 
    prompt: str, 
    model: str,
    json_output: bool = True
) -> str:
    """
    Call LLM with a role prompt and input content.
    
    Args:
        input_content: Input text or JSON object to process
        role_name: Name of the role to use
        model: Model to use with the LLM
        json_output: Whether to expect JSON output
        
    Returns:
        String response from the LLM
    
    Raises:
        ValueError: If role file doesn't exist
        subprocess.CalledProcessError: If LLM call fails
    """
    
    # Convert input to string if it's JSON
    if isinstance(input_content, (dict, list)):
        import json
        input_str = json.dumps(input_content)
    else:
        input_str = str(input_content)
    

    result = None
    # Currently using fabric, but this could be replaced with any LLM client
    #save the composed prompt to a temp file
    with tempfile.NamedTemporaryFile(mode='w', delete=True) as f:
        f.write(prompt)
        prompt_file=Path(f.name).resolve()
        print(prompt_file)
        result = subprocess.run(
            ["fabric", "--model", model, "-p", str(prompt_file)],
            input=input_str,
            capture_output=True,
            text=True,
            check=True
        )
    print(result.stdout)
    # Return the raw output
    return result.stdout

def parse_llm_response(response: str, expect_json: bool = True) -> Any:
    """
    Parse the LLM response, handling JSON if needed.
    
    Args:
        response: Raw text response from LLM
        expect_json: Whether to parse as JSON
        
    Returns:
        Parsed JSON object or raw string
    
    Raises:
        json.JSONDecodeError: If JSON parsing fails
    """
    if not expect_json:
        return response
    
    import json
    try:
        result = json.loads(response)
        return result
    except json.JSONDecodeError:
        # Try to extract JSON from text if it's embedded
        import re
        json_pattern = r'(\{[\s\S]*\}|\[[\s\S]*\])'
        match = re.search(json_pattern, response)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # If all parsing fails, raise the original error
        raise json.JSONDecodeError(f"Failed to parse JSON from LLM response", response, 0)

def llm_process_with_prompt(
    input_content: Union[str, Dict, List],
    prompt: str,
    model: str,
    expect_json: bool = True
) -> Any:
    """
    Process input with a role and return parsed results.
    Convenience function combining call_llm_with_role and parse_llm_response.
    
    Args:
        input_content: Input text or JSON to process
        role_file: Path to the role prompt file
        model: Model to use with the LLM
        expect_json: Whether to parse output as JSON
        
    Returns:
        Processed result (parsed JSON or raw string)
    """
    config=get_config()
    # Use model from config if not provided
    model = model or config.llm.generative_model

    # Call LLM with role and get raw response
    response = call_llm_with_prompt(input_content, prompt, model)

    # Parse the response
    parsed_response = parse_llm_response(response, expect_json)
    return parsed_response



def get_role_file(role_name: str, role_type: str) -> Path:
    """
    Get the file path for a specific role.
    
    Args:
        role_name: Name of the role (e.g., "KeywordExtractor").
        role_type: Type of the role (e.g., "Helper", "Observer").
    
    Returns:
        Path to the role file.
    
    Raises:
        FileNotFoundError: If the role file does not exist.
        ValueError: If the role type is invalid.
    """
    config = get_config()
    roles_path = Path(config.roles.path)
    
    # Validate role type
    if role_type not in ["Helper", "Observer"]:
        raise ValueError(f"Invalid role type: {role_type}. Must be 'Helper' or 'Observer'.")
    
    if role_type=="Observer":
        roles=config.roles.Observers
        
    target_role = next((role for role in roles if role.name == role_name), None)

    # Construct the role file path
    role_file = Path(roles_path / target_role.file).resolve()
    print(role_file)
        
    # Check if the file exists
    if not role_file.exists():
        raise FileNotFoundError(f"Role file not found: {role_file}")
    
    return role_file

def get_role_prompt(role_name, role_type):
    role_file = get_role_file(role_name, role_type)
    print(role_file)
    with open(role_file, "r") as f:
        prompt=f.read()
        if role_type == "observers":
            #insert the schema into the prompt observers
            return replace_token_with_file_contents(prompt, "{{schema}}", Path(CONFIG.roles.path) / CONFIG.database.schema_file)
        
        return prompt


def replace_token_with_file_contents(content, token, file_path):
    with open(file_path, "r") as f:
        file_content = f.read()
    return content.replace(token, file_content)