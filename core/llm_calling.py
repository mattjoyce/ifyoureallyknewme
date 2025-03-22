"""
LLM utility functions for TheInterview project.
Provides abstracted interface for LLM operations to make it easy to switch providers.
"""

import subprocess
import tempfile
import logging
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import llm

from .config import ConfigSchema

# Set up logging
logger = logging.getLogger(__name__)


def call_llm_with_prompt(
    input_content: Union[str, Dict, List], 
    prompt: str, 
    model_name: str,
    json_output: bool = True
) -> Any:  
    """
    Call LLM with a role prompt and input content.
    """
    
    # Convert input to string if it's JSON
    if isinstance(input_content, (dict, list)):
        input_str = json.dumps(input_content)
    else:
        input_str = str(input_content)
    
    logger.debug(f"Calling LLM with model: {model_name}")

    model = llm.get_model(model_name or"gpt-4o-mini")
    context=f"{prompt} \n\n {input_str}"
    response=model.prompt(prompt=context)

    
    # Debug the response type
    logger.debug(f"Response type: {type(response)}")
    
    # Get the response content
    response_dict = response.json()
    logger.debug(f"Response dict keys: {response_dict.keys()}")
    
    # Extract the content
    response_content = response_dict['content']
    logger.debug(f"Response content: {response_content[:100]}")
    logger.debug(f"Response content type: {type(response_content)}")

    # Try parsing here directly for debugging
    try:
        parsed = json.loads(response_content)
        logger.debug(f"Successfully parsed content as JSON, type: {type(parsed)}")
        
        # If we successfully parsed it and json_output is True, return the parsed object directly
        if json_output:
            return parsed
    except Exception as e:
        logger.info(f"Error parsing content as JSON: {str(e)}")
    
    # If parsing failed or json_output is False, return the raw content
    if json_output:
        return parse_llm_response(response_content, expect_json=True)
    else:
        return response_content


def parse_llm_response(response: Any, expect_json: bool = True) -> Any:
    """
    Parse the LLM response, handling JSON if needed.
    """
    logger.info(f"parse_llm_response called with type: {type(response)}")
    
    if not expect_json:
        return response
        
    # If response is already a dict or list, return it directly
    if isinstance(response, (dict, list)):
        print(" Response is already a dict or list, returning directly")
        return response
    
    try:
        result = json.loads(response)
        return result
    except json.JSONDecodeError as e:
        logger.info(f"JSON decode error: {str(e)}")
        # Try to extract JSON from text if it's embedded
        import re
        json_pattern = r'(\{[\s\S]*\}|\[[\s\S]*\])'
        match = re.search(json_pattern, response)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # If all parsing fails, print the response for debugging
        logger.info(f"Failed to parse response as JSON: {response}")
        raise

def llm_process_with_prompt(config:ConfigSchema,
    input_content: Union[str, Dict, List],
    prompt: str,
    model: str,
    expect_json: bool = True
) -> Any:
    """
    Process input with a role and return parsed results.
    """
    # Use model from config if not provided
    model = model or config.llm.generative_model

    # Call LLM with role and get response
    response = call_llm_with_prompt(input_content, prompt, model, json_output=expect_json)
    
    # Debug the response
    logger.info(f"Response from call_llm_with_prompt type: {type(response)}")
    
    # If response is already a dict or list, return it directly
    if isinstance(response, (dict, list)):
        logger.debug("Response is already a dict or list, returning directly")
        return response
        
    # Otherwise, try to parse it
    if expect_json:
        return parse_llm_response(response, expect_json)
    else:
        return response



def get_role_file(config: ConfigSchema,role_name: str, role_type: str) -> Path:
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


    
    # Validate role type
    if role_type not in ["Helper", "Observer"]:
        raise ValueError(f"Invalid role type: {role_type}. Must be 'Helper' or 'Observer'.")
    
    if role_type=="Observer":
        roles=config.roles.Observers

    if role_type=="Helper":
        roles=config.roles.Helpers

    target_role = next((role for role in roles if role.name == role_name), None)

    # Construct the role file path
    role_file = Path(f"{config.roles.path}/{target_role.file}").resolve()
        
    # Check if the file exists
    if not role_file.exists():
        raise FileNotFoundError(f"Role file not found: {role_file}")
    
    return role_file

def get_role_prompt(config:ConfigSchema, role_name : str, role_type : str):
    role_file = get_role_file(config, role_name, role_type)
    with open(role_file, "r") as f:
        prompt=f.read()
        if role_type == "Observer":
            #insert the schema into the prompt observers
            composed_prompt= replace_token_with_file_contents(prompt, "{{schema}}", Path(f"{config.roles.path}/{config.roles.schema_file}").resolve())
            return composed_prompt
        return prompt


def replace_token_with_file_contents(content, token, file_path):
    with open(file_path, "r") as f:
        file_content = f.read()
    return content.replace(token, file_content)


def get_embedding(config:ConfigSchema, text: str, model: Optional[str] = None) -> any:
    embedding_model=llm.get_embedding_model(model or config.llm.embedding_model)
    vector = embedding_model.embed(text)