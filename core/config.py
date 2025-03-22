# core/config.py
"""
Core configuration module for TheInterview project.
Provides configuration management with support for both YAML and environment variables.
Features Pydantic models for schema validation and dot notation access.
"""
import logging
import os
from pathlib import Path
from typing import Optional, List
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

# Load environment variables from .env file
load_dotenv()
# Global variable to cache the config file path
_cached_config_path = None

class Helper(BaseModel):
    name: str
    file: str

class Observer(BaseModel):
    name: str
    file: str

class Roles(BaseModel):
    path: str
    schema_file: str
    Observers: List[Observer]
    Helpers: List[Helper]

class Database(BaseModel):
    path: str
    schema_path: str

class LLM(BaseModel):
    generative_model: str
    embedding_model: str

class Content(BaseModel):
    default_author: str

class Logging(BaseModel):
    level: str
    file: str

class ConfigSchema(BaseModel):
    database: Database
    llm: LLM
    roles: Roles
    content: Content
    logging: Logging

def load_config(config_path: str) -> ConfigSchema:
    """
    Load configuration from a YAML file and validate it against the Pydantic schema.
    
    Args:
        config_path: Path to the YAML configuration file.
    
    Returns:
        A validated ConfigSchema instance.
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f) or {}
        try:
            return ConfigSchema(**config_dict)
        except ValidationError as e:
            raise ValueError(f"Configuration validation error: {e}")

def get_config(config_path: Optional[str] = None) -> ConfigSchema:
    """
    Get the configuration object, optionally loading from a specific path.
    
    Args:
        config_path: Optional path to config file.
    
    Returns:
        A validated ConfigSchema instance.
    """
    # Determine the configuration path
    global _cached_config_path
    config_path = config_path or _cached_config_path

    if config_path is None:
        logging.info("No config path provided, checking environment variable")
        config_path = os.getenv('KNOWME_CONFIG_PATH', None)
    
    resolved_path = Path(config_path).resolve()
    logging.info(f"Loading configuration from {resolved_path}")

    # Load and validate the configuration
    if not resolved_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved_path}")
    
    _cached_config_path = str(resolved_path)
    return load_config(str(resolved_path))

def configure_logging(level: Optional[str] = None, log_file: Optional[str] = None) -> None:
    """
    Configure logging for the application.
    
    Args:
        level: Optional override for log level.
        log_file: Optional override for log file path.
               - If set to '-', only logs to console
               - If None, uses default "knowme.log"
    """
    from rich.logging import RichHandler
    
    level = level or "WARNING"
    log_file = log_file or "knowme.log"
    
    # Reset root logger
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    
    # Set the root logger level
    root.setLevel(getattr(logging, level.upper()))
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Always add a Rich console handler
    console_handler = RichHandler(rich_tracebacks=True, markup=True)
    root.addHandler(console_handler)
    
    # Add file handler if needed
    if log_file != '-':
        try:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            # Add file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(file_formatter)
            root.addHandler(file_handler)
        except Exception as e:
            print(f"Error setting up file logging: {e}")
    
    # Log a test message to confirm configuration
    logging.debug(f"Logging configured at level {level} with {'console and file' if log_file != '-' else 'console only'}")