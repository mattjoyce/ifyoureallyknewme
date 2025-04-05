#!/usr/bin/env python3
import json
import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import Context, FastMCP

# Set up logging
log_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(log_dir, "knowme_mcp_server.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("knowme_mcp")


# Create stats tracking
class ServerStats:
    def __init__(self):
        self.api_calls = 0
        self.resource_requests = 0
        self.tool_calls = 0
        self.errors = 0
        self.start_time = datetime.now()

    def log_status(self):
        uptime = datetime.now() - self.start_time
        hours, remainder = divmod(uptime.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

        logger.info(
            f"SERVER STATUS: Uptime: {uptime_str}, API Calls: {self.api_calls}, "
            f"Resource Requests: {self.resource_requests}, Tool Calls: {self.tool_calls}, "
            f"Errors: {self.errors}"
        )


# Initialize stats
stats = ServerStats()

# Load environment variables
load_dotenv()

logger.info("Starting KnowMe MCP Server")

# Create an MCP server
mcp = FastMCP("KnowMe MCP")

# # Constants
# API_BASE_URL = os.getenv("SENECHAL_API_BASE_URL")
# API_KEY = os.getenv("SENECHAL_API_KEY")

# if not API_BASE_URL:
#     logger.error("No API URL found. Please set SENECHAL_API_BASE_URL environment variable in .env file")
#     raise ValueError("SENECHAL_API_BASE_URL environment variable is required")

# if not API_KEY:
#     logger.error("No API key found. Please set SENECHAL_API_KEY environment variable in .env file")
#     raise ValueError("SENECHAL_API_KEY environment variable is required")

# logger.info(f"Using API URL: {API_BASE_URL}")
# logger.info(f"API Key: {'*' * len(API_KEY)}")


# Helper class for response types
@dataclass
class Question:
    markdown: str


@dataclass
class Profile:
    markdown: str


@dataclass
class Transcript:
    markdown: str


@dataclass
class Coverage:
    markdown: str


# Resources
## fixme move to config
@mcp.resource(
    name="GetRandomQuestion",
    description="Get a random question from the Book of Questions.",
)
def get_random_question() -> Question:
    """Get a random question from the Book of Questions."""

    async def impl():
        stats.resource_requests += 1
        logger.info(f"Resource Request: BookOfQuestions")

        try:
            with open("BookOfQuestions.json") as f:
                questions = json.load(f)
                question = random.choice(questions)
                return Question(markdown=question)
        except Exception as error:
            logger.error(f"error in BookOfQuestions: {str(error)}")
            # Provide fallback data in case API fails
            return Question(
                markdown="# Error\n\nUnable to retrieve question data. "
            )

    return impl

# # Tools
# @mcp.tool()
# async def get_question() -> str:
#     """ 
#     Get a random question from the Book of Questions.
#     """

#     stats.tool_calls += 1
#     logger.info("Tool Call: get question")

#     try:
#         # Call the actual API, expecting markdown text
#         data = await make_api_request("health/profile", expect_json=False)
#         return data
#     except Exception as e:
#         logger.error(f"API error in fetch_health_profile: {str(e)}")
#         # Provide fallback data in case API fails
#         return "# Error\n\nUnable to retrieve health profile data. Please check your API key and connection."


# Prompts
@mcp.prompt(name="ConductInterview", description="Conduct an interview.")
def conduct_interview() -> str:
    """
    Conduct an interview.
    """
    logger.info("Prompt Requested: ConductInterview")
    question = get_random_question()

    with open("roles/Role-Interviewer-mcp.md") as f:
        prompt = f.read()
    
    prompt.replace("{{question}}", question.markdown)


    return prompt


# Run the server
if __name__ == "__main__":
    logger.info("Starting MCP server")
    mcp.run()
