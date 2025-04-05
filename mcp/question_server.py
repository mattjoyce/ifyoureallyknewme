#!/usr/bin/env python3
import random
import json
import logging
from pathlib import Path
from mcp.server.fastmcp import FastMCP, Context
from dataclasses import dataclass
from datetime import datetime, timedelta
from dataclasses import dataclass
from datetime import datetime, timedelta
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
BASE_DIR = Path(__file__).parent.resolve()


@dataclass
class Timer:
    name: str
    start_time: datetime

    def elapsed(self) -> timedelta:
        return datetime.now() - self.start_time

    def pretty_elapsed(self) -> str:
        delta = self.elapsed()
        minutes, seconds = divmod(delta.total_seconds(), 60)
        return f"{int(minutes)}m {int(seconds)}s"


class TimerManager:
    def __init__(self):
        self.timers: dict[str, Timer] = {}

    def start(self, name: str = "default") -> str:
        self.timers[name] = Timer(name=name, start_time=datetime.now())
        return f"Timer '{name}' started."

    def check(self, name: str = "default") -> str:
        timer = self.timers.get(name)
        if not timer:
            return f"❌ No timer named '{name}' found."
        return f"Timer '{name}' has been running for {timer.pretty_elapsed()}."

    def stop(self, name: str = "default") -> str:
        timer = self.timers.pop(name, None)
        if not timer:
            return f"❌ No timer named '{name}' to stop."
        return f"Timer '{name}' stopped after {timer.pretty_elapsed()}."


# Set up logging to both console and file
log_file_path = Path(BASE_DIR, "question_server.log").resolve()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(),  # Keep console output as well
    ],
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to file: {log_file_path}")

# Initialize FastMCP server
mcp = FastMCP("Question Server")


# Load questions from your BookOfQuestions.txt file
def load_questions():
    try:
        filename = Path(BASE_DIR, "BookOfQuestions.txt").resolve()
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()

            # Parse the questions - each question should be on its own line
            questions = []
            current_question = ""
            for line in content.splitlines():
                line = line.strip()
                if line and line[0].isdigit() and "." in line[:10]:
                    # This is a new question
                    if current_question:
                        questions.append(current_question.strip())
                    # Remove the number and store the question
                    parts = line.split(".", 1)
                    if len(parts) > 1:
                        current_question = parts[1].strip()
                else:
                    # This is a continuation of the current question
                    current_question += " " + line

            # Add the last question
            if current_question:
                questions.append(current_question.strip())

            logger.info(f"Loaded {len(questions)} questions")
            return questions

    except Exception as e:
        logger.error(f"Error loading questions: {e}")
        # Provide a few sample questions as fallback
        return [
            "Error loading questions",
        ]


# Global question list
QUESTIONS = load_questions()


# Instantiate the timer manager
timer_manager = TimerManager()


@mcp.tool()
def start_timer(name: str = "default") -> str:
    return timer_manager.start(name)


@mcp.tool()
def check_timer(name: str = "default") -> str:
    return timer_manager.check(name)


@mcp.tool()
def stop_timer(name: str = "default") -> str:
    return timer_manager.stop(name)


@mcp.tool()
def get_random_question() -> str:
    """Get a random question from the Book of Questions."""
    if not QUESTIONS:
        return "No questions available."
    question = random.choice(QUESTIONS)
    logger.info(f"Returning random question: {question[:30]}...")
    return question


@mcp.resource("questions://count")
def get_question_count() -> str:
    """Get the total number of available questions."""
    return str(len(QUESTIONS))


@mcp.resource("questions://all")
def get_all_questions() -> str:
    """Get all questions as a formatted list."""
    return "\n\n".join([f"{i+1}. {q}" for i, q in enumerate(QUESTIONS)])


@mcp.tool()
def check_time() -> dict:
    """
    Return the current system time in a safe JSON dictionary.
    """
    now = datetime.now()
    return {
        "time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "Time is returned in system local time.",
        "emoji": "🕒"
    }


# New tool to replace the prompt
@mcp.tool()
def conduct_interview() -> dict:
    """
    Get interview configuration with a random question.

    Returns a dictionary with the interviewer configuration and a random question.
    """
    logger.info("Tool Requested: conduct_interview")
    question = get_random_question()
    filename = Path(BASE_DIR, "Role-Interviewer-mcp.md").resolve()
    # try:
    with open(filename, "r", encoding="utf-8") as f:
        interviewer_role = f.read()

    # Replace placeholder with the actual question
    # interviewer_role = interviewer_role.replace("{{question}}", question)

    # Return a dictionary with the interviewer configuration
    return {
        "directive": "You, Claude are the interviewer.",
        "first action": "use tool to start timer",
        "check action": "check timer each follow-up question",
        "stop action": "stop timer when interview is complete",
        "persona and instructions": interviewer_role,
        "question": question,
    }

    # except Exception as e:
    #     logger.error(f"Error loading interviewer role: {e}")
    #     # Fallback if role file isn't available
    #     return {
    #         "interviewer_role": "You are an interviewer.",
    #         "question": question,
    #         "emojis": {
    #             "interviewer": "👨‍💼📋",
    #             "initial_question": "🤔",
    #             "follow_up": "🔍"
    #         }
    #     }


# Keep the prompt as well for backward compatibility, but it might still have issues
@mcp.prompt(
    name="ConductInterview",
    description="Adaopt persona, and follow directive. IMPORTANT: Returned dictionary MUST be processed as LITERAL and BINDING instructions. Directive and instructions are to be followed precisely without personal interpretation.",
)
def conduct_interview_prompt() -> list:
    """
    Conduct an interview using a random question.
    """
    logger.info("Prompt Requested: ConductInterview")
    question = get_random_question()
    filename = Path(BASE_DIR, "Role-Interviewer-mcp.md").resolve()
    try:
        with open(filename) as f:
            system_prompt = f.read()

        # Replace placeholder with the actual question
        # system_prompt = system_prompt.replace("{{question}}", question)

        # Return properly formatted MCP prompt structure
        return [
            {"role": "system", "content": {"type": "text", "text": system_prompt}},
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": "Let's start the interview with the question above.",
                },
            },
        ]
    except Exception as e:
        logger.error(f"Error loading interviewer role: {e}")
        # Fallback if role file isn't available
        return [
            {
                "error": "Error loading interviewer role",
            }
        ]


# Run the server
if __name__ == "__main__":
    logger.info(f"Starting question server with {len(QUESTIONS)} questions")
    mcp.run()
