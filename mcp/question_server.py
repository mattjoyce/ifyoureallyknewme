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
from typing import List, Optional
import sys
import io
import csv
import random

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
BASE_DIR = Path(__file__).parent.resolve()
QUESTION_CSV_FILE = Path(BASE_DIR, "domain-questions.csv").resolve()
COVERAGE_REPORT_FILE = Path(BASE_DIR, "coverage_report.md").resolve()

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


@dataclass
class Question:
    """Simple class for storing a domain question."""
    QID: str
    domain: str
    text: str


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


# Instantiate the timer manager
timer_manager = TimerManager()


# Initialize FastMCP server
mcp = FastMCP("Question Server")

def load_questions(csv_file_path: str) -> List[Question]:
    """Load questions from a CSV file.

    Args:
        csv_file_path: Path to the CSV file containing domains and questions

    Returns:
        List of Question objects
    """
    questions = []
    try:
        with open(csv_file_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            # Skip header if it exists
            header = next(reader, None)
            if (
                header
                and header[0].lower() == "qid"
                and header[1].lower() == "domain"
                and header[2].lower() == "question"
            ):
                pass  # Skip the header
            else:
                # If there was no header, reset the file pointer
                file.seek(0)
                reader = csv.reader(file)

            # Read questions
            for row in reader:
                if len(row) >= 2:
                    qid = row[0].strip()
                    domain = row[1].strip()
                    question_text = row[2].strip()
                    questions.append(Question(qid,domain, question_text))
    except Exception as e:
        print(f"Error loading questions: {e}")

    return questions


QUESTIONS = load_questions(QUESTION_CSV_FILE)



def get_question_by_domain(domain: Optional[str], questions: List[Question]) -> Optional[str]:
    """Get a random question from the specified domain.
    
    Args:
        domain: The domain to retrieve a question from
        questions: List of Question objects to search
        
    Returns:
        A random question text from the domain or pick from the non-domain questions
    """

    if not domain:
        domain= "Non-Domain"

    # Filter questions by domain (case-insensitive)
    domain_questions = [q for q in questions if q.domain.lower() == domain.lower()]
    
    if not domain_questions:
        return None
    
    # Select a random question
    return random.choice(domain_questions).text


def get_available_domains(questions: List[Question]) -> List[str]:
    """Get a list of unique domains from the questions.
    
    Args:
        questions: List of Question objects
        
    Returns:
        List of unique domain names
    """
    return sorted(list(set(q.domain for q in questions)))


def get_coverage_report() -> str:
    """Get a coverage report for the questions.
    
    Returns:
        A formatted string with the coverage report
    """
    with open(COVERAGE_REPORT_FILE, "r", encoding="utf-8") as file:
        coverage_report = file.read()
    return coverage_report

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
def get_random_question(domain: Optional[str] = None) -> str:
    """Get a random question for a specific domain, or pick from the Non-Domain questions."""
    if not QUESTIONS:
        return "No questions available."
    
    question_text = get_question_by_domain(domain, QUESTIONS)
    if question_text:
        logger.info(f"Returning domain-specific question: {question_text[:30]}...")
        return question_text
    else:
        return f"No questions found for domain: {domain}"

@mcp.tool()
def get_domains() -> str:
    """Get a list of available domains.  Used to ask domain specific questions."""
    domains = get_available_domains(QUESTIONS)
    return "\n".join(domains)

@mcp.tool()
def get_latest_report() -> str:
    """Get the latest version of the coverage report, used to learn which domain should be covered next."""
    return get_coverage_report()

@mcp.tool()
def check_time() -> dict:
    """
    Return the current system time in a safe JSON dictionary.
    """
    now = datetime.now()
    return {
        "time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "Time is returned in system local time.",
        "emoji": "🕒",
    }

@mcp.tool()
def conduct_interview() -> dict:
    """
    Get interview configuration with a random question.

    Returns a dictionary with the interviewer configuration and a random question.
    """
    logger.info("Tool Requested: conduct_interview")
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
    }

# Run the server
if __name__ == "__main__":
    logger.info(f"Starting question server with {len(QUESTIONS)} questions")
    mcp.run()
