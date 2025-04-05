#!/usr/bin/env python3
import random
import json
import logging
from pathlib import Path
from mcp.server.fastmcp import FastMCP, Context

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("Question Server")

# Load questions from your BookOfQuestions.txt file
def load_questions():
    try:
        with open("BookOfQuestions.txt", "r") as f:
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
            "What would constitute a perfect day for you?",
            "For what in your life do you feel most grateful?",
            "If you could change anything about the way you were raised, what would it be?"
        ]

# Global question list
QUESTIONS = load_questions()

@mcp.tool()
def get_random_question() -> str:
    """Get a random question from the Book of Questions."""
    if not QUESTIONS:
        return "No questions available."
    question = random.choice(QUESTIONS)
    logger.info(f"Returning random question: {question[:30]}...")
    return question

@mcp.tool()
def get_question_by_number(number: int) -> str:
    """Get a specific question by its number (1-267).
    
    Args:
        number: The question number (starting from 1)
    """
    if not QUESTIONS:
        return "No questions available."
    
    if number < 1 or number > len(QUESTIONS):
        return f"Question number must be between 1 and {len(QUESTIONS)}."
    
    question = QUESTIONS[number - 1]
    logger.info(f"Returning question #{number}: {question[:30]}...")
    return question

@mcp.resource("questions://count")
def get_question_count() -> str:
    """Get the total number of available questions."""
    return str(len(QUESTIONS))

@mcp.resource("questions://all")
def get_all_questions() -> str:
    """Get all questions as a formatted list."""
    return "\n\n".join([f"{i+1}. {q}" for i, q in enumerate(QUESTIONS)])

# Prompts
@mcp.prompt(name="ConductInterview", description="Conduct an interview.")
def conduct_interview() -> list:
    """
    Conduct an interview using a random question.
    """
    logger.info("Prompt Requested: ConductInterview")
    question = get_random_question()
    
    try:
        with open("../roles/Role-Interviewer-mcp.md") as f:
            system_prompt = f.read()
        
        # Replace placeholder with the actual question
        system_prompt = system_prompt.replace("{{question}}", question)
        
        # Return properly formatted MCP prompt structure
        return [
            {
                "role": "system",
                "content": {
                    "type": "text",
                    "text": system_prompt
                }
            },
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": "Let's start the interview with the question above."
                }
            }
        ]
    except Exception as e:
        logger.error(f"Error loading interviewer role: {e}")
        # Fallback if role file isn't available
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"You are an interviewer. Ask me this question: {question}"
                }
            }
        ]


# Run the server
if __name__ == "__main__":
    logger.info(f"Starting question server with {len(QUESTIONS)} questions")
    mcp.run()