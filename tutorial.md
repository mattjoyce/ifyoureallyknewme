# If You Really Knew Me

## Overview

The goal of this project is :
1. Produce profile summaries that can be used by LLMs to learn more about the person.
2. Over time produce a high fidelity profile, to be able to simulate the person's responses.
3. Personal growth.

This application, is a personal knowledge base application, it 's focus is a singular person, maybe you.
It processes various input content, and uses LLM models to tease out observations about the person's personality and also facts.
It attempts to attribute facts and observations to a life stage and knowledge domain.

## Life Stage Epoch 
- NOT KNOWN
- CHILDHOOD: 0-12 (combining early childhood and school age)
- ADOLESCENCE: 13-19
- EARLY_ADULTHOOD: 20-29 (typically education/early career)
- EARLY_CAREER: 30-39
- MID_CAREER: 40-49
- LATE_CAREER: 50+

## Domains 
- Personal History
- Professional Evolution
- Psychological and Behavioral Evolution
- Relationships and Networks
- Community and Ideological Engagement
- Daily Routines and Health
- Values, Beliefs, and Goals
- Active Projects and Learning

## Technical
- written in phyton
- uses SQLite3 as the database
- uses openai for embeddings
- uses fabric for llm calls (this can be changed)

## Getting Started

The main application is CLI only and is called *knowme.py*

```bash
❯ python3 knowme.py --help
Usage: knowme.py [OPTIONS] COMMAND [ARGS]...

  KnowMe - Personal knowledge management and analysis system.

Options:
  --version      Show the version and exit.
  --config PATH  Path to config file
  --help         Show this message and exit.

Commands:
  analyze  Run expert analysis on content.
  init     Initialize a new knowledge database.
  merge    Find clusters of similar notes/facts and create consensus...
  profile  Generate a profile from the knowledge base.
  queue    Add content to the analysis queue or process QA transcripts.
```


It uses a yaml config file, here's an example.

```yaml
# Database configuration
database:
  path: private/test.db
  schema_path: schema.sql

# LLM configuration
llm:
  generative_model: gpt-4
  embedding_model: text-embedding-ada-002

# Role configuration
roles:
  path: ./roles
  schema_file: Prompt-Schema.md  
  Observers:
    - name: Psychologist
      file: Role-Psychologist.md
    - name: Demographer
      file: Role-Demographer.md
    - name: BehavioralEconomist
      file: Role-BehavioralEconomist.md
    - name: PoliticalScientist
      file: Role-PoliticalScientist.md
    - name: FactExtractor
      file: Role-FactExtractor.md
  Helpers:
    - name: ConsensusMaker
      file: Role-ConsensusMaker.md
    - name: CoverageAssessor
      file: Role-CoverageAssessor.md
    - name: KeywordExtractor
      file: Role-KeywordExtractor.md
    - name: Interviewer
      file: Role-Interviewer.md
  

content:
  default_author: "Matt"

# Logging configuration
logging:
  level: INFO
  file: private/knowme.log 


```

My advice is you create a private folder, and use that for you files you'll be uploading and the database.
Ok, let's start a new database.

```bash
python3 knowme.py --config test_config.yaml init
```

ok, use a text editor, and copy the qa-template.md to your private folder.
Edit it, and answer the question as fully as you can.

Let's load that QA document into the system.

```bash

```

