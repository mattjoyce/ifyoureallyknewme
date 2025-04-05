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
- written in python
- uses SQLite3 as the database
- uses openai for embeddings
- uses Simon Wilson's llm library calls (this can be changed)

## Getting Started

The main application is CLI only and is called *knowme.py*

Usage: knowme.py [OPTIONS] COMMAND [ARGS]...

  KnowMe - Personal knowledge management and analysis system.

Options:
  --version       Show the version and exit.
  --config PATH   Path to config file
  --db-path PATH  Override database path from config
  --model TEXT    Override LLM model from config
  --help          Show this message and exit.

Commands:
  analyze    Run expert analysis on content in the queue.
  consensus  Manage consensus records.
  init       Initialize a new knowledge database.
  load       Add documents as sources, and update the queue.
  merge      Find clusters of similar notes/facts and create consensus...
  profile    Generate a profile from the knowledge base.


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

My advice is to create a private folder, and use that for you files you'll be uploading and the database.
Ok, let's start a new database.

```bash
python3 knowme.py --config test_config.yaml init
```


Use a text editor (Text for Life!) to create a document your want to load.
For example start with this seed question, spend 15 minutes really getting detailed.

"Tell me the story of your life. Start from the beginning -- from your childhood, to education, to family and relationships, and to any major life events you may have had."

Put that at the top of the document and then put your answer underneath.
Save your file and let's load it into the system.

```bash

```
```
```
