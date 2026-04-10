# KnowMe System - Coding Guidelines

## Commands
- Run CLI: `uv run knowme [command]` or `python knowme.py --config config.yaml [command]`
- Initialize DB: `knowme init`
- Load content: `knowme load [file_patterns] [--name NAME] [--lifestage STAGE]`
- Analyze queue: `knowme analyze --queue [--limit N] [--model MODEL]`
- Merge clusters: `knowme merge [--threshold T] [--dryrun]`
- Manage consensus: `knowme consensus [--threshold T] [--reset]`
- Generate profile: `knowme profile [--output FILE] [--format FORMAT] [--mode MODE]`
- Queue content: `knowme queue [file_patterns] [--qa] [--type TYPE]`
- Regen embeddings: `knowme regen-embeddings [--batch-size N] [--dryrun]`

## Style Guidelines
- **Single file**: All code lives in `knowme.py`
- **Imports**: Standard lib first, third-party second
- **Type annotations**: Required for all functions (PEP 604 union syntax)
- **Formatting**: ruff format (black-compatible, 95 char line width)
- **Linting**: ruff check with select rules (see pyproject.toml)
- **Config**: loaden library for YAML config loading with `loaden.get(config, "key.path")`
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Packaging**: uv for dependency management

## Project Architecture
- Single-file CLI (`knowme.py`) using Click framework
- Configuration via `config.yaml` (loaded by loaden with .env support)
- Database schema in `schema.sql`
- Role definitions in `roles/` directory
- Private data in `private/` (gitignored)

## Configuration
- Default config file: `config.yaml`
- Override with `--config` option
- Access values with `loaden.get(config, "section.key", default)`
- Environment variables loaded via `loaden_env: .env` in config.yaml
