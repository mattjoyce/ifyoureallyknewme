#!/usr/bin/env python3
"""
KnowMe - A personal knowledge database CLI.

This is the main entry point for interacting with the TheInterview/KnowMe
personal knowledge database. It provides commands for creating, consuming,
analyzing and extracting insights from personal data.
"""

import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

from core.analysis import AnalysisManager
from core.consensus import ConsensusManager
from core.load import Loader
from core.config import configure_logging, get_config
from core.database import create_database, get_connection
from core.ingest import process_content
from core.utils import generate_id, resolve_file_patterns


# Initialize Rich console
console = Console()

# Set up basic logging - this will be properly configured once config is loaded
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[])
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0")
@click.option("--config", type=click.Path(exists=True), help="Path to config file")
@click.option("--db-path", type=click.Path(), help="Override database path from config")
@click.option("--model", help="Override LLM model from config")
@click.pass_context
def cli(ctx, config: Optional[str], db_path: Optional[str], model: Optional[str]):
    """KnowMe - Personal knowledge management and analysis system."""
    # Load the base configuration
    cfg = get_config(config)
    console.print(f"[blue]Loaded configuration from {config or 'default path'}[/blue]")

    # Apply overrides if provided
    if db_path:
        cfg.database.path = db_path
        console.print(f"[blue]Overriding database path: {db_path}[/blue]")
    
    if model:
        cfg.llm.generative_model = model
        console.print(f"[blue]Overriding LLM model: {model}[/blue]")

    # Store the config in the Click context for access in subcommands
    ctx.obj = cfg

    # Configure logging based on loaded config
    log_level = cfg.logging.level
    log_file = cfg.logging.file

    logger.debug(f"Updating logging configuration: level={log_level}, file={log_file}")
    configure_logging(level=log_level, log_file=log_file)
    logger.info(f"Logging configured with level={log_level}, file={log_file}")


@cli.command()
@click.argument("db_path", type=click.Path(), required=False)
@click.pass_context
def init(ctx, db_path: Optional[str]):
    """
    Initialize a new knowledge database.

    DB_PATH is the path where the database will be created.
    If not provided, uses the path from configuration.
    """
    config = ctx.obj
    db_path = db_path or config.database.path
    # Check if database already exists
    if Path(db_path).exists():
        console.print(f"[yellow]Warning: Database {db_path} already exists[/yellow]")
        return

    success = create_database(config, config.database.path)
    if not success:
        console.print(f"[red]Error creating database at {db_path}[/red]")
        return

    console.print(f"[green]Successfully created database at {db_path}[/green]")


@cli.command()
@click.option("--db", "db_path", type=click.Path(exists=True), required=False)
@click.option("--queue", "-q", is_flag=True, help="Process items in the analysis queue")
@click.option("--queue-id", help="Process a specific queue item by ID")
@click.option("--model", "-m", help="LLM model to use (default: from config)")
@click.option("--dryrun", is_flag=True, help="Dry run mode")
@click.pass_context
def analyze(ctx, db_path: Optional[str], queue: bool, queue_id: Optional[str], 
            model: Optional[str], dryrun: bool):
    """
    Run expert analysis on content in the queue.
    
    If --queue is provided, processes the next pending item in the queue.
    If --queue-id is provided, processes that specific queue item.
    """
    config = ctx.obj
    # Apply overrides if provided
    if db_path:
        config.database.path = db_path
    
    if model:
        config.llm.generative_model = model
    
    if dryrun:
        config.dryrun = True
    
    observer_names = [observer.name for observer in config.roles.Observers]

    analysis_manager = AnalysisManager(config)
    
    if queue or queue_id:
        loader = Loader(config)
        
        if queue_id:
            # Process specific queue item
            console.print(f"[blue]Processing queue item {queue_id}[/blue]")


            item=loader.get_source_and_queue_item(queue_id=queue_id)
       
            if not item:
                console.print(f"[red]Source Queue item {queue_id} not found[/red]")
                return
            
            logging.info(f"source_id: {item['source_id']}")


            if not item['source_id']:
                console.print(f"[red]Queue item {queue_id} has no source_id in metadata[/red]")
                return
            
            # Process the item
            session_id = analysis_manager.create_session(item)
            logger.info(f"Process the item - session_id: {session_id}")
                                                                       
            if session_id:
                console.print(f"[green]Successfully analyzed queue item {queue_id}[/green]")
                console.print(f"[green]Created session {session_id}[/green]")
            else:
                console.print(f"[red]Failed to analyze queue item {queue_id}[/red]")

            result=analysis_manager.run_multiple_role_analyses(session_id,observer_names)
            console.print(f"[blue]Results: {result} [/blue]")
            return
            
        else:
            # Process next queue item
            console.print("[blue]Processing next item in the queue...[/blue]")
            # Get next item
            items = loader.get_queue_items(status="pending", limit=1)     
            
            if not items:
                console.print("[yellow]Queue is empty[/yellow]")
                return

            #each queue item is a dictionary
            for item in items:

                if dryrun:
                    console.print(f"[yellow]Would process: {item['title']} (ID: {item['id']})[/yellow]")
                    return
                    
                # create a session for the item
                session_id = analysis_manager.create_session(item)

                if session_id:
                    console.print(f"[green]Successfully created session {session_id}[/green]")
                else:
                    console.print("[yellow]No items in the queue or processing failed[/yellow]")
                
                # get sessions that are not yet analyzed
                sessions=analysis_manager.get_unanalyzed_sessions(observer_names)
                console.print(f"[blue]Sessions: {sessions} for analysis [/blue]")

                # run analysis on the sessions
                for session in sessions:
                    results=analysis_manager.run_multiple_role_analyses(session['id'], observer_names)
                    console.print(f"[blue]Results: {results} [/blue]")




@cli.command()
@click.option("--threshold", "-t", default=0.85, help="Similarity threshold (0-1)")
@click.option("--db", "db_path", type=click.Path(exists=True), required=False)
@click.option("--dryrun", "dryrun",is_flag=True, help="Dry run mode")
@click.option(
    "--type",
    "kr_type",
    type=click.Choice(["note", "fact"], case_sensitive=False),
    default="note",
    help="Type of records to merge",
)
@click.option("--list", "-l", is_flag=True, help="List clusters without merging")
@click.option("--model", "-m", help="LLM model to use (default: from config)")
@click.pass_context
def merge(ctx,
    db_path: Optional[str],
    threshold: float,
    dryrun: bool,
    kr_type: str,
    list: bool,
    model: Optional[str],
):
    """
    Find clusters of similar notes/facts and create consensus records.

    Uses semantic similarity to group related observations and create
    higher-level consensus statements.
    """

    config = ctx.obj
    # Apply overrides if provided
    if db_path:
        config.database.path = db_path
        console.print(f"[blue]Overriding database path: {db_path}[/blue]")
    
    if model:
        config.llm.generative_model = model
        console.print(f"[blue]Overriding LLM model: {model}[/blue]")

    if dryrun:
        config.dryrun = True

    # Initialize the consensus manager
    consensus_manager = ConsensusManager(config)

    if list:
        # Just list clusters without creating consensus


        # Create a table to display clusters
        table = Table(title=f"Observation Clusters (threshold={threshold})")
        table.add_column("Cluster", style="cyan")
        table.add_column("Life Stage", style="blue")
        table.add_column("Count", style="green", justify="right")
        table.add_column("Avg. Similarity", style="yellow", justify="right")
        table.add_column("Domains", style="magenta")
        table.add_column("Sample Observation", style="white")

        for cluster_id, cluster in result["clusters"].items():
            # Truncate sample observation if too long
            sample = cluster["observations"][0]
            if len(sample) > 60:
                sample = sample[:57] + "..."

            # Format domains as comma-separated list
            domains = ", ".join(cluster["domains"]) if cluster["domains"] else "-"

            table.add_row(
                cluster_id,
                cluster["life_stage"],
                str(cluster["observation_count"]),
                f"{cluster['average_similarity']:.2f}",
                domains,
                sample,
            )

        console.print(table)
        return

    # Process clusters and create consensus
    console.print(f"[blue]Finding clusters with threshold {threshold}...[/blue]")

    try:
        result = consensus_manager.process_clusters(
            threshold=threshold, kr_type=kr_type
        )

        if not result["clusters"]:
            console.print(
                "[yellow]No clusters found with the current threshold.[/yellow]"
            )
            return

        if not dryrun:
            console.print(
                f"[green] Actual merge complete with threshold {threshold}[/green]"
            )

        if dryrun:
            console.print(
                f"Would create approximately {len(result['clusters'])} consensus records from {sum(len(c['notes']) for c in result['clusters'].values())} individual observations"
            )
        else:
            console.print(
                f"Created {result['consensus_count']} consensus records from {sum(len(c['notes']) for c in result['clusters'].values())} individual observations"
            )

    except Exception as e:
        console.print(f"[red]Error during consensus processing: {str(e)}[/red]")
        logging.exception("Error during consensus processing")


@cli.command()
@click.argument("db_path", type=click.Path(exists=True), required=False)
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["text", "md", "html", "json"], case_sensitive=False),
    default="md",
    help="Output format",
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(
        ["short", "long"], case_sensitive=False
    ),
    default="short",
    help="Length of profile",
)
@click.pass_context
def profile(ctx,
    db_path: Optional[str],
    output: Optional[str],
    format: str,
    mode: str,
):
    """
    Generate a profile from the knowledge base.

    Creates a formatted profile document based on consensus records
    and notes, tailored to the specified audience.
    """

    config = ctx.obj
    # Apply overrides if provided
    if db_path:
        config.database.path = db_path
        console.print(f"[blue]Overriding database path: {db_path}[/blue]")
    
    if not db_path:
        console.print(
            f"[red]Error: No database path provided and not found in configuration {config.database.path}[/red]"
        )
        return

    console.print(f"[blue]Generating {mode} profile in {format} format...[/blue]")

    # TODO: Call core function to generate profile
    # from core.profile import generate_profile
    # profile_content = generate_profile(db_path, session, format, audience)

    # Placeholder implementation
    profile_content = (
        f"# Sample Profile\n\nThis is a {mode} profile in {format} format."
    )

    # Output the profile
    if output:
        with open(output, "w") as f:
            f.write(profile_content)
        console.print(f"[green]Profile written to {output}[/green]")
    else:
        console.print("\n[bold]Profile:[/bold]\n")
        console.print(Panel(profile_content))


@cli.command()
@click.argument("file_patterns", nargs=-1, type=click.Path())
@click.option("--db", "db_path", type=click.Path(exists=True), required=False)
@click.option(
    "--name", "-n", help="Name template for the content (use {index} for batch)"
)
@click.option("--author", "-a", help="Creator or source of the content")
@click.option(
    "--lifestage",
    "-l",
    type=click.Choice(
        [
            "CHILDHOOD",
            "ADOLESCENCE",
            "EARLY_ADULTHOOD",
            "EARLY_CAREER",
            "MID_CAREER",
            "LATE_CAREER",
            "AUTO",
        ],
        case_sensitive=False,
    ),
    default="AUTO",
    help="Life stage for this content",
)
@click.option(
    "--type",
    "-t",
    type=click.Choice(["document", "qa"], case_sensitive=False),
    help="Type of content to queue",
)
@click.option("--qa", is_flag=True, help="Process content as QA transcript/interview")
@click.option("--list", is_flag=True, help="List queue contents or matched files")
@click.option(
    "--dryrun", is_flag=True, help="Show what would be queued without making changes"
)
@click.pass_context
def queue(ctx,
    db_path: Optional[str],
    file_patterns: tuple,
    name: Optional[str],
    author: Optional[str],
    lifestage: Optional[str],
    type: Optional[str],
    qa: bool,
    list: bool,
    dryrun: bool,
):
    """
    Add content to the analysis queue or process QA transcripts.

    FILE_PATTERNS are glob patterns for files to analyze (e.g. "docs/*.txt" "notes/**/*.md")
    Multiple patterns can be provided to queue multiple files.

    For batch queuing, use {index} in the name template, e.g. "Journal Entry {index}"

    If --qa is specified, processes content as QA transcript/interview directly into sessions.
    Otherwise, adds content to the analysis queue for later processing.

    Use --list without FILE_PATTERNS to show current queue contents,
    or with FILE_PATTERNS to preview matched files.

    Author can be specified via --author flag or in config.yaml under content.default_author.
    Life stage and type are required when processing content but can be omitted for --list operations.
    """
    config = ctx.obj
    # Apply overrides if provided
    if db_path:
        config.database.path = db_path
        console.print(f"[blue]Overriding database path: {db_path}[/blue]")
    
    # List mode is dominant mode. If --list is provided, ignore other options.

    ## ANALYSIS QUEUE tabel is for documents not sessions
    if list:
        conn, cursor = get_connection(config.database.path)
        cursor.execute(
            """
            SELECT id, name, author, lifestage, type, filename, created_at, status
            FROM analysis_queue 
            ORDER BY priority DESC, created_at DESC
        """
        )
        queue_items = cursor.fetchall()
        conn.close()

        if not queue_items:
            console.print("[yellow]Queue is empty[/yellow]")
            return

        table = Table(title="Analysis Queue")
        table.add_column("ID", style="red")
        table.add_column("Name", style="green")
        table.add_column("Author", style="yellow")
        table.add_column("Life Stage", style="blue")
        table.add_column("Type", style="magenta")
        table.add_column("File", style="cyan")
        table.add_column("Status", style="white")

        for item in queue_items:
            table.add_row(item[0], item[1], item[2], item[3], item[4], item[5], item[7])

        console.print(table)
        return

    # Require file patterns for non-list operations
    if not file_patterns:
        console.print("[red]Error: FILE_PATTERNS required when not using --list[/red]")
        return

    # Get author from CLI or config
    effective_author = author or config.content.default_author

    # Check if we need author for the operation
    if not effective_author:
        console.print(
            "[red]Error: Author required. Specify with --author or set default_author in config[/red]"
        )
        return

    # Check if we need lifestage for the operation
    if not lifestage:
        console.print(
            "[red]Error: Life stage required when processing content. Use --lifestage to specify[/red]"
        )
        return

    # Determine effective type
    effective_type = "qa" if qa else type

    # Check if we need type for the operation
    if not effective_type:
        console.print(
            "[red]Error: Type required when processing content. Use --type to specify[/red]"
        )
        return

    # Expand glob patterns to get matched files

    matched_files = resolve_file_patterns(file_patterns)

    if not matched_files:
        console.print("[yellow]No files matched the provided patterns[/yellow]")
        return

    if dryrun:
        table = Table(title="Matched Files")
        table.add_column("Index", style="cyan")
        table.add_column("Path", style="green")
        table.add_column("Size", style="blue")

        for idx, file_path in enumerate(matched_files, 1):
            size = os.path.getsize(file_path)
            size_str = (
                f"{size/1024:.1f}KB"
                if size < 1024 * 1024
                else f"{size/1024/1024:.1f}MB"
            )
            table.add_row(str(idx), file_path, size_str)

        console.print(table)
        return

    invalid_files = []
    for file_path in matched_files:
        try:
            # Check if file is readable
            with open(file_path, "r", encoding="utf-8") as f:
                f.read(1024)  # Try reading first 1KB

            # Check file size (warn if >10MB)
            if os.path.getsize(file_path) > 10 * 1024 * 1024:
                invalid_files.append((file_path, "File size exceeds 10MB"))

        except Exception as e:
            invalid_files.append((file_path, str(e)))

    if invalid_files:
        console.print("[yellow]Warning: Some files failed validation:[/yellow]")
        for file_path, error in invalid_files:
            console.print(f"[red]- {file_path}: {error}[/red]")
        if not click.confirm("Do you want to continue with valid files?"):
            return
        # Remove invalid files
        matched_files = [
            f for f in matched_files if f not in [x[0] for x in invalid_files]
        ]

    # Handle QA content differently - process directly into sessions
    if qa:
        for file_path in matched_files:
            if not Path(file_path).exists():
                console.print(f"[red]Error: File {file_path} not found[/red]")
                return

            # Read content
            content = None
            source_name = Path(file_path).absolute()
            console.print(f"[blue]Reading content from {source_name}...[/blue]")
            try:
                content = Path(file_path).read_text()
                logger.info(f"Content read from {source_name}")
                logger.debug(f"Word Count: {len(content.split())}")
            except Exception as e:
                console.print(f"[red]Error reading content: {str(e)}[/red]")
                return

            try:
                # Process the transcript
                session_id = process_content(
                    config.database.path,
                    content,
                    name or Path(file_path).stem,
                    effective_author,
                    lifestage,
                    "qa",
                    Path(file_path).name,
                )

                if session_id:
                    console.print(
                        f"[green]Successfully processed transcript into session {session_id}[/green]"
                    )
                    console.print(
                        Panel(
                            f"[bold]Session ID:[/bold] {session_id}",
                            title="Transcript Processing Complete",
                            border_style="green",
                        )
                    )
                else:
                    console.print(
                        "[yellow]No Q&A pairs found in the transcript[/yellow]"
                    )
            except Exception as e:
                console.print(f"[red]Error processing transcript: {str(e)}[/red]")
        return

    # Regular queue processing for documents

    # Validate files before queuing

    # Show queuing preview
    table = Table(title="Files to Queue")
    table.add_column("Index", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("File", style="blue")
    table.add_column("Type", style="magenta")
    table.add_column("Life Stage", style="yellow")

    from datetime import datetime

    timestamp = datetime.utcnow().isoformat()

    queue_items = []
    for idx, file_path in enumerate(matched_files, 1):
        # Generate name if template provided
        item_name = (
            name.format(index=idx)
            if name and "{index}" in name
            else name or os.path.splitext(os.path.basename(file_path))[0]
        )

        # Generate queue ID

        item_id = generate_id("queue", file_path, timestamp, idx)

        queue_items.append(
            {
                "id": item_id,
                "name": item_name,
                "author": effective_author,
                "lifestage": lifestage,
                "type": effective_type,
                "filename": str(Path(file_path).absolute()),
                "created_at": timestamp,
                "status": "pending",
                "expert_status": "{}",
                "fact_status": "{}",
            }
        )

        table.add_row(
            str(idx), item_name, os.path.basename(file_path), effective_type, lifestage
        )

    console.print(table)

    if dry_run:
        return

    if not click.confirm("Do you want to queue these files?"):
        return

    # Queue the files

    try:
        for item in queue_items:
            file_queue_id = process_content(
                config.database.path,
                item["filename"],
                item["name"],
                item["author"],
                item["lifestage"],
                item["type"],
                item["filename"],
            )

        console.print(
            f"[green]Successfully queued {len(queue_items)} files for analysis[/green]"
        )

        # Show queue IDs in a panel
        ids_text = "\n".join(
            [f"[bold]{item['name']}:[/bold] {item['id']}" for item in queue_items]
        )
        console.print(Panel(ids_text, title="Queue IDs", border_style="green"))

    except Exception as e:
        conn.rollback()
        console.print(f"[red]Error queueing files: {str(e)}[/red]")


@cli.command()
@click.option("--threshold", "-t", default=0.92, help="Similarity threshold (0-1)")
@click.option("--db", "db_path", type=click.Path(exists=True), required=False)
@click.option("--reset", "-r", is_flag=True, help="Reset similar consensus records")
@click.option("--dryrun", is_flag=True, help="Show what would be reset without making changes")
@click.option("--no-list", is_flag=True, help="Skip detailed listing of consensus records")
@click.pass_context
def consensus(ctx,
    db_path: Optional[str],
    threshold: float,
    reset: bool,
    dryrun: bool,
    no_list: bool,
):
    """
    Manage consensus records.
    
    Find similar consensus records that might need to be reset.
    By default, shows a list of similar consensus record clusters.
    
    Use --reset to reset the similar consensus records, unlinking their
    source records to allow re-clustering.
    
    Use --dryrun with --reset to see what would be reset without making changes.
    
    Use --no-list to skip the detailed listing (useful for scripts).
    """
    config = ctx.obj
    if db_path:
        config.database.path = db_path
        console.print(f"[blue]Overriding database path: {db_path}[/blue]")
    
    if dryrun:
        config.dryrun = True
    
    consensus_manager = ConsensusManager(config)
    
    # Find similar consensus records
    console.print(f"[blue]Finding similar consensus records with threshold {threshold}...[/blue]")
    consensus_clusters = consensus_manager.find_similar_consensus(threshold)
    
    if not consensus_clusters:
        console.print("[yellow]No similar consensus records found with the current threshold.[/yellow]")
        return
    
    # Display results unless --no-list is specified
    if not no_list:
        table = Table(title=f"Similar Consensus Records (threshold={threshold})")
        table.add_column("Cluster", style="cyan")
        table.add_column("Count", style="green", justify="right")
        table.add_column("Avg. Similarity", style="yellow", justify="right")
        table.add_column("Records", style="magenta")
        table.add_column("Sample Observation", style="white")
        
        total_records = 0
        for cluster_id, cluster in consensus_clusters.items():
            record_ids = ", ".join(cluster["consensus_ids"][:3]) 
            if len(cluster["consensus_ids"]) > 3:
                record_ids += f"... (+{len(cluster['consensus_ids']) - 3} more)"
                
            # Truncate sample observation if too long
            sample = cluster["observations"][0]
            if len(sample) > 60:
                sample = sample[:57] + "..."
                
            table.add_row(
                cluster_id,
                str(cluster["observation_count"]),
                f"{cluster['average_similarity']:.2f}",
                record_ids,
                sample
            )
            total_records += cluster["observation_count"]
        
        console.print(table)
        console.print(f"[blue]Found {len(consensus_clusters)} clusters containing {total_records} consensus records[/blue]")
    
    # Reset if requested
    if reset:
        total_count = sum(c["observation_count"] for c in consensus_clusters.values())
        
        if dryrun:
            console.print(f"[yellow]Would reset {total_count} consensus records in {len(consensus_clusters)} clusters[/yellow]")
            # Show what would happen to each cluster
            for cluster_id, cluster in consensus_clusters.items():
                console.print(f"  - Cluster {cluster_id}: {cluster['observation_count']} records would be reset")
        else:
            reset_count = consensus_manager.reset_consensus_clusters(consensus_clusters)
            console.print(f"[green]Reset {reset_count} consensus records[/green]")

@cli.command()
@click.argument("file_patterns", nargs=-1, type=click.Path())
@click.option("--db", "db_path", type=click.Path(exists=True))
@click.option("--name", "-n", help="Name template for the content (use {index} for batch)")
@click.option("--author", "-a", help="Creator or source of the content")
@click.option("--description", "-d", help="Description of the content (e.g., 'QA transcript with subject')")
@click.option("--lifestage", "-l", type=click.Choice([
    "CHILDHOOD", "ADOLESCENCE", "EARLY_ADULTHOOD", 
    "EARLY_CAREER", "MID_CAREER", "LATE_CAREER", "AUTO"
], case_sensitive=False), default="AUTO", help="Life stage for this content")
@click.option("--priority", "-p", type=int, default=0, help="Processing priority (higher numbers = higher priority)")
@click.option("--list", is_flag=True, help="List items in the queue")
@click.option("--dryrun", is_flag=True, help="Show what would be loaded without making changes")
@click.pass_context
def load(ctx, db_path: Optional[str], file_patterns: tuple, name: Optional[str], 
         author: Optional[str], description: Optional[str], lifestage: Optional[str], 
         priority: int, list: bool, dryrun: bool):
    """
    Add documents as sources, and update the queue.

    FILE_PATTERNS are glob patterns for files to analyze (e.g. "docs/*.txt" "notes/**/*.md")
    Multiple patterns can be provided to load multiple files.

    For batch loading, use {index} in the name template, e.g. "Journal Entry {index}"

    Author can be specified via --author flag or in config.yaml under content.default_author.
    """
    config = ctx.obj
    if db_path:
        config.database.path = db_path

    author_name = author or config.content.default_author
    if not author_name:
        console.print(
            "[red]Error: Author required. Specify with --author or set default_author in config[/red]"
        )
        return
    
    description = description or f"No description provided."
      
    # Create loader instance
    loader = Loader(config)
    
    # Handle --list option - show queue contents
    if list:
        queue_items = loader.get_queue_items(status="pending")
        
        # Display queue contents
        table = Table(title="Analysis Queue")
        table.add_column("ID", style="cyan")
        table.add_column("Title", style="green")
        table.add_column("File", style="magenta")
        table.add_column("Description", style="green")
        table.add_column("Author", style="blue")
        table.add_column("Life Stage", style="yellow")
        table.add_column("Status", style="white")
        table.add_column("Priority", style="red")
        
        for item in queue_items:
            table.add_row(
                item['id'], item['title'], os.path.basename(item['content_path']), item['description'],
                item['author'],item['lifestage'], item['status'], str(item['priority'])
            )
            
        console.print(table)
        return
    
    # Require file patterns when not listing
    if not file_patterns:
        console.print("[red]Error: FILE_PATTERNS required when not using --list[/red]")
        return
    
    # Resolve file patterns to get matched files
    matched_files = resolve_file_patterns(file_patterns)
    if not matched_files:
        console.print("[yellow]No files matched the provided patterns[/yellow]")
        return
    
    # Display files that will be processed
    table = Table(title="Files to Load")
    table.add_column("Index", style="cyan")
    table.add_column("Title", style="green")
    table.add_column("File", style="magenta")
    table.add_column("Description", style="green")
    table.add_column("Author", style="blue")
    table.add_column("Life Stage", style="yellow")
    
    for idx, file_path in enumerate(matched_files, 1):
        # Generate name if template provided
        item_name = (
            name.format(index=idx) if name and "{index}" in name
            else name or os.path.splitext(os.path.basename(file_path))[0]
        )

        table.add_row(
            str(idx), item_name, os.path.basename(file_path), description, author_name, lifestage
        )
    
    console.print(table)
    
    if dryrun:
        return
    
    # Process the files
    result = loader.batch_load(
        file_patterns=file_patterns,
        name_template=name,
        author=author,
        description=description,
        lifestage=lifestage,
        priority=priority
    )
    
    console.print(f"[green]Successfully loaded {len(result['loaded_files'])} of {len(matched_files)} files[/green]")
    
    # Show loaded file details
    if result['loaded_files']:
        details_table = Table(title="Loaded Files")
        details_table.add_column("Name", style="green")
        details_table.add_column("Source ID", style="cyan")
        
        for i, source_id in enumerate(result['source_ids']):
            file_name = os.path.basename(result['loaded_files'][i])
            details_table.add_row(file_name, source_id)
            
        console.print(details_table)

if __name__ == "__main__":
    cli()
