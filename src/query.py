import duckdb
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
import polars as pl
from rich.columns import Columns
from rich.tree import Tree
from rich import print as rprint
import os


app = typer.Typer(help="Golden Record Explorer CLI - High Performance Parquet Query Tool")
console = Console()

# --- CONFIGURATION ---
PARQUET_PATH = "data/output/verified_golden_table.parquet"
LOG_PATH = "data/output/query_log.txt"

def get_conn():
    """Connect to duckdb and register the parquet file."""
    if not os.path.exists(PARQUET_PATH):
        console.print(f"[bold red]Error:[/] File not found at {PARQUET_PATH}")
        console.print("[yellow]Please run the enrichment pipeline first.[/]")
        raise typer.Exit()
    return duckdb.connect(database=':memory:')

def format_domain(domain):
    """Return a styled domain string."""
    if not domain:
        return "[dim italic red]Missing[/]"
    return f"[bold green]{domain}[/]"

@app.command()
def search(query: str = typer.Argument(..., help="Company name to search for")):
    """Search for companies by name (Case-insensitive partial match)."""
    conn = get_conn()
    
    # DuckDB Query - Optimized for Parquet
    # Searches both Primary Name and Aliases (for better coverage)
    sql = f"""
        SELECT 
            canonical_id as cid, 
            primary_company_name, 
            domain, 
            COALESCE(description, 'N/A') as description,
            aliases,
            COALESCE(record_count, 1) as record_count,
            logo_url
        FROM read_parquet('{PARQUET_PATH}')
        WHERE 
            primary_company_name ILIKE '%{query}%' 
            OR array_to_string(aliases, ',') ILIKE '%{query}%'
            OR domain ILIKE '%{query}%'
        LIMIT 20
    """
    
    try:
        results = conn.execute(sql).fetchall()
        
        if not results:
            console.print(f"\n[yellow]No results found for '[bold]{query}[/]'[/]\n")
            return

        table = Table(title=f"Search Results for '{query}'", header_style="bold magenta", border_style="dim")
        table.add_column("ID", justify="right", style="cyan")
        table.add_column("Company Name", style="white")
        table.add_column("Domain", style="green")
        table.add_column("Logo", style="blue")
        table.add_column("Records", justify="right", style="yellow")
        table.add_column("Description", style="dim")

        for row in results:
            # Highlight target query in name
            name = row[1]
            highlighted_name = name.replace(query, f"[bold yellow]{query}[/]") if query.lower() in name.lower() else name
            
            # Show aliases in a sub-line if they exist
            aliases = row[4]
            if aliases and len(aliases) > 0:
                alias_str = f"\n[dim italic]Aliases: {', '.join(aliases[:2])}...[/]" if len(aliases) > 2 else f"\n[dim italic]Aliases: {', '.join(aliases)}[/]"
                highlighted_name += alias_str

            table.add_row(
                row[0],
                highlighted_name,
                format_domain(row[2]),
                f"[link={row[6]}]🔗[/link]" if row[6] else "[dim]-[/]",
                f"{row[5]}",
                row[3][:80] + "..." if len(row[3]) > 80 else row[3]
            )

        console.print(table)
        console.print(f"[dim]Showing top {len(results)} results.[/]")

    except Exception as e:
        console.print(f"[red]Query error:[/] {e}")

@app.command()
def list_all(limit: int = typer.Option(20, help="Number of records to show")):
    """List the first N records from the verified golden table."""
    conn = get_conn()
    
    sql = f"""
        SELECT 
            canonical_id as cid, 
            primary_company_name, 
            domain, 
            COALESCE(description, 'N/A') as description,
            aliases,
            COALESCE(record_count, 1) as record_count,
            logo_url
        FROM read_parquet('{PARQUET_PATH}')
        LIMIT {limit}
    """
    
    try:
        results = conn.execute(sql).fetchall()
        
        table = Table(title=f"First {len(results)} Master Clusters", header_style="bold magenta", border_style="dim")
        table.add_column("ID", justify="right", style="cyan")
        table.add_column("Company Name", style="white")
        table.add_column("Domain", style="green")
        table.add_column("Logo", style="blue")
        table.add_column("Records", justify="right", style="yellow")
        table.add_column("Description", style="dim")

        for row in results:
            table.add_row(
                row[0],
                row[1],
                format_domain(row[2]),
                f"[link={row[6]}]🔗[/link]" if row[6] else "[dim]-[/]",
                f"{row[5]}",
                row[3][:80] + "..." if len(row[3]) > 80 else row[3]
            )

        console.print(table)
    except Exception as e:
        console.print(f"[red]List error:[/] {e}")

@app.command()
def stats():
    """Display enrichment performance and data distribution dashboard."""
    conn = get_conn()
    
    # Calculate Metrics
    try:
        total = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{PARQUET_PATH}')").fetchone()[0]
        # Valid means not 'Missing' and not 'Information not found'
        with_domain = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{PARQUET_PATH}') WHERE domain != 'Missing'").fetchone()[0]
        with_desc = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{PARQUET_PATH}') WHERE description NOT LIKE 'Information not found%'").fetchone()[0]
        
        # We know from raw analysis there were 500,000 records in source_C.parquet
        RAW_TOTAL = 500000 
        
        fill_rate = (with_domain / total) * 100 if total > 0 else 0
        desc_rate = (with_desc / total) * 100 if total > 0 else 0
        overall_enrichment = (fill_rate + desc_rate) / 2
        dedupe_rate = ((RAW_TOTAL - total) / RAW_TOTAL) * 100 if RAW_TOTAL > 0 else 0

        # Build Dashboard Components
        metric_panel = Panel(
            f"""
[bold cyan]Total Clusters:[/]       [white]{total:,}[/]
[bold cyan]Raw Records Analyzed:[/] [white]{RAW_TOTAL:,}[/]
[bold green]Deduplication Eff:[/]   [bold green]{dedupe_rate:.2f}%[/]
[bold magenta]Domain Coverage:[/]      [white]{fill_rate:.1f}%[/]
[bold magenta]Description Quality:[/]  [white]{desc_rate:.1f}%[/]
            """,
            title="[bold]Golden Statistics[/]",
            expand=False
        )
        
        # 2. Progress Bar for Enrichment Completion
        progress_bar = Table.grid(expand=True)
        progress_bar.add_row(f"Overall Enrichment Completion: [bold]{overall_enrichment:.1f}%[/]")
        
        # Render Dashboard
        console.print("\n", Panel.fit("[bold white]Final Golden Record Dashboard[/]", style="on blue"), "\n")
        console.print(metric_panel)
        
        # Simple Visual Progress Bar
        console.print(progress_bar)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            progress.add_task("[magenta]Enrichment Completion", total=100, completed=overall_enrichment)
        
        print("\n")

    except Exception as e:
        console.print(f"[red]Stats error:[/] {e}")

@app.command()
def inspect(canonical_id: str = typer.Argument(..., help="The canonical_id to inspect")):
    """Inspect all available fields for a specific company record."""
    conn = get_conn()
    
    try:
        # Fetch all columns to show raw data
        df = conn.execute(f"SELECT * FROM read_parquet('{PARQUET_PATH}') WHERE canonical_id = '{canonical_id}'").pl()
        
        if df.is_empty():
            console.print(f"[red]No record found with ID: {canonical_id}[/]")
            return

        row_dict = df.to_dicts()[0]
        
        tree = Tree(f"[bold blue]Entity:[/] {row_dict.get('primary_company_name', 'Unknown')}")
        tree.add(f"[cyan]Canonical ID:[/] {canonical_id}")
        
        details = tree.add("[bold magenta]Attributes")
        for key, val in row_dict.items():
            if key not in ['primary_company_name', 'canonical_id']:
                if key == 'logo_url' and val:
                    val_str = f"[link={val}][bold blue]🔗 {val}[/bold blue][/link]"
                else:
                    val_str = f"[green]{val}[/]" if val else "[dim red]N/A[/]"
                details.add(f"{key}: {val_str}")
        
        console.print("\n", tree, "\n")
        
    except Exception as e:
        console.print(f"[red]Inspect error:[/] {e}")

if __name__ == "__main__":
    app()
