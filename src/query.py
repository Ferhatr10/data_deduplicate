import duckdb
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.columns import Columns
from rich.tree import Tree
from rich import print as rprint
import os


app = typer.Typer(help="Golden Record Explorer CLI - High Performance Parquet Query Tool")
console = Console()

PARQUET_PATH = "data/output/refined_golden_table.parquet"
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
            COALESCE(industry_tags, 'Unknown') as industry,
            COALESCE(source_dataset, 'N/A') as source,
            aliases,
            COALESCE(record_count, 1) as record_count
        FROM read_parquet('{PARQUET_PATH}')
        WHERE 
            primary_company_name ILIKE '%{query}%' 
            OR array_to_string(aliases, ',') ILIKE '%{query}%'
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
        table.add_column("Records", justify="right", style="yellow")
        table.add_column("Industry", style="blue")
        table.add_column("Source", style="dim")

        for row in results:
            # Highlight target query in name
            name = row[1]
            highlighted_name = name.replace(query, f"[bold yellow]{query}[/]") if query.lower() in name.lower() else name
            
            # Show aliases in a sub-line if they exist
            aliases = row[5]
            if aliases and len(aliases) > 0:
                alias_str = f"\n[dim italic]Aliases: {', '.join(aliases[:2])}...[/]" if len(aliases) > 2 else f"\n[dim italic]Aliases: {', '.join(aliases)}[/]"
                highlighted_name += alias_str

            table.add_row(
                row[0],
                highlighted_name,
                format_domain(row[2]),
                f"{row[6]}",
                row[3]
            )

        console.print(table)
        console.print(f"[dim]Showing top {len(results)} results.[/]")

    except Exception as e:
        console.print(f"[red]Query error:[/] {e}")

@app.command()
def stats():
    """Display enrichment performance and data distribution dashboard."""
    conn = get_conn()
    
    # Calculate Metrics
    try:
        total = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{PARQUET_PATH}')").fetchone()[0]
        with_domain = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{PARQUET_PATH}') WHERE domain IS NOT NULL").fetchone()[0]
        raw_total = conn.execute(f"SELECT SUM(record_count) FROM read_parquet('{PARQUET_PATH}')").fetchone()[0]
        
        fill_rate = (with_domain / total) * 100 if total > 0 else 0
        dedupe_rate = ((raw_total - total) / raw_total) * 100 if raw_total > 0 else 0
        
        # Source distribution
        sources = conn.execute(f"""
            SELECT source_dataset, COUNT(*) 
            FROM read_parquet('{PARQUET_PATH}') 
            GROUP BY source_dataset 
            ORDER BY 2 DESC
        """).fetchall()

        # Build Dashboard Components
        metric_panel = Panel(
            f"[bold cyan]Total Clusters:[/] {total:,}\n"
            f"[bold cyan]Raw Records Analyzed:[/] {raw_total:,}\n"
            f"[bold green]Deduplication Efficiency:[/] {dedupe_rate:.1f}%\n"
            f"[bold yellow]Enrichment Fill Rate:[/] {fill_rate:.2f}%",
            title="Golden Statistics",
            expand=False
        )

        # 2. Progress Bar for Fill Rate
        progress_bar = Table.grid(expand=True)
        progress_bar.add_row(f"Domain Quality Score: [bold]{fill_rate:.1f}%[/]")
        
        # 3. Source Distribution Table
        src_table = Table(title="Source Distribution", box=None, header_style="bold green")
        src_table.add_column("Source")
        src_table.add_column("Records", justify="right")
        for s in sources:
            src_table.add_row(str(s[0]) if s[0] else "Unknown", f"{s[1]:,}")

        # Render Dashboard
        console.print("\n", Panel.fit("[bold white]Golden Record Dashboard[/]", style="on blue"), "\n")
        console.print(Columns([metric_panel, src_table]))
        
        # Simple Visual Progress Bar
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task("[green]Enrichment Completion", total=100)
            progress.update(task, completed=fill_rate)
        
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
                val_str = f"[green]{val}[/]" if val else "[dim red]N/A[/]"
                details.add(f"{key}: {val_str}")
        
        console.print("\n", tree, "\n")
        
    except Exception as e:
        console.print(f"[red]Inspect error:[/] {e}")

if __name__ == "__main__":
    app()
