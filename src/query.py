import duckdb
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
import os

app = typer.Typer(help="Golden Record Explorer CLI - High Performance Parquet Query Tool")
console = Console()

# --- CONFIGURATION ---
PARQUET_PATH = "data/04_golden/master.parquet"

def get_actual_path():
    if os.path.exists(PARQUET_PATH): return PARQUET_PATH
    # Fallback for compatibility during transitions
    alt_paths = [
        "data/output/company_master_records.parquet",
        "data/04_golden/enriched.parquet"
    ]
    for p in alt_paths:
        if os.path.exists(p): return p
    return PARQUET_PATH

def get_conn():
    """Connect to duckdb."""
    path = get_actual_path()
    if not os.path.exists(path):
        console.print(f"[bold red]Error:[/] File not found at {path}")
        raise typer.Exit()
    return duckdb.connect(database=':memory:')

def format_value(val):
    if val is None: return "[dim]-[/]"
    return str(val)

@app.command()
def search(query: str = typer.Argument(..., help="Company name to search for")):
    """Search for companies by name (Case-insensitive partial match)."""
    conn = get_conn()
    
    sql = f"""
        SELECT 
            canonical_id, 
            primary_company_name, 
            domain, 
            record_count,
            aliases,
            operating_countries,
            logo_url
        FROM read_parquet('{get_actual_path()}')
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

        table = Table(title=f"Search Results for '{query}'", header_style="bold magenta")
        table.add_column("ID", style="cyan")
        table.add_column("Company Name", style="white")
        table.add_column("Domain", style="green")
        table.add_column("Records", justify="right", style="yellow")
        table.add_column("Countries", style="dim")

        for row in results:
            table.add_row(
                row[0],
                row[1],
                format_value(row[2]),
                str(row[3]),
                ", ".join(row[5][:3]) + ("..." if len(row[5]) > 3 else "") if row[5] else "-"
            )

        console.print(table)
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")

@app.command()
def list_all(limit: int = typer.Option(50, help="Number of records to show")):
    """List processed records from the verified golden table."""
    conn = get_conn()
    
    sql = f"SELECT * FROM read_parquet('{get_actual_path()}') LIMIT {limit}"
    
    try:
        results = conn.execute(sql).fetchall()
        # Get column names from cursor description
        cols = [desc[0] for desc in conn.description]
        
        table = Table(title=f"Dataset Preview (Top {len(results)})", header_style="bold magenta")
        for col in cols:
            table.add_column(col)

        for row in results:
            table.add_row(*[format_value(val) for val in row])

        console.print(table)
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")

@app.command()
def inspect(canonical_id: str = typer.Argument(..., help="The canonical_id to inspect")):
    """Inspect all fields for a specific record."""
    conn = get_conn()
    
    try:
        res = conn.execute(f"SELECT * FROM read_parquet('{get_actual_path()}') WHERE canonical_id = '{canonical_id}'").fetchone()
        if not res:
            console.print(f"[red]No record found with ID: {canonical_id}[/]")
            return

        cols = [desc[0] for desc in conn.description]
        tree = Tree(f"[bold blue]Entity:[/] {res[1]}")
        
        for i, col in enumerate(cols):
            tree.add(f"[cyan]{col}:[/] {format_value(res[i])}")
        
        console.print(tree)
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")

@app.command()
def stats():
    """Show detailed deduplication and enrichment statistics."""
    raw_path = "data/02_standardized/firms.parquet"
    final_path = get_actual_path()
    
    if not os.path.exists(raw_path) or not os.path.exists(final_path):
        console.print("[red]Error:[/] Required files not found for stats evaluation.")
        return

    conn = duckdb.connect()
    
    # Sequential Statistics SQL
    stats_sql = f"""
        WITH raw AS (
            SELECT count(*) as total, count(DISTINCT company_name) as names
            FROM read_parquet('{raw_path}')
        ),
        final AS (
            SELECT 
                count(*) as clusters, 
                count(domain) FILTER (WHERE domain <> '' AND domain <> '-' AND domain <> 'Missing' AND domain <> 'unknown') as domains,
                count(logo_url) FILTER (WHERE logo_url <> '' AND logo_url <> '-') as logos,
                count(description) FILTER (WHERE description <> 'Information not found.' AND description <> 'Information not found' AND description <> 'Missing' AND description IS NOT NULL) as descriptions
            FROM read_parquet('{final_path}')
        )
        SELECT * FROM raw, final
    """
    
    try:
        res = conn.execute(stats_sql).fetchone()
        raw_rows, raw_names, final_clusters, final_domains, final_logos, final_descs = res
        
        # 1. Sequential Progress Flow
        table = Table(title="[bold blue]Data Deduplication & Enrichment Flow[/bold blue]", box=None, show_header=True)
        table.add_column("Processing Stage", style="cyan", width=30)
        table.add_column("Resulting Count", justify="right", style="white", width=20)
        table.add_column("Efficiency / Fill Rate", style="yellow", width=25)
        
        table.add_row("Initial Raw Records", f"{raw_rows:,} items", "100.0% (Base)")
        
        table.add_row(
            "   └─ After Deduplication", 
            f"{final_clusters:,} entities", 
            f"{raw_rows/final_clusters:.1f}x Consolidation"
        )
        
        table.add_row(
            "   └─ With Verified Domains", 
            f"{final_domains:,} websites", 
            f"{(final_domains/final_clusters*100):.1f}% Web Presence"
        )

        table.add_row(
            "   └─ With Brand Logos", 
            f"{final_logos:,} logos", 
            f"{(final_logos/final_clusters*100):.1f}% Visual Match"
        )
        
        table.add_row(
            "   └─ With AI Descriptions", 
            f"{final_descs:.0f} summaries", 
            f"{(final_descs/final_clusters*100):.1f}% Content Richness"
        )
        
        console.print(table)
        console.print("\n")

        # 2. Top 10 Global Entities
        top_sql = f"""
            SELECT primary_company_name, record_count, len(operating_countries) as c_count
            FROM read_parquet('{final_path}')
            ORDER BY record_count DESC
            LIMIT 10
        """
        top_results = conn.execute(top_sql).fetchall()
        
        top_table = Table(title="[bold blue]Top 10 Global Entities (by Record Volume)[/bold blue]", box=None, show_header=True)
        top_table.add_column("Company Entity", style="white bold", width=25)
        top_table.add_column("Records Merged", justify="right", style="magenta", width=20)
        top_table.add_column("Country Footprint", justify="right", style="green", width=20)
        
        for row in top_results:
            top_table.add_row(row[0], f"{row[1]:,}", f"{row[2]} Countries")
            
        console.print(top_table)
        
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")

if __name__ == "__main__":
    app()
