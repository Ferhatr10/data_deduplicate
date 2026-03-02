import duckdb
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServingLayer:
    def __init__(self, db_path=":memory:"):
        self.con = duckdb.connect(db_path)

    def generate_golden_table(self, cluster_path, output_path, metrics=None):
        """
        Takes the clustered data and produces a 'golden table' where
        each cluster is represented by a single canonical record.
        """
        logger.info(f"Generating golden table from {cluster_path}")
        if metrics:
            metrics.start_timer("serving_process")
        
        # Load clusters
        self.con.execute(f"CREATE OR REPLACE TABLE clusters AS SELECT * FROM read_parquet('{cluster_path}')")
        
        # Get count of independent clusters
        cluster_count = self.con.execute("SELECT COUNT(DISTINCT cluster_id) FROM clusters").fetchone()[0]
        if metrics:
            metrics.set_metric("serving_input_clusters", cluster_count)

        # Logic for Golden Record:
        # In a real-world scenario, this might involve source priority or data completeness.
        # Here, we use a simple 'first record in cluster' or 'most complete record' approach.
        # We'll calculate a 'completeness_score' based on non-null fields.
        
        golden_query = """
        CREATE OR REPLACE TABLE golden_table AS
        WITH cluster_stats AS (
            SELECT 
                cluster_id,
                mode(company_name) as primary_company_name,
                mode(address) as canonical_address,
                mode(city) as canonical_city,
                mode(website) as primary_website,
                mode(industry_tags) as canonical_industry,
                mode(certifications) as canonical_certifications,
                mode(employees) as canonical_employees,
                mode(founded_year) as canonical_founded_year,
                ARRAY_AGG(DISTINCT country) FILTER (WHERE country IS NOT NULL) as operating_countries,
                ARRAY_AGG(DISTINCT company_name) FILTER (WHERE company_name IS NOT NULL) as all_names,
                MAX(scraped_at) as latest_scraped_at,
                mode(source_dataset) as primary_source
            FROM clusters
            GROUP BY cluster_id
        )
        SELECT 
            cluster_id as canonical_id,
            primary_company_name,
            list_filter(all_names, x -> x <> primary_company_name) as aliases,
            canonical_address as address,
            canonical_city as city,
            operating_countries,
            primary_website,
            canonical_industry as industry_tags,
            canonical_certifications as certifications,
            canonical_employees as employees,
            canonical_founded_year as founded_year,
            primary_source as source_dataset,
            latest_scraped_at as scraped_at
        FROM cluster_stats
        """
        
        self.con.execute(golden_query)
        
        # Get count of golden records
        golden_count = self.con.execute("SELECT COUNT(*) FROM golden_table").fetchone()[0]
        if metrics:
            metrics.set_metric("serving_golden_records", golden_count)

        # Save output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.con.execute(f"COPY golden_table TO '{output_path}' (FORMAT PARQUET)")
        
        if metrics:
            metrics.stop_timer("serving_process")
        
        logger.info(f"Golden table saved to {output_path} ({golden_count} records)")
        return output_path

if __name__ == "__main__":
    serving = ServingLayer()
    try:
        serving.generate_golden_table("data/processed/clusters.parquet", "data/output/golden_table.parquet")
    except Exception as e:
        logger.error(f"Serving failed: {e}")
