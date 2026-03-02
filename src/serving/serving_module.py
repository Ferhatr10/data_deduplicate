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
        WITH scored_records AS (
            SELECT 
                *,
                (CASE WHEN company_name IS NOT NULL THEN 1 ELSE 0 END +
                 CASE WHEN address IS NOT NULL THEN 1 ELSE 0 END +
                 CASE WHEN city IS NOT NULL THEN 1 ELSE 0 END +
                 CASE WHEN country IS NOT NULL THEN 1 ELSE 0 END +
                 CASE WHEN website IS NOT NULL THEN 1 ELSE 0 END +
                 CASE WHEN certifications IS NOT NULL THEN 1 ELSE 0 END +
                 CASE WHEN industry_tags IS NOT NULL THEN 1 ELSE 0 END +
                 CASE WHEN founded_year IS NOT NULL THEN 1 ELSE 0 END) as completeness_score
            FROM clusters
        ),
        ranked_records AS (
            SELECT 
                cluster_id,
                unique_id,
                company_name,
                address,
                city,
                country,
                website,
                industry_tags,
                certifications,
                employees,
                founded_year,
                source_dataset,
                scraped_at,
                ROW_NUMBER() OVER (PARTITION BY cluster_id ORDER BY completeness_score DESC, unique_id ASC) as rank
            FROM scored_records
        )
        SELECT 
            cluster_id as canonical_id,
            company_name,
            address,
            city,
            country,
            website,
            industry_tags,
            certifications,
            employees,
            founded_year,
            source_dataset,
            scraped_at
        FROM ranked_records
        WHERE rank = 1
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
