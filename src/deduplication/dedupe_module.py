import splink.comparison_library as cl
import splink.blocking_rule_library as brl
from splink import DuckDBAPI, Linker
import duckdb
import os
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeduplicationLayer:
    def __init__(self, settings=None):
        self.settings = settings or self._default_settings()

    def _default_settings(self):
        # Splink v4 settings structure
        return {
            "link_type": "dedupe_only",
            "blocking_rules_to_generate_predictions": [
                brl.block_on("country", "website"),
                brl.block_on("country", "company_name"),
                brl.block_on("country", "address"),
            ],
            "comparisons": [
                cl.ExactMatch("country"),
                cl.LevenshteinAtThresholds("company_name", [2]),
                cl.JaroWinklerAtThresholds("address", [0.9]),
                cl.LevenshteinAtThresholds("website", [2]),
            ],
            "retain_matching_columns": True,
            "retain_intermediate_calculation_columns": True,
        }

    def run_deduplication(self, input_path, output_path, metrics=None):
        logger.info(f"Loading data for deduplication from {input_path}")
        if metrics:
            metrics.start_timer("deduplication_process")
        
        # Load data into DuckDB 
        con = duckdb.connect()
        # Explicitly cast unique_id to VARCHAR to avoid UUID type issues with Arrow/Splink
        df = con.execute(f"SELECT * EXCLUDE (unique_id), CAST(unique_id AS VARCHAR) as unique_id FROM read_parquet('{input_path}')").df()
        
        if metrics:
            metrics.set_metric("dedupe_input_records", len(df))

        db_api = DuckDBAPI(connection=con)
        
        linker = Linker(df, self.settings, db_api)
        
        # 1. EM Training (Unsupervised)
        logger.info("Starting EM Training...")
        linker.training.estimate_u_using_random_sampling(max_pairs=1e6)
        
        # Estimate m for specific columns
        # Using country as a blocking rule for training is safer for small datasets
        linker.training.estimate_parameters_using_expectation_maximisation(brl.block_on("country"))
        
        # In a real pipeline, we'd add more rules, but for this prototype, 
        # we'll stick to country to ensure convergence with tiny test data.

        # 2. Predict Clusters
        logger.info("Predicting matches...")
        df_predict = linker.inference.predict(threshold_match_probability=0.9)
        
        # 3. Clustering
        logger.info("Generating clusters...")
        df_clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(df_predict, threshold_match_probability=0.9)
        
        # Get metrics from clusters
        clusters_df = df_clusters.as_pandas_dataframe()
        # Ensure IDs are strings to avoid Arrow conversion issues with UUIDs
        clusters_df["cluster_id"] = clusters_df["cluster_id"].astype(str)
        clusters_df["unique_id"] = clusters_df["unique_id"].astype(str)
        
        num_clusters = clusters_df["cluster_id"].nunique()
        if metrics:
            metrics.set_metric("dedupe_total_clusters", num_clusters)

        # Save cluster results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Convert to pandas/parquet for portability
        clusters_df.to_parquet(output_path, index=False)
        
        if metrics:
            metrics.stop_timer("deduplication_process")
        
        logger.info(f"Clustering complete. Found {num_clusters} clusters. Results saved to {output_path}")
        return output_path

if __name__ == "__main__":
    # Local test
    dedupe = DeduplicationLayer()
    try:
        dedupe.run_deduplication("data/processed/standardized.parquet", "data/processed/clusters.parquet")
    except Exception as e:
        logger.error(f"Deduplication failed: {e}")
        raise e
