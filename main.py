import os
import logging
import sys
import time
from src.etl.etl_module import ETLLayer
from src.deduplication.dedupe_module import DeduplicationLayer
from src.serving.serving_module import ServingLayer
from src.utils.metrics import MetricsCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PipelineRunner")

def run_pipeline():
    # 0. Initialize Metrics
    metrics = MetricsCollector(log_file="data/pipeline_metrics.json")
    metrics.start_timer("total_pipeline")

    # 1. Paths
    raw_data_dir = "data/raw"
    standardized_path = "data/processed/standardized.parquet"
    clusters_path = "data/processed/clusters.parquet"
    golden_table_path = "data/output/golden_table.parquet"

    # 2. ETL Layer
    logger.info("=== STEP 1: ETL ===")
    etl = ETLLayer()
    etl.ingest_and_standardize(raw_data_dir, standardized_path, metrics=metrics)

    # 3. Deduplication Layer
    logger.info("=== STEP 2: DEDUPLICATION ===")
    dedupe = DeduplicationLayer()
    dedupe.run_deduplication(standardized_path, clusters_path, metrics=metrics)

    # 4. Serving Layer
    logger.info("=== STEP 3: SERVING ===")
    serving = ServingLayer()
    serving.generate_golden_table(clusters_path, golden_table_path, metrics=metrics)

    # 5. Finalize Metrics
    metrics.stop_timer("total_pipeline")
    logger.info("=== PIPELINE COMPLETE ===")
    logger.info(f"Final output: {golden_table_path}")
    
    # Log and save metrics
    metrics.log_summary()
    metrics.save_to_file()

if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)
