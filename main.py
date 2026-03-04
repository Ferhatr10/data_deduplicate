import os
import logging
import sys
import time
from src.etl.etl_module import ETLLayer
from src.deduplication.dedupe_module import DeduplicationLayer
from src.deduplication.second_pass_dedupe import run_second_pass
from src.serving.serving_module import ServingLayer
from src.enrichment.logo_enrichment import LogoEnricher
from src.utils.metrics import MetricsCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PipelineRunner")

def run_pipeline():
    # 0. Initialize Metrics
    metrics_dir = "data/metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    metrics = MetricsCollector(log_file=os.path.join(metrics_dir, "pipeline.json"))
    metrics.start_timer("total_pipeline")

    # 1. Paths
    raw_data_dir = "data/01_raw"
    standardized_path = "data/02_standardized/firms.parquet"
    clusters_path = "data/03_deduplicated/clusters.parquet"
    entities_path = "data/03_deduplicated/entities.parquet"
    refined_entities_path = "data/03_deduplicated/refined_entities.parquet"
    enriched_path = "data/04_golden/enriched.parquet"
    master_path = "data/04_golden/master.parquet"

    # 2. ETL Layer
    logger.info("=== STEP 1: ETL (Data Ingestion) ===")
    etl = ETLLayer()
    etl.ingest_and_standardize(raw_data_dir, standardized_path, metrics=metrics)

    # 3. Deduplication Layer (First Pass)
    logger.info("=== STEP 2: DEDUPLICATION (First Pass - Semantic Hybrid) ===")
    dedupe = DeduplicationLayer()
    # dedupe_module produces clusters_path and entities_path (internal logic updated)
    dedupe.run_deduplication(standardized_path, clusters_path, metrics=metrics)

    # 4. Second Pass Deduplication (PolyFuzz Refinement)
    logger.info("=== STEP 3: REFINED DEDUPLICATION (Second Pass - PolyFuzz) ===")
    run_second_pass(entities_path, refined_entities_path)

    # 5. Mega Enrichment (Bulk CSV + Kaggle + Logo)
    logger.info("=== STEP 4: MEGA ENRICHMENT (Bulk Search + Domain Unification) ===")
    from src.enrichment.enrich_pipeline import main as run_mega_enrich
    import asyncio
    asyncio.run(run_mega_enrich()) # This uses internal config pointing to refined_entities -> enriched

    # 6. AI Verification (Gemini Descriptions)
    logger.info("=== STEP 5: AI VERIFICATION (Gemini Pro) ===")
    from src.enrichment.verify_gemini import CompanyVerifier
    try:
        verifier = CompanyVerifier()
        asyncio.run(verifier.process_all()) # Internal config points to enriched -> master
    except Exception as e:
        logger.warning(f"Gemini verification failed or skipped: {e}")
        # If gemini fails, master is same as enriched for now
        import shutil
        if os.path.exists(enriched_path) and not os.path.exists(master_path):
            shutil.copy(enriched_path, master_path)

    # 7. Finalize Metrics
    metrics.stop_timer("total_pipeline")
    logger.info("=== PIPELINE COMPLETE ===")
    logger.info(f"Final Golden Master: {master_path}")
    
    metrics.log_summary()
    metrics.save_to_file()

if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)
