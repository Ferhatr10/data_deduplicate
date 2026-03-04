import duckdb
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ETLLayer:
    def __init__(self, db_path=":memory:"):
        self.con = duckdb.connect(db_path)
    
    def ingest_and_standardize(self, raw_data_dir, output_path, metrics=None):
        """
        Ingests CSV, JSONL, and Parquet files from raw_data_dir,
        standardizes them to a unified schema, and saves to output_path.
        """
        logger.info(f"Starting ingestion from {raw_data_dir}")
        if metrics:
            metrics.start_timer("etl_ingestion")
        
        # SQL templates for different formats
        queries = []
        file_counts = {}
        
        # CSV
        csv_files = [f for f in os.listdir(raw_data_dir) if f.endswith(".csv")]
        if csv_files:
            queries.append(f"SELECT * FROM read_csv_auto('{raw_data_dir}/*.csv', header=True, ignore_errors=True, union_by_name=True)")
            file_counts["csv"] = len(csv_files)
            
        # JSONL
        json_files = [f for f in os.listdir(raw_data_dir) if f.endswith(".jsonl")]
        if json_files:
            queries.append(f"SELECT * FROM read_json_auto('{raw_data_dir}/*.jsonl', format='newline_delimited')")
            file_counts["jsonl"] = len(json_files)
            
        # Parquet
        parquet_files = [f for f in os.listdir(raw_data_dir) if f.endswith(".parquet")]
        if parquet_files:
            queries.append(f"SELECT * FROM read_parquet('{raw_data_dir}/*.parquet')")
            file_counts["parquet"] = len(parquet_files)
            
        if not queries:
            raise ValueError(f"No valid files found in {raw_data_dir}")
            
        if metrics:
            metrics.set_metric("etl_files_processed", sum(file_counts.values()))
            for fmt, count in file_counts.items():
                metrics.set_metric(f"etl_{fmt}_files", count)

        format_views = []
        
        # Helper to get columns for a temporary table or view
        def get_cols(name):
            return [col[1] for col in self.con.execute(f"PRAGMA table_info('{name}')").fetchall()]

        # CSV Processing
        if csv_files:
            self.con.execute(f"CREATE OR REPLACE VIEW raw_csv AS SELECT * FROM read_csv_auto('{raw_data_dir}/*.csv', header=True, ALL_VARCHAR=True, ignore_errors=True, union_by_name=True)")
            cols = get_cols('raw_csv')
            
            def safe(options, default="CAST(NULL AS VARCHAR)"):
                for opt in options:
                    if opt in cols: return f"CAST({opt} AS VARCHAR)"
                return default

            self.con.execute(f"""
                CREATE OR REPLACE VIEW std_csv AS
                SELECT 
                    md5(concat_ws('_', 'csv', row_number() over(), {safe(['company_name', 'scraped_name', 'name'])})) as raw_id,
                    {safe(['company_name', 'scraped_name', 'name'])} as raw_company_name,
                    {safe(['website', 'url'])} as raw_website,
                    {safe(['country', 'country_code'])} as raw_country,
                    {safe(['city'])} as raw_city,
                    {safe(['address'])} as raw_address,
                    {safe(['industry_tags', 'tags', 'industry'])} as raw_industry,
                    {safe(['certifications'])} as raw_certs,
                    {safe(['employees'])} as raw_employees,
                    {safe(['founded_year'])} as raw_founded_year,
                    {safe(['source_dataset', 'source'], "'csv'")} as raw_source,
                    {safe(['scraped_at'], 'current_date')} as raw_date
                FROM raw_csv
            """)
            format_views.append("std_csv")
            
        # JSONL Processing
        if json_files:
            self.con.execute(f"CREATE OR REPLACE VIEW raw_json AS SELECT * FROM read_json_auto('{raw_data_dir}/*.jsonl', format='newline_delimited')")
            cols = get_cols('raw_json')
            
            def safe(options, default="CAST(NULL AS VARCHAR)"):
                for opt in options:
                    if opt in cols: return f"CAST({opt} AS VARCHAR)"
                    # Handle nested (location.country)
                    if "." in opt:
                        base = opt.split(".")[0]
                        if base in cols: return f"CAST({opt} AS VARCHAR)"
                return default

            # For JSON, we specifically handle ARRAY/STRUCT to VARCHAR conversion
            self.con.execute(f"""
                CREATE OR REPLACE VIEW std_json AS
                SELECT 
                    md5(concat_ws('_', 'jsonl', row_number() over(), {safe(['company_name', 'scraped_name', 'name'])})) as raw_id,
                    {safe(['company_name', 'scraped_name', 'name'])} as raw_company_name,
                    {safe(['website', 'url'])} as raw_website,
                    {safe(['country', 'location.country', 'country_code'])} as raw_country,
                    {safe(['city', 'location.city'])} as raw_city,
                    {safe(['address', 'location.address'])} as raw_address,
                    {safe(['industry_tags', 'tags', 'industry'])} as raw_industry,
                    {safe(['certifications'])} as raw_certs,
                    {safe(['employees', 'employee_count'])} as raw_employees,
                    {safe(['founded_year', 'est_year'])} as raw_founded_year,
                    {safe(['source_dataset', 'source'], "'jsonl'")} as raw_source,
                    {safe(['scraped_at'], 'current_date')} as raw_date
                FROM raw_json
            """)
            format_views.append("std_json")
            
        # Parquet Processing
        if parquet_files:
            self.con.execute(f"CREATE OR REPLACE VIEW raw_parquet AS SELECT * FROM read_parquet('{raw_data_dir}/*.parquet')")
            cols = get_cols('raw_parquet')
            
            def safe(options, default="CAST(NULL AS VARCHAR)"):
                for opt in options:
                    if opt in cols: return f"CAST({opt} AS VARCHAR)"
                return default

            self.con.execute(f"""
                CREATE OR REPLACE VIEW std_parquet AS
                SELECT 
                    md5(concat_ws('_', 'parquet', row_number() over(), {safe(['company_name', 'scraped_name', 'name'])})) as raw_id,
                    {safe(['company_name', 'scraped_name', 'name'])} as raw_company_name,
                    {safe(['website', 'url'])} as raw_website,
                    {safe(['country', 'country_code'])} as raw_country,
                    {safe(['city'])} as raw_city,
                    {safe(['address'])} as raw_address,
                    {safe(['industry_tags', 'tags', 'industry'])} as raw_industry,
                    {safe(['certifications'])} as raw_certs,
                    {safe(['employees'])} as raw_employees,
                    {safe(['founded_year'])} as raw_founded_year,
                    {safe(['source_dataset', 'source'], "'parquet'")} as raw_source,
                    {safe(['scraped_at'], 'current_date')} as raw_date
                FROM raw_parquet
            """)
            format_views.append("std_parquet")

        if not format_views:
            raise ValueError("No formats to process")

        # 3. Combine standardized views
        union_query = "\nUNION ALL\n".join([f"SELECT * FROM {v}" for v in format_views])
        self.con.execute(f"CREATE OR REPLACE TABLE normalized_temp AS {union_query}")

        # Final Standardization with Advanced Cleaning
        standardization_query = """
        CREATE OR REPLACE TABLE standardized_suppliers AS
        SELECT 
            raw_id as unique_id,
            raw_company_name as original_name,
            REGEXP_REPLACE(LOWER(TRIM(raw_company_name)), '\\s+', ' ', 'g') as company_name,
            REGEXP_REPLACE(LOWER(TRIM(raw_website)), '^(https?://)?(www\\.)?', '') as website,
            CASE 
                WHEN UPPER(TRIM(raw_country)) IN ('DE', 'DEUTSCHLAND', 'GERMANY') THEN 'GERMANY'
                WHEN UPPER(TRIM(raw_country)) IN ('JP', 'JAPAN') THEN 'JAPAN'
                WHEN UPPER(TRIM(raw_country)) IN ('FR', 'FRANCE') THEN 'FRANCE'
                WHEN UPPER(TRIM(raw_country)) IN ('US', 'USA', 'UNITED STATES') THEN 'UNITED STATES'
                WHEN UPPER(TRIM(raw_country)) IN ('CA', 'CANADA') THEN 'CANADA'
                WHEN UPPER(TRIM(raw_country)) IN ('SE', 'SWEDEN') THEN 'SWEDEN'
                WHEN UPPER(TRIM(raw_country)) IN ('MX', 'MEXICO') THEN 'MEXICO'
                WHEN UPPER(TRIM(raw_country)) IN ('ES', 'SPAIN') THEN 'SPAIN'
                WHEN UPPER(TRIM(raw_country)) IN ('KR', 'SOUTH KOREA', 'KOREA', 'REPUBLIC OF KOREA') THEN 'SOUTH KOREA'
                WHEN UPPER(TRIM(raw_country)) IN ('IE', 'IRELAND') THEN 'IRELAND'
                WHEN UPPER(TRIM(raw_country)) IN ('IN', 'INDIA') THEN 'INDIA'
                WHEN UPPER(TRIM(raw_country)) IN ('TR', 'TURKEY', 'TÜRKİYE') THEN 'TURKEY'
                ELSE UPPER(TRIM(raw_country)) 
            END as country,
            LOWER(TRIM(raw_city)) as city,
            LOWER(TRIM(raw_address)) as address,
            LOWER(TRIM(raw_industry)) as industry_tags,
            UPPER(TRIM(raw_certs)) as certifications,
            TRIM(raw_employees) as employees,
            TRY_CAST(raw_founded_year AS INTEGER) as founded_year,
            raw_source as source_dataset,
            TRY_CAST(raw_date AS DATE) as scraped_at
        FROM normalized_temp
        WHERE raw_company_name IS NOT NULL AND TRIM(raw_company_name) != ''
        """
        
        logger.info("Executing robust standardization query...")
        self.con.execute(standardization_query)
        
        # Get count of standardized records
        result = self.con.execute("SELECT COUNT(*) FROM standardized_suppliers").fetchone()
        row_count = result[0] if result else 0
        if metrics:
            metrics.set_metric("etl_total_rows", row_count)

        # Save to processed directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.con.execute(f"COPY standardized_suppliers TO '{output_path}' (FORMAT PARQUET)")
        
        if metrics:
            metrics.stop_timer("etl_ingestion")
        
        logger.info(f"Standardized data saved to {output_path} ({row_count} rows)")
        return output_path

if __name__ == "__main__":
    # Local test
    etl = ETLLayer()
    # Assuming generate_test_data.py was run
    try:
        etl.ingest_and_standardize("data/01_raw", "data/02_standardized/firms.parquet")
    except Exception as e:
        logger.error(f"ETL failed: {e}")
