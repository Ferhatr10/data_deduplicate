import os
import logging
import asyncio
import polars as pl
import duckdb
import lancedb
from lancedb.pydantic import LanceModel, Vector
import kagglehub
import torch
from sentence_transformers import SentenceTransformer
from urllib.parse import urlparse
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_PARQUET = "data/output/enriched_golden_table.parquet"
OUTPUT_PARQUET = "data/output/final_enriched_table.parquet"
VECTOR_STORE_PATH = "data/vectorstore"
BULK_CSV_22M = "free_company_dataset (2).csv"
CHUNK_SIZE = 10000  # Increased for faster batching
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Ensure directories
os.makedirs("data/output", exist_ok=True)
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VectorEnricher")

# --- UTILS ---

def clean_domain(url):
    """Extract clean domain from URL, filtering utility sites."""
    if not url: return None
    try:
        if not isinstance(url, str): return None
        url = url.strip().lower()
        if not url.startswith(('http://', 'https://')): url = 'http://' + url
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain.startswith('www.'): domain = domain[4:]
        if '.' not in domain or len(domain.split('.')[-1]) < 2: return None
        
        # Filter out utility/social domains
        bad_domains = {'linkedin.com', 'facebook.com', 'twitter.com', 'wikipedia.org', 
                       'google.com', 'baidu.com', 'github.com', 'youtube.com', 'instagram.com',
                       'juraforum.de', 'weforum.org', 'motogp.com'}
        if domain in bad_domains: return None
        return domain
    except: return None

def is_automotive(name):
    """Check if company name suggests automotive industry."""
    keywords = ["auto", "motor", "vehicle", "car", "truck", "automotive", "parts", "racing", "tire"]
    name_lower = name.lower()
    return any(kw in name_lower for kw in keywords)

# --- VECTOR STORE MANAGEMENT ---

class VectorManager:
    def __init__(self, db_path):
        self.db = lancedb.connect(db_path)
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
        
    def build_index(self, force=False):
        """Unified reference data indexing into LanceDB (Focus on Automotive)."""
        if "automotive_ref" in self.db.table_names() and not force:
            logger.info("Automotive Index already exists. Skipping build.")
            return

        logger.info("Building Automotive Index...")
        con = duckdb.connect()
        
        con.execute(f"CREATE VIEW csv_22m AS SELECT name as ref_name, website as ref_domain, industry as ref_industry FROM read_csv_auto('{BULK_CSV_22M}', ignore_errors=true, strict_mode=false, all_varchar=true)")
        
        import kagglehub
        path_7m = kagglehub.dataset_download("peopledatalabssf/free-7-million-company-dataset")
        path_big = kagglehub.dataset_download("mfrye0/bigpicture-company-dataset")
        
        con.execute(f"CREATE VIEW kaggle_7m AS SELECT name as ref_name, domain as ref_domain, industry as ref_industry FROM read_csv('{path_7m}/*.csv')")
        con.execute(f"CREATE VIEW kaggle_big AS SELECT name as ref_name, website as ref_domain, industry as ref_industry FROM read_csv('{path_big}/*.csv')")
        
        con.execute("""
            CREATE TABLE unified_ref AS 
            SELECT DISTINCT ref_name, ref_domain, ref_industry 
            FROM (
                SELECT * FROM csv_22m UNION ALL 
                SELECT * FROM kaggle_7m UNION ALL 
                SELECT * FROM kaggle_big
            ) 
            WHERE ref_name IS NOT NULL AND ref_domain IS NOT NULL
        """)
        
        # Only Automotive
        auto_df = con.execute("SELECT ref_name, ref_domain FROM unified_ref WHERE regexp_matches(ref_industry, '(?i)auto|motor|vehicle')").pl()
        self._create_table("automotive_ref", auto_df)
        logger.info("Automotive Index build complete.")

    def _create_table(self, name, df):
        """Helper to create embedded table in LanceDB."""
        logger.info(f"Indexing {name} ({len(df)} records)...")
        
        # Iterator for batching
        def data_gen():
            pbar = tqdm(total=len(df), desc=f"Indexing {name}", unit="rec")
            for i in range(0, len(df), CHUNK_SIZE):
                batch = df.slice(i, CHUNK_SIZE)
                names = batch["ref_name"].to_list()
                embeddings = self.model.encode(names, convert_to_tensor=False, show_progress_bar=False)
                records = []
                for name_val, domain_val, vector in zip(names, batch["ref_domain"].to_list(), embeddings):
                    records.append({"name": name_val, "domain": domain_val, "vector": vector})
                yield records
                pbar.update(len(batch))
            pbar.close()

        list_tables = self.db.table_names()
        if name in list_tables:
            self.db.drop_table(name)
            
        self.db.create_table(name, data=data_gen())

    def search(self, name, table_name, limit=1):
        """Semantic search in specified table with Cosine Similarity."""
        table = self.db.open_table(table_name)
        vector = self.model.encode([name], convert_to_tensor=False, show_progress_bar=False)[0]
        # Search using Cosine Distance (1 - similarity)
        # Proper LanceDB syntax: .search(vector).metric("cosine")
        query = table.search(vector).metric("cosine").limit(limit)
        results = query.to_list()
        if results:
            res = results[0]
            # distance in cosine metric is 1 - similarity
            score = 1 - res["_distance"]
            return res["domain"], score
        return None, 0

# --- MAIN ENGINE ---

class VectorEnricher:
    def __init__(self):
        self.vm = VectorManager(VECTOR_STORE_PATH)
        self.df = None

    def load_data(self):
        if not os.path.exists(INPUT_PARQUET):
            logger.error(f"Input file not found: {INPUT_PARQUET}")
            return False
            
        self.df = pl.read_parquet(INPUT_PARQUET)
        # Ensure domain column exists
        if "domain" not in self.df.columns:
            self.df = self.df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("domain"))
        return True

    def enrich(self):
        mask_missing = (pl.col("domain").is_null()) | (pl.col("domain") == "")
        missing_df = self.df.with_row_count("row_nr").filter(mask_missing)
        
        if len(missing_df) == 0:
            logger.info("No missing domains to enrich. Stopping.")
            return

        missing_indices = missing_df["row_nr"].to_list()
        names_to_process = missing_df["primary_company_name"].to_list()

        logger.info(f"Semantic Search: Processing {len(missing_indices)} missing domains...")
        
        # We'll store updates in a dict to avoid slow row-level updates in Polars
        updates = {}
        new_domains_count = 0
        pbar = tqdm(total=len(missing_indices))
        
        for idx_in_missing, company_name in enumerate(names_to_process):
            row_idx = missing_indices[idx_in_missing]
            
            if not company_name or len(str(company_name)) < 2:
                pbar.update(1)
                continue
                
            found_domain = None
            
            # Step 1: Automotive Search
            domain, score = self.vm.search(company_name, "automotive_ref")
            
            if score > 0.94:
                found_domain = domain
            elif score > 0.90:
                if is_automotive(company_name) or is_automotive(str(domain)):
                    found_domain = domain
            
            # Step 2: Global Search is SKIPPED as per request
            
            if found_domain:
                cleaned = clean_domain(found_domain)
                if cleaned:
                    updates[row_idx] = cleaned
                    new_domains_count += 1
            
            pbar.update(1)
            if (idx_in_missing + 1) % 100 == 0:
                logger.info(f"Progress: Processed {idx_in_missing+1}/{len(missing_indices)}, Found {new_domains_count} domains.")

        # Apply all updates at once
        if updates:
            # Create a dataframe from updates
            update_df = pl.DataFrame({
                "row_nr": list(updates.keys()),
                "new_domain": list(updates.values())
            }).with_columns(pl.col("row_nr").cast(pl.UInt32))
            
            self.df = self.df.with_row_count("row_nr").join(
                update_df, on="row_nr", how="left"
            ).with_columns(
                pl.coalesce([pl.col("domain"), pl.col("new_domain")]).alias("domain")
            ).drop(["row_nr", "new_domain"])

        logger.info(f"Vector Enrichment Complete. Found {new_domains_count} new domains.")

    def save(self):
        self.df.write_parquet(OUTPUT_PARQUET)
        logger.info(f"Results saved to {OUTPUT_PARQUET}")

if __name__ == "__main__":
    enricher = VectorEnricher()
    if enricher.load_data():
        # User confirms automotive is indexed, so force=False by default
        enricher.vm.build_index(force=False) 
        enricher.enrich()
        enricher.save()
