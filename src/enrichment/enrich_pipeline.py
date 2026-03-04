import asyncio
import polars as pl
import kagglehub
import os
import logging
import time
import networkx as nx
from duckduckgo_search import DDGS
from urllib.parse import urlparse
from rapidfuzz import process, utils as fuzz_utils
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# --- CONFIGURATION ---
INPUT_PARQUET = "data/output/refined_golden_table.parquet"
OUTPUT_PARQUET = "data/output/enriched_golden_table.parquet"
RAW_DIR = "data/raw"
BULK_CSV_22M = os.path.join(RAW_DIR, "free_company_dataset (2).csv")

# Kaggle Datasets
KAGGE_7M = "peopledatalabssf/free-7-million-company-dataset"
KAGGE_BIGPICTURE = "mfrye0/bigpicture-company-dataset"

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MegaEnricher")

# --- UTILS ---

def clean_name(name_col):
    """Normalize company name: lowercase and strip."""
    return name_col.str.to_lowercase().str.strip_chars()

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
        
        # Filter out utility/social domains that often appear as false positives
        bad_domains = {'linkedin.com', 'facebook.com', 'twitter.com', 'wikipedia.org', 
                       'google.com', 'baidu.com', 'github.com', 'youtube.com', 'instagram.com',
                       'juraforum.de', 'weforum.org', 'motogp.com'}
        if domain in bad_domains: return None
        return domain
    except: return None

# --- FUZZY MATCHING (Optimized) ---

def match_batch(batch, ref_names, ref_domains, threshold):
    """Worker function for parallel fuzzy matching."""
    results = {}
    for name in batch:
        match = process.extractOne(
            name, 
            ref_names, 
            processor=fuzz_utils.default_process, 
            score_cutoff=threshold
        )
        if match:
            _, score, idx = match
            results[name] = ref_domains[idx]
    return results

def parallel_fuzzy_match(missing_names, ref_names, ref_domains, threshold=92):
    """Orchestrate parallel fuzzy matching across processes."""
    if not missing_names or not ref_names: return {}
    
    num_workers = os.cpu_count() or 4
    batch_size = max(1, len(missing_names) // (num_workers * 2))
    batches = [missing_names[i:i + batch_size] for i in range(0, len(missing_names), batch_size)]
    
    logger.info(f"Fuzzy Match: Matching {len(missing_names)} names against {len(ref_names)} ref samples...")
    results_map = {}
    start = time.time()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        func = partial(match_batch, ref_names=ref_names, ref_domains=ref_domains, threshold=threshold)
        for i, partial_results in enumerate(executor.map(func, batches)):
            results_map.update(partial_results)
            if (i+1) % 10 == 0 or (i+1) == len(batches):
                logger.debug(f"Progress: {i+1}/{len(batches)} batches done.")

    logger.info(f"Fuzzy matching finished in {time.time()-start:.1f}s. New domains found: {len(results_map)}")
    return results_map

# --- CORE LOGIC ---

class MegaEnricher:
    def __init__(self):
        self.df = None
        self.ref_auto = None  # Unified automotive reference
        self.ref_global = None # Unified global reference
        
    def load_cached_state(self):
        """Resume from output file if it exists, otherwise start fresh."""
        if os.path.exists(OUTPUT_PARQUET):
            logger.info(f"Resuming from existing enrichment: {OUTPUT_PARQUET}")
            self.df = pl.read_parquet(OUTPUT_PARQUET)
        else:
            logger.info(f"Starting fresh enrichment from: {INPUT_PARQUET}")
            self.df = pl.read_parquet(INPUT_PARQUET)
            if "domain" not in self.df.columns:
                # Prioritize existing website columns in golden table
                cols = self.df.columns
                pref_website = next((c for c in ["primary_website", "website", "domain"] if c in cols), None)
                if pref_website:
                    self.df = self.df.with_columns(pl.col(pref_website).alias("domain"))
                else:
                    self.df = self.df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("domain"))

    def download_data(self):
        """Ensure all Kaggle datasets are present."""
        logger.info("Verifying/Downloading Kaggle datasets...")
        try:
            self.path_7m = kagglehub.dataset_download(KAGGE_7M)
            self.path_big = kagglehub.dataset_download(KAGGE_BIGPICTURE)
            logger.info("Kaggle data ready.")
        except Exception as e:
            logger.error(f"Kaggle download failed: {e}")
            raise

    def build_unified_reference(self):
        """Scan all 3 bulk sources and build Automotive/Global reference sets."""
        logger.info("Scanning bulk sources (22M + Kaggle) into Unified Reference...")
        
        # 1. Load 22M CSV (SKIPPED for memory safety)
        # lf_22m = pl.scan_csv(BULK_CSV_22M, quote_char=None, truncate_ragged_lines=True, infer_schema_length=0, encoding="utf8-lossy").select([
        #     pl.col("name").alias("ref_name"),
        #     pl.col("website").alias("ref_domain"),
        #     pl.col("industry").alias("ref_industry")
        # ])
        
        # 2. Load Kaggle 7M (Assuming standard names in that dataset)
        # Note: We look for CSVs in the downloaded path
        csv_7m = next(f for f in os.listdir(self.path_7m) if f.endswith('.csv'))
        lf_7m = pl.scan_csv(os.path.join(self.path_7m, csv_7m), infer_schema_length=0).select([
            pl.col("name").alias("ref_name"),
            pl.col("domain").alias("ref_domain"),
            pl.col("industry").alias("ref_industry")
        ])

        # 3. Load Kaggle BigPicture
        csv_big = next(f for f in os.listdir(self.path_big) if f.endswith('.csv'))
        lf_big = pl.scan_csv(os.path.join(self.path_big, csv_big), infer_schema_length=0).select([
            pl.col("name").alias("ref_name"),
            pl.col("website").alias("ref_domain"),
            pl.col("industry").alias("ref_industry")
        ])

        # Combine all (excluding 22M CSV)
        lf_unified = pl.concat([lf_7m, lf_big]).unique(subset=["ref_name"]).drop_nulls(subset=["ref_name", "ref_domain"])
        
        # Stage A: Automotive Subset (Fuzzy target)
        self.ref_auto = lf_unified.filter(
            pl.col("ref_industry").str.contains("(?i)auto|motor|vehicle")
        ).collect()
        
        # Stage B: Global Global (Direct Join target)
        self.ref_global = lf_unified
        
        logger.info(f"Reference logic built. Automotive sample size: {len(self.ref_auto)}")

    def run_stage_bulk_csv_chunked(self, chunk_size=3_000_000):
        """Memory-safe chunked scan of the 22M CSV for missing domains."""
        logger.info(f"Stage 1.5: Chunked Scan of 22M CSV (Chunk Size: {chunk_size})...")
        mask_missing = (pl.col("domain").is_null()) | (pl.col("domain") == "")
        missing_df = self.df.filter(mask_missing)
        
        if missing_df.height == 0: return
        
        # Create a lookup map for missing names (clean -> original)
        missing_map = {n.lower().strip(): n for n in missing_df["primary_company_name"].to_list()}
        matches_found = {}

        try:
            reader = pl.read_csv_batched(
                BULK_CSV_22M, 
                quote_char=None, 
                truncate_ragged_lines=True, 
                infer_schema_length=0, 
                encoding="utf8-lossy"
            )
            
            chunk_idx = 0
            while True:
                batches = reader.next_batches(5) # 5 batches per iteration
                if not batches: break
                
                chunk_idx += 1
                df_chunk = pl.concat(batches).select([
                    pl.col("name").alias("ref_name"),
                    pl.col("website").alias("ref_domain")
                ]).drop_nulls()
                
                # Fast inner join/overlap check
                df_chunk = df_chunk.with_columns(clean_name(pl.col("ref_name")).alias("match_key"))
                
                # Check for hits
                for row in df_chunk.iter_rows(named=True):
                    key = row["match_key"]
                    if key in missing_map and key not in matches_found:
                        matches_found[key] = row["ref_domain"]
                
                # Early exit if all found
                if len(matches_found) == len(missing_map): break
                
                if chunk_idx % 2 == 0:
                    logger.info(f"Processed ~{chunk_idx * 5}M rows... Found: {len(matches_found)}/{len(missing_map)}")

            # Update main DF
            if matches_found:
                self.df = self.df.with_columns(
                    pl.struct(["primary_company_name", "domain"]).map_elements(
                        lambda x: matches_found.get(x["primary_company_name"].lower().strip(), x["domain"]),
                        return_dtype=pl.Utf8
                    ).alias("domain")
                )
                logger.info(f"Chunked scan completed. Found {len(matches_found)} new domains.")
        
        except Exception as e:
            logger.error(f"Chunked scan failed: {e}")

    def run_stage_direct_join(self):
        """Ultra-fast exact name matching against global dataset."""
        logger.info("Stage 1: Global Direct Join (Exact match)...")
        mask_missing = (pl.col("domain").is_null()) | (pl.col("domain") == "")
        count_before = self.df.filter(mask_missing).height
        
        if count_before == 0: return

        # Prep lazy join
        lf_data = self.df.lazy().with_columns(clean_name(pl.col("primary_company_name")).alias("match_key"))
        lf_ref = self.ref_global.with_columns(clean_name(pl.col("ref_name")).alias("match_key")).select(["match_key", "ref_domain"])
        
        res = lf_data.join(lf_ref, on="match_key", how="left").collect()
        
        # FIX: Ensure we don't multiply rows if ref has duplicates
        res = res.unique(subset=["canonical_id"], keep="first")
        
        self.df = res.with_columns(
            pl.coalesce([pl.col("domain"), pl.col("ref_domain")]).alias("domain")
        ).drop(["match_key", "ref_domain"])
        
        count_after = self.df.filter((pl.col("domain").is_null()) | (pl.col("domain") == "")).height
        logger.info(f"Direct Join filled {count_before - count_after} gaps.")

    def run_stage_fuzzy_automotive(self, threshold=92):
        """High-precision fuzzy matching against prioritized automotive subset."""
        logger.info("Stage 2: Automotive Fuzzy Join (Priority Match)...")
        mask_missing = (pl.col("domain").is_null()) | (pl.col("domain") == "")
        df_missing = self.df.filter(mask_missing)
        
        if len(df_missing) == 0: return
        
        unique_names = df_missing["primary_company_name"].unique().to_list()
        ref_names = self.ref_auto["ref_name"].to_list()
        ref_domains = self.ref_auto["ref_domain"].to_list()
        
        match_map = parallel_fuzzy_match(unique_names, ref_names, ref_domains, threshold)
        
        if match_map:
            self.df = self.df.with_columns(
                pl.struct(["primary_company_name", "domain"]).map_elements(
                    lambda x: match_map.get(x["primary_company_name"], x["domain"]) if (not x["domain"] or x["domain"] == "") else x["domain"],
                    return_dtype=pl.Utf8
                ).alias("domain")
            )


    def run_stage_domain_unification(self):
        """
        The Data Seal: 
        Merges master clusters that share the same domain after enrichment.
        """
        logger.info("Stage 4: Post-Enrichment Domain Unification...")
        mask_has_domain = (pl.col("domain").is_not_null()) & (pl.col("domain") != "")
        df_rich = self.df.filter(mask_has_domain)
        
        if df_rich.height == 0: return

        # Group by domain to find potential merges
        domain_groups = df_rich.group_by("domain").agg(pl.col("canonical_id")).filter(pl.col("canonical_id").list.len() > 1)
        
        if domain_groups.height == 0:
            logger.info("No new domain-based merges found.")
            return

        logger.info(f"Found {domain_groups.height} domain groups requiring unification.")
        
        # Build a graph of canonical_ids that need to be merged
        G = nx.Graph()
        G.add_nodes_from(self.df["canonical_id"].to_list())
        for row in domain_groups.iter_rows(named=True):
            ids_to_merge = row["canonical_id"]
            for i in range(len(ids_to_merge) - 1):
                G.add_edge(ids_to_merge[i], ids_to_merge[i+1])
        
        # Recalculate clusters
        new_clusters = list(nx.connected_components(G))
        new_map = {cid: next(iter(ids)) for ids in new_clusters for cid in ids if len(ids) > 1}
        
        if not new_map: 
            logger.info("No overlaps found in search results.")
            return

        # Apply unification and re-aggregate
        before_count = self.df.height
        self.df = self.df.with_columns(
            pl.col("canonical_id").map_dict(new_map, default=pl.col("canonical_id")).alias("canonical_id")
        )
        
        # Re-aggregate to collapse the newly merged clusters
        self.df = self.df.group_by("canonical_id").agg([
            pl.col("primary_company_name").mode().first(),
            pl.col("aliases").flatten().unique(),
            pl.col("record_count").sum(),
            pl.col("operating_countries").flatten().unique(),
            pl.col("domain").mode().first(),
            pl.col("industry_tags").mode().first(),
            pl.col("source_dataset").mode().first()
        ])
        
        after_count = self.df.height
        logger.info(f"Domain Unification complete. {before_count} -> {after_count} clusters.")

    def save(self):
        """Write current state to output, cleaning domains on the way."""
        self.df = self.df.with_columns(pl.col("domain").map_elements(clean_domain, return_dtype=pl.Utf8).alias("domain"))
        self.df.write_parquet(OUTPUT_PARQUET)
        
        filled = self.df["domain"].is_not_null().sum()
        total = len(self.df)
        logger.info(f"Checkpointed to {OUTPUT_PARQUET}. Current Fill: {filled}/{total} ({filled/total*100:.2f}%)")

async def main():
    enricher = MegaEnricher()
    
    # Preparations
    enricher.load_cached_state()
    enricher.download_data()
    enricher.build_unified_reference()
    
    # Stage 1: Fast Exact Match (Multi-Source Global - Kaggle Only)
    enricher.run_stage_direct_join()
    enricher.save()
    
    # Stage 1.5: Chunked Scan (22M CSV Bulk)
    enricher.run_stage_bulk_csv_chunked()
    enricher.save()
    
    # Stage 2: Precision Fuzzy Match (Automotive Subset)
    enricher.run_stage_fuzzy_automotive(threshold=94)
    enricher.save()

    # Stage 3: Final Domain Unification
    enricher.run_stage_domain_unification()
    enricher.save()
    
    logger.info("Mega-Enricher Flow Complete.")

if __name__ == "__main__":
    asyncio.run(main())
