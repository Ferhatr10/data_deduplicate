import os
import logging
import pandas as pd
import numpy as np
import duckdb
import torch
import math
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import re
from rapidfuzz import fuzz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_company_name(name):
    """
    Standardizes company names by removing common suffixes and department tags.
    Used to improve matching accuracy by reducing noise.
    """
    if not name or not isinstance(name, str):
        return ""
    
    # 1. Lowercase and strip whitespace
    n = name.lower().strip()
    
    # 2. Remove common department tags (e.g., " - automotive division")
    dept_pattern = r'\s*-\s*automotive\s*division|\s+automotive\s+division|\s+automotive'
    n = re.sub(dept_pattern, '', n).strip()
    
    # 3. Remove legal suffixes (Inc, Ltd, LLC, GmbH, AG, SA, Co, Corp, Group)
    # Loop to ensure nested suffixes are handled (e.g., "Group Ltd")
    suffix_pattern = r'\b(inc|ltd|llc|gmbh|ag|sa|co|corp|group|plc|solutions|as)\b\.*$'
    prev_n = None
    while n != prev_n:
        prev_n = n
        n = re.sub(suffix_pattern, '', n).strip()
        n = n.rstrip('., ')
        
    return n.upper() # Standardize to uppercase for consistency

class DeduplicationLayer:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2', similarity_threshold=0.88, batch_size=256):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        # Cosine Similarity -> L2 distance conversion for normalized vectors:
        # L2 = sqrt(2 * (1 - cos_sim))
        # Note: The user mentioned sqrt(2 * (1 - 0.92)) but also 0.88 threshold.
        # I will use the similarity_threshold variable for consistency.
        self.l2_threshold = math.sqrt(2 * (1 - self.similarity_threshold))
        self.batch_size = batch_size
        
        # MAC MINI (M1/M2/M3) GPU DESTEĞİ
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
            
        logger.info(f"DeduplicationLayer initialized: {model_name} | Device: {self.device}")
        logger.info(f"Thresholds: Cosine Similarity > {self.similarity_threshold} (L2 Distance < {self.l2_threshold:.4f})")
        self.model = SentenceTransformer(model_name, device=self.device)

    def generate_embeddings_with_cache(self, names, cache_path="data/embeddings_cache.npy"):
        """Loads embeddings from cache or generates them using the model."""
        if os.path.exists(cache_path):
            logger.info(f"Cache found! Loading from {cache_path}...")
            return np.load(cache_path)

        logger.info(f"Generating embeddings for {len(names)} records...")
        embeddings = self.model.encode(
            names, 
            batch_size=self.batch_size, 
            show_progress_bar=True, 
            convert_to_numpy=True,
            normalize_embeddings=True # Required for Cosine Similarity conversion
        )
        
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, embeddings)
        logger.info(f"Embeddings saved to {cache_path}.")
        return embeddings

    def find_matches_sklearn(self, unique_ids, company_names, embeddings, k=10, match_cache="data/matches_cache.parquet"):
        """
        Performs nearest neighbor search using BallTree and applies hybrid scoring.
        Semantic (Vector) + Lexical (RapidFuzz) validation.
        """
        
        # 0. Load from disk if already computed
        if os.path.exists(match_cache):
            logger.info(f"Matches cache found ({match_cache}), skipping BallTree...")
            # Return as list as expected by subsequent logic
            return pd.read_parquet(match_cache).values.tolist()

        logger.info(f"Building NearestNeighbors (BallTree) index (k={k}, metric=l2)...")
        # Prepare data as float32 contiguous array
        data = np.ascontiguousarray(embeddings).astype('float32')
        
        # BallTree is more stable for larger datasets
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree', metric='l2', n_jobs=-1)
        nn.fit(data)
        
        logger.info("Scanning for similarity matches...")
        distances, indices = nn.kneighbors(data)
        
        match_pairs = []
        n = len(unique_ids)
        for i in range(n):
            for j in range(1, k + 1): # Exclude self-match
                dist = distances[i][j]
                if dist < self.l2_threshold:
                    idx_r = indices[i][j]
                    if idx_r == -1 or idx_r >= n: continue
                    
                    id_l, id_r = unique_ids[i], unique_ids[idx_r]
                    name_l, name_r = company_names[i], company_names[idx_r]
                    
                    # CRITICAL STEP: Lexical Similarity Check
                    # Vector similarity alone can be too aggressive (prevents over-merging)
                    lexical_sim = fuzz.token_sort_ratio(name_l, name_r)
                    
                    if lexical_sim > 81: # Threshold to prevent false matches like AUTO PARTS vs MOTO PARTS
                        if id_l < id_r:
                            match_pairs.append((id_l, id_r))
                    else:
                        logger.debug(f"Over-merge prevented: {name_l} <-> {name_r} (Lexical FAIL: {lexical_sim})")
        
        # 5. Save results to cache
        os.makedirs(os.path.dirname(match_cache), exist_ok=True)
        pd.DataFrame(match_pairs, columns=['id_l', 'id_r']).to_parquet(match_cache)
        
        logger.info(f"Found {len(match_pairs)} matches. Saved to {match_cache}.")
        return match_pairs

    def run_deduplication(self, input_path, output_path, metrics=None):
        logger.info(f"Loading data from {input_path}")
        con = duckdb.connect()
        # CAST(unique_id AS VARCHAR) is clean, unlike pandas .astype(str) for bytes
        query = f"SELECT * EXCLUDE(unique_id), CAST(unique_id AS VARCHAR) as unique_id FROM read_parquet('{input_path}')"
        df = con.execute(query).df()
        
        # ID ve İsim tutarlılığı
        unique_ids = df['unique_id'].tolist()
        company_names = df['company_name'].fillna('').tolist()

        # 1. Name Normalization
        logger.info("Normalizing company names...")
        normalized_names = [normalize_company_name(n) for n in company_names]

        # 2. Vector Generation (Cache + GPU)
        embeddings = self.generate_embeddings_with_cache(normalized_names)

        # 3. Hybrid Match Search
        match_pairs = self.find_matches_sklearn(unique_ids, normalized_names, embeddings)

        # 3. Clustering
        logger.info("Generating clusters via NetworkX...")
        G = nx.Graph()
        G.add_nodes_from(unique_ids)
        G.add_edges_from(match_pairs)
        
        clusters = list(nx.connected_components(G))
        cluster_map = {uid: f"v_{i:06d}" for i, cluster in enumerate(clusters) for uid in cluster}
        df['cluster_id'] = df['unique_id'].map(cluster_map)

        # 4. Golden Record Aggregation
        logger.info("Collating Golden Records...")
        con.register("df_clustered", df)
        golden_table_sql = """
            SELECT 
                cluster_id as canonical_id,
                MODE(company_name) as primary_company_name,
                -- Deduplicated list of countries
                ARRAY_AGG(DISTINCT country) FILTER (WHERE country IS NOT NULL) as operating_countries,
                -- Aliases: Keep unique variations differing from primary name
                LIST_FILTER(
                    ARRAY_AGG(DISTINCT original_name), 
                    x -> x IS NOT NULL AND 
                         LOWER(TRIM(x)) <> LOWER(TRIM(MODE(company_name))) AND
                         -- Exclude very similar variants (e.g., legal suffix differences)
                         levenshtein(LOWER(TRIM(x)), LOWER(TRIM(MODE(company_name)))) > 3
                ) as aliases,
                MODE(website) as primary_website,
                count(*) as record_count
            FROM df_clustered
            GROUP BY cluster_id
        """
        golden_table = con.execute(golden_table_sql).df()

        # 5. Kaydet
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False)
        
        golden_out = os.path.join(os.path.dirname(os.path.dirname(output_path)), "output", "golden_table.parquet")
        os.makedirs(os.path.dirname(golden_out), exist_ok=True)
        golden_table.to_parquet(golden_out, index=False)
        
        logger.info(f"Deduplication complete! Golden Table available at: {golden_out}")
        
        if metrics:
            metrics.set_metric("dedupe_input_records", len(df))
            metrics.set_metric("dedupe_total_clusters", len(golden_table))
            metrics.set_metric("vector_model", self.model_name)
            
        return output_path

if __name__ == "__main__":
    layer = DeduplicationLayer(similarity_threshold=0.88)
    try:
        layer.run_deduplication("data/processed/standardized.parquet", "data/processed/clusters.parquet")
    except Exception as e:
        logger.error(f"Deduplication failed: {e}")
        raise e