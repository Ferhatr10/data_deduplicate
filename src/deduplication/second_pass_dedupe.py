import os
import logging
import duckdb
import pandas as pd
import networkx as nx
from deduplication.dedupe_module import DeduplicationLayer, normalize_company_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_second_pass(input_path, output_path):
    """
    Performs second-pass deduplication on the 81k golden records.
    Focuses on merging similar names (Ltd vs Corp) that survived exact matching.
    """
    logger.info(f"Starting second-pass deduplication on {input_path}")
    
    # 1. Load the golden records
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{input_path}')").df()
    
    # We need a unique ID for the clustering phase. 
    # Since canonical_id (v_000...) is already unique per row, we use it.
    df['temp_id'] = range(len(df))
    ids = df['temp_id'].tolist()
    names = df['primary_company_name'].tolist()
    
    # 2. İsim Normalizasyonu
    logger.info("Normalization applying...")
    normalized_names = [normalize_company_name(n) for n in names]

    # 3. Use DeduplicationLayer for fuzzy matching
    # Optimized threshold (0.85)
    deduper = DeduplicationLayer(similarity_threshold=0.85)
    
    embeddings = deduper.generate_embeddings_with_cache(
        normalized_names, 
        cache_path="data/second_pass_embeddings_v5.npy"
    )
    
    # K=100 is deep enough to find variations
    match_pairs = deduper.find_matches_sklearn(
        ids, 
        normalized_names,
        embeddings, 
        k=100, 
        match_cache="data/second_pass_matches_v5.parquet"
    )
    
    # 3. Graph-Based Clustering (Final Merging Strategy)
    logger.info("Building Final Consolidation Graph (Fuzzy + Exact + Alias + Domain)...")
    G = nx.Graph()
    G.add_nodes_from(ids)

    # Rule A: Fuzzy Similarity (Deep Vector Search)
    G.add_edges_from(match_pairs)
    
    # Rule B: Suffix-Aware Core Name Bridge
    # Core name logic is now encapsulated in normalize_company_name
    core_name_to_ids = {}
    for i, name in enumerate(names):
        cn = normalize_company_name(name)
        if len(cn) > 3: 
            if cn not in core_name_to_ids: core_name_to_ids[cn] = []
            core_name_to_ids[cn].append(ids[i])
            
    for cn, group in core_name_to_ids.items():
        if len(group) > 1:
            for j in range(len(group) - 1):
                G.add_edge(group[j], group[j+1])

    # Rule C: Alias Overlap (Transitive Consistency)
    # If Record A's alias matches Record B's name/alias, they are the same entity
    alias_map = {}
    for i, row in df.iterrows():
        tid = row['temp_id']
        variants = set()
        variants.add(str(row['primary_company_name']).lower().strip())
        if 'aliases' in df.columns and isinstance(row['aliases'], list):
            for a in row['aliases']:
                if a: variants.add(str(a).lower().strip())
        
        for v in variants:
            if v not in alias_map: alias_map[v] = []
            alias_map[v].append(tid)

    for group in alias_map.values():
        if len(group) > 1:
            for j in range(len(group) - 1):
                G.add_edge(group[j], group[j+1])

    # Rule D: Domain-Based Anchor (For records that already have domains)
    if 'primary_website' in df.columns:
        domain_map = {}
        for i, row in df.iterrows():
            d = str(row['primary_website']).lower().strip()
            if d and d != 'nan' and '.' in d:
                if d not in domain_map: domain_map[d] = []
                domain_map[d].append(row['temp_id'])
        
        for group in domain_map.values():
            if len(group) > 1:
                for j in range(len(group) - 1):
                    G.add_edge(group[j], group[j+1])
    
    # Finalize Clusters
    clusters = list(nx.connected_components(G))
    cluster_map = {node_id: f"m_{i:06d}" for i, cluster in enumerate(clusters) for node_id in cluster}
    df['master_cluster_id'] = df['temp_id'].map(cluster_map)
    
    # 4. Final Aggregation
    # Merging the already merged clusters
    con.register("df_second_pass", df)
    
    # Check if record_count exists, if not default to 1
    has_count = 'record_count' in df.columns
    count_expr = "SUM(record_count)" if has_count else "COUNT(*)"

    # Re-aggregating aliases and summing record counts
    # We also keep the original canonical_id as a reference
    refined_sql = f"""
        SELECT 
            master_cluster_id as canonical_id,
            MODE(primary_company_name) as primary_company_name,
            -- Aggregate all variations
            LIST_DISTINCT(
                flatten(ARRAY_AGG(aliases)) || ARRAY_AGG(primary_company_name)
            ) as all_variations,
            COALESCE({count_expr}, 0) as total_record_count,
            LIST_DISTINCT(flatten(ARRAY_AGG(operating_countries))) as countries,
            MODE(primary_website) as domain,
            MODE(industry_tags) as industry_tags,
            MODE(source_dataset) as source_dataset
        FROM df_second_pass
        GROUP BY master_cluster_id
    """
    
    logger.info("Aggregating refined golden records...")
    refined_df = con.execute(refined_sql).df()
    
    # Final cleanup of aliases: Remove primary name AND very similar variations
    def cleanup_aliases(row):
        p_name = str(row['primary_company_name']).lower().strip()
        p_norm = normalize_company_name(p_name)
        
        clean_variants = []
        for a in row['all_variations']:
            if not a: continue
            a_str = str(a).lower().strip()
            a_norm = normalize_company_name(a_str)
            
            # Kural 1: Ana isimden farklı olmalı
            if a_str == p_name: continue
            
            # Kural 2: Normalize hali ana ismin normalize halinden farklı olmalı
            # (Bu sayede GmbH/Inc gibi farklar elenir)
            if a_norm == p_norm: continue
            
            # Kural 3: Harfsel olarak çok yakın olanları da ele (Levenshtein)
            from rapidfuzz.distance import Levenshtein
            if Levenshtein.distance(a_str, p_name) <= 3: continue
            
            clean_variants.append(a)
        return list(set(clean_variants))

    refined_df['aliases'] = refined_df.apply(cleanup_aliases, axis=1)
    
    # Drop temp columns and rename for final schema
    final_df = refined_df.rename(columns={
        'countries': 'operating_countries',
        'total_record_count': 'record_count'
    }).drop(columns=['all_variations'])
    
    # Sort by record weight
    final_df = final_df.sort_values('record_count', ascending=False)
    
    # 5. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_parquet(output_path, index=False)
    
    logger.info(f"Second-pass complete! Refined Table: {output_path}")
    logger.info(f"Reduced {len(df)} clusters to {len(final_df)} master clusters.")
    
    return output_path

if __name__ == "__main__":
    INPUT = "data/output/golden_table.parquet"
    OUTPUT = "data/output/refined_golden_table.parquet"
    
    if os.path.exists(INPUT):
        run_second_pass(INPUT, OUTPUT)
    else:
        logger.error(f"Input file not found: {INPUT}")
