import os
import logging
import duckdb
import pandas as pd
import networkx as nx
from deduplication.dedupe_module import DeduplicationLayer

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
    
    # 2. Use DeduplicationLayer for fuzzy matching
    # Optimized threshold (0.85) - High precision while allowing fuzzy variations
    deduper = DeduplicationLayer(similarity_threshold=0.85)
    
    # Force re-calculation by adding _v3 to cache paths
    embeddings = deduper.generate_embeddings_with_cache(
        names, 
        cache_path="data/second_pass_embeddings_v4.npy"
    )
    
    # K=100 is deep enough to find variations
    match_pairs = deduper.find_matches_sklearn(
        ids, 
        embeddings, 
        k=100, 
        match_cache="data/second_pass_matches_v4.parquet"
    )
    
    # 3. Clustering
    logger.info("Building second-pass clusters (NetworkX)...")
    G = nx.Graph()
    G.add_nodes_from(ids)
    G.add_edges_from(match_pairs)
    
    # NEW: Safety net - Force merge records with EXACT SAME primary names
    # This prevents the K-search limit from leaving duplicates
    name_to_ids = {}
    for i, name in enumerate(names):
        n = name.lower().strip()
        if n not in name_to_ids: name_to_ids[n] = []
        name_to_ids[n].append(ids[i])
        
    for group_ids in name_to_ids.values():
        for j in range(len(group_ids) - 1):
            G.add_edge(group_ids[j], group_ids[j+1])
    
    clusters = list(nx.connected_components(G))
    cluster_map = {node_id: f"m_{i:06d}" for i, cluster in enumerate(clusters) for node_id in cluster}
    df['master_cluster_id'] = df['temp_id'].map(cluster_map)
    
    # 4. Platinum-Tier Aggregation
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
            -- Combine existing aliases with primary names that are now being merged
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
    
    # Final cleanup of aliases: Remove the primary name from the alias list
    def cleanup_aliases(row):
        p_name = row['primary_company_name'].lower().strip()
        return [a for a in row['all_variations'] if a.lower().strip() != p_name]

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
