import os
import logging
import duckdb
import pandas as pd
import networkx as nx
from polyfuzz import PolyFuzz
from polyfuzz.models import RapidFuzz
from rapidfuzz import fuzz
from .dedupe_module import normalize_company_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_second_pass(input_path, output_path):
    """
    Performs second-pass deduplication using PolyFuzz + RapidFuzz (Token Sort).
    Focuses on merging similar names (e.g., "Koza Bilişim" vs "Bilişim Koza") 
    that survived vector matching.
    """
    logger.info(f"Starting REFINED second-pass deduplication on {input_path}")
    
    # 1. Load the golden records
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{input_path}')").df()
    
    # Unique IDs and Names
    df['temp_id'] = range(len(df))
    ids = df['temp_id'].tolist()
    names = df['primary_company_name'].tolist()
    
    # 2. Name Normalization (Core Name Search)
    logger.info("Applying core name normalization...")
    normalized_names = [normalize_company_name(n) for n in names]

    # 3. PolyFuzz with RapidFuzz (Token Sort)
    # This specifically addresses "Bilişim Koza" vs "Koza Bilişim"
    logger.info("Running PolyFuzz clustering (RapidFuzz Token Sort)...")
    
    # We use a custom RapidFuzz model with token_sort_ratio
    rf_model = RapidFuzz(n_jobs=-1, scorer=fuzz.token_sort_ratio)
    model = PolyFuzz(rf_model)
    
    # Clustering 40k names
    model.match(normalized_names, normalized_names)
    
    # Extract match results
    # PolyFuzz returns a dataframe with 'From', 'To', and 'Similarity'
    matches = model.get_matches()
    
    # 4. Graph-Based Clustering
    logger.info("Building Final Consolidation Graph...")
    G = nx.Graph()
    G.add_nodes_from(ids)

    # Rule A: PolyFuzz Matches (Threshold 0.90 for high confidence token-sort)
    # Note: PolyFuzz matches are 0-1, token_sort_ratio is 0-100
    for _, row in matches.iterrows():
        if row['Similarity'] > 0.90:
            idx_from = normalized_names.index(row['From'])
            idx_to = normalized_names.index(row['To'])
            if idx_from != idx_to:
                G.add_edge(ids[idx_from], ids[idx_to])

    # Rule B: Exact Core Name Bridge (Reinforced)
    core_name_to_ids = {}
    for i, cn in enumerate(normalized_names):
        if len(cn) > 3: 
            if cn not in core_name_to_ids: core_name_to_ids[cn] = []
            core_name_to_ids[cn].append(ids[i])
            
    for group in core_name_to_ids.values():
        if len(group) > 1:
            for j in range(len(group) - 1):
                G.add_edge(group[j], group[j+1])

    # Rule C: Alias Overlap
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

    # Rule D: Domain-Based Anchor - REMOVED AS REQUESTED
    
    # Finalize Clusters
    clusters = list(nx.connected_components(G))
    cluster_map = {node_id: f"m_{i:06d}" for i, cluster in enumerate(clusters) for node_id in cluster}
    df['master_cluster_id'] = df['temp_id'].map(cluster_map)
    
    # 5. Final Aggregation
    con.register("df_second_pass", df)
    
    # Check if record_count exists
    has_count = 'record_count' in df.columns
    count_expr = "SUM(record_count)" if has_count else "COUNT(*)"

    refined_sql = f"""
        SELECT 
            master_cluster_id as canonical_id,
            MODE(primary_company_name) as primary_company_name,
            LIST_DISTINCT(
                flatten(ARRAY_AGG(aliases)) || ARRAY_AGG(primary_company_name)
            ) as all_variations,
            COALESCE({count_expr}, 0) as record_count,
            LIST_DISTINCT(flatten(ARRAY_AGG(operating_countries))) as operating_countries,
            MODE(primary_website) as domain
        FROM df_second_pass
        GROUP BY master_cluster_id
    """
    
    logger.info("Aggregating refined golden records...")
    refined_df = con.execute(refined_sql).df()
    
    # Final cleanup: Apply normalization to primary name and deduplicate aliases
    def refine_record(row):
        # 1. Clean Primary Name (Strip suffixes for final output)
        raw_primary = row['primary_company_name']
        clean_primary = normalize_company_name(raw_primary)
        
        # 2. Deduplicate Aliases (Case-insensitive unique set)
        p_name_lower = clean_primary.lower().strip()
        seen = {p_name_lower}
        unique_aliases = []
        
        # We sort by length to keep "Inc" variants as aliases if they are different from root
        # but the user said "tekilleştirilmiş... bir örneği kalsın"
        # So we keep original casing but only one version per string
        for a in row['all_variations']:
            if not a: continue
            a_str = str(a).strip()
            a_lower = a_str.lower()
            
            # Skip if already seen (case insensitive)
            if a_lower in seen: continue
            
            # Skip if too similar to primary (to avoid minor typos/punctuation diffs)
            if fuzz.ratio(a_lower, p_name_lower) > 98: 
                seen.add(a_lower)
                continue
                
            unique_aliases.append(a_str)
            seen.add(a_lower)
            
        return pd.Series([clean_primary, unique_aliases])

    refined_df[['primary_company_name', 'aliases']] = refined_df.apply(refine_record, axis=1)
    
    # Drop temp columns
    final_df = refined_df.drop(columns=['all_variations'])
    final_df = final_df.sort_values('record_count', ascending=False)
    
    # 6. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_parquet(output_path, index=False)
    
    logger.info(f"Second-pass complete! Master Clusters: {len(final_df)}")
    return output_path

if __name__ == "__main__":
    INPUT = "data/03_deduplicated/entities.parquet"
    OUTPUT = "data/03_deduplicated/refined_entities.parquet"
    if os.path.exists(INPUT):
        run_second_pass(INPUT, OUTPUT)
    else:
        logger.error(f"Input file not found: {INPUT}")
