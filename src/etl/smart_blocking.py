import pandas as pd
import duckdb
import re
import logging

logger = logging.getLogger(__name__)

def apply_smart_blocking(df: pd.DataFrame, threshold: int = 50000) -> pd.DataFrame:
    """
    Applies a smart token-based blocking strategy to a DataFrame of company names.
    
    1. Normalizes company names (lowercase, remove punctuation, remove B2B suffixes).
    2. Calculates token frequencies across the entire dataset.
    3. Assigns the first token with frequency < threshold as the 'blocking_key'.
    4. Marks records with only frequent tokens for manual review.
    """
    
    # 0. Connect to DuckDB
    con = duckdb.connect()
    con.register("input_df", df)
    
    # 1. Define Suffixes and Punctuation for Regex
    # Common Global B2B suffixes to remove
    legal_suffixes = [
        "gmbh", "ltd", "inc", "corp", "ag", "sa", "group", "holding", 
        "limited", "incorporated", "corporation", "co", "llc", "plc",
        "bv", "nv", "srl", "sas", "kgaa", "gmbh & co kg",
    ]
    suffixes_pattern = r'\b(' + '|'.join(legal_suffixes) + r')\b'
    
    logger.info("Normalizing names and calculating token frequencies...")
    
    # 2. SQL to normalize, tokenize, and count
    # We use DuckDB for performance on millions of rows
    sql = f"""
    -- Create a normalized version of the name
    WITH normalized AS (
        SELECT 
            unique_id,
            company_name as original_name,
            -- Lowercase, remove punctuation, remove legal suffixes
            REGEXP_REPLACE(
                REGEXP_REPLACE(
                    LOWER(company_name), 
                    '{suffixes_pattern}', 
                    '', 
                    'g'
                ),
                '[^a-z0-9 ]', 
                ' ', 
                'g'
            ) AS clean_name
        FROM input_df
    ),
    -- Tokenize and unnest
    tokens_unnested AS (
        SELECT 
            unique_id,
            TRIM(t) as token
        FROM (
            SELECT 
                unique_id, 
                unnest(string_split(REGEXP_REPLACE(clean_name, '\\s+', ' ', 'g'), ' ')) as t 
            FROM normalized
        )
        WHERE length(token) > 1 -- Skip single character tokens
    ),
    -- Count global token frequencies
    token_counts AS (
        SELECT 
            token, 
            count(*) as freq
        FROM tokens_unnested
        GROUP BY 1
    ),
    -- Join tokens with their frequencies and rank them by appearance in clean_name
    ranked_tokens AS (
        SELECT 
            t.unique_id,
            t.token,
            tc.freq,
            -- We need to maintain the order of tokens as they appear in the name
            -- DuckDB's unnest maintains order, so we can use row_number
            row_number() OVER (PARTITION BY t.unique_id) as pos
        FROM tokens_unnested t
        JOIN token_counts tc ON t.token = tc.token
    ),
    -- Select the first token with freq < threshold
    best_tokens AS (
        SELECT 
            unique_id,
            token as blocking_key,
            freq as token_freq,
            pos
        FROM ranked_tokens
        WHERE freq < {threshold}
        QUALIFY row_number() OVER (PARTITION BY unique_id ORDER BY pos) = 1
    )
    -- Final Join
    SELECT 
        i.*,
        bt.blocking_key,
        bt.token_freq,
        CASE 
            WHEN bt.blocking_key IS NULL THEN TRUE 
            ELSE FALSE 
        END as needs_manual_review
    FROM input_df i
    LEFT JOIN best_tokens bt ON i.unique_id = bt.unique_id
    """
    
    result_df = con.execute(sql).df()
    
    # Logging stats
    manual_review_count = result_df['needs_manual_review'].sum()
    logger.info(f"Smart blocking complete. Total: {len(result_df)}, Manual Review Needed: {manual_review_count}")
    
    return result_df

if __name__ == "__main__":
    # Test block
    logging.basicConfig(level=logging.INFO)
    test_data = pd.DataFrame({
        "unique_id": ["1", "2", "3", "4"],
        "company_name": [
            "Apple Inc.", 
            "Apple Europe GmbH", 
            "The Great Generic Holding Group", 
            "UniqueZyxel Trading"
        ]
    })
    
    # For testing, use a very low threshold like 2
    processed_df = apply_smart_blocking(test_data, threshold=2)
    print(processed_df[["company_name", "blocking_key", "needs_manual_review"]])
