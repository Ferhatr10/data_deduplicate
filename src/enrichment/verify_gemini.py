import polars as pl
import os
import logging
import json
import time
from tqdm import tqdm
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv() # Load API key from .env file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
INPUT_PARQUET = "data/04_golden/enriched.parquet"
OUTPUT_PARQUET = "data/04_golden/master.parquet"
GEMINI_MODEL = "gemini-2.0-flash" 

# Configure Gemini with environment variable
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    logger.error("GOOGLE_API_KEY not found in .env file or environment!")
else:
    genai.configure(api_key=api_key)

class CompanyVerifier:
    def __init__(self):
        if not os.path.exists(INPUT_PARQUET):
            raise FileNotFoundError(f"Input file not found: {INPUT_PARQUET}")
        self.df = pl.read_parquet(INPUT_PARQUET)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        
    def generate_prompt(self, batch):
        """Creates a structured prompt for a batch of companies."""
        prompt = """
    Perform the following tasks for the provided list of companies:
    1. Check if an official corporate website exists. If 'current_domain' is invalid or missing, try to find it.
    2. Write a professional, concise summary in English (max 160 characters) ONLY if you can verify information from the company's official website.
    3. If no official website information is found, return "Information not found" for the description and "Missing" for the verified_domain.
    4. Return the result ONLY as a valid JSON array.
    """
        data_to_send = []
        for row in batch:
            data_to_send.append({
                "id": row["canonical_id"],
                "name": row["primary_company_name"],
                "current_domain": row["domain"]
            })
        
        return prompt + json.dumps(data_to_send, indent=2) + "\n\nOutput Format: [{\"id\": \"...\", \"verified_domain\": \"...\", \"description\": \"...\", \"is_automotive\": true}]"

    async def process_all(self, batch_size=10):
        records = self.df.to_dicts()
        results = []
        
        logger.info(f"Starting Gemini Verification for {len(records)} records...")
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            prompt = self.generate_prompt(batch)
            
            try:
                response = self.model.generate_content(prompt)
                # Clean JSON response (strip markdown code blocks if present)
                text = response.text.strip()
                if text.startswith("```json"):
                    text = text[7:-3].strip()
                elif text.startswith("```"):
                    text = text[3:-3].strip()
                
                batch_results = json.loads(text)
                results.extend(batch_results)
                logger.info(f"Processed {min(i+batch_size, len(records))}/{len(records)}")
                
                # Simple rate limit protection for free tier
                time.sleep(2) 
            except Exception as e:
                logger.error(f"Error processing batch at {i}: {e}")
                # Fill with defaults on error to keep IDs aligned
                for r in batch:
                    results.append({"id": r["canonical_id"], "verified_domain": "Missing", "description": "Information not found.", "is_automotive": True})

        # Merge results back to Polars
        res_df = pl.from_dicts(results)
        
        # Final join and cleanup
        self.df = self.df.join(res_df, left_on="canonical_id", right_on="id", how="left")
        
        # Update domain and add description
        self.df = self.df.with_columns([
            pl.coalesce([pl.col("verified_domain"), pl.col("domain")]).alias("domain"),
            pl.col("description").fill_null("Information not found.")
        ]).select([
            "canonical_id", "primary_company_name", "domain", "description", 
            "record_count", "operating_countries", "aliases", "logo_url"
        ])
        
        self.df.write_parquet(OUTPUT_PARQUET)
        logger.info(f"Final verified dataset saved to {OUTPUT_PARQUET}")

if __name__ == "__main__":
    verifier = CompanyVerifier()
    import asyncio
    asyncio.run(verifier.process_all())
