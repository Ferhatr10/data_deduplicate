import polars as pl
import os
import logging
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LogoEnricher")

class LogoEnricher:
    def __init__(self):
        self.primary_api_template = "https://api.companyenrich.com/logo/{domain}"
        self.fallback_api_template = "https://www.google.com/s2/favicons?domain={domain}&sz=256"

    def check_url_valid(self, url):
        """Checks if the logo URL actually returns a valid image/response."""
        try:
            # Short timeout to avoid hanging the pipeline
            response = requests.head(url, timeout=3, allow_redirects=True)
            return response.status_code == 200
        except:
            return False

    def get_logo_url(self, domain):
        """
        Attempts to find a logo URL for a given domain.
        1. CompanyEnrich API
        2. Google Favicon (Fallback)
        3. None
        """
        if not domain or domain == "Missing" or domain == "nan":
            return None

        domain = domain.strip().lower()
        
        # Strategy 1: CompanyEnrich
        primary_url = self.primary_api_template.format(domain=domain)
        if self.check_url_valid(primary_url):
            return primary_url

        # Strategy 2: Google Favicon Fallback
        fallback_url = self.fallback_api_template.format(domain=domain)
        if self.check_url_valid(fallback_url):
            return fallback_url

        return None

    def enrich_dataframe(self, df, domain_col="domain"):
        """Enriches a Polars DataFrame with logo URLs."""
        if domain_col not in df.columns:
            logger.warning(f"Column {domain_col} not found in DataFrame. Skipping logo enrichment.")
            return df

        domains = df[domain_col].to_list()
        
        logger.info(f"Enriching logos for {len(domains)} records...")
        
        # Parallel processing for speed (IO bound)
        with ThreadPoolExecutor(max_workers=10) as executor:
            logo_urls = list(tqdm(executor.map(self.get_logo_url, domains), total=len(domains)))

        return df.with_columns(pl.Series("logo_url", logo_urls))

    def enrich_parquet(self, input_path, output_path):
        """Reads parquet, enriches logos, and saves back."""
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            return

        df = pl.read_parquet(input_path)
        enriched_df = self.enrich_dataframe(df)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        enriched_df.write_parquet(output_path)
        logger.info(f"Logo enriched data saved to {output_path}")
        return output_path

if __name__ == "__main__":
    # Test with sample domains
    enricher = LogoEnricher()
    test_domains = ["google.com", "apple.com", "nonexistent-domain-12345.com"]
    for d in test_domains:
        logo = enricher.get_logo_url(d)
        print(f"Domain: {d} -> Logo: {logo}")
