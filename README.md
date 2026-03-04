# ⚙️ AI-Powered B2B Data Deduplication Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![DuckDB](https://img.shields.io/badge/database-DuckDB-orange.svg)](https://duckdb.org/)
[![Sentence-Transformers](https://img.shields.io/badge/NLP-Sentence--Transformers-red.svg)](https://www.sbert.net/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview

This project is a high-performance **autonomous data engineering pipeline** designed to standardize and deduplicate B2B company data (firmographics) from disparate, messy sources including **CSV, JSONL, and Parquet**.

Unlike traditional deterministic rule engines, this system leverages **Natural Language Processing (NLP)**, **Aggressive Suffix Normalization**, and **Advanced Clustering (PolyFuzz)** to resolve enterprise entities at scale. It consolidates nearly **half a million** records into a high-fidelity **"Golden Table"** (~142 unique global entities) with enriched domains, logos, and AI-generated insights.

---

## 🛤️ Architectural Evolution (Roadmap)

| Version / Approach | Technology Stack | Trade-off & Decision |
| --- | --- | --- |
| **v1. Rule-Based** | *Splink* | High variation in global B2B metadata made manual rule-tuning too complex. |
| **v2. Brute-Force** | *DuckDB (Jaro)* | Faced $O(N^2)$ complexity. Comparing 477k records directly exhausted RAM. |
| **v3. Vectorial** | *FAISS/HNSW* | Fast but unstable on Apple M-series (Segmentation Faults). |
| **v4. Spatial Tree** | *BallTree (K-NN)* | Efficient first-pass partitioning. Identified 1.1M candidate matches. |
| **v5. Intelligent Clustering** | **PolyFuzz + RapidFuzz** | **(Current)** Uses `fuzz.token_sort_ratio` to handle word permutations (e.g., "Bosch Gmbh" vs "Gmbh Bosch") and aggressive suffix normalization for maximum recall. |

---

## 🛠️ System Architecture

### 1. ETL & Standardisation (`etl_module.py`)
- **Aggressive Normalization:** Strips legal suffixes (Inc, Ltd, GmbH, S.A.), divisions, and noise from company names to find "root" entities.
- **Vectorized Engine:** Powered by **DuckDB** for memory-efficient ingestion of 477k+ records.

### 2. High-Fidelity Deduplication (`dedupe_module.py` & `second_pass_dedupe.py`)
- **Semantic First Pass:** Maps names to **384D vectors** for spatial proximity search.
- **PolyFuzz Second Pass:** Uses Token Sort logic to group name variations that semantic search might miss due to word ordering.
- **Intelligent Aliasing:** Retains unique name variations as aliases while keeping the primary name clean.

### 3. Mega-Enrichment Pipeline (`enrich_pipeline.py`)
- **Domain Discovery:** Scans **22M+ company records** and Kaggle datasets to find missing official websites.
- **Domain-Based Unification:** Further collapses entities that share a verified website, reaching **~142 global master entities**.
- **Visual Assets:** Enriches logos via CompanyEnrich API with Google Favicon fallbacks.
- **AI Verification:** (Optional) Uses **Gemini 1.5/2.0** to generate professional company summaries and verify automotive industry classification.

---

## 🔍 Data Exploration (CLI)

The project includes a powerful CLI tool (`src/query.py`) for real-time interaction:

- **`stats`**: Live dashboard showing the full **Sequential Progress Flow** (Raw -> Deduplicated -> Enriched).
- **`search <query>`**: Partial match search across primary names, aliases, and domains.
- **`inspect <canonical_id>`**: Detailed tree-view of an entity's complete history, countries, and enrichment status.
- **`list-all`**: paginated view of the final Golden Records.

---

## 📊 Performance Metrics

*Latest autonomous pipeline results:*

- **Raw Records Processed:** 477,600 rows
- **Consolidated Master Entities:** **142**
- **Consolidation Ratio:** **~3363x**
- **Domain Coverage (Fill Rate):** **>80%**
- **Noise Reduction:** **~88%** (Unique names 1,166 -> 142)
- **Total Pipeline Execution:** ~480 Seconds (Cold Start) / < 10 Seconds (Cached)

---

## 🚀 Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env and add GOOGLE_API_KEY for Gemini enrichment
cp .env.example .env

# Run the complete pipeline
python main.py

# 🔍 Explorer CLI Usage
python src/query.py stats
python src/query.py search "Bosch"
python src/query.py inspect "m_000001"
```

---
*Optimized for High-Performance Entity Resolution and Automotive Supply Chain Intelligence.*
