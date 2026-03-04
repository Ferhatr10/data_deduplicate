# ⚙️ AI-Powered B2B Data Deduplication Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![DuckDB](https://img.shields.io/badge/database-DuckDB-orange.svg)](https://duckdb.org/)
[![Sentence-Transformers](https://img.shields.io/badge/NLP-Sentence--Transformers-red.svg)](https://www.sbert.net/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview

This project is a high-performance **autonomous data engineering pipeline** designed to standardize and deduplicate B2B company data (firmographics) from disparate, messy sources including **CSV, JSONL, and Parquet**.

Unlike traditional deterministic rule engines, this system leverages **Natural Language Processing (NLP)** for semantic vectorization, **Spatial Indexing** for efficient retrieval, and **Graph Theory** for entity resolution. It processes nearly **half a million** complex records using in-memory analytical engines to produce a unified, consolidated **"Golden Table"** of unique corporate entities.

---

## 🛤️ Architectural Evolution (Roadmap)

To reach the current stable and high-performance architecture, several approaches were tested against the dataset's scale (**477k+ records**) and hardware constraints (**Apple Silicon ARM architecture**).

| Version / Approach | Technology Stack | Trade-off & Decision |
| --- | --- | --- |
| **v1. Rule-Based Linkage** | *Splink* | High variation in global B2B metadata (languages, missing fields) made manual rule-tuning too complex. Switched to a "Zero-Shot" semantic NLP approach. |
| **v2. Character Fuzzy Join** | *DuckDB (Jaro-Winkler)* | Faced $O(N^2)$ complexity. Comparing 477k records directly exhausted RAM, leading to `OOM - Killed` errors. |
| **v3. Vectorial HNSW** | *FAISS* | Excellent search speed, but FAISS's C++ memory alignment requirements caused `Segmentation Faults` on Apple M-series chips. Stability was insufficient. |
| **v4. Spatial Tree (Current)** | *Scikit-Learn (BallTree)* | Switched to **BallTree** for Python-native memory management. Successfully identified **1.1M matches** without memory overflow. Integrated **MD5 Hashing** for deterministic ID consistency. |

---

## 🛠️ System Architecture

The system is designed as 3 independent, memory-safe layers:

### 1. ETL Layer (`etl_module.py`)
The entry point of the pipeline. To bypass Python/Pandas memory overhead, all ingestion and transformation is handled by **DuckDB's vectorized SQL engine**.
- **Lazy Evaluation:** Ingests CSV, JSONL, and Parquet directly from disk without full-memory loading.
- **Sanitization:** Uses regex to clean whitespace and HTTP prefixes. Standardizes country names (e.g., 'DE', 'GERMANY') via SQL mapping.
- **Deterministic IDs:** Uses MD5 hashing of source metadata to ensure **idempotency** across pipeline runs, preventing broken graph links.

### 2. AI Deduplication Layer (`dedupe_module.py`)
The core semantic engine. Replaces brute-force text matching with spatial vector mathematics.
- **Vectorization:** Uses `paraphrase-multilingual-MiniLM-L12-v2` to map names into **384-dimensional vectors**. Accelerated by Apple Silicon GPU (**MPS**).
- **BallTree K-NN:** Performs neighbor searches in $O(N \log N)$ time. Uses `ball_tree` for optimal performance with high-dimensional embeddings.
- **Hybrid Scoring:** Combines semantic similarity with **RapidFuzz** lexical validation to prevent "over-merging" of similar but distinct entities.

### 3. Enrichment & Verification Layer
Finalizes entity resolution and enriches records with high-fidelity corporate metadata.
- **Bulk Data Enrichment (`enrich_pipeline.py`):** Automatically completes missing domains and metadata by scanning **22M+ company records** and curated Kaggle datasets using both exact and high-precision fuzzy joins.
- **AI Verification (`verify_gemini.py`):** Uses **Gemini 2.0 Flash** to autonomously research and verify the most complex entities, generating professional English summaries and validating corporate domains.
- **Logo Enrichment (`logo_enrichment.py`):** Fetches official corporate logos via **CompanyEnrich API** with a **Google Favicon** fallback mechanism.
- **Graph Clustering:** Uses `NetworkX` to group millions of matching pairs into closed subgraphs (entities), collapsed into Golden Records via DuckDB `MODE` and `ARRAY_AGG` functions.

---

## 🔍 Data Exploration (CLI)

The project includes a high-performance CLI tool (`src/query.py`) built with **Typer** and **Rich** for real-time interaction with the Golden Table.

- **`search <name>`**: Fast partial matching across primary names, aliases, and domains.
- **`list-all`**: paginated view of consolidated entities.
- **`inspect <id>`**: Detailed tree-view of a specific entity, including all its variations and metadata.
- **`stats`**: Live dashboard showing deduplication efficiency and enrichment coverage.

---

## 📊 Performance Metrics

*Latest autonomous pipeline run results (recorded via `metrics_module.py`):*

- **Raw Records Processed:** 477,600 rows
- **Consolidated Golden Records:** 81,129 entities
- **Deduplication / Compression Rate:** **~83%**
- **Hardware Acceleration:** Apple Metal Performance Shaders (MPS)
- **Total Pipeline Execution (Cold Start):** **471.57 Seconds** (~7.8 minutes)
- **Cached Execution Time:** **< 10 Seconds**

---

## 🚀 Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
python main.py

# 🔍 Explorer CLI Usage
python src/query.py search "Google"
python src/query.py stats
python src/query.py inspect "m_001234"
```

---
*Created for High-Performance Entity Resolution and B2B Data Quality.*
