---
name: splink-duckdb
description: >
  Expert agent for probabilistic record linkage and deduplication using Splink with DuckDB backend.
  Use this skill whenever the user wants to: deduplicate datasets, link records across tables,
  find matching entities without unique IDs, perform entity resolution, fuzzy match records,
  or use Splink with DuckDB. Trigger even for partial requests like "match these two tables",
  "find duplicate records", "entity resolution", or "record linkage". If the user uploads CSVs
  or DataFrames and asks to find matches/duplicates, always use this skill.
---

# Splink + DuckDB Agent

Splink is a Python library for **probabilistic record linkage** (entity resolution) — deduplication and cross-table matching for data without unique identifiers. DuckDB is the default and recommended backend: fast, parallel, runs on a laptop.

## Core Concepts (Must Know)

| Concept | Description |
|---|---|
| `link_type` | `"dedupe_only"` (one table), `"link_only"` (two+ tables, no dedup), `"link_and_dedupe"` |
| **Blocking Rules** | Narrow down candidate pairs before comparison (performance critical) |
| **Comparisons** | How to score similarity between fields (fuzzy, exact, date, etc.) |
| **Training** | Estimate `m` and `u` probabilities via EM algorithm |
| **Inference** | `linker.inference.predict()` → pairwise match scores |
| **Clustering** | Group predictions into unique entity clusters |

## Standard Workflow

### 1. Install & Import
```python
pip install splink
```
```python
import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on
import pandas as pd
```

### 2. Initialize DuckDB Backend
```python
db_api = DuckDBAPI()
# Optional: pass an existing DuckDB connection
# import duckdb; con = duckdb.connect(); db_api = DuckDBAPI(con)
```

### 3. Define Settings
```python
settings = SettingsCreator(
    link_type="dedupe_only",  # or "link_only", "link_and_dedupe"
    comparisons=[
        cl.NameComparison("first_name"),
        cl.JaroWinklerAtThresholds("surname", [0.9, 0.7]),
        cl.DateOfBirthComparison("dob", input_is_string=True),
        cl.ExactMatch("city").configure(term_frequency_adjustments=True),
        cl.EmailComparison("email"),
    ],
    blocking_rules_to_generate_predictions=[
        block_on("first_name"),
        block_on("surname"),
        block_on("dob"),
    ],
)
```

### 4. Create Linker
```python
# Single table (dedupe)
linker = Linker(df, settings, db_api)

# Two tables (link_only)
linker = Linker(
    [df_left, df_right],
    settings,
    db_api,
    input_table_aliases=["source_a", "source_b"],
)
```

### 5. Train Model
```python
# Step 1: Estimate baseline match probability
linker.training.estimate_probability_two_random_records_match(
    [block_on("first_name", "surname")],
    recall=0.7,
)

# Step 2: Estimate u-probabilities (non-match rates)
linker.training.estimate_u_using_random_sampling(max_pairs=1e6)

# Step 3: Estimate m-probabilities via EM (run multiple with different blocking rules)
linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("first_name", "surname")
)
linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("dob")
)
```

### 6. Predict & Cluster
```python
# Get pairwise predictions
pairwise_predictions = linker.inference.predict(threshold_match_probability=0.8)
# or use threshold_match_weight=-5 for broader capture

# Convert to pandas
df_predictions = pairwise_predictions.as_pandas_dataframe()

# Cluster into unique entities
clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
    pairwise_predictions, threshold_match_probability=0.95
)
df_clusters = clusters.as_pandas_dataframe()
```

---

## Comparison Library Reference

Read `references/comparisons.md` for full details. Quick reference:

| Use Case | Class |
|---|---|
| Names (persons) | `cl.NameComparison("col")` |
| Surnames / strings | `cl.JaroWinklerAtThresholds("col", [0.9, 0.7])` |
| Dates of birth | `cl.DateOfBirthComparison("col", input_is_string=True)` |
| Email addresses | `cl.EmailComparison("col")` |
| Exact match only | `cl.ExactMatch("col")` |
| Distances/numbers | `cl.DistanceFunctionAtThresholds("col", "levenshtein", [1, 2])` |
| Postcodes | `cl.PostcodeComparison("col")` |

Term frequency adjustments (for skewed columns like city, country):
```python
cl.ExactMatch("city").configure(term_frequency_adjustments=True)
```

---

## Blocking Rules Guide

Read `references/blocking.md` for performance guidance. Key patterns:

```python
# Simple field match
block_on("surname")

# Multi-field (AND logic)
block_on("first_name", "dob")

# SQL expression blocking (substring)
block_on("substr(first_name, 1, 3)", "substr(surname, 1, 4)")

# Analyze blocking before running
from splink.blocking_analysis import cumulative_comparisons_to_be_scored_from_blocking_rules_chart
cumulative_comparisons_to_be_scored_from_blocking_rules_chart(
    table_or_tables=df,
    blocking_rules=[block_on("surname"), block_on("dob")],
    db_api=db_api,
    link_type="dedupe_only",
)
```

**Rule of thumb:** Aim for < 100M candidate pairs total. Use the cumulative chart to check.

---

## Exploratory Analysis (Pre-Linker)

```python
from splink.exploratory import completeness_chart, profile_columns

# Check data completeness
completeness_chart(df, cols=["first_name", "surname", "dob"], db_api=DuckDBAPI())

# Profile column distributions
profile_columns(df, db_api=DuckDBAPI(), cols=["surname", "dob"])
```

---

## Deterministic (Rules-Based) Dedup

When probabilistic is overkill or for quick wins:

```python
settings = SettingsCreator(
    link_type="dedupe_only",
    blocking_rules_to_generate_predictions=[
        block_on("first_name", "surname", "dob"),
        block_on("surname", "dob", "postcode"),
    ],
    retain_intermediate_calculation_columns=True,
)
linker = Linker(df, settings, db_api=db_api)
df_predict = linker.inference.deterministic_link()
```

---

## Saving & Loading Models

```python
# Save trained model
linker.misc.save_model_to_json("my_model.json")

# Load and reuse
linker2 = Linker(df, "my_model.json", db_api)
preds = linker2.inference.predict(threshold_match_probability=0.9)
```

---

## Visualisations & Diagnostics

```python
# Match weight chart (post-training)
linker.visualisations.match_weights_chart()

# Comparison viewer (sample of pairs)
linker.visualisations.comparison_viewer_dashboard(
    pairwise_predictions, "dashboard.html", overwrite=True
)

# Cluster studio (inspect entity clusters)
linker.visualisations.cluster_studio_dashboard(
    pairwise_predictions, clusters, "clusters.html",
    sampling_method="by_cluster_size", overwrite=True
)
```

---

## Common Pitfalls

1. **Missing `unique_id` column** — Input DataFrames must have a `unique_id` column. If absent, add one: `df["unique_id"] = range(len(df))`
2. **Splink 3 vs Splink 4** — API changed significantly. This skill targets **Splink 4** (`pip install splink` gets v4+). Old syntax like `DuckDBLinker(df, settings)` is Splink 3.
3. **EM training blocked columns** — EM cannot estimate m-values for columns used in the training blocking rule. Use multiple training sessions with different blocking rules.
4. **Too many comparisons** — If blocking is too loose, performance tanks. Always check with `cumulative_comparisons_to_be_scored_from_blocking_rules_chart`.
5. **Threshold tuning** — `threshold_match_probability=0.9` is strict; `threshold_match_weight=-5` is loose. Use ROC charts for tuning if ground truth is available.

---

## Reference Files

- `references/comparisons.md` — Full comparison library with all options and parameters
- `references/blocking.md` — Advanced blocking patterns and performance optimization
