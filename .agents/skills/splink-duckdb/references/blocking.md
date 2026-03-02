# Blocking Rules — Advanced Guide

## Why Blocking Matters

Without blocking, every record is compared to every other: O(n²) comparisons.
Blocking reduces this to manageable pairs by only comparing records that share at least one blocking key.

**Target:** < 100M candidate pairs for DuckDB on a laptop. < 1B for large machines.

---

## Basic Patterns

```python
from splink import block_on

# Single column
block_on("surname")

# Multiple columns (AND logic — both must match)
block_on("first_name", "dob")

# SQL expression (substring)
block_on("substr(first_name, 1, 3)", "substr(surname, 1, 4)")
block_on("substr(postcode, 1, 3)", "dob")

# Year only from date string
block_on("substr(dob, 1, 4)", "first_name")
```

---

## Blocking Rule Design Strategy

Use **multiple loose rules** (OR logic) rather than one strict rule:
- Each rule catches different error types
- Union of all rules = candidate pairs

```python
blocking_rules = [
    block_on("substr(first_name, 1, 3)", "substr(surname, 1, 4)"),  # name prefix
    block_on("surname", "dob"),           # surname + dob
    block_on("first_name", "dob"),        # firstname + dob
    block_on("postcode", "first_name"),   # local + name
    block_on("dob", "birth_place"),       # dob + place
]
```

---

## Analyze Before Running

Always check pair counts before running the full linkage:

```python
from splink.blocking_analysis import (
    cumulative_comparisons_to_be_scored_from_blocking_rules_chart,
)

chart = cumulative_comparisons_to_be_scored_from_blocking_rules_chart(
    table_or_tables=df,
    blocking_rules=blocking_rules,
    db_api=db_api,
    link_type="dedupe_only",   # or "link_only"
)
chart  # Shows bar chart — cumulative pair count per rule
```

---

## Training Blocking Rules vs Prediction Blocking Rules

These are different:
- **Prediction blocking rules** → `blocking_rules_to_generate_predictions` in `SettingsCreator`. Defines which pairs get a match score.
- **Training blocking rules** → passed to `estimate_parameters_using_expectation_maximisation()`. Temporary, just for EM training.

Training rules should be **tight** (few pairs, mostly true matches):
```python
linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("first_name", "surname")   # training rule: tight
)
```

Prediction rules should be **loose** (high recall):
```python
SettingsCreator(
    blocking_rules_to_generate_predictions=[
        block_on("surname"),   # loose — catches transpositions
        block_on("first_name"),
        block_on("dob"),
    ]
)
```

---

## SQL String Blocking Rules

For complex logic, use raw SQL strings:
```python
settings = SettingsCreator(
    blocking_rules_to_generate_predictions=[
        "l.surname = r.surname",
        "l.dob = r.dob",
        "levenshtein(l.first_name, r.first_name) <= 1 and l.surname = r.surname",
    ]
)
```

---

## Performance Tips for DuckDB

1. **Substring blocking** reduces comparisons without losing much recall:
   `block_on("substr(first_name, 1, 2)", "substr(dob, 1, 4)")`

2. **Sort by unique_id** before loading — DuckDB uses sorted joins more efficiently.

3. **Larger datasets (>1M rows):** Set `max_pairs` higher for u-sampling:
   ```python
   linker.training.estimate_u_using_random_sampling(max_pairs=5e6)
   ```

4. **DuckDB memory:** For very large jobs, set memory limit:
   ```python
   import duckdb
   con = duckdb.connect()
   con.execute("SET memory_limit='8GB'")
   con.execute("SET threads=8")
   db_api = DuckDBAPI(con)
   ```

5. **Persist intermediate tables** for large jobs:
   ```python
   db_api = DuckDBAPI(connection=":memory:")  # default
   # or use file-based DuckDB for persistence:
   db_api = DuckDBAPI(connection="my_linkage.duckdb")
   ```
