# Splink Comparison Library — Full Reference

## Import
```python
import splink.comparison_library as cl
```

---

## Name Comparisons

### `cl.NameComparison(col)`
Best for first names. Uses Jaro-Winkler + phonetic fallback.
```python
cl.NameComparison("first_name")
```

### `cl.JaroWinklerAtThresholds(col, thresholds)`
General string similarity. Good for surnames, city names.
```python
cl.JaroWinklerAtThresholds("surname", [0.9, 0.7])
# Levels: Exact > JW >= 0.9 > JW >= 0.7 > else
```

### `cl.JaroAtThresholds(col, thresholds)`
```python
cl.JaroAtThresholds("surname", [0.9, 0.7])
```

---

## Date Comparisons

### `cl.DateOfBirthComparison(col, ...)`
```python
cl.DateOfBirthComparison(
    "dob",
    input_is_string=True,          # True if stored as "YYYY-MM-DD" string
    datetime_metrics=["year", "month"],
    datetime_thresholds=[1, 1],    # 1 year off OR 1 month off
    invalid_dates_as_null=True,    # Treat unparseable dates as null
)
```

---

## Email Comparisons

### `cl.EmailComparison(col)`
Handles exact match, username match, domain match.
```python
cl.EmailComparison("email")
```

---

## Postcode / Geographic

### `cl.PostcodeComparison(col)`
```python
cl.PostcodeComparison("postcode")
# Levels: Exact full > sector match > district match > area match > else
```

---

## Exact Match

### `cl.ExactMatch(col)`
```python
cl.ExactMatch("gender")
# With term frequency (for skewed distributions):
cl.ExactMatch("city").configure(term_frequency_adjustments=True)
```

---

## Distance-Based

### `cl.LevenshteinAtThresholds(col, thresholds)`
Edit distance. Good for short strings with typos.
```python
cl.LevenshteinAtThresholds("name", [1, 2])
```

### `cl.DistanceFunctionAtThresholds(col, distance_function, thresholds)`
Use any DuckDB-supported distance function.
```python
cl.DistanceFunctionAtThresholds("address", "levenshtein", [3, 5])
```

---

## Numeric / Array Comparisons

### `cl.AbsoluteDifferenceAtThresholds(col, thresholds)`
```python
cl.AbsoluteDifferenceAtThresholds("age", [1, 5])
```

---

## Custom Comparisons

Build from scratch using comparison levels:
```python
from splink.comparison_level_library import (
    ExactMatchLevel,
    LevenshteinLevel,
    NullLevel,
    ElseLevel,
)
from splink import Comparison

custom = Comparison(
    comparison_levels=[
        NullLevel("name"),
        ExactMatchLevel("name"),
        LevenshteinLevel("name", distance_threshold=2),
        ElseLevel(),
    ],
    output_column_name="name",
)
```

---

## Configuration Methods

All comparisons support `.configure()`:
```python
cl.ExactMatch("city").configure(
    term_frequency_adjustments=True,  # Adjusts for skewed value frequencies
    m_probabilities=[0.9, 0.1],       # Override m values (optional)
    u_probabilities=[0.01, 0.99],     # Override u values (optional)
)
```
