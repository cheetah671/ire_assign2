# News Recommendation Project

This project implements a complete news recommendation pipeline for two datasets, MIND-small and EB-NeRD-demo. The goal is to turn raw data into a clean, reproducible workflow that supports two ranking baselines, a shared evaluation harness, and a report-ready comparison of results.

The assignment is split into these parts:

- Q1: build the data pipeline
- Q2: rank articles with BM25
- Q3: rank articles with embeddings
- Q4: evaluate ranking quality and beyond-accuracy metrics
- Q5: generate Codabench submissions
- Q6: write the design note / report
- Q7: push the project to GitHub Classroom with a working README
- Q8: commit work regularly
- Q9: add a leakage test to verify time-based correctness

## What an Impression Means

An impression is one recommendation event: a user sees a candidate set of articles at a specific time, and the dataset records which article(s) were clicked. In practice, each impression contains:

- a user ID
- a timestamp
- a candidate list of article IDs shown to the user
- one or more clicked article IDs, or none

Impressions are the core unit for ranking and evaluation because the model must predict which candidate the user will click based only on information available before that moment.

## High-Level Workflow

1. Download the raw MIND and EB-NeRD files.
2. Parse the original TSV/JSON/embedding files.
3. Standardize both datasets into the same internal schema.
4. Split impressions by time into train, validation, and test.
5. Build BM25 and embedding-based ranking baselines.
6. Evaluate both methods using recall and ranking metrics.
7. Produce submission files in the format required by Codabench.
8. Write the final report and keep the code reproducible.

## Suggested Project Structure

This is a clean layout you can follow:

```text
project/
  data/
    raw/
    processed/
  src/
    data/
    features/
    ranking/
    evaluation/
    submission/
  tests/
  build_pipeline.py
  README.md
```

You do not need to keep exactly this structure, but the important idea is to separate raw data handling, feature generation, ranking, evaluation, and submission creation.

## Q1: Data Pipeline

### Goal

Convert both datasets into one consistent format so every later step can reuse the same code.

### What to produce

- `articles.csv`: article metadata and text
- `history.csv`: user click history before each impression
- `impressions.csv`: candidate articles and click labels for each impression
- time-based train/validation/test splits

### How to approach it

1. Read the raw files from each dataset.
2. Normalize field names and timestamps.
3. Extract the article text you will use later, usually title and abstract.
4. Build per-user click histories in temporal order.
5. Split impressions chronologically, not randomly.
6. Save the cleaned output to disk as CSV or Parquet.

### Useful libraries and functions

- `pandas.read_csv()` and `pandas.read_json()` for loading raw data
- `pandas.DataFrame.groupby()` for aggregating user histories
- `pandas.to_datetime()` for timestamp handling
- `pandas.sort_values()` for chronological ordering
- `csv`, `json`, or `pyarrow` if the raw files require special handling
- `pathlib.Path` for file paths
- `os.makedirs()` or `Path.mkdir()` for output directories

### Leakage test for Q9

Add a test that checks time safety:

- any feature built at time T must only use clicks strictly before T
- no impression in train should contain future clicks from validation or test
- no user-history row should include a click that happens after the impression timestamp

This test is important because time leakage can silently inflate offline metrics.