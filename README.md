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

## Assignment 2: Learning from Click-Logs

A2 adds behavioural signals on top of the A1 retrieval stage: a two-stage
retrieve-then-rank pipeline with a LightGBM re-ranker, an ablation with paired
bootstrap CIs, and a serving/scale analysis.

### Reproduce (in this order)

```bash
pip install -r requirements.txt

# 1. Build the feature store from raw files (A1)
python build_pipeline.py

# 2. Stage-1 retrieval baselines (A1)
python run_bm25.py       --dataset both
python run_embeddings.py --dataset both

# 3. Stage-2 re-ranker + ablation (A2 Q1-Q3)
#    Scores stage-1 on train AND val, then trains three models:
#      baseline  - behavioural features only
#      improved  - + stage-1 semantic score   (the ablation contrast)
#      serving   - improved minus features unavailable at serving time (Q9)
#    Also writes stage1_val_predictions.parquet = the "before re-ranking" run.
python run_reranker.py --dataset mind   --ranker emb
python run_reranker.py --dataset ebnerd --ranker emb

# 3b. Reproduced NRMS baseline (A2 Q3.1)
python run_nrms.py --dataset mind   --epochs 2
python run_nrms.py --dataset ebnerd --epochs 2

# 3c. Is position bias real in these datasets? (A2 Q1.2)
python scripts/analyse_position_bias.py

# 4. Extended evaluation with bootstrap + paired bootstrap CIs (A2 Q5)
python run_evaluation.py --dataset both --ranker all

# 5. Serving and scale analysis (A2 Q4)
python run_serving_analysis.py

# 6. Leakage / behaviour-window tests (A2 Q9)
pytest tests/ -v

# 7. Codabench submission for MIND-large test
python src/scripts/run_reranker_inference.py --dataset mind_large
```

### Two-stage design

Stage 1 retrieves and scores candidates (BM25 or sentence-transformer
embeddings). Stage 2 is a LightGBM `lambdarank` model over behavioural features
**plus** the stage-1 score.

The stage-1 score is computed on the training impressions with the same code
path used for validation and inference. This matters: an earlier version filled
the training column with a constant `0.0` placeholder, which gave it zero
variance, so LightGBM never split on it and the feature was silently inert.
`run_reranker.py` now asserts the training variance is non-zero and aborts if not.

### Position bias: tested, then deliberately not used

Q1.2 lists position bias as a session feature. Pooling all impressions makes it
look strong — click rate falls monotonically with position. That is an artifact
of mixing candidate-list lengths: a longer list has a lower per-candidate click
rate by construction, and only long lists reach deep positions.

Holding list length fixed, click rate is flat (mean position/CTR correlation
+0.03 on MIND, -0.04 on EB-NeRD). Both datasets shuffle the in-view list, so
position carries no display-order information and a position feature would be
noise. `scripts/analyse_position_bias.py` reproduces the check and writes
`results/position_bias_check.csv`.

### Features available at serving time

MIND's test `behaviors.tsv` has no per-click timestamps and `news.tsv` has no
publication date, so `user_hist_recency_sum`, `session_clicks_1h` and
`freshness_days` cannot be reconstructed at serving time. They are listed in
`SERVING_UNSAFE` in `run_reranker.py`, kept in the `improved` model for the Q9
"with vs. without" comparison, and dropped from the `serving` model that
actually produces the Codabench submission.

Each model is saved next to a `.features.json` listing the exact feature order
it was trained with; the inference script loads that file and refuses to run if
it disagrees with the model's own feature names.

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