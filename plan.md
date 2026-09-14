Implementation Plan — CS4.406 Assignment 1
Datasets in hand:

MINDsmall_train.zip + MINDsmall_dev.zip → MIND-small corpus
ebnerd_demo.zip + ebnerd_small.zip → EB-NeRD corpus
Ekstra_Bladet_word2vec.zip → Pre-trained Word2Vec embeddings for EB-NeRD (optional)
google_bert_base_multilingual_cased.zip → Multilingual BERT embeddings for EB-NeRD (optional)
Q1: Reproducible Data Pipeline
Goal
Convert raw zip files from both datasets into a unified, consistent schema that every downstream step (BM25, embeddings, evaluation) can reuse without touching raw files again.

Raw File Structures
MIND-small
Both MINDsmall_train/ and MINDsmall_dev/ contain:

news.tsv — article metadata: news_id | category | subcategory | title | abstract | url | title_entities | abstract_entities
behaviors.tsv — impression logs: impression_id | user_id | time | history | impressions
history = space-separated list of past clicked news_ids
impressions = space-separated list of news_id-label (label: 1=clicked, 0=not clicked)
EB-NeRD (demo / small)
Each bundle contains:

articles.parquet — article metadata: article_id, title, abstract, body, category, published_time, etc.
train/behaviors.parquet — impression logs with impression_id, user_id, impression_time, article_ids_inview (candidate list), article_ids_clicked
train/history.parquet — per-user click histories with timestamps
validation/behaviors.parquet + validation/history.parquet
Pre-computed embeddings: train/article_embeddings.parquet (optional)
Unified Schema (Output Files)
articles.csv
Column	Description
article_id	Unique article identifier
dataset	MIND or EBNERD
title	Article title
abstract	Article abstract (may be empty)
body	Article body text (if available)
category	Top-level category
published_time	Publication timestamp (UTC, ISO format)
history.csv
Column	Description
user_id	User identifier
dataset	MIND or EBNERD
article_id	Clicked article ID
click_time	Timestamp of the click (or impression time proxy for MIND)
impressions.csv
Column	Description
impression_id	Unique impression identifier
user_id	User identifier
dataset	MIND or EBNERD
impression_time	When the impression was shown (UTC)
article_id	Candidate article ID
label	1 if clicked, 0 if not
split	train, val, or test
Processing Steps
Step 1 — Unzip raw data

data/raw/MIND/train/      ← MINDsmall_train.zip contents
data/raw/MIND/dev/        ← MINDsmall_dev.zip contents
data/raw/EBNERD/demo/     ← ebnerd_demo.zip contents
data/raw/EBNERD/small/    ← ebnerd_small.zip contents
data/raw/EBNERD/word2vec/ ← Ekstra_Bladet_word2vec.zip
data/raw/EBNERD/bert/     ← google_bert_base_multilingual_cased.zip
Step 2 — Parse & normalize MIND
Read news.tsv from train + dev; deduplicate by news_id → articles.csv
Read behaviors.tsv from train:
Expand history column → history.csv (one row per user-article pair)
Expand impressions column → impressions.csv (one row per candidate)
Label split = train
Read behaviors.tsv from dev:
Same expansion, label split = val
Step 3 — Parse & normalize EB-NeRD
Read articles.parquet → append to articles.csv
Read train/behaviors.parquet + train/history.parquet → split = train
Read validation/behaviors.parquet + validation/history.parquet → split = val
Step 4 — Temporal split verification
Sort impressions by impression_time
Verify no user history row has click_time >= impression_time of its impression (leakage check)
Step 5 — Feature store
Save to data/processed/:

articles.csv / .parquet
history.csv / .parquet
impressions.csv / .parquet
Step 6 — One-command rebuild
python build_pipeline.py runs steps 1–5 end-to-end from raw zips.

Libraries
pandas — read_csv, read_parquet, merge, groupby, to_datetime
pathlib.Path — file path management
zipfile — extracting archives
tqdm — progress bars
Key Decisions
Never random-split interaction data; always sort by timestamp
MIND has no per-click timestamps in history → use impression_time as proxy for click_time
EB-NeRD history.parquet has actual click timestamps → use them directly
Store both raw and processed; gitignore both
Q2: BM25 Baseline
Goal
Rank candidate articles using keyword overlap between user history and article text.

Approach
Build BM25 index over title + abstract from articles.csv
Construct query = concatenate titles of user's recently clicked articles (before impression_time)
Score all candidates in the impression using BM25
Sort by score descending → ranked list
Compute recall@K for K ∈ {50, 100, 200} on val split
Run on both MIND and EB-NeRD
Libraries
rank_bm25.BM25Okapi — BM25 scoring
re / str.split — tokenization
numpy.argsort — ranking
Q3: Semantic Embedding Baseline
Goal
Rank candidates by semantic similarity using dense vector representations.

Approach
EB-NeRD: Load embeddings from Ekstra_Bladet_word2vec.zip or article_embeddings.parquet
MIND: Compute embeddings using sentence-transformers (e.g., all-MiniLM-L6-v2)
User vector = mean-pool of clicked article embeddings (before impression_time)
Score candidates via cosine similarity between user vector and candidate embedding
Optionally use FAISS ANN index for speed
Report recall@K for K ∈ {50, 100, 200}
Compare with BM25 — which slices does each win?
Libraries
sentence_transformers.SentenceTransformer — MIND embeddings
numpy.mean, numpy.linalg.norm — user vector + cosine similarity
sklearn.metrics.pairwise.cosine_similarity
faiss (optional)
Q4: Offline Evaluation Harness
Metrics
Ranking Metrics
AUC — ROC curve, clicked vs. not-clicked
MRR — mean reciprocal rank of first clicked article
nDCG@5 — normalized discounted cumulative gain at cutoff 5
nDCG@10 — same at cutoff 10
Beyond-Accuracy
Intra-list Diversity (ILD) — avg pairwise distance between top-K recommended articles
Novelty — negative log of article popularity
Coverage — fraction of catalog recommended at least once
Slicing
Cold-start users (≤5 clicks) vs. Warm users (>5 clicks)
Optional: Head articles (top 20% by popularity) vs. Tail articles
Bootstrap Confidence Intervals
N=1000 bootstrap samples of users with replacement
2.5th and 97.5th percentiles → 95% CI
Libraries
sklearn.metrics.roc_auc_score
numpy — bootstrap, nDCG, MRR
pandas — slicing via groupby
Q5: Codabench Submission
MIND Format
File: prediction.txt
One line per impression: impression_id [news_id_1,news_id_2,...]
EB-NeRD Format
Check Codabench competition page for exact schema
Typically: impression_id, ranked_article_ids
Approach
Run BM25 or embedding ranker on test impressions
Write predictions in competition format
Upload and screenshot leaderboard
Q6: Design Note (≤4 pages)
Pipeline end-to-end description
Design choices: time split, query construction, embedding strategy
BM25 vs. semantic results and analysis
Dataset differences (English MIND vs. Danish EB-NeRD)
Where it breaks at 10× scale
Q9: Leakage Test
Rule
Any feature at time T must only use clicks strictly before T.

Tests (pytest)
For every impression row, assert all history clicks for that user have click_time < impression_time
Assert train impressions are all earlier than val impressions per user
Assert embeddings are built from article metadata only (not future click counts)
Project Directory Layout

IRE_A1/
  data/
    raw/
      MIND/train/
      MIND/dev/
      EBNERD/demo/
      EBNERD/small/
      EBNERD/word2vec/
      EBNERD/bert/
    processed/
      articles.csv
      history.csv
      impressions.csv
      article_embeddings/
  src/
    data/
      parse_mind.py
      parse_ebnerd.py
      schema.py
      split.py
    features/
      bm25_index.py
      embedding_index.py
    ranking/
      bm25_ranker.py
      embedding_ranker.py
    evaluation/
      metrics.py
      bootstrap.py
      slicing.py
    submission/
      format_mind.py
      format_ebnerd.py
  tests/
    test_leakage.py
  build_pipeline.py
  implementation.md
  README.md
  .gitignore
.gitignore Essentials

data/
*.zip
*.pt
*.ckpt
*.npy
__pycache__/
*.pyc
.env