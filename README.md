# News Recommendation Pipeline — MIND & EB-NeRD

A complete, two-stage news recommendation system built across CS4.406 Assignments 1 and 2. The project takes raw click-log datasets (Microsoft's **MIND** and Ekstra Bladet's **EB-NeRD**), turns them into a shared feature store, retrieves candidates with lexical/semantic search, re-ranks them with a behavioural-feature GBDT model, reproduces a neural baseline (NRMS) for comparison, and evaluates everything with a statistically rigorous, leakage-safe harness — including serving-cost and 10× scale analysis, and Codabench submission generation for both competitions.

This README is written so a newcomer can understand the whole system and reproduce every result end to end.

---

## 1. What's in this repo

| Stage | What it does | Key modules |
|---|---|---|
| **Data pipeline** | Unzips and normalizes MIND + EB-NeRD into one shared schema (`articles`, `history`, `impressions`) | `build_pipeline.py`, `src/data/` |
| **Stage-1 retrieval** | BM25 lexical search and sentence-embedding / Word2Vec semantic search, both producing top-K candidates per impression | `run_bm25.py`, `run_embeddings.py`, `src/ranking/bm25_ranker.py`, `src/features/embedding_index.py` |
| **Behavioural features** | Click-history recency, session activity, dwell time, article popularity/freshness, category affinity — all leakage-safe | `src/features/behavioural_features.py` |
| **Stage-2 re-ranker** | LightGBM `lambdarank` model over stage-1 score + behavioural features | `run_reranker.py`, `src/ranking/reranker.py` |
| **Neural baseline** | A from-scratch NRMS (multi-head self-attention news/user encoder) reproduction, trained on both datasets | `run_nrms.py`, `src/ranking/nrms.py` |
| **Evaluation harness** | AUC, MRR, nDCG@5/10, novelty, coverage, intra-list diversity, with cold/warm and head/tail slicing and bootstrap 95% CIs (including *paired* bootstrap for significance testing) | `run_evaluation.py`, `src/evaluation/` |
| **Serving & scale analysis** | Measured index memory, p50/p95/p99 latency breakdown, cost-per-1000-queries, and a 10× scaling argument | `run_serving_analysis.py` |
| **Submission generation** | Codabench-format `prediction.txt` files for both the MIND and RecSys-2024/EB-NeRD leaderboards, including large-scale streaming inference | `run_submission.py`, `src/submission/`, `src/scripts/run_reranker_inference.py`, `src/scripts/run_ebnerd_reranker_inference.py`, `make_mindlarge_submission.py` |
| **Tests** | Automated checks for temporal leakage and train/serve feature parity | `tests/` |

---

## 2. Datasets

- **MIND** (Microsoft News Dataset, English): `MINDsmall_train.zip` / `MINDsmall_dev.zip` for development, `MINDlarge_test.zip` for the Codabench leaderboard submission (2.37M impressions, no labels).
- **EB-NeRD** (Ekstra Bladet, Danish): `ebnerd_demo.zip` / `ebnerd_small.zip` for development, `ebnerd_testset.zip` (the large test bundle, 13.5M impressions) for the RecSys 2024 Codabench leaderboard.
- **Pre-computed embeddings**: `Ekstra_Bladet_word2vec.zip` (300-dim Word2Vec document vectors for EB-NeRD articles); MIND articles are embedded on the fly with `sentence-transformers` (`all-MiniLM-L6-v2`).

Raw zip files stay untouched in the repo root; `build_pipeline.py` extracts and normalizes them into `data/processed/` without ever needing to be re-run against the originals again.

---

## 3. Repository layout

```
ire_assign2/
├── build_pipeline.py            # Q1: raw zips -> unified schema
├── run_bm25.py                  # Stage-1: BM25 lexical retrieval
├── run_embeddings.py            # Stage-1: semantic embedding retrieval
├── run_reranker.py              # Stage-2: LightGBM re-ranker + ablation
├── run_nrms.py                  # Neural baseline (NRMS) training
├── run_evaluation.py            # Full evaluation harness with bootstrap CIs
├── run_serving_analysis.py      # Index memory / latency / cost / scale
├── run_submission.py            # Codabench prediction.txt generation (small/demo scale)
├── run_mind_reranker_inference.py / make_mindlarge_submission.py
│                                 # Large-scale MIND submission pipeline
├── scripts/
│   └── analyse_position_bias.py # Position-bias sanity check
├── src/
│   ├── data/                    # Parsers + shared schema (parse_mind.py, parse_ebnerd.py, parse_mind_large.py, schema.py)
│   ├── features/                # behavioural_features.py, embedding_index.py
│   ├── ranking/                 # bm25_ranker.py, nrms.py, reranker.py
│   ├── evaluation/               # metrics.py, bootstrap.py, slicing.py
│   ├── submission/               # format_mind.py, format_ebnerd.py
│   └── scripts/                  # large-scale / streaming inference entry points
├── tests/                        # test_leakage.py, test_feature_parity.py, test_bootstrap.py
├── data/                         # raw/, processed/, cache/, predictions/  (gitignored)
├── models/                       # trained LightGBM boosters + feature-order sidecars
├── results/                      # all metric CSVs and logs
├── submissions/                  # generated Codabench prediction files
└── design_note.pdf, design_note_2.pdf   # A1 / A2 design notes
```

---

## 4. Quickstart — reproduce everything

```bash
pip install -r requirements.txt

# 1. Build the shared feature store from raw zips
python build_pipeline.py --dataset both --scale both

# 2. Stage-1 retrieval baselines
python run_bm25.py       --dataset both
python run_embeddings.py --dataset both

# 3. Stage-2 re-ranker + ablation (trains baseline / improved / serving-safe models)
python run_reranker.py --dataset mind   --ranker emb
python run_reranker.py --dataset ebnerd --ranker emb

# 3b. Reproduced NRMS neural baseline
python run_nrms.py --dataset mind   --epochs 2
python run_nrms.py --dataset ebnerd --epochs 2

# 3c. Position-bias sanity check
python scripts/analyse_position_bias.py

# 4. Extended evaluation: all metrics, slices, bootstrap + paired bootstrap CIs
python run_evaluation.py --dataset both --ranker all

# 5. Serving & scale analysis
python run_serving_analysis.py --dataset both --ranker emb

# 6. Leakage / train-serve parity tests
pytest tests/ -v

# 7. Codabench submissions
python src/scripts/run_reranker_inference.py --dataset mind_large        # MIND large test
python src/scripts/run_ebnerd_reranker_inference.py                      # EB-NeRD large test
```

Each script is independently runnable with `--help` for its exact CLI options; the sequence above is the dependency order (each stage reads the previous stage's output from `data/processed/`, `data/predictions/`, or `models/`).

---

## 5. Stage-by-stage detail

### 5.1 Data pipeline (`build_pipeline.py`, `src/data/`)

Both datasets are parsed into three shared tables — `articles`, `history`, `impressions` — with a common schema (`src/data/schema.py`) so every downstream component is dataset-agnostic:

- **MIND**: `news.tsv` → articles, `behaviors.tsv` → per-user history + impressions. Since MIND-small doesn't expose per-click timestamps, `impression_time` is used as a documented click-time proxy.
- **EB-NeRD**: `articles.parquet`, `history.parquet`, `behaviors.parquet` are parsed directly, preserving real click timestamps, read-time, and scroll-depth where available.
- Impressions are split **chronologically** (train → validation), never randomly, to reflect real deployment and avoid temporal leakage.
- Output lands in `data/processed/<DATASET>/{articles,history,impressions}.parquet` (plus CSV for inspection), which every later stage reads exclusively — nothing downstream touches the raw zips again.

### 5.2 Stage-1 retrieval

- **BM25** (`src/ranking/bm25_ranker.py`): a user's query is the concatenated titles of their most recent clicks *strictly before* the impression timestamp. Tokenization is a simple regex that works for both English and Danish without extra language tooling. Cold-start users (no prior history) get an empty query and rank at the bottom.
- **Embeddings** (`src/features/embedding_index.py`): MIND articles are encoded with `sentence-transformers/all-MiniLM-L6-v2` (title + subtitle, L2-normalized so cosine similarity = dot product); EB-NeRD uses the provided pre-computed Word2Vec document vectors. Both are cached to `data/cache/` so re-runs don't re-encode.
- `run_bm25.py` / `run_embeddings.py` report recall@K (`results/bm25_recall_at_k.csv`, `results/emb_recall_at_k.csv`) and produce ranked candidate lists consumed by the re-ranker.

### 5.3 Behavioural features (`src/features/behavioural_features.py`)

`BehaviouralFeatureExtractor` pre-sorts each user's click history and computes, for every impression, features derived only from clicks strictly before that impression's timestamp (enforced via `np.searchsorted`, an O(log n) binary search rather than a table scan):

- **Click-history**: `user_click_count`, `user_hist_recency_sum` (exponential decay, 7-day half-life)
- **Session**: `session_clicks_1h`, `session_clicks_24h`; dwell-time features `hist_mean_read_time` / `hist_mean_scroll` where EB-NeRD provides them
- **Article**: `article_popularity` (train-split click count), `freshness_days` (time since publish), `category_match` / `cat_affinity` (overlap with the user's recent category history)
- History is capped at the most recent 50 clicks — matching what's reconstructible at serving time on MIND's truncated test history.
- **Position bias** was investigated (`scripts/analyse_position_bias.py`) and found to be a list-length artifact once controlled for — both datasets shuffle the candidate list, so it's deliberately excluded as a feature, with the analysis documented and reproducible.

### 5.4 Stage-2 re-ranker (`run_reranker.py`, `src/ranking/reranker.py`)

`LightGBMReranker` trains a `lambdarank` objective (optimizing NDCG@5/10 directly) over stage-1 score + all behavioural features, using impression-grouped training data. `run_reranker.py` orchestrates three model variants per dataset:

- **baseline** — behavioural features only
- **improved** — baseline + the stage-1 semantic/BM25 score (the ablation contrast)
- **serving** — improved, minus any feature that can't be reconstructed at serving time (see below)

Every trained model is saved with a `.features.json` sidecar recording its exact feature order; inference scripts refuse to run if their feature list disagrees with what the model was trained on (guarded by `tests/test_feature_parity.py`).

### 5.5 Neural baseline — NRMS (`run_nrms.py`, `src/ranking/nrms.py`)

A from-scratch PyTorch reproduction of NRMS (Wu et al., 2019): a news encoder (embedding → multi-head self-attention → additive attention pooling) and a user encoder (the same architecture applied over the user's clicked-article vectors), scored by dot product between user and candidate vectors. Trained with the paper's negative-sampling softmax objective (1 click + K sampled non-clicks per impression, per-impression cross-entropy). Runs on both MIND and EB-NeRD; results land in `results/evaluation_nrms.csv`.

### 5.6 Evaluation harness (`run_evaluation.py`, `src/evaluation/`)

- **`metrics.py`**: fully vectorised (no per-impression Python loops) computation of AUC, MRR, nDCG@5, nDCG@10, plus beyond-accuracy metrics — **novelty** (−log2 article popularity), **coverage** (fraction of the catalog ever recommended), and **intra-list diversity** (category-pair diversity within the top-K list).
- **`slicing.py`**: splits evaluation rows into **cold-start vs. warm** users (≤5 vs. >5 train clicks) and **head vs. tail** articles (top-20%-popularity vs. rest), so metrics can be reported per slice, not just in aggregate.
- **`bootstrap.py`**: `bootstrap_ci` resamples users with replacement to produce 95% CIs for every metric; `paired_bootstrap_ci` resamples the *same* users for two models simultaneously (baseline vs. improved) so the CI on the *difference* isn't inflated by independent sampling noise — this is what backs the statistical-significance claims in `results/paired_bootstrap_ci.csv`.
- `run_evaluation.py --ranker all` runs every stage (stage-1, baseline, improved, serving, NRMS) through the same harness and writes one CSV per ranker/dataset to `results/`.

### 5.7 Serving & scale analysis (`run_serving_analysis.py`)

Simulates the pipeline exactly as it would serve a real request, using the actual trained LightGBM booster (not a stand-in):

1. **Index memory** — measured byte counts for the embedding matrix, id→row map, click-history store, article store, and GBDT model.
2. **Latency** — single-request, end-to-end timing broken into stage-1 retrieval / feature extraction / GBDT scoring, reported at p50/p95/p99.
3. **Cost/QPS** — sustained queries-per-second per core and per instance, and cost per 1,000 / 1M queries at a p99 < 100ms SLA.
4. **10× scaling argument** — grounded in the measured numbers: identifies the in-process pandas history store as the first thing to break at 10× users, notes embedding lookup stays O(candidates) not O(catalog) so a 10× catalog doesn't hurt latency, and that 10× QPS scales horizontally but multiplies per-node memory cost.

Results are written to `results/serving_analysis.csv`.

### 5.8 Codabench submissions (`src/submission/`, `src/scripts/`)

- `format_mind.py` / `format_ebnerd.py` write the exact `prediction.txt` format each competition expects (ranked article-ID lists per impression line).
- Large-scale inference is memory-efficient by design: `src/scripts/run_reranker_inference.py` (MIND) and `src/scripts/run_ebnerd_reranker_inference.py` (EB-NeRD) stream the multi-GB test bundles in batches — articles/categories/publish-times as flat numpy arrays for O(1) candidate lookup, and per-user history stored CSR-style — rather than loading everything into memory at once.
- Generated submissions live under `submissions/` (e.g. `submissions/MIND_LARGE/{bm25,emb,reranker}/prediction.txt`), zipped and uploaded to the respective Codabench leaderboards.

### 5.9 Tests (`tests/`)

- **`test_leakage.py`**: verifies the behaviour-window boundary (no future clicks in a feature window), that every user's train impressions strictly precede their validation impressions, and basic label/ID integrity.
- **`test_feature_parity.py`**: guards the train/serve contract — checks a model's feature order matches its `.features.json` sidecar, that the serving model excludes any feature unreconstructible at serving time (`SERVING_UNSAFE` in `run_reranker.py`), and that stage-1/behavioural features actually carry signal (non-degenerate variance, vary within an impression).
- **`test_bootstrap.py`**: sanity-checks the bootstrap CI machinery itself.

Run all of them with `pytest tests/ -v`.

---

## 6. Where results live

| What | File(s) |
|---|---|
| Stage-1 recall@K | `results/bm25_recall_at_k.csv`, `results/emb_recall_at_k.csv` |
| Stage-1 vs. re-ranker metrics | `results/evaluation_stage1.csv`, `results/evaluation_bm25.csv`, `results/evaluation_emb.csv` |
| Baseline vs. improved vs. serving | `results/evaluation_baseline.csv`, `results/evaluation_improved.csv`, `results/evaluation_serving.csv` |
| NRMS neural baseline | `results/evaluation_nrms.csv`, `results/_nrms_ebnerd.log` |
| Significance testing | `results/paired_bootstrap_ci.csv` |
| Position-bias check | `results/position_bias_check.csv` |
| Serving/scale analysis | `results/serving_analysis.csv`, `results/_serving.log` |
| Trained models | `models/*.txt` (LightGBM boosters) + `*.features.json` (feature order) |
| Codabench submissions | `submissions/` |
| Design notes | `design_note.pdf` (A1), `design_note_2.pdf` (A2) |

---

## 7. Key design decisions worth knowing

- **No random splitting, ever.** Every train/val/test boundary is chronological, per user, matching how the system would actually be deployed.
- **Behaviour-window enforcement is structural, not a filter bolted on.** `BehaviouralFeatureExtractor` uses a binary search against pre-sorted per-user history, so a feature literally cannot see a click that hasn't happened yet by construction.
- **Serving-safety is tracked explicitly per dataset.** `SERVING_UNSAFE` in `run_reranker.py` lists exactly which features MIND's test set can't provide (no per-click timestamps, no publish dates) versus EB-NeRD's (which ships everything the model was trained on) — and a dedicated "serving" model variant is trained without the unsafe features, then compared against the full "improved" model to quantify the cost of serving-safety.
- **Position bias was tested rather than assumed.** The pipeline includes the actual investigation script and result that justified leaving it out as a feature.
- **Large-scale inference is streaming by design**, not just "for correctness" — it's what actually let the 13.5M-impression EB-NeRD test set and 2.37M-impression MIND test set run without loading either into memory whole.
