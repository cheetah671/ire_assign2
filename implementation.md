# Implementation Plan — CS4.406 Assignment 1

## Datasets in Hand

* `MINDsmall_train.zip` + `MINDsmall_dev.zip` → **MIND-small corpus**
* `ebnerd_demo.zip` + `ebnerd_small.zip` → **EB-NeRD corpus**
* `Ekstra_Bladet_word2vec.zip` → Pre-trained Word2Vec embeddings for EB-NeRD *(optional)*
* `google_bert_base_multilingual_cased.zip` → Multilingual BERT embeddings for EB-NeRD *(optional)*

---

# Q1: Reproducible Data Pipeline

## Goal

Convert the raw ZIP files from both datasets into a **unified, consistent schema** that every downstream step — BM25, embeddings, and evaluation — can reuse without accessing the raw files again.

---

## 1. Raw File Structures

### 1.1 MIND-small

Both `MINDsmall_train/` and `MINDsmall_dev/` contain:

#### `news.tsv`

| Column              | Description                      |
| ------------------- | -------------------------------- |
| `news_id`           | Unique article identifier        |
| `category`          | Top-level category               |
| `subcategory`       | Article subcategory              |
| `title`             | Article title                    |
| `abstract`          | Article abstract                 |
| `url`               | Article URL                      |
| `title_entities`    | Entities extracted from title    |
| `abstract_entities` | Entities extracted from abstract |

#### `behaviors.tsv`

| Column          | Description                                         |
| --------------- | --------------------------------------------------- |
| `impression_id` | Unique impression identifier                        |
| `user_id`       | User identifier                                     |
| `time`          | Impression timestamp                                |
| `history`       | Space-separated list of previously clicked news IDs |
| `impressions`   | Space-separated list of `news_id-label` pairs       |

For example:

```text
history:
N123 N456 N789

impressions:
N111-0 N222-1 N333-0
```

where:

* `1` = clicked
* `0` = not clicked

---

### 1.2 EB-NeRD

The `demo` and `small` bundles contain:

#### `articles.parquet`

Article metadata including:

* `article_id`
* `title`
* `abstract`
* `body`
* `category`
* `published_time`
* and other metadata

#### `train/behaviors.parquet`

Training impression data containing:

* `impression_id`
* `user_id`
* `impression_time`
* `article_ids_inview`
* `article_ids_clicked`

#### `train/history.parquet`

Per-user click histories containing article IDs and click timestamps.

#### `validation/behaviors.parquet`

Validation impression data.

#### `validation/history.parquet`

Validation click histories.

#### Optional pre-computed embeddings

* `train/article_embeddings.parquet`
* Ekstra Bladet Word2Vec embeddings
* Google BERT multilingual embeddings

---

# 2. Unified Schema

The pipeline will normalize both datasets into three common tables:

* `articles`
* `history`
* `impressions`

This allows downstream retrieval and evaluation code to operate on the same schema regardless of the source dataset.

---

## 2.1 `articles.csv`

| Column           | Description                             |
| ---------------- | --------------------------------------- |
| `article_id`     | Unique article identifier               |
| `dataset`        | `MIND` or `EBNERD`                      |
| `title`          | Article title                           |
| `abstract`       | Article abstract; may be empty          |
| `body`           | Article body text, if available         |
| `category`       | Top-level category                      |
| `published_time` | Publication timestamp in UTC ISO format |

---

## 2.2 `history.csv`

| Column       | Description            |
| ------------ | ---------------------- |
| `user_id`    | User identifier        |
| `dataset`    | `MIND` or `EBNERD`     |
| `article_id` | Clicked article ID     |
| `click_time` | Timestamp of the click |

### Timestamp handling

* **MIND-small:** Does not provide individual click timestamps in the history field. Therefore, `impression_time` is used as a proxy for `click_time`.
* **EB-NeRD:** Uses the actual click timestamps provided by `history.parquet`.

---

## 2.3 `impressions.csv`

| Column            | Description                        |
| ----------------- | ---------------------------------- |
| `impression_id`   | Unique impression identifier       |
| `user_id`         | User identifier                    |
| `dataset`         | `MIND` or `EBNERD`                 |
| `impression_time` | Time when the impression was shown |
| `article_id`      | Candidate article ID               |
| `label`           | `1` if clicked, `0` otherwise      |
| `split`           | `train`, `val`, or `test`          |

---

# 3. Processing Pipeline

## Step 1 — Unzip Raw Data

The raw ZIP files will be extracted into the following directory structure:

```text
data/
└── raw/
    ├── MIND/
    │   ├── train/
    │   │   ├── news.tsv
    │   │   └── behaviors.tsv
    │   └── dev/
    │       ├── news.tsv
    │       └── behaviors.tsv
    │
    └── EBNERD/
        ├── demo/
        ├── small/
        ├── word2vec/
        └── bert/
```

The raw data will remain unchanged so that the complete pipeline can always be reproduced from the original files.

---

## Step 2 — Parse and Normalize MIND

### 2.1 Build the article table

Read `news.tsv` from both:

```text
data/raw/MIND/train/news.tsv
data/raw/MIND/dev/news.tsv
```

Then:

1. Rename `news_id` → `article_id`.
2. Add `dataset = MIND`.
3. Normalize the article fields.
4. Combine the two files.
5. Deduplicate articles by `article_id`.
6. Save the result as part of `articles`.

---

### 2.2 Build the history table

Read `behaviors.tsv` from the MIND training split.

The `history` column contains space-separated article IDs:

```text
N123 N456 N789
```

Expand this into one row per user-article interaction:

```text
user_id,dataset,article_id,click_time
U1,MIND,N123,...
U1,MIND,N456,...
U1,MIND,N789,...
```

For MIND, use the corresponding impression timestamp as the `click_time` proxy.

---

### 2.3 Build the impressions table

Parse the `impressions` column:

```text
N111-0 N222-1 N333-0
```

Expand it into:

```text
impression_id,user_id,dataset,impression_time,article_id,label,split
```

For the training data:

```text
split = train
```

Repeat the same process for `MINDsmall_dev/behaviors.tsv`, setting:

```text
split = val
```

---

# 4. Step 3 — Parse and Normalize EB-NeRD

## 4.1 Build the article table

Read:

```text
articles.parquet
```

from the EB-NeRD dataset.

Normalize its fields to the unified article schema:

```text
article_id
dataset
title
abstract
body
category
published_time
```

Set:

```text
dataset = EBNERD
```

Append the normalized EB-NeRD articles to the MIND article table.

---

## 4.2 Build the training history

Read:

```text
train/history.parquet
```

and convert the records into:

```text
user_id,dataset,article_id,click_time
```

Use the actual click timestamps provided by EB-NeRD.

---

## 4.3 Build the training impressions

Read:

```text
train/behaviors.parquet
```

and expand `article_ids_inview` into one row per candidate article.

Determine the label using `article_ids_clicked`:

```text
label = 1
```

if the candidate article appears in the clicked list, otherwise:

```text
label = 0
```

Set:

```text
split = train
```

---

## 4.4 Build the validation data

Read:

```text
validation/history.parquet
validation/behaviors.parquet
```

Normalize them using the same procedure as the training data.

Set:

```text
split = val
```

This produces a consistent EB-NeRD representation compatible with MIND.

---

# 5. Step 4 — Temporal Split Verification

The pipeline must verify that no future information is accidentally used when constructing user histories.

First, convert all timestamps to a consistent UTC representation and sort impressions chronologically:

```text
impression_time ASC
```

For every impression, verify:

```text
click_time < impression_time
```

for all history interactions used before that impression.

Any history interaction satisfying:

```text
click_time >= impression_time
```

should be flagged as a potential data-leakage violation.

### Important exception

For MIND-small, individual click timestamps are unavailable. Therefore, `impression_time` is used as the click-time proxy, and the limitation should be documented explicitly rather than treated as an exact timestamp.

---

# 6. Step 5 — Feature Store

After processing, save the normalized datasets under:

```text
data/
└── processed/
    ├── articles.csv
    ├── articles.parquet
    ├── history.csv
    ├── history.parquet
    ├── impressions.csv
    └── impressions.parquet
```

The Parquet versions will be used for efficient downstream processing, while CSV files provide a human-readable representation for inspection and debugging.

---

# 7. Step 6 — One-Command Rebuild

The entire pipeline should be reproducible through a single command:

```bash
python build_pipeline.py
```

The script should perform:

```text
Raw ZIP files
      ↓
Extraction
      ↓
MIND parsing
      ↓
EB-NeRD parsing
      ↓
Schema normalization
      ↓
Timestamp normalization
      ↓
Leakage verification
      ↓
Deduplication
      ↓
Feature-store generation
      ↓
data/processed/
```

No downstream component should need to access the original ZIP files directly.

---

# 8. Suggested Project Structure

```text
project/
│
├── data/
│   ├── raw/
│   │   ├── MIND/
│   │   │   ├── train/
│   │   │   └── dev/
│   │   │
│   │   └── EBNERD/
│   │       ├── demo/
│   │       ├── small/
│   │       ├── word2vec/
│   │       └── bert/
│   │
│   └── processed/
│       ├── articles.csv
│       ├── articles.parquet
│       ├── history.csv
│       ├── history.parquet
│       ├── impressions.csv
│       └── impressions.parquet
│
├── src/
│   ├── __init__.py
│   ├── mind_parser.py
│   ├── ebnerd_parser.py
│   ├── normalize.py
│   ├── validation.py
│   └── utils.py
│
├── build_pipeline.py
├── requirements.txt
└── README.md
```

---

# 9. Libraries

The implementation will primarily use:

```text
pandas
pathlib
zipfile
tqdm
```

### `pandas`

Used for:

* Reading TSV files
* Reading Parquet files
* Data transformation
* Merging
* Grouping
* Deduplication
* Timestamp conversion

### `pathlib.Path`

Used for:

* Cross-platform path management
* Creating directories
* Managing raw and processed data locations

### `zipfile`

Used for:

* Extracting the raw ZIP archives
* Reproducing the extraction process programmatically

### `tqdm`

Used for:

* Progress bars during large-scale history and impression expansion

---

# 10. Key Design Decisions

## 10.1 No Random Interaction Splitting

Interaction data will **not** be randomly split into training and validation sets.

Instead, the pipeline preserves the temporal ordering of user interactions:

```text
past interactions → training
future interactions → validation/test
```

This better reflects the real-world news recommendation setting and avoids temporal leakage.

---

## 10.2 MIND Click-Time Approximation

MIND-small does not provide individual timestamps for historical clicks.

Therefore:

```text
click_time = impression_time
```

is used as a proxy.

This limitation will be documented in the processed-data metadata and considered when performing temporal leakage checks.

---

## 10.3 EB-NeRD Click Timestamps

EB-NeRD provides actual historical click timestamps through `history.parquet`.

Therefore, these timestamps are preserved directly:

```text
click_time = actual EB-NeRD click timestamp
```

---

## 10.4 Dataset Identifier

Every normalized row contains a `dataset` field:

```text
MIND
EBNERD
```

This prevents ID collisions and allows experiments to filter or compare datasets easily.

---

## 10.5 Raw Data Preservation

The raw ZIP files and extracted raw files should never be modified.

The processing pipeline should be deterministic:

```text
raw data + build_pipeline.py
              ↓
       processed data
```

This ensures that the complete feature store can be regenerated whenever required.

---

# 11. Optional Embeddings

The following resources can be integrated later without changing the core data pipeline:

### Word2Vec

```text
data/raw/EBNERD/word2vec/
```

Source:

```text
Ekstra_Bladet_word2vec.zip
```

### Multilingual BERT

```text
data/raw/EBNERD/bert/
```

Source:

```text
google_bert_base_multilingual_cased.zip
```

These embeddings should be treated as optional downstream features rather than dependencies of the core data-normalization pipeline.

---

# 12. Final Pipeline Contract

After successful execution of:

```bash
python build_pipeline.py
```

the following guarantees should hold:

* [ ] All raw archives have been extracted reproducibly.
* [ ] MIND and EB-NeRD articles use the same article schema.
* [ ] MIND and EB-NeRD histories use the same history schema.
* [ ] MIND and EB-NeRD impressions use the same impression schema.
* [ ] Dataset identifiers distinguish MIND from EB-NeRD.
* [ ] Timestamps are normalized to UTC.
* [ ] MIND history timestamps are explicitly documented as impression-time proxies.
* [ ] EB-NeRD uses actual click timestamps.
* [ ] Interaction data is kept in temporal order.
* [ ] Potential temporal leakage is detected and reported.
* [ ] Duplicate articles are removed using `article_id`.
* [ ] Both CSV and Parquet feature stores are generated.
* [ ] Downstream BM25, embedding, and evaluation stages can operate exclusively on `data/processed/`.
* [ ] The complete pipeline can be rebuilt with one command.

---

## Expected Output

The final reusable feature store is:

```text
data/processed/
├── articles.csv
├── articles.parquet
├── history.csv
├── history.parquet
├── impressions.csv
└── impressions.parquet
```

This processed layer becomes the **single source of truth** for all downstream experiments, including BM25 retrieval, semantic/embedding-based retrieval, ranking, and evaluation.
