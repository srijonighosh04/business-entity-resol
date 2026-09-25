# Business Entity Resolution Pipeline

This folder contains the complete, self-contained end-to-end Machine Learning pipeline for the Business Entity Resolution Challenge.

## Pipeline Architecture Overview

The solution consists of two primary stages designed to maximize macro-averaged $F_{0.5}$:

1. **Part A — Candidate Generation & Blocking (`src/blocker.py`)**:
   - Performs field-level text normalization (legal suffix stripping, address parsing, digit extraction).
   - Partitions business records by country (`US`, `India`, `France`).
   - Computes character 3-gram and 4-gram TF-IDF cosine similarity matrices via sparse matrix multiplication to generate candidate pairs.
   - Applies an inverted index on first tokens for exact prefix matching fallbacks.

2. **Part B — Feature Engineering & Matching Model (`src/features.py`, `src/train_matcher.py`, `src/scoring.py`, `src/generate_submission.py`)**:
   - **Feature Engineering**: Computes 19 numerical features (Levenshtein, Jaccard, token sort/set ratios, digit overlap, and within-group relative ranking metrics like `rank_in_group` and `score_gap_to_best`).
   - **Classifier**: Trains a LightGBM gradient boosted decision tree classifier (`is_unbalance=True`, `num_leaves=31`).
   - **Cross-Validation**: 5-fold `GroupKFold` grouped by `source1_entity_id` to eliminate data leakage across folds.
   - **Threshold Sweeping**: Sweeps probability thresholds on out-of-fold predictions specifically to maximize macro $F_{0.5}$.
   - **Inference & Submission**: Generates `matching_results.tsv` and `candidate_pairs.tsv` and runs format validation.

---

## Directory Structure

```
code/business_entity_resolution/
├── src/
│   ├── blocker.py               # Part A: Candidate generation / blocking engine
│   ├── features.py              # Part B: Feature extraction & precomputation
│   ├── train_matcher.py         # Part B: LightGBM GroupKFold model training
│   ├── scoring.py               # Part B: Macro F_0.5 threshold sweeper
│   ├── generate_submission.py   # Part B: Test inference & matching TSV generator
│   └── run_pipeline.py          # Master runner for end-to-end execution
├── README.md                    # This reproducibility guide
└── requirements.txt             # Pinned Python dependencies
```

---

## Environment Setup

### Prerequisites
- Python 3.10+
- Minimum 8 GB RAM (16 GB recommended for large dataset TF-IDF matrices)

### Installation

Install all required packages:

```bash
pip install -r requirements.txt
```

---

## How to Run (End-to-End Reproducibility)

### Quick Run (Single Command)

To execute the entire pipeline from raw data to final validated outputs:

```bash
python src/run_pipeline.py
```

### Step-by-Step Execution Guide

If you prefer to run each stage individually:

#### Step 1: Run Candidate Generation (Part A) on Training Data
```bash
python src/blocker.py \
    --source1 ../../dataset/train/train_source1.tsv \
    --source2 ../../dataset/train/train_source2.tsv \
    --source3 ../../dataset/train/train_source3.tsv \
    --out ../../candidate_pairs_train.tsv \
    --ground-truth ../../dataset/train/train_ground_truth.tsv \
    --top-k 15 \
    --min-sim 0.20
```

#### Step 2: Train Model & Compute OOF Predictions (Part B)
```bash
python src/train_matcher.py \
    --source1 ../../dataset/train/train_source1.tsv \
    --source2 ../../dataset/train/train_source2.tsv \
    --source3 ../../dataset/train/train_source3.tsv \
    --candidates ../../candidate_pairs_train.tsv \
    --ground-truth ../../dataset/train/train_ground_truth.tsv \
    --model-out ../../model.txt \
    --n-folds 5 \
    --n-jobs 4
```

#### Step 3: Sweep Decision Threshold for Macro $F_{0.5}$
```bash
python src/scoring.py --val-predictions ../../val_predictions.tsv
```

#### Step 4: Run Candidate Generation (Part A) on Test Set
```bash
python src/blocker.py \
    --source1 ../../dataset/test/test_source1.tsv \
    --source2 ../../dataset/test/test_source2.tsv \
    --source3 ../../dataset/test/test_source3.tsv \
    --out ../../output/candidate_pairs.tsv \
    --top-k 15 \
    --min-sim 0.20
```

#### Step 5: Generate Test Set Match Predictions
```bash
python src/generate_submission.py \
    --source1 ../../dataset/test/test_source1.tsv \
    --source2 ../../dataset/test/test_source2.tsv \
    --source3 ../../dataset/test/test_source3.tsv \
    --candidates ../../output/candidate_pairs.tsv \
    --model ../../model.txt \
    --threshold 0.62 \
    --out ../../output/matching_results.tsv \
    --n-jobs 4
```

---

## Output Verification

Generated files in `output/`:
- **`output/matching_results.tsv`**: Leaderboard submission file (columns: `source1_entity_id`, `matched_entity_ids`).
- **`output/candidate_pairs.tsv`**: Candidate blocking set fed to the matcher (columns: `source1_entity_id`, `candidate_entity_ids`).
