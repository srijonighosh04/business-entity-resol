# Business Entity Resolution Pipeline

This repository contains the complete end-to-end Machine Learning pipeline for the Business Entity Resolution Challenge.

## Pipeline Architecture

- **Part A (`src/blocker.py`)**: High-recall candidate generation / blocking using TF-IDF character n-gram cosine similarity (sparse matrix multiplication) and first-token inverted indices grouped by country.
- **Part B (`src/features.py`, `src/train_matcher.py`, `src/scoring.py`, `src/generate_submission.py`)**: 
  - Feature engineering (Levenshtein, Jaccard, token sort/set ratios, digit overlap, group-relative features).
  - LightGBM classifier trained with 5-fold `GroupKFold` grouped by `source1_entity_id`.
  - Probability threshold sweeping targeting macro-averaged $F_{0.5}$.
  - Submission generation and verification against `utils/validate_submission.py`.

## Quick Start / How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Full Pipeline End-to-End**:
   ```bash
   python src/run_pipeline.py
   ```

3. **Output Files**:
   - `output/matching_results.tsv` (Final entity resolution matches for leaderboard submission)
   - `output/candidate_pairs.tsv` (Blocking candidate set)
