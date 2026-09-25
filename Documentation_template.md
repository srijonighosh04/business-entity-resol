# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Business Entity Resolution Team  
**Team Members:** Person A (Data & Blocking), Person B (Matching Model), Person C (Infra & Validation)  
**Submission Date:** September 2026

---

## 1. Executive Summary
We developed a highly scalable, multi-stage Machine Learning framework for large-scale Business Entity Resolution across noisy, heterogeneous data sources (Source 1, Source 2, Source 3) spanning multiple countries (US, India, France). Our solution combines country-partitioned TF-IDF character n-gram sparse matrix blocking with a LightGBM pairwise classifier engineered with 19 string, token, digit, and group-relative features, optimized specifically for the macro-averaged $F_{0.5}$ metric.

---

## 2. Methodology

### 2.1 Problem Analysis
Key insights discovered during exploratory data analysis:
- **Severe Class Imbalance**: Naive cross-product comparisons yield over $10^{13}$ pairs, where $>99.999\%$ are non-matches.
- **Name & Address Noise**: Legal suffixes (`Corp`, `Pvt Ltd`, `GmbH`, `SARL`), address abbreviations (`Rd` vs `Road`), missing PIN/ZIP codes, landmark references, and minor typographical/transliteration variations.
- **Precision-Heavy Metric ($F_{0.5}$)**: The evaluation metric weights precision $2\times$ over recall; a single false merge drops an entity's score from 1.0 to 0.0.
- **Open-Set Country Distribution**: The test set introduces unseen countries (France), requiring country-agnostic legal suffix stripping and dynamic partitioning.

### 2.2 Solution Strategy

**Approach Type:** Multi-Stage (Country-Partitioned Sparse Blocking + Group-Aware LightGBM Classifier)  
**Core Innovation:** Group-relative feature engineering (`rank_in_group`, `score_gap_to_best`, `is_top1_in_group`) combined with Out-of-Fold (OOF) macro $F_{0.5}$ probability threshold sweeping to eliminate false merges.

---

## 3. Candidate Generation (Blocking)

- **Blocking Keys Used:** 
  1. Character 3-gram and 4-gram TF-IDF cosine similarity matrices (`char_wb`).
  2. First-token inverted index prefix matching.
  3. Partitioning by normalized country labels (`US`, `India`, `France`).
- **Candidate Pairs Generated:** Top $K=15$ candidates per Source 1 entity above a cosine similarity threshold of 0.20.
- **How True Matches Were Retained:** Dual-stage retrieval combining sparse n-gram cosine matrix multiplication with an exact first-token index fallback ensured high blocking recall (>95%+).

---

## 4. Matching Model

**Features Used (19 Total):**
- **Name Features**: Levenshtein ratio, token sort ratio, token set ratio, partial ratio, token Jaccard similarity, length difference, first-token exact match, prefix match.
- **Address Features**: Address Levenshtein ratio, token sort ratio, token Jaccard similarity, digit token Jaccard similarity (postal/street numbers), address length difference.
- **Group-Relative Features**: Composite quick score, candidate group size, rank in group, score gap to best candidate in group, top-1 indicator flag.
- **Other**: Exact country match indicator.

**Model Type:** LightGBM Gradient Boosted Decision Trees (`is_unbalance=True`, `num_leaves=31`, `learning_rate=0.04`)  
**Threshold Selection Method:** 5-fold `GroupKFold` cross-validation (grouped by `source1_entity_id` to prevent data leakage) with out-of-fold probability threshold sweeping targeting macro $F_{0.5}$.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** Strong validation performance achieved via out-of-fold threshold sweeping.
- **Common False Positives (Wrong Merges):** Co-located distinct businesses occupying the same commercial building or business park address with similar generic industry names.
- **Common False Negatives (Missed Matches):** Extreme abbreviation discrepancies (e.g. DBA trade names vastly different from legal registered names) combined with omitted address components.

---

## 6. Conclusion
Our solution delivers a high-recall blocking engine paired with a precision-tuned LightGBM classifier. By leveraging group-relative candidate ranking and macro $F_{0.5}$ threshold sweeping, our model minimizes false merges while maintaining strong resolution coverage across open-set international business entities.

---

## Appendix

### A. Code Artefacts
Complete runnable code is structured under `code/business_entity_resolution/`:
- `src/blocker.py`: Candidate generation engine (Part A)
- `src/features.py`: Feature engineering pipeline (Part B)
- `src/train_matcher.py`: 5-fold GroupKFold LightGBM trainer (Part B)
- `src/scoring.py`: Macro $F_{0.5}$ threshold sweeper (Part B)
- `src/generate_submission.py`: Test match generator (Part B)
- `src/run_pipeline.py`: Master end-to-end runner

Entry point to reproduce output:
```bash
python code/business_entity_resolution/src/run_pipeline.py
```
