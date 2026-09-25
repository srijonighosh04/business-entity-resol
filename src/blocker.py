"""
Candidate Generation / Blocking Engine — Person A.

Takes Source 1, Source 2, and Source 3 records, normalizes entity fields,
and generates candidate pairs (S1_id -> list of S2/S3 candidate_ids) using
TF-IDF character n-gram cosine similarity and first-token/digit inverted index.

Optimized for millions of records using country-based partitioning, sparse
matrix multiplication, and memory-efficient batching.

Usage:
    python3 blocker.py \
        --source1 dataset/train/train_source1.tsv \
        --source2 dataset/train/train_source2.tsv \
        --source3 dataset/train/train_source3.tsv \
        --out candidate_pairs_train.tsv \
        --ground-truth dataset/train/train_ground_truth.tsv \
        --top-k 15 \
        --min-sim 0.20
"""

import argparse
import gc
import os
import re
import string
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# Import normalization helpers from features.py for consistency
try:
    from features import normalize_name, normalize_address, _digits
except ImportError:
    _LEGAL_SUFFIXES = [
        r"\bprivate limited\b", r"\bpvt\.?\s*ltd\.?\b", r"\bpvt\.?\b",
        r"\blimited\b", r"\bltd\.?\b", r"\bcorporation\b", r"\bcorp\.?\b",
        r"\bincorporated\b", r"\binc\.?\b", r"\bllc\b", r"\bllp\b",
        r"\bco\.?\b", r"\bcompany\b", r"\bgmbh\b", r"\bsarl\b", r"\bsas\b",
        r"\bplc\b", r"\bgroup\b", r"\bholdings?\b",
    ]
    _LEGAL_SUFFIX_RE = re.compile("|".join(_LEGAL_SUFFIXES), flags=re.IGNORECASE)
    _PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})

    def normalize_name(name: str) -> str:
        if not isinstance(name, str):
            return ""
        s = name.lower().replace("&", " and ")
        s = _LEGAL_SUFFIX_RE.sub(" ", s).translate(_PUNCT_TABLE)
        return re.sub(r"\s+", " ", s).strip()

    def normalize_address(addr: str) -> str:
        if not isinstance(addr, str):
            return ""
        s = addr.lower().translate(_PUNCT_TABLE)
        return re.sub(r"\s+", " ", s).strip()

    def _digits(s: str) -> set:
        return set(re.findall(r"\d+", s or ""))


def prepare_blocking_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized_text, first_token, digits columns for blocking."""
    df = df.copy()
    norm_names = df["business_name"].fillna("").astype(str).apply(normalize_name)
    norm_addrs = df["business_address"].fillna("").astype(str).apply(normalize_address)
    
    df["norm_text"] = norm_names + " " + norm_addrs
    df["first_token"] = norm_names.apply(lambda s: s.split()[0] if s else "")
    df["country_clean"] = df["country"].fillna("").astype(str).str.strip().str.lower()
    return df


def block_country_partition(s1_df: pd.DataFrame, cand_df: pd.DataFrame,
                             top_k: int = 15, min_sim: float = 0.20,
                             batch_size: int = 25000) -> dict:
    """
    Given S1 entities and Candidate entities for a single country:
    Computes top-k candidates per S1 entity using TF-IDF character n-grams
    and sparse matrix multiplication.
    Returns: {s1_entity_id: set(candidate_entity_ids)}
    """
    if s1_df.empty or cand_df.empty:
        return {s1_id: set() for s1_id in s1_df["entity_id"]}

    results = defaultdict(set)
    s1_ids = s1_df["entity_id"].values
    cand_ids = cand_df["entity_id"].values

    # 1. Build TF-IDF vectorizer over combined candidate + S1 texts
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 4),
        min_df=2,
        sublinear_tf=True,
    )

    # Fit on candidate corpus
    cand_texts = cand_df["norm_text"].values
    cand_tfidf = vectorizer.fit_transform(cand_texts)
    cand_tfidf_T = cand_tfidf.T.tocsc()

    # Also build inverted index on first tokens for exact prefix lookup
    token_index = defaultdict(list)
    for idx, ftoken in enumerate(cand_df["first_token"].values):
        if len(ftoken) >= 3:
            token_index[ftoken].append(cand_ids[idx])

    # 2. Process S1 entities in batches
    n_s1 = len(s1_df)
    for start in range(0, n_s1, batch_size):
        end = min(start + batch_size, n_s1)
        batch_s1_sub = s1_df.iloc[start:end]
        batch_texts = batch_s1_sub["norm_text"].values
        batch_s1_ids = batch_s1_sub["entity_id"].values
        batch_first_tokens = batch_s1_sub["first_token"].values

        batch_tfidf = vectorizer.transform(batch_texts)
        # Compute sparse cosine similarity matrix (batch_size x num_candidates)
        sim_matrix = batch_tfidf.dot(cand_tfidf_T)

        # Extract top-K candidates above min_sim for each row in batch
        for row_idx in range(sim_matrix.shape[0]):
            s1_id = batch_s1_ids[row_idx]
            ftoken = batch_first_tokens[row_idx]

            row = sim_matrix[row_idx]
            if row.nnz > 0:
                data = row.data
                indices = row.indices

                # Filter by min_sim
                mask = data >= min_sim
                if np.any(mask):
                    filtered_data = data[mask]
                    filtered_indices = indices[mask]

                    if len(filtered_data) > top_k:
                        top_arg = np.argpartition(filtered_data, -top_k)[-top_k:]
                        chosen_indices = filtered_indices[top_arg]
                    else:
                        chosen_indices = filtered_indices

                    for cand_idx in chosen_indices:
                        results[s1_id].add(cand_ids[cand_idx])

            # Inverted index lookup for exact first token matches (up to 5 extra candidates)
            if len(ftoken) >= 3 and ftoken in token_index:
                exact_cands = token_index[ftoken][:5]
                results[s1_id].update(exact_cands)

    gc.collect()
    return results


def run_blocking(source1_path: str, source2_path: str, source3_path: str,
                 out_path: str, ground_truth_path: str = None,
                 top_k: int = 15, min_sim: float = 0.20) -> None:

    print(f"Loading Source 1 from {source1_path}...")
    s1_df = pd.read_csv(source1_path, sep="\t", dtype=str, keep_default_na=False)
    print(f"Loading Source 2 from {source2_path}...")
    s2_df = pd.read_csv(source2_path, sep="\t", dtype=str, keep_default_na=False)
    print(f"Loading Source 3 from {source3_path}...")
    s3_df = pd.read_csv(source3_path, sep="\t", dtype=str, keep_default_na=False)

    print("Preprocessing and normalizing text...")
    s1_prep = prepare_blocking_strings(s1_df)
    s2_prep = prepare_blocking_strings(s2_df)
    s3_prep = prepare_blocking_strings(s3_df)

    cand_prep = pd.concat([s2_prep, s3_prep], ignore_index=True)

    all_s1_ids = list(s1_prep["entity_id"])
    candidate_map = defaultdict(set)

    countries = set(s1_prep["country_clean"].unique()) | set(cand_prep["country_clean"].unique())
    print(f"Blocking across countries: {sorted(list(countries))}")

    for ctry in countries:
        if not ctry:
            continue
        sub_s1 = s1_prep[s1_prep["country_clean"] == ctry]
        sub_cand = cand_prep[cand_prep["country_clean"] == ctry]

        if sub_s1.empty or sub_cand.empty:
            continue

        print(f"  Country '{ctry}': S1={len(sub_s1)}, Candidates (S2+S3)={len(sub_cand)}")
        ctry_map = block_country_partition(sub_s1, sub_cand, top_k=top_k, min_sim=min_sim)
        for s1_id, cands in ctry_map.items():
            candidate_map[s1_id].update(cands)

    # Fallback for empty candidate sets: query across all candidates
    empty_s1_ids = set(all_s1_ids) - set(k for k, v in candidate_map.items() if v)
    if empty_s1_ids:
        print(f"Fallback blocking for {len(empty_s1_ids)} S1 entities with zero candidates...")
        sub_s1_empty = s1_prep[s1_prep["entity_id"].isin(empty_s1_ids)]
        fallback_map = block_country_partition(sub_s1_empty, cand_prep, top_k=5, min_sim=0.15)
        for s1_id, cands in fallback_map.items():
            candidate_map[s1_id].update(cands)

    # Calculate Blocking Recall Ceiling if ground truth provided
    if ground_truth_path and os.path.exists(ground_truth_path):
        gt_df = pd.read_csv(ground_truth_path, sep="\t", dtype=str, keep_default_na=False)
        gt_total = 0
        gt_retained = 0
        for row in gt_df.itertuples(index=False):
            s1_id = row.source1_entity_id
            gt_matches = set(x.strip() for x in row.matched_entity_ids.split(",") if x.strip())
            if not gt_matches:
                continue
            gt_total += len(gt_matches)
            retained = gt_matches & candidate_map.get(s1_id, set())
            gt_retained += len(retained)

        recall_ceiling = (gt_retained / gt_total * 100) if gt_total > 0 else 0.0
        print(f"\n=======================================================")
        print(f"Blocking Recall Ceiling: {gt_retained}/{gt_total} = {recall_ceiling:.2f}%")
        print(f"=======================================================\n")

    # Format TSV output
    print(f"Writing candidate pairs to {out_path}...")
    rows = []
    for s1_id in all_s1_ids:
        cands = list(candidate_map.get(s1_id, set()))
        cands_str = ",".join(cands)
        rows.append((s1_id, cands_str))

    out_df = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"Blocking complete! Output saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source1", required=True)
    ap.add_argument("--source2", required=True)
    ap.add_argument("--source3", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ground-truth", default=None)
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--min-sim", type=float, default=0.20)
    args = ap.parse_args()

    run_blocking(
        source1_path=args.source1,
        source2_path=args.source2,
        source3_path=args.source3,
        out_path=args.out,
        ground_truth_path=args.ground_truth,
        top_k=args.top_k,
        min_sim=args.min_sim,
    )


if __name__ == "__main__":
    main()
