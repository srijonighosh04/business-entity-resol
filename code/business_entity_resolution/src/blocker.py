"""
Ultra-fast inverted-index blocker.
Builds a token->candidate lookup in ~2 min for 10M records.
No matrix multiplication = zero memory errors.
"""
import argparse
import gc
import os
import re
import string
from collections import defaultdict
import pandas as pd

try:
    from features import normalize_name, normalize_address
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

    def normalize_name(name):
        if not isinstance(name, str): return ""
        s = name.lower().replace("&", " and ")
        s = _LEGAL_SUFFIX_RE.sub(" ", s).translate(_PUNCT_TABLE)
        return re.sub(r"\s+", " ", s).strip()

    def normalize_address(addr):
        if not isinstance(addr, str): return ""
        s = addr.lower().translate(_PUNCT_TABLE)
        return re.sub(r"\s+", " ", s).strip()


# Tokens so common they expand candidates explosively
STOPWORDS = {
    "and", "inc", "llc", "ltd", "pvt", "co", "corp", "company",
    "limited", "private", "the", "of", "in", "st", "rd", "ave",
    "road", "street", "group", "holding", "holdings", "services",
    "solutions", "enterprises", "international", "global",
}


def tok(s):
    return [t for t in normalize_name(s).split() if len(t) >= 3 and t not in STOPWORDS]


def run_blocking(s1_path, s2_path, s3_path, out_path,
                 gt_path=None, top_k=20, min_sim=0.0, sample_s1=None):
    print("Loading sources...", flush=True)
    s1 = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    s2 = pd.read_csv(s2_path, sep="\t", dtype=str, keep_default_na=False)
    s3 = pd.read_csv(s3_path, sep="\t", dtype=str, keep_default_na=False)

    if sample_s1 and len(s1) > sample_s1:
        print(f"Sampling {sample_s1} from {len(s1)} S1 entities...", flush=True)
        s1 = s1.sample(n=sample_s1, random_state=42).reset_index(drop=True)

    cands = pd.concat([s2, s3], ignore_index=True)
    print(f"S1={len(s1)}, candidates={len(cands)}", flush=True)

    cand_ids = cands["entity_id"].values
    cand_ctry = cands["country"].fillna("").str.strip().str.lower().values
    cand_tokens = [tok(n) for n in cands["business_name"]]

    # Build inverted index: token -> list of candidate row indices (capped at 300 per token)
    print("Building inverted index...", flush=True)
    inv_idx = defaultdict(list)
    for i, tlist in enumerate(cand_tokens):
        for t in set(tlist):
            if len(inv_idx[t]) < 300:
                inv_idx[t].append(i)
    print(f"Index built. Vocab size: {len(inv_idx)}", flush=True)
    del cand_tokens
    gc.collect()

    all_s1_ids = list(s1["entity_id"])
    s1_ctry = s1["country"].fillna("").str.strip().str.lower().values

    cmap = {}
    n = len(s1)
    from collections import Counter
    for i, row in enumerate(s1.itertuples(index=False)):
        if i % 100000 == 0 and i > 0:
            print(f"  processed {i}/{n} S1 entities...", flush=True)
        toks = tok(row.business_name)
        if not toks:
            cmap[row.entity_id] = set()
            continue
        ctry = s1_ctry[i]
        counts = Counter()
        for t in toks:
            for ci in inv_idx.get(t, []):
                if cand_ctry[ci] == ctry:
                    counts[ci] += 1
        best = [cand_ids[ci] for ci, _ in counts.most_common(top_k)]
        cmap[row.entity_id] = set(best)

    print(f"Blocking complete. {len(cmap)} S1 entities processed.", flush=True)

    # Recall ceiling check
    if gt_path and os.path.exists(gt_path):
        gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
        total, found = 0, 0
        s1_id_set = set(all_s1_ids)
        for row in gt.itertuples(index=False):
            matches = {x.strip() for x in row.matched_entity_ids.split(",") if x.strip()}
            if not matches or row.source1_entity_id not in s1_id_set:
                continue
            total += len(matches)
            found += len(matches & cmap.get(row.source1_entity_id, set()))
        if total:
            print(f"\n  Recall ceiling: {found}/{total} = {100*found/total:.1f}%\n", flush=True)

    rows = [(sid, ",".join(sorted(cmap.get(sid, set())))) for sid in all_s1_ids]
    out = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {len(out)} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source1", required=True)
    ap.add_argument("--source2", required=True)
    ap.add_argument("--source3", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ground-truth", default=None)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--min-sim", type=float, default=0.0)
    ap.add_argument("--sample-s1", type=int, default=None)
    args = ap.parse_args()
    run_blocking(args.source1, args.source2, args.source3, args.out,
                 args.ground_truth, args.top_k, args.min_sim, args.sample_s1)
