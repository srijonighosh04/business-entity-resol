"""
Ultra-fast streaming pipeline for test set.
ASCII only output, batched LightGBM inference, fast string comparisons.
"""
import re, string, os, sys
from collections import defaultdict, Counter
import lightgbm as lgb
import numpy as np
from rapidfuzz import fuzz, distance as rfz_dist

# ── normalisation (same as features.py) ──────────────────────────────────────
_LEGAL = re.compile(
    r"\b(private limited|pvt\.?\s*ltd\.?|pvt\.?|limited|ltd\.?|corporation|corp\.?|"
    r"incorporated|inc\.?|llc|llp|l\.l\.c\.?|co\.?|company|gmbh|sarl|sas|s\.a\.?|"
    r"plc|group|holdings?)\b", re.I)
_PUNCT = str.maketrans({c: " " for c in string.punctuation})

def norm_name(s):
    if not s: return ""
    s = s.lower().replace("&", " and ")
    s = _LEGAL.sub(" ", s).translate(_PUNCT)
    return re.sub(r"\s+", " ", s).strip()

def get_tokens(name):
    n = norm_name(name)
    return [t for t in n.split() if len(t) >= 3]

# ── paths ─────────────────────────────────────────────────────────────────────
base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
test_s1  = os.path.join(base, "dataset", "test", "test_source1.tsv")
test_s2  = os.path.join(base, "dataset", "test", "test_source2.tsv")
test_s3  = os.path.join(base, "dataset", "test", "test_source3.tsv")
model_f  = os.path.join(base, "model.txt")
cand_out = os.path.join(base, "output", "candidate_pairs.tsv")
match_out = os.path.join(base, "output", "matching_results.tsv")
os.makedirs(os.path.join(base, "output"), exist_ok=True)

THRESHOLD = 0.94
TOP_K = 10
MAX_PER_TOKEN = 100

# ── STEP A: build inverted index from S2 + S3 ────────────────────────────────
print("Building inverted index from S2+S3...", flush=True)
cand_ids   = []
cand_ctry  = []
inv_idx    = defaultdict(list)
cand_name  = []
cand_addr  = []

for path in [test_s2, test_s3]:
    with open(path, encoding="utf-8", errors="ignore") as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4: continue
            eid, name, addr, ctry = parts[0], parts[1], parts[2], parts[3].strip().lower()
            idx = len(cand_ids)
            cand_ids.append(eid)
            cand_ctry.append(ctry)
            cand_name.append(norm_name(name))
            cand_addr.append(addr.lower().translate(_PUNCT))
            for t in set(get_tokens(name)):
                lst = inv_idx[t]
                if len(lst) < MAX_PER_TOKEN:
                    lst.append(idx)

cand_ids  = np.array(cand_ids,  dtype=object)
cand_ctry = np.array(cand_ctry, dtype=object)
cand_name = np.array(cand_name, dtype=object)
cand_addr = np.array(cand_addr, dtype=object)
print(f"  Indexed {len(cand_ids)} candidates, vocab={len(inv_idx)}", flush=True)

# ── STEP B: Fast feature function ─────────────────────────────────────────────
def pair_features(n1, a1, ctry1, n2, a2, ctry2):
    nt1 = frozenset(n1.split()); nt2 = frozenset(n2.split())
    at1 = frozenset(a1.split()); at2 = frozenset(a2.split())
    def jac(a, b):
        if not a and not b: return 1.0
        if not a or not b: return 0.0
        return len(a & b) / len(a | b)
    dig1 = frozenset(re.findall(r"\d+", a1))
    dig2 = frozenset(re.findall(r"\d+", a2))
    name_lev = rfz_dist.Levenshtein.normalized_similarity(n1, n2) if n1 or n2 else 0.0
    name_tsr = fuzz.token_sort_ratio(n1, n2) / 100.0
    name_tst = fuzz.token_set_ratio(n1, n2) / 100.0
    name_pr  = fuzz.partial_ratio(n1, n2) / 100.0
    name_jac = jac(nt1, nt2)
    name_ld  = abs(len(n1) - len(n2))
    ft1 = n1.split()[0] if n1 else ""
    ft2 = n2.split()[0] if n2 else ""
    name_ft  = 1.0 if (ft1 and ft1 == ft2) else 0.0
    addr_lev = rfz_dist.Levenshtein.normalized_similarity(a1, a2) if a1 or a2 else 0.0
    addr_tsr = fuzz.token_sort_ratio(a1, a2) / 100.0
    addr_jac = jac(at1, at2)
    addr_dj  = jac(dig1, dig2)
    addr_ld  = abs(len(a1) - len(a2))
    ctry_m   = 1.0 if (ctry1 and ctry1 == ctry2) else 0.0
    pfx      = 1.0 if (n1 and n2 and (n1 in n2 or n2 in n1)) else 0.0
    qs = name_tsr * 0.5 + addr_tsr * 0.3 + ctry_m * 0.2
    return [name_lev, name_tsr, name_tst, name_pr, name_jac, name_ld, name_ft,
            addr_lev, addr_tsr, addr_jac, addr_dj, addr_ld, ctry_m, pfx, qs]

print("Loading LightGBM model...", flush=True)
model = lgb.Booster(model_file=model_f)

print("Streaming S1 -> blocking -> scoring -> writing output...", flush=True)

# Open output files for direct streaming write
match_f = open(match_out, "w", encoding="utf-8")
cand_f  = open(cand_out, "w", encoding="utf-8")
match_f.write("source1_entity_id\tmatched_entity_ids\n")
cand_f.write("source1_entity_id\tcandidate_entity_ids\n")

BATCH_SIZE = 10000
s1_items = []
n_total = 0
n_matched = 0

def process_batch(items):
    global n_matched
    # items: list of (s1_id, n1, a1, ctry1, best_cis)
    all_feat_rows = []
    ranges = []  # (s1_id, best_cis, start_idx, end_idx)
    
    for s1_id, n1, a1, ctry1, best_cis in items:
        if not best_cis:
            ranges.append((s1_id, [], 0, 0))
            continue
        start_idx = len(all_feat_rows)
        for ci in best_cis:
            all_feat_rows.append(pair_features(n1, a1, ctry1, cand_name[ci], cand_addr[ci], cand_ctry[ci]))
        end_idx = len(all_feat_rows)
        ranges.append((s1_id, best_cis, start_idx, end_idx))
    
    if all_feat_rows:
        X = np.array(all_feat_rows, dtype=np.float32)
        # We need group features for each item's candidate set
        group_feats = np.zeros((len(all_feat_rows), 4), dtype=np.float32)
        for s1_id, best_cis, start_idx, end_idx in ranges:
            gsz = end_idx - start_idx
            if gsz == 0: continue
            qs_vals = X[start_idx:end_idx, 14]
            qs_max = qs_vals.max()
            ranks = (gsz + 1) - np.argsort(np.argsort(-qs_vals))
            gap = qs_max - qs_vals
            top1 = (ranks == 1).astype(np.float32)
            group_feats[start_idx:end_idx, 0] = gsz
            group_feats[start_idx:end_idx, 1] = ranks
            group_feats[start_idx:end_idx, 2] = gap
            group_feats[start_idx:end_idx, 3] = top1
        
        full_X = np.column_stack([X, group_feats])
        probs = model.predict(full_X)
    else:
        probs = np.array([])
    
    # Write results directly
    for s1_id, best_cis, start_idx, end_idx in ranges:
        if not best_cis:
            match_f.write(f"{s1_id}\t\n")
            cand_f.write(f"{s1_id}\t\n")
        else:
            item_probs = probs[start_idx:end_idx]
            matched = [cand_ids[ci] for ci, p in zip(best_cis, item_probs) if p >= THRESHOLD]
            all_cands = [cand_ids[ci] for ci in best_cis]
            if matched:
                n_matched += 1
            match_f.write(f"{s1_id}\t{','.join(matched)}\n")
            cand_f.write(f"{s1_id}\t{','.join(all_cands)}\n")

with open(test_s1, encoding="utf-8", errors="ignore") as f:
    f.readline()  # header
    for line in f:
        parts = line.rstrip("\r\n").split("\t")
        if len(parts) < 4: continue
        eid, name, addr, ctry = parts[0], parts[1], parts[2], parts[3].strip().lower()
        n1 = norm_name(name)
        a1 = addr.lower().translate(_PUNCT)
        toks = get_tokens(name)
        if not toks:
            s1_items.append((eid, n1, a1, ctry, []))
        else:
            counts = Counter()
            for t in toks:
                for ci in inv_idx.get(t, []):
                    if cand_ctry[ci] == ctry:
                        counts[ci] += 1
            best_cis = [ci for ci, _ in counts.most_common(TOP_K)]
            s1_items.append((eid, n1, a1, ctry, best_cis))
        
        n_total += 1
        if len(s1_items) >= BATCH_SIZE:
            process_batch(s1_items)
            s1_items = []
            if n_total % 100000 == 0:
                print(f"  Processed {n_total} / 1732544 S1 entities...", flush=True)

if s1_items:
    process_batch(s1_items)

match_f.close()
cand_f.close()

print(f"\nALL DONE! Total {n_total} S1 records processed, {n_matched} entities matched.", flush=True)
print(f"  Output matching_results: {match_out}", flush=True)
print(f"  Output candidate_pairs: {cand_out}", flush=True)
