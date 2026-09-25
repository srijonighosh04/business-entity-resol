import re
import string
from functools import partial
from multiprocessing import Pool, cpu_count
import pandas as pd
from rapidfuzz import fuzz, distance

_LEGAL_SUFFIXES = [
    r"\bprivate limited\b", r"\bpvt\.?\s*ltd\.?\b", r"\bpvt\.?\b",
    r"\blimited\b", r"\bltd\.?\b", r"\bcorporation\b", r"\bcorp\.?\b",
    r"\bincorporated\b", r"\binc\.?\b", r"\bllc\b", r"\bllp\b", r"\bl\.l\.c\.?\b",
    r"\bco\.?\b", r"\bcompany\b", r"\bgmbh\b", r"\bsarl\b", r"\bsas\b", r"\bs\.a\.?\b",
    r"\bplc\b", r"\bgroup\b", r"\bholdings?\b",
]
_LEGAL_SUFFIX_RE = re.compile("|".join(_LEGAL_SUFFIXES), flags=re.IGNORECASE)
_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})

_ADDR_ABBREV = {
    r"\brd\b": "road", r"\bst\b": "street", r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard", r"\bapt\b": "apartment", r"\bfl\b": "floor",
    r"\bno\.?\b": "number", r"\bnr\b": "near",
}
_ADDR_ABBREV_RE = [(re.compile(p, re.IGNORECASE), r) for p, r in _ADDR_ABBREV.items()]


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.lower().replace("&", " and ")
    s = _LEGAL_SUFFIX_RE.sub(" ", s).translate(_PUNCT_TABLE)
    return re.sub(r"\s+", " ", s).strip()


def normalize_address(addr: str) -> str:
    if not isinstance(addr, str):
        return ""
    s = addr.lower()
    for pat, repl in _ADDR_ABBREV_RE:
        s = pat.sub(repl, s)
    s = s.translate(_PUNCT_TABLE)
    return re.sub(r"\s+", " ", s).strip()


def _digits(s: str) -> set:
    return set(re.findall(r"\d+", s or ""))


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_entity_record(name: str, address: str, country: str) -> dict:
    norm_name = normalize_name(name)
    norm_addr = normalize_address(address)
    return {
        "norm_name": norm_name,
        "norm_addr": norm_addr,
        "name_tokens": frozenset(norm_name.split()) if norm_name else frozenset(),
        "addr_tokens": frozenset(norm_addr.split()) if norm_addr else frozenset(),
        "addr_digits": frozenset(_digits(norm_addr)),
        "first_token": norm_name.split()[0] if norm_name else "",
        "name_len": len(norm_name),
        "addr_len": len(norm_addr),
        "country": country.strip().lower() if isinstance(country, str) else "",
    }


def load_lookup(path: str) -> dict:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {
        row.entity_id: build_entity_record(row.business_name, row.business_address, row.country)
        for row in df.itertuples(index=False)
    }


BASE_FEATURE_NAMES = [
    "name_levenshtein_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_jaccard",
    "name_len_diff",
    "name_first_token_match",
    "addr_levenshtein_ratio",
    "addr_token_sort_ratio",
    "addr_jaccard",
    "addr_digit_jaccard",
    "addr_len_diff",
    "country_match",
    "name_is_prefix",
]

GROUP_FEATURE_NAMES = [
    "quick_score",
    "group_size",
    "rank_in_group",
    "score_gap_to_best",
    "is_top1_in_group",
]

FEATURE_NAMES = BASE_FEATURE_NAMES + GROUP_FEATURE_NAMES


def _pair_features_from_records(r1: dict, r2: dict) -> list:
    n1, n2 = r1["norm_name"], r2["norm_name"]
    a1, a2 = r1["norm_addr"], r2["norm_addr"]
    return [
        distance.Levenshtein.normalized_similarity(n1, n2) if n1 or n2 else 0.0,
        fuzz.token_sort_ratio(n1, n2) / 100.0,
        fuzz.token_set_ratio(n1, n2) / 100.0,
        fuzz.partial_ratio(n1, n2) / 100.0,
        _jaccard(r1["name_tokens"], r2["name_tokens"]),
        abs(r1["name_len"] - r2["name_len"]),
        1.0 if (r1["first_token"] and r1["first_token"] == r2["first_token"]) else 0.0,
        distance.Levenshtein.normalized_similarity(a1, a2) if a1 or a2 else 0.0,
        fuzz.token_sort_ratio(a1, a2) / 100.0,
        _jaccard(r1["addr_tokens"], r2["addr_tokens"]),
        _jaccard(r1["addr_digits"], r2["addr_digits"]),
        abs(r1["addr_len"] - r2["addr_len"]),
        1.0 if (r1["country"] and r1["country"] == r2["country"]) else 0.0,
        1.0 if (n1 and n2 and (n1 in n2 or n2 in n1)) else 0.0,
    ]


def pair_features(name1, addr1, country1, name2, addr2, country2) -> list:
    r1 = build_entity_record(name1, addr1, country1)
    r2 = build_entity_record(name2, addr2, country2)
    return _pair_features_from_records(r1, r2)


def _compute_chunk(id_pairs, s1_lookup, s2s3_lookup):
    empty = build_entity_record("", "", "")
    return [
        _pair_features_from_records(s1_lookup.get(s1_id, empty), s2s3_lookup.get(cand_id, empty))
        for s1_id, cand_id in id_pairs
    ]


def add_group_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["quick_score"] = (
        df["name_token_sort_ratio"] * 0.5
        + df["addr_token_sort_ratio"] * 0.3
        + df["country_match"] * 0.2
    )
    grp = df.groupby("source1_entity_id")["quick_score"]
    df["group_size"] = grp.transform("size")
    df["rank_in_group"] = grp.rank(method="min", ascending=False)
    df["score_gap_to_best"] = grp.transform("max") - df["quick_score"]
    df["is_top1_in_group"] = (df["rank_in_group"] == 1).astype(float)
    return df


def build_feature_frame(pairs_df: pd.DataFrame, s1_lookup: dict, s2s3_lookup: dict,
                         n_jobs: int = 1) -> pd.DataFrame:
    id_pairs = list(zip(pairs_df["source1_entity_id"], pairs_df["candidate_entity_id"]))

    if n_jobs > 1 and len(id_pairs) > 20_000:
        n_jobs = min(n_jobs, cpu_count())
        chunk_size = max(1, len(id_pairs) // n_jobs)
        chunks = [id_pairs[i:i + chunk_size] for i in range(0, len(id_pairs), chunk_size)]
        worker = partial(_compute_chunk, s1_lookup=s1_lookup, s2s3_lookup=s2s3_lookup)
        with Pool(n_jobs) as pool:
            results = pool.map(worker, chunks)
        rows = [r for chunk in results for r in chunk]
    else:
        rows = _compute_chunk(id_pairs, s1_lookup, s2s3_lookup)

    base_df = pd.DataFrame(rows, columns=BASE_FEATURE_NAMES)
    out = pairs_df.reset_index(drop=True).join(base_df)
    out = add_group_relative_features(out)
    return out[["source1_entity_id", "candidate_entity_id"] + FEATURE_NAMES]


def expand_candidate_pairs(candidate_pairs_path: str) -> pd.DataFrame:
    df = pd.read_csv(candidate_pairs_path, sep="\t", dtype=str, keep_default_na=False)
    records = []
    for row in df.itertuples(index=False):
        cand_str = getattr(row, "candidate_entity_ids", "")
        if not cand_str:
            continue
        for cand_id in cand_str.split(","):
            cand_id = cand_id.strip()
            if cand_id:
                records.append((row.source1_entity_id, cand_id))
    return pd.DataFrame(records, columns=["source1_entity_id", "candidate_entity_id"])
