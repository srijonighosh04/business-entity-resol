"""
Generate matching_results.tsv from test candidate_pairs.tsv — Person B.

Loads the trained model, scores every (S1, candidate) pair in the test
candidate set, keeps pairs above threshold, and writes the required
two-column output — including one row per S1 entity even if it has zero
matches (singleton), since the spec requires exactly one row per S1 entity.

Usage:
    python3 generate_submission.py \
        --source1 dataset/test/test_source1.tsv \
        --source2 dataset/test/test_source2.tsv \
        --source3 dataset/test/test_source3.tsv \
        --candidates output/candidate_pairs.tsv \
        --model model.txt \
        --threshold 0.62 \
        --out output/matching_results.tsv
"""
import argparse
import pandas as pd

from features import build_feature_frame, expand_candidate_pairs, load_lookup, FEATURE_NAMES

try:
    import lightgbm as lgb
    HAVE_LGB = True
except ImportError:
    HAVE_LGB = False


def load_model(path: str):
    if HAVE_LGB:
        return lgb.Booster(model_file=path), "lgb"
    import joblib
    return joblib.load(path), "sklearn"


def predict_proba(model, kind, X):
    if kind == "lgb":
        return model.predict(X)  # raw booster -> probabilities for binary objective
    return model.predict_proba(X)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source1", required=True)
    ap.add_argument("--source2", required=True)
    ap.add_argument("--source3", required=True)
    ap.add_argument("--candidates", required=True,
                     help="output/candidate_pairs.tsv from Person A's blocker (test set)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--threshold", type=float, required=True,
                     help="from scoring.py's threshold sweep on validation")
    ap.add_argument("--out", default="output/matching_results.tsv")
    ap.add_argument("--n-jobs", type=int, default=1,
                     help="parallel workers for feature computation (see features.py)")
    args = ap.parse_args()

    s1_lookup = load_lookup(args.source1)
    s2_lookup = load_lookup(args.source2)
    s3_lookup = load_lookup(args.source3)
    s2s3_lookup = {**s2_lookup, **s3_lookup}

    all_s1_ids = list(s1_lookup.keys())  # every S1 test entity needs a row, even w/o candidates

    pairs = expand_candidate_pairs(args.candidates)

    model, kind = load_model(args.model)

    if not pairs.empty:
        feat_df = build_feature_frame(pairs[["source1_entity_id", "candidate_entity_id"]],
                                       s1_lookup, s2s3_lookup, n_jobs=args.n_jobs)
        feat_df["score"] = predict_proba(model, kind, feat_df[FEATURE_NAMES])
        kept = feat_df[feat_df["score"] >= args.threshold]
    else:
        kept = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "score"])

    matches = kept.groupby("source1_entity_id")["candidate_entity_id"].apply(
        lambda ids: ",".join(dict.fromkeys(ids))  # de-dupe, preserve order
    ).to_dict()

    rows = [(s1_id, matches.get(s1_id, "")) for s1_id in all_s1_ids]
    out_df = pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, sep="\t", index=False)

    n_with_matches = (out_df["matched_entity_ids"] != "").sum()
    print(f"Wrote {len(out_df)} rows -> {args.out}")
    print(f"  {n_with_matches} entities with >=1 match, "
          f"{len(out_df) - n_with_matches} singletons at threshold={args.threshold}")
    print("Run utils/validate_submission.py before uploading.")


if __name__ == "__main__":
    main()
