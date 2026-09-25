"""
Train the pairwise matching classifier — Person B.

Input (from Person A / Person C):
  - dataset/train/train_source1.tsv, train_source2.tsv, train_source3.tsv
  - candidate_pairs_train.tsv   (Person A's blocking output on TRAIN data,
                                  same 2-column format as the real candidate_pairs.tsv)
  - dataset/train/train_ground_truth.tsv

Design choices (why this is stronger than a single 80/20 split):
  - GroupKFold (grouped by source1_entity_id, so an S1 entity's pairs never
    span train/val within a fold — no leakage) with out-of-fold (OOF)
    predictions collected across ALL folds. The threshold in scoring.py is
    then tuned against predictions covering the ENTIRE training set, not
    just one lucky 20% slice — much lower variance in the chosen threshold.
  - Final deployed model is retrained on 100% of the training data once CV
    is done, since more training data only helps at inference time.
  - Class imbalance is severe (most candidate pairs are non-matches), so
    LightGBM's is_unbalance flag is used instead of naive re-weighting,
    which tends to over-correct on this kind of skew.

Usage:
    python3 train_matcher.py \
        --source1 dataset/train/train_source1.tsv \
        --source2 dataset/train/train_source2.tsv \
        --source3 dataset/train/train_source3.tsv \
        --candidates candidate_pairs_train.tsv \
        --ground-truth dataset/train/train_ground_truth.tsv \
        --model-out model.txt \
        --n-folds 5 --n-jobs 4
"""
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# eval_set/eval_X naming has shifted across LightGBM releases; keep eval_set
# for broad compatibility across teammates' installed versions and just
# silence the resulting deprecation noise.
warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")

from features import build_feature_frame, expand_candidate_pairs, load_lookup, FEATURE_NAMES

try:
    import lightgbm as lgb
    HAVE_LGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAVE_LGB = False


def load_ground_truth(path: str) -> dict:
    """source1_entity_id -> set(matched_entity_ids)"""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    gt = {}
    for row in df.itertuples(index=False):
        matches = set(x.strip() for x in row.matched_entity_ids.split(",") if x.strip())
        gt[row.source1_entity_id] = matches
    return gt


def _fit_lgb(X_train, y_train, X_val, y_val, seed):
    model = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.04, num_leaves=31,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        is_unbalance=True, random_state=seed, verbosity=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              eval_metric="average_precision",
              callbacks=[lgb.early_stopping(40, verbose=False)])
    return model


def _fit_sklearn(X_train, y_train, seed):
    model = GradientBoostingClassifier(random_state=seed)
    model.fit(X_train, y_train)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source1", required=True)
    ap.add_argument("--source2", required=True)
    ap.add_argument("--source3", required=True)
    ap.add_argument("--candidates", required=True,
                     help="candidate_pairs.tsv generated on the TRAIN set by Person A's blocker")
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--model-out", default="model.txt")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=1,
                     help="parallel workers for feature computation (see features.py)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    s1_lookup = load_lookup(args.source1)
    s2_lookup = load_lookup(args.source2)
    s3_lookup = load_lookup(args.source3)
    s2s3_lookup = {**s2_lookup, **s3_lookup}

    gt = load_ground_truth(args.ground_truth)

    pairs = expand_candidate_pairs(args.candidates)
    if pairs.empty:
        raise SystemExit("No candidate pairs found — check --candidates path/format.")

    pairs["label"] = [
        1 if cand in gt.get(s1, set()) else 0
        for s1, cand in zip(pairs["source1_entity_id"], pairs["candidate_entity_id"])
    ]

    print(f"Total candidate pairs: {len(pairs)}  |  positive: {pairs['label'].sum()}  "
          f"({100 * pairs['label'].mean():.2f}%)")

    all_true_pairs = sum(len(v) for v in gt.values())
    true_pairs_present = pairs["label"].sum()
    if all_true_pairs:
        print(f"Blocking recall ceiling: {true_pairs_present}/{all_true_pairs} "
              f"= {100 * true_pairs_present / all_true_pairs:.2f}% "
              f"(this caps your max achievable recall downstream — flag Person A if low)")

    feat_df = build_feature_frame(pairs[["source1_entity_id", "candidate_entity_id"]],
                                   s1_lookup, s2s3_lookup, n_jobs=args.n_jobs)
    feat_df["label"] = pairs["label"].values

    groups = feat_df["source1_entity_id"].values
    X_all, y_all = feat_df[FEATURE_NAMES], feat_df["label"]

    n_folds = max(2, min(args.n_folds, feat_df["source1_entity_id"].nunique()))
    gkf = GroupKFold(n_splits=n_folds)

    oof_score = np.zeros(len(feat_df))
    fold_importances = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_all, y_all, groups)):
        X_train, y_train = X_all.iloc[train_idx], y_all.iloc[train_idx]
        X_val, y_val = X_all.iloc[val_idx], y_all.iloc[val_idx]

        if HAVE_LGB:
            model = _fit_lgb(X_train, y_train, X_val, y_val, args.seed + fold)
            oof_score[val_idx] = model.predict_proba(X_val)[:, 1]
            fold_importances.append(model.feature_importances_)
        else:
            model = _fit_sklearn(X_train, y_train, args.seed + fold)
            oof_score[val_idx] = model.predict_proba(X_val)[:, 1]
            fold_importances.append(model.feature_importances_)

        fold_pos = int(y_val.sum())
        print(f"  fold {fold + 1}/{n_folds}: val pairs={len(val_idx)}  positives={fold_pos}")

    feat_df["score"] = oof_score
    oof_out = feat_df[["source1_entity_id", "candidate_entity_id", "label", "score"]]
    oof_out.to_csv("val_predictions.tsv", sep="\t", index=False)
    print(f"\nSaved OOF predictions covering all {feat_df['source1_entity_id'].nunique()} "
          "training S1 entities -> val_predictions.tsv (use with scoring.py)")

    print("\nMean feature importances across folds:")
    mean_imp = np.mean(fold_importances, axis=0)
    for name, imp in sorted(zip(FEATURE_NAMES, mean_imp), key=lambda x: -x[1]):
        print(f"  {name:22s} {imp:.1f}")

    # Final model: retrain on ALL training data for deployment/inference.
    print("\nRetraining final model on full training data...")
    if HAVE_LGB:
        final_model = lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.04, num_leaves=31,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            is_unbalance=True, random_state=args.seed, verbosity=-1,
        )
        final_model.fit(X_all, y_all)
        final_model.booster_.save_model(args.model_out)
    else:
        final_model = _fit_sklearn(X_all, y_all, args.seed)
        import joblib
        joblib.dump(final_model, args.model_out)

    print(f"Saved final model -> {args.model_out}")
    print("Next: python3 scoring.py --val-predictions val_predictions.tsv")


if __name__ == "__main__":
    main()
