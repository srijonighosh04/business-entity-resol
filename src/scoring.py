"""
Macro F_0.5 scoring + threshold tuning — Person B.

F_0.5 is computed PER source1 entity, then averaged (macro). A singleton
(no true matches) scores 1.0 if you predict empty, 0.0 if you predict anything.
This module reads val_predictions.tsv from train_matcher.py (columns:
source1_entity_id, candidate_entity_id, label, score) and finds the
probability threshold that maximizes macro F_0.5 on the validation split.

Usage:
    python3 scoring.py --val-predictions val_predictions.tsv
"""
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta * beta
    denom = (b2 * precision) + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def macro_f05(predicted: dict, truth: dict) -> float:
    """
    predicted / truth: {source1_entity_id: set(entity_ids)}
    Every S1 entity in `truth` must have a (possibly empty) entry in `predicted`.
    """
    scores = []
    for s1_id, true_set in truth.items():
        pred_set = predicted.get(s1_id, set())
        if not true_set and not pred_set:
            scores.append(1.0)
            continue
        if not pred_set:
            # recall = 0 regardless of true_set being empty or not (empty handled above)
            scores.append(0.0 if true_set else 1.0)
            continue
        tp = len(pred_set & true_set)
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(true_set) if true_set else 0.0
        if not true_set:
            # true_set empty but pred_set non-empty -> false merge -> 0.0
            scores.append(0.0)
        else:
            scores.append(f_beta(precision, recall))
    return float(np.mean(scores)) if scores else 0.0


def sweep_threshold(val_df: pd.DataFrame, thresholds=None):
    """
    val_df: columns source1_entity_id, candidate_entity_id, label, score
    Returns (best_threshold, best_f05, full_sweep_results).
    Includes every S1 entity that appears in val_df (even if all its
    candidates are negative) so singletons are scored correctly.
    """
    if thresholds is None:
        thresholds = np.arange(0.30, 0.96, 0.02)  # precision-heavy metric -> bias high

    truth = defaultdict(set)
    for row in val_df.itertuples(index=False):
        if row.label == 1:
            truth[row.source1_entity_id].add(row.candidate_entity_id)
    # ensure every S1 id in val_df has a truth entry (possibly empty set)
    for s1_id in val_df["source1_entity_id"].unique():
        truth.setdefault(s1_id, set())

    results = []
    best = (None, -1.0)
    for t in thresholds:
        predicted = defaultdict(set)
        for row in val_df[val_df["score"] >= t].itertuples(index=False):
            predicted[row.source1_entity_id].add(row.candidate_entity_id)
        score = macro_f05(predicted, truth)
        results.append((round(float(t), 3), score))
        if score > best[1]:
            best = (round(float(t), 3), score)

    return best[0], best[1], results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-predictions", default="val_predictions.tsv")
    args = ap.parse_args()

    val_df = pd.read_csv(args.val_predictions, sep="\t")
    best_t, best_f05, sweep = sweep_threshold(val_df)

    print("Threshold sweep (macro F_0.5):")
    for t, s in sweep:
        marker = "  <-- best" if t == best_t else ""
        print(f"  {t:.2f}  {s:.4f}{marker}")

    print(f"\nBest threshold: {best_t}  |  Best validation macro F_0.5: {best_f05:.4f}")
    print("\nRemember: F_0.5 weights precision 2x recall — if two thresholds tie, "
          "prefer the higher one.")


if __name__ == "__main__":
    main()
