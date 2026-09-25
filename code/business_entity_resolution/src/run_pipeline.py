"""
Optimized end-to-end pipeline.
Key speedup: sample 100K S1 entities for training (plenty for LightGBM),
block full test set for submission.
"""
import os
import sys
import subprocess


def run(cmd):
    print(f"\n>>> {cmd}\n", flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    r = subprocess.run(cmd, shell=True, env=env)
    if r.returncode != 0:
        print(f"FAILED (exit {r.returncode}): {cmd}", flush=True)
        sys.exit(r.returncode)


def main():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src = os.path.dirname(__file__)

    train_s1 = os.path.join(base, "dataset", "train", "train_source1.tsv")
    train_s2 = os.path.join(base, "dataset", "train", "train_source2.tsv")
    train_s3 = os.path.join(base, "dataset", "train", "train_source3.tsv")
    train_gt = os.path.join(base, "dataset", "train", "train_ground_truth.tsv")
    test_s1 = os.path.join(base, "dataset", "test", "test_source1.tsv")
    test_s2 = os.path.join(base, "dataset", "test", "test_source2.tsv")
    test_s3 = os.path.join(base, "dataset", "test", "test_source3.tsv")

    cand_train = os.path.join(base, "candidate_pairs_train.tsv")
    cand_test = os.path.join(base, "output", "candidate_pairs.tsv")
    match_out = os.path.join(base, "output", "matching_results.tsv")
    model_f = os.path.join(base, "model.txt")
    val_pred = os.path.join(base, "val_predictions.tsv")

    blocker = os.path.join(src, "blocker.py")
    trainer = os.path.join(src, "train_matcher.py")
    scorer = os.path.join(src, "scoring.py")
    submitter = os.path.join(src, "generate_submission.py")
    validator = os.path.join(base, "utils", "validate_submission.py")

    # Step 1: Block SAMPLED train set (25K S1 entities — fast & lightweight)
    print("=" * 60, flush=True)
    print("STEP 1: Blocking sampled train set (25K S1 entities)", flush=True)
    print("=" * 60, flush=True)
    run(f'python "{blocker}" --source1 "{train_s1}" --source2 "{train_s2}" '
        f'--source3 "{train_s3}" --out "{cand_train}" --ground-truth "{train_gt}" '
        f'--top-k 10 --min-sim 0.15 --sample-s1 25000')

    # Step 2: Train classifier
    print("=" * 60, flush=True)
    print("STEP 2: Training LightGBM classifier", flush=True)
    print("=" * 60, flush=True)
    run(f'python "{trainer}" --source1 "{train_s1}" --source2 "{train_s2}" '
        f'--source3 "{train_s3}" --candidates "{cand_train}" '
        f'--ground-truth "{train_gt}" --model-out "{model_f}" --n-folds 3 --n-jobs 2')

    # Step 3: Find best threshold
    print("=" * 60, flush=True)
    print("STEP 3: Sweeping F_0.5 threshold", flush=True)
    print("=" * 60, flush=True)
    run(f'python "{scorer}" --val-predictions "{val_pred}"')

    # read best threshold from val_predictions
    import pandas as pd
    sys.path.insert(0, src)
    from scoring import sweep_threshold
    vdf = pd.read_csv(val_pred, sep="\t")
    best_t, best_f05, _ = sweep_threshold(vdf)
    print(f"\nBest threshold: {best_t}  |  Val F_0.5: {best_f05:.4f}\n", flush=True)

    # Step 4: Block FULL test set (no sampling — every S1 entity needs a row)
    print("=" * 60, flush=True)
    print("STEP 4: Blocking full test set", flush=True)
    print("=" * 60, flush=True)
    run(f'python "{blocker}" --source1 "{test_s1}" --source2 "{test_s2}" '
        f'--source3 "{test_s3}" --out "{cand_test}" --top-k 10 --min-sim 0.15')

    # Step 5: Generate submission
    print("=" * 60, flush=True)
    print("STEP 5: Generating matching_results.tsv", flush=True)
    print("=" * 60, flush=True)
    run(f'python "{submitter}" --source1 "{test_s1}" --source2 "{test_s2}" '
        f'--source3 "{test_s3}" --candidates "{cand_test}" '
        f'--model "{model_f}" --threshold {best_t} --out "{match_out}" --n-jobs 2')

    # Step 6: Validate
    print("=" * 60, flush=True)
    print("STEP 6: Validating submission", flush=True)
    print("=" * 60, flush=True)
    if os.path.exists(validator):
        run(f'python "{validator}" --matching "{match_out}" --candidate "{cand_test}" '
            f'--test-dir "{os.path.join(base, "dataset", "test")}"')

    print("\n" + "=" * 60, flush=True)
    print("DONE! Outputs ready in output/", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
