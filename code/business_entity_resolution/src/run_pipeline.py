"""
Optimized end-to-end pipeline.
Key speedup: sample 100K S1 entities for training (plenty for LightGBM),
block full test set for submission.

AWS S3 integration (optional):
  Set BER_S3_BUCKET to automatically download datasets from S3 before
  running and upload model + outputs to S3 when finished.
  See src/aws_utils.py for the full list of environment variables.
"""
import os
import sys
import subprocess

# AWS integration – import is optional; failures are caught at call time.
try:
    from aws_utils import download_dataset_from_s3, upload_artifacts_to_s3
    _AWS_AVAILABLE = True
except Exception:
    _AWS_AVAILABLE = False


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

    # ------------------------------------------------------------------
    # AWS S3 — Download dataset files (no-op if BER_S3_BUCKET not set)
    # ------------------------------------------------------------------
    if _AWS_AVAILABLE:
        download_dataset_from_s3(base)
    else:
        print("[aws_utils] aws_utils not available – skipping S3 download.", flush=True)

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

    # ------------------------------------------------------------------
    # AWS S3 — Upload model + outputs (no-op if BER_S3_BUCKET not set)
    # ------------------------------------------------------------------
    print("=" * 60, flush=True)
    print("STEP 7: Uploading artifacts to S3 (if configured)", flush=True)
    print("=" * 60, flush=True)
    if _AWS_AVAILABLE:
        upload_artifacts_to_s3(
            model_path=model_f,
            output_dir=os.path.join(base, "output"),
        )
    else:
        print("[aws_utils] aws_utils not available – skipping S3 upload.", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("DONE! Outputs ready in output/", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
