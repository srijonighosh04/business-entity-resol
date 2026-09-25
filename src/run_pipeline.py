"""
End-to-End Execution Pipeline for Business Entity Resolution (Part A + Part B).

Runs:
1. Part A Blocking on Train Set (generates candidate_pairs_train.tsv)
2. Part B Classifier Training (trains LightGBM & outputs val_predictions.tsv)
3. Part B Threshold Tuning (sweeps macro F_0.5 on OOF predictions)
4. Part A Blocking on Test Set (generates output/candidate_pairs.tsv)
5. Part B Inference (generates output/matching_results.tsv)
6. Submissions Validator (runs utils/validate_submission.py)

Usage:
    python3 run_pipeline.py
"""

import os
import sys
import subprocess


def run_cmd(cmd: str):
    print(f"\n=======================================================", flush=True)
    print(f"RUNNING: {cmd}", flush=True)
    print(f"=======================================================\n", flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    res = subprocess.run(cmd, shell=True, env=env)
    if res.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {res.returncode}: {cmd}", flush=True)
        sys.exit(res.returncode)


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.dirname(__file__))

    train_s1 = os.path.join(base_dir, "dataset", "train", "train_source1.tsv")
    train_s2 = os.path.join(base_dir, "dataset", "train", "train_source2.tsv")
    train_s3 = os.path.join(base_dir, "dataset", "train", "train_source3.tsv")
    train_gt = os.path.join(base_dir, "dataset", "train", "train_ground_truth.tsv")

    test_s1 = os.path.join(base_dir, "dataset", "test", "test_source1.tsv")
    test_s2 = os.path.join(base_dir, "dataset", "test", "test_source2.tsv")
    test_s3 = os.path.join(base_dir, "dataset", "test", "test_source3.tsv")

    cand_train = os.path.join(base_dir, "candidate_pairs_train.tsv")
    cand_test = os.path.join(base_dir, "output", "candidate_pairs.tsv")
    match_test = os.path.join(base_dir, "output", "matching_results.tsv")
    model_file = os.path.join(base_dir, "model.txt")
    val_pred = os.path.join(base_dir, "val_predictions.tsv")

    src_dir = os.path.dirname(__file__)

    # Step 1: Part A Blocking on Train Set
    print("\n>>> STEP 1: Running Part A Candidate Generation on Train Set...")
    cmd1 = f'python "{os.path.join(src_dir, "blocker.py")}" --source1 "{train_s1}" --source2 "{train_s2}" --source3 "{train_s3}" --out "{cand_train}" --ground-truth "{train_gt}" --top-k 15 --min-sim 0.20'
    run_cmd(cmd1)

    # Step 2: Part B Training
    print("\n>>> STEP 2: Running Part B Classifier Training & CV...")
    cmd2 = f'python "{os.path.join(src_dir, "train_matcher.py")}" --source1 "{train_s1}" --source2 "{train_s2}" --source3 "{train_s3}" --candidates "{cand_train}" --ground-truth "{train_gt}" --model-out "{model_file}" --n-folds 5 --n-jobs 4'
    run_cmd(cmd2)

    # Step 3: Part B Threshold Sweeping
    print("\n>>> STEP 3: Sweeping Macro F_0.5 Threshold on Validation...")
    from scoring import sweep_threshold
    import pandas as pd
    val_df = pd.read_csv(val_pred, sep="\t")
    best_t, best_f05, _ = sweep_threshold(val_df)
    print(f"\n[RESULT] Best Threshold: {best_t} | Best Validation Macro F_0.5: {best_f05:.4f}")

    # Step 4: Part A Blocking on Test Set
    print("\n>>> STEP 4: Running Part A Candidate Generation on Test Set...")
    cmd4 = f'python "{os.path.join(src_dir, "blocker.py")}" --source1 "{test_s1}" --source2 "{test_s2}" --source3 "{test_s3}" --out "{cand_test}" --top-k 15 --min-sim 0.20'
    run_cmd(cmd4)

    # Step 5: Part B Inference on Test Set
    print("\n>>> STEP 5: Generating Test Predictions...")
    cmd5 = f'python "{os.path.join(src_dir, "generate_submission.py")}" --source1 "{test_s1}" --source2 "{test_s2}" --source3 "{test_s3}" --candidates "{cand_test}" --model "{model_file}" --threshold {best_t} --out "{match_test}" --n-jobs 4'
    run_cmd(cmd5)

    # Step 6: Validation
    print("\n>>> STEP 6: Validating Submission Files...")
    val_script = os.path.join(base_dir, "utils", "validate_submission.py")
    if os.path.exists(val_script):
        cmd6 = f'python "{val_script}" --matching "{match_test}" --candidate "{cand_test}" --test-dir "{os.path.join(base_dir, "dataset", "test")}"'
        run_cmd(cmd6)

    print("\n=======================================================")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Final outputs ready in: {os.path.join(base_dir, 'output')}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
