"""
AWS S3 utilities for the Business Entity Resolution pipeline.

Two responsibilities:
  1. download_dataset_from_s3  – pull raw TSV files from S3 into the local
                                  dataset/ directory before the pipeline runs.
  2. upload_artifacts_to_s3   – push the trained model + both output TSVs
                                  back to S3 after the pipeline finishes.

Environment variables (all optional – S3 steps are skipped if BUCKET is not set):
  BER_S3_BUCKET   : S3 bucket name  (e.g. "my-entity-resolution-bucket")
  BER_S3_PREFIX   : key prefix inside the bucket (default: "business_entity_resolution")
  AWS_REGION      : AWS region for the boto3 session (default: "us-east-1")

The expected S3 layout (under BER_S3_PREFIX) is:
  <prefix>/dataset/train/train_source1.tsv
  <prefix>/dataset/train/train_source2.tsv
  <prefix>/dataset/train/train_source3.tsv
  <prefix>/dataset/train/train_ground_truth.tsv
  <prefix>/dataset/test/test_source1.tsv
  <prefix>/dataset/test/test_source2.tsv
  <prefix>/dataset/test/test_source3.tsv

Artifacts uploaded after a run:
  <prefix>/artifacts/model.txt
  <prefix>/artifacts/output/matching_results.tsv
  <prefix>/artifacts/output/candidate_pairs.tsv
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _boto3_client(region: str):
    """Return a boto3 S3 client; raises ImportError with a clear message if
    boto3 is not installed."""
    try:
        import boto3
        return boto3.client("s3", region_name=region)
    except ImportError:
        raise ImportError(
            "boto3 is required for AWS S3 integration. "
            "Install it with:  pip install boto3"
        )


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


# ---------------------------------------------------------------------------
# Dataset files to sync (s3_relative_path, local_relative_path)
# ---------------------------------------------------------------------------

DATASET_FILES: List[Tuple[str, str]] = [
    ("dataset/train/train_source1.tsv",      "dataset/train/train_source1.tsv"),
    ("dataset/train/train_source2.tsv",      "dataset/train/train_source2.tsv"),
    ("dataset/train/train_source3.tsv",      "dataset/train/train_source3.tsv"),
    ("dataset/train/train_ground_truth.tsv", "dataset/train/train_ground_truth.tsv"),
    ("dataset/test/test_source1.tsv",        "dataset/test/test_source1.tsv"),
    ("dataset/test/test_source2.tsv",        "dataset/test/test_source2.tsv"),
    ("dataset/test/test_source3.tsv",        "dataset/test/test_source3.tsv"),
]


def download_dataset_from_s3(base_dir: str | Path) -> bool:
    """
    Download all dataset TSV files from S3 into *base_dir*/dataset/.

    Returns True if downloads were attempted, False if BER_S3_BUCKET is not
    set (pipeline falls back to local files silently).

    Skips individual files that already exist locally to avoid redundant
    transfers on repeated runs.
    """
    bucket = _env("BER_S3_BUCKET")
    if not bucket:
        print("[aws_utils] BER_S3_BUCKET not set – skipping S3 download, "
              "using local dataset files.", flush=True)
        return False

    prefix = _env("BER_S3_PREFIX", "business_entity_resolution")
    region = _env("AWS_REGION", "us-east-1")
    base_dir = Path(base_dir)

    client = _boto3_client(region)
    downloaded = 0

    print(f"[aws_utils] Downloading dataset from s3://{bucket}/{prefix}/", flush=True)
    for s3_rel, local_rel in DATASET_FILES:
        local_path = base_dir / local_rel
        if local_path.exists():
            print(f"  [skip – already exists] {local_rel}", flush=True)
            continue

        s3_key = f"{prefix}/{s3_rel}"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading  s3://{bucket}/{s3_key}  →  {local_path}", flush=True)
        client.download_file(bucket, s3_key, str(local_path))
        downloaded += 1

    print(f"[aws_utils] Done – {downloaded} file(s) downloaded from S3.", flush=True)
    return True


def upload_artifacts_to_s3(
    model_path: str | Path,
    output_dir: str | Path,
) -> bool:
    """
    Upload the trained model and output TSVs to S3 after the pipeline finishes.

    Files uploaded:
      - model_path                       →  <prefix>/artifacts/model.txt
      - output_dir/matching_results.tsv  →  <prefix>/artifacts/output/matching_results.tsv
      - output_dir/candidate_pairs.tsv   →  <prefix>/artifacts/output/candidate_pairs.tsv

    Returns True on success, False if BER_S3_BUCKET is not set.
    """
    bucket = _env("BER_S3_BUCKET")
    if not bucket:
        print("[aws_utils] BER_S3_BUCKET not set – skipping S3 upload.", flush=True)
        return False

    prefix = _env("BER_S3_PREFIX", "business_entity_resolution")
    region = _env("AWS_REGION", "us-east-1")
    model_path = Path(model_path)
    output_dir = Path(output_dir)

    client = _boto3_client(region)

    artifacts: List[Tuple[Path, str]] = [
        (model_path,                              f"{prefix}/artifacts/model.txt"),
        (output_dir / "matching_results.tsv",     f"{prefix}/artifacts/output/matching_results.tsv"),
        (output_dir / "candidate_pairs.tsv",      f"{prefix}/artifacts/output/candidate_pairs.tsv"),
    ]

    print(f"[aws_utils] Uploading artifacts to s3://{bucket}/{prefix}/artifacts/", flush=True)
    uploaded = 0
    for local_path, s3_key in artifacts:
        if not local_path.exists():
            print(f"  [skip – file not found] {local_path}", flush=True)
            continue
        print(f"  Uploading  {local_path}  →  s3://{bucket}/{s3_key}", flush=True)
        client.upload_file(str(local_path), bucket, s3_key)
        uploaded += 1

    print(f"[aws_utils] Done – {uploaded} artifact(s) uploaded to S3.", flush=True)
    return True
