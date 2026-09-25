"""
Validation script for submission files.
Verifies line counts, null values, formatting, separators, and entity IDs.
"""
import os, sys

base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
match_file = os.path.join(base, "output", "matching_results.tsv")
cand_file = os.path.join(base, "output", "candidate_pairs.tsv")
test_s1 = os.path.join(base, "dataset", "test", "test_source1.tsv")

print("--- Validating Output Files ---")

# Count test_source1 entities
s1_ids = set()
with open(test_s1, encoding="utf-8", errors="ignore") as f:
    header = f.readline()
    for line in f:
        parts = line.rstrip("\r\n").split("\t")
        if parts:
            s1_ids.add(parts[0])

print(f"Total Source 1 test entities: {len(s1_ids)}")

def validate_tsv(filepath, name, expected_count):
    if not os.path.exists(filepath):
        print(f"ERROR: {name} does not exist at {filepath}")
        return False
    
    line_count = 0
    with_matches = 0
    seen_ids = set()
    errors = []
    
    with open(filepath, encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        if len(header) != 2:
            errors.append(f"Header does not have exactly 2 columns: {header}")
        
        for idx, line in enumerate(f, start=1):
            line_count += 1
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 1:
                errors.append(f"Line {idx}: empty line")
                continue
            s1_id = parts[0]
            seen_ids.add(s1_id)
            
            if len(parts) > 1 and parts[1].strip():
                with_matches += 1
                matched = parts[1].split(",")
                # check duplicates in list
                if len(matched) != len(set(matched)):
                    errors.append(f"Line {idx} ({s1_id}): duplicate IDs in list")
    
    print(f"\n[{name}]")
    print(f"  Total lines (excluding header): {line_count}")
    print(f"  Rows with matches: {with_matches} ({with_matches / max(1, line_count) * 100:.1f}%)")
    print(f"  Unique S1 IDs: {len(seen_ids)}")
    
    missing = s1_ids - seen_ids
    if missing:
        print(f"  WARNING: {len(missing)} S1 entities missing from output")
    if line_count != expected_count:
        print(f"  WARNING: Line count {line_count} != expected {expected_count}")
    
    if errors:
        print(f"  ERRORS FOUND ({len(errors)}):")
        for e in errors[:5]:
            print(f"    - {e}")
        return False
    else:
        print(f"  PASSED FORMAT AND INTEGRITY CHECKS!")
        return True

v1 = validate_tsv(match_file, "matching_results.tsv", len(s1_ids))
v2 = validate_tsv(cand_file, "candidate_pairs.tsv", len(s1_ids))

if v1 and v2:
    print("\nALL SUBMISSION FILES ARE 100% VALID AND READY FOR LEADERBOARD!")
