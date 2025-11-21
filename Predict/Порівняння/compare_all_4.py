import pandas as pd
import numpy as np
from pathlib import Path
import sys

# --- CONFIGURATION ---
# Raw Files
RAW1_PATH = Path(r"extended_customer_data_2025-01-17.csv")
RAW2_PATH = Path(r"db1.csv")

# Processed Files
PROC1_PATH = Path(r"b2b.csv")
PROC2_PATH = Path(r"b2b_for_ml.csv")

FILES = {
    "Raw1 (Extended)": RAW1_PATH,
    "Raw2 (DB1)": RAW2_PATH,
    "Proc1 (b2b)": PROC1_PATH,
    "Proc2 (b2b_ml)": PROC2_PATH
}


def normalize_id(val):
    s = str(val).strip().replace('"', '').replace("'", "")
    # Remove SO prefix if present
    if s.upper().startswith('SO'):
        s = s[2:]

    try:
        i = int(s)
        # Handle 1,000,000 offset
        if i > 1000000:
            i -= 1000000
        return i
    except:
        return s


def load_data(name, path):
    print(f"Loading {name} from {path}...")
    try:
        df = pd.read_csv(path)
        print(f"  -> Loaded {len(df)} rows, {len(df.columns)} cols")

        # Normalize order_id
        if 'order_id' in df.columns:
            df['order_id'] = df['order_id'].apply(normalize_id)
            print(f"  -> Normalized order_ids (Sample: {df['order_id'].iloc[0]})")

        return df
    except Exception as e:
        print(f"  [ERROR] Failed to load {name}: {e}")
        return None


def compare_pair(name1, df1, name2, df2, common_ids):
    print(f"\n--- Comparing {name1} vs {name2} ---")

    # Filter to common IDs
    d1 = df1[df1['order_id'].isin(common_ids)].sort_values('order_id').reset_index(drop=True)
    d2 = df2[df2['order_id'].isin(common_ids)].sort_values('order_id').reset_index(drop=True)

    # Compare Columns
    cols1 = set(d1.columns)
    cols2 = set(d2.columns)
    common_cols = sorted(list(cols1.intersection(cols2)))

    print(f"Common Columns: {len(common_cols)}")
    print(f"Unique to {name1}: {len(cols1 - cols2)}")
    print(f"Unique to {name2}: {len(cols2 - cols1)}")

    # Value Comparison for Common Columns
    mismatch_count = 0
    for col in common_cols:
        if col == 'order_id': continue

        s1 = d1[col]
        s2 = d2[col]

        # Handle numeric vs string
        if pd.api.types.is_numeric_dtype(s1) and pd.api.types.is_numeric_dtype(s2):
            # Numeric comparison
            is_close = np.isclose(s1.fillna(-9999), s2.fillna(-9999), rtol=1e-05, atol=1e-08)
            match_pct = (is_close.sum() / len(is_close)) * 100
        else:
            # String comparison
            s1_str = s1.astype(str).str.strip().fillna("")
            s2_str = s2.astype(str).str.strip().fillna("")
            matches = (s1_str == s2_str)
            match_pct = (matches.sum() / len(matches)) * 100

        if match_pct < 100:
            mismatch_count += 1
            print(f"  [DIFF] {col}: {match_pct:.2f}% match")
            # Show sample diff
            if match_pct < 99.9:  # Only show if significant difference
                diff_mask = ~np.isclose(s1.fillna(-9999), s2.fillna(-9999), rtol=1e-05,
                                        atol=1e-08) if pd.api.types.is_numeric_dtype(
                    s1) and pd.api.types.is_numeric_dtype(s2) else (s1.astype(str) != s2.astype(str))
                # print(d1.loc[diff_mask, ['order_id', col]].rename(columns={col: f'{col}_{name1}'}).join(d2.loc[diff_mask, [col]].rename(columns={col: f'{col}_{name2}'})).head(1))

    if mismatch_count == 0:
        print("  [OK] All common columns match perfectly!")
    else:
        print(f"  Found mismatches in {mismatch_count} columns.")


def main():
    dfs = {}
    for name, path in FILES.items():
        dfs[name] = load_data(name, path)
        if dfs[name] is None:
            return

    # Check order_id existence
    for name, df in dfs.items():
        if 'order_id' not in df.columns:
            print(f"[ERROR] 'order_id' missing in {name}")
            return

    # Find Common IDs across ALL 4
    ids_sets = [set(df['order_id']) for df in dfs.values()]
    common_ids = set.intersection(*ids_sets)
    print(f"\nTotal Common Order IDs across ALL 4 files: {len(common_ids)}")

    if len(common_ids) == 0:
        print("[WARNING] No common IDs found across all 4 files! Checking pairwise...")
        # Fallback to pairwise checks if needed, but for now let's see.

    # Generate all unique pairs
    import itertools
    pairs = list(itertools.combinations(dfs.keys(), 2))

    print(f"\n[INFO] Starting full all-vs-all comparison ({len(pairs)} pairs)...")

    for name1, name2 in pairs:
        df1 = dfs[name1]
        df2 = dfs[name2]

        # Find common IDs for this specific pair
        ids1 = set(df1['order_id'])
        ids2 = set(df2['order_id'])
        common_ids = ids1.intersection(ids2)

        if len(common_ids) == 0:
            print(f"\n--- Comparing {name1} vs {name2} ---")
            print("  [WARNING] No common order_ids found. Skipping value comparison.")
            continue

        compare_pair(name1, df1, name2, df2, common_ids)


if __name__ == "__main__":
    main()
