import pandas as pd
import numpy as np
from pathlib import Path
import sys

# --- CONFIGURATION ---
FILES = {
    "Raw1 (Extended)": Path(r"extended_customer_data_2025-01-17.csv"),
    "Raw2 (DB1)": Path(r"db1.csv"),
    "Proc1 (b2b)": Path(r"b2b.csv"),
    "Proc2 (b2b_ml)": Path(r"b2b_for_ml.csv")
}


def normalize_id(val):
    """Normalizes order_id to a common integer format."""
    s = str(val).strip().replace('"', '').replace("'", "")
    if s.upper().startswith('SO'):
        s = s[2:]
    try:
        i = int(s)
        # Handle 1,000,000 offset (if ID > 1M, subtract 1M to align with raw)
        if i > 1000000:
            i -= 1000000
        return i
    except:
        return s


def preprocess_data(df):
    """Applies the same preprocessing as the build script to ensure fair comparison."""
    df = df.copy()

    # 1. Normalize Dates
    date_cols = ['create_date', 'date_order']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # 2. Add/Normalize Calculated Columns (if source columns exist)
    # Logic from build_b2b_from_pgadmin_export.py

    # is_successful
    if 'state' in df.columns and 'is_successful' not in df.columns:
        successful_states = {"sale", "done"}
        df["is_successful"] = df["state"].astype(str).str.lower().isin(successful_states).astype(int)

    # order_amount
    if 'total_amount' in df.columns and 'order_amount' not in df.columns:
        df["order_amount"] = df["total_amount"].astype(float)
    elif 'order_amount' in df.columns:
        df["order_amount"] = df["order_amount"].astype(float)

    # order_messages
    if 'messages_count' in df.columns and 'order_messages' not in df.columns:
        df["order_messages"] = pd.to_numeric(df["messages_count"], errors="coerce").fillna(0).astype(float)
    elif 'order_messages' in df.columns:
        df["order_messages"] = pd.to_numeric(df["order_messages"], errors="coerce").fillna(0).astype(float)

    # order_changes
    if 'changes_count' in df.columns and 'order_changes' not in df.columns:
        df["order_changes"] = pd.to_numeric(df["changes_count"], errors="coerce").fillna(0).astype(float)
    elif 'order_changes' in df.columns:
        df["order_changes"] = pd.to_numeric(df["order_changes"], errors="coerce").fillna(0).astype(float)

    return df


def load_data(name, path):
    print(f"Loading {name}...")
    if not path.exists():
        print(f"  [ERROR] File not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
        # Normalize order_id
        if 'order_id' in df.columns:
            df['order_id'] = df['order_id'].apply(normalize_id)

        # Apply Preprocessing
        df = preprocess_data(df)

        return df
    except Exception as e:
        print(f"  [ERROR] Failed to load: {e}")
        return None


def smart_compare_series(s1, s2, col_name):
    """Compares two series with smart handling for types and NaNs."""

    # 1. Handle Numeric
    if pd.api.types.is_numeric_dtype(s1) and pd.api.types.is_numeric_dtype(s2):
        # Fill NaNs with 0 for fair comparison (Raw vs Processed)
        v1 = s1.fillna(0)
        v2 = s2.fillna(0)

        # Check for 1M offset (e.g. partner_id)
        # Try direct match first
        is_close = np.isclose(v1, v2, rtol=1e-05, atol=1e-08)
        match_pct = (is_close.sum() / len(is_close)) * 100

        if match_pct < 100 and 'id' in col_name.lower():
            # Try subtracting 1M from s2 (assuming s2 is Processed/Offset)
            v2_offset = v2 - 1_000_000
            is_close_offset = np.isclose(v1, v2_offset, rtol=1e-05, atol=1e-08)
            match_pct_offset = (is_close_offset.sum() / len(is_close_offset)) * 100

            if match_pct_offset > match_pct:
                return match_pct_offset, " (with 1M offset)"

        return match_pct, ""

    # 2. Handle String/Object
    s1_str = s1.astype(str).str.strip().replace('nan', '')
    s2_str = s2.astype(str).str.strip().replace('nan', '')

    # Handle boolean represented as string/int
    if {'true', 'false'}.intersection(set(s1_str.str.lower().unique())):
        s1_str = s1_str.str.lower().replace({'true': '1', 'false': '0', '1.0': '1', '0.0': '0'})
        s2_str = s2_str.str.lower().replace({'true': '1', 'false': '0', '1.0': '1', '0.0': '0'})

    matches = (s1_str == s2_str)
    match_pct = (matches.sum() / len(matches)) * 100
    return match_pct, ""


def compare_pair(name1, df1, name2, df2):
    print(f"\n{'=' * 60}")
    print(f"COMPARING: {name1} vs {name2}")
    print(f"{'=' * 60}")

    # Find common IDs
    common_ids = set(df1['order_id']).intersection(set(df2['order_id']))
    print(f"Common Order IDs: {len(common_ids)}")

    if len(common_ids) == 0:
        print("[WARNING] No common IDs found. Skipping.")
        return

    # Align Data
    d1 = df1[df1['order_id'].isin(common_ids)].sort_values('order_id').reset_index(drop=True)
    d2 = df2[df2['order_id'].isin(common_ids)].sort_values('order_id').reset_index(drop=True)

    # Find Common Columns
    cols1 = set(d1.columns)
    cols2 = set(d2.columns)
    common_cols = sorted(list(cols1.intersection(cols2)))

    print(f"Common Columns: {len(common_cols)}")

    results = []
    for col in common_cols:
        if col == 'order_id': continue

        pct, note = smart_compare_series(d1[col], d2[col], col)
        results.append((col, pct, note))

    # Print Results
    # Sort by match percentage (ascending) to show mismatches first
    results.sort(key=lambda x: x[1])

    for col, pct, note in results:
        status = "[OK]" if pct > 99.9 else "[DIFF]"
        if pct < 99.9:
            print(f"{status} {col:<30} {pct:>6.2f}% match{note}")
        else:
            # Optional: Don't print perfect matches to reduce noise, or print condensed
            pass

    # Summary of perfect matches
    perfect_cols = [c for c, p, n in results if p > 99.9]
    if perfect_cols:
        print(f"[OK] Perfect matches ({len(perfect_cols)} cols): {', '.join(perfect_cols)}")


def main():
    dfs = {}
    for name, path in FILES.items():
        dfs[name] = load_data(name, path)
        if dfs[name] is None: return

    import itertools
    pairs = list(itertools.combinations(dfs.keys(), 2))

    for name1, name2 in pairs:
        compare_pair(name1, dfs[name1], name2, dfs[name2])


if __name__ == "__main__":
    main()
