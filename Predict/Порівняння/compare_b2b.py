import pandas as pd
import numpy as np
import sys

# Paths provided by the user
FILE_NEW = r"b2b_for_ml.csv"
FILE_OLD = r"b2b.csv"


def compare_b2b():
    print(f"Loading New File: {FILE_NEW}")
    try:
        df_new = pd.read_csv(FILE_NEW)
    except Exception as e:
        print(f"Error loading New File: {e}")
        return

    print(f"Loading Old File: {FILE_OLD}")
    try:
        df_old = pd.read_csv(FILE_OLD)
    except Exception as e:
        print(f"Error loading Old File: {e}")
        return

    print(f"\n--- Shape Analysis ---")
    print(f"New: {df_new.shape}")
    print(f"Old: {df_old.shape}")

    # Column check
    cols_new = set(df_new.columns)
    cols_old = set(df_old.columns)

    if cols_new != cols_old:
        print(f"\n[WARNING] Column mismatch!")
        print(f"In New only: {cols_new - cols_old}")
        print(f"In Old only: {cols_old - cols_new}")
    else:
        print(f"\n[OK] Columns match exactly ({len(cols_new)} columns).")

    # Ensure order_id is present
    if 'order_id' not in df_new.columns:
        print("Error: 'order_id' missing.")
        return

    # Sort by order_id to align
    df_new.sort_values('order_id', inplace=True)
    df_old.sort_values('order_id', inplace=True)

    # Initial reset index
    df_new.reset_index(drop=True, inplace=True)
    df_old.reset_index(drop=True, inplace=True)

    # Check if order_ids match
    ids_new = set(df_new['order_id'])
    ids_old = set(df_old['order_id'])

    if ids_new != ids_old:
        print(f"\n[WARNING] order_id mismatch!")
        print(f"Unique to New: {len(ids_new - ids_old)}")
        print(f"Unique to Old: {len(ids_old - ids_new)}")

        # Intersect for value comparison
        common_ids = ids_new.intersection(ids_old)
        print(f"Proceeding with {len(common_ids)} common rows...")

        # Filter and Sort
        df_new = df_new[df_new['order_id'].isin(common_ids)].sort_values('order_id')
        df_old = df_old[df_old['order_id'].isin(common_ids)].sort_values('order_id')

        # CRITICAL FIX: Reset index again after filtering so they align perfectly for comparison
        df_new.reset_index(drop=True, inplace=True)
        df_old.reset_index(drop=True, inplace=True)

    else:
        print(f"\n[OK] All {len(ids_new)} order_ids match.")

    print("\n--- Value Comparison ---")
    # Compare columns
    common_cols = sorted(list(cols_new.intersection(cols_old)))

    for col in common_cols:
        if col == 'order_id': continue

        s_new = df_new[col]
        s_old = df_old[col]

        # Check dtype
        if s_new.dtype != s_old.dtype:
            if pd.api.types.is_numeric_dtype(s_new) and pd.api.types.is_numeric_dtype(s_old):
                pass
            else:
                print(f"Column '{col}': Dtype mismatch ({s_new.dtype} vs {s_old.dtype})")

        # Comparison logic
        if pd.api.types.is_numeric_dtype(s_new) and pd.api.types.is_numeric_dtype(s_old):
            # Numeric comparison with tolerance
            v_new = s_new.fillna(-9999)
            v_old = s_old.fillna(-9999)

            is_close = np.isclose(v_new, v_old, rtol=1e-05, atol=1e-08)
            match_pct = (is_close.sum() / len(is_close)) * 100

            if match_pct < 100:
                print(f"Column '{col}': {match_pct:.2f}% match (Numeric)")
                # Show diff
                diff_mask = ~is_close
                diff_df = pd.DataFrame({
                    'order_id': df_new.loc[diff_mask, 'order_id'],
                    'New': s_new[diff_mask],
                    'Old': s_old[diff_mask],
                    'Diff': s_new[diff_mask] - s_old[diff_mask]
                })
                print(diff_df.head(3).to_string(index=False))
            else:
                print(f"Column '{col}': 100% match")

        else:
            # String/Object comparison
            s_new = s_new.astype(str).str.strip()
            s_old = s_old.astype(str).str.strip()

            matches = (s_new == s_old)
            match_pct = (matches.sum() / len(matches)) * 100

            if match_pct < 100:
                print(f"Column '{col}': {match_pct:.2f}% match (String)")
                diff_mask = ~matches
                diff_df = pd.DataFrame({
                    'order_id': df_new.loc[diff_mask, 'order_id'],
                    'New': s_new[diff_mask],
                    'Old': s_old[diff_mask]
                })
                print(diff_df.head(3).to_string(index=False))
            else:
                print(f"Column '{col}': 100% match")


if __name__ == "__main__":
    compare_b2b()
