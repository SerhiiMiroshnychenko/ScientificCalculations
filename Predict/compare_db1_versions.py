import pandas as pd
from pathlib import Path

# Paths configuration
OLD_DB1_PATH = Path(r"D:\WINDSURF\DATABASES\Еталони\Дані\Сирі\db1.csv")
NEW_DB1_PATH = Path(r"D:\WINDSURF\DATABASES\Еталони\Дані\Нові\db1.csv")
CLEANEST_DATA_PATH = Path(r"D:\WINDSURF\DATABASES\Для-порівняння\cleanest_data\1\cleanest_data.csv")


def load_csv(path, name):
    print(f"\n[INFO] Loading {name} from {path}...")
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
        print(f"[INFO] {name}: {df.shape} rows, columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to read {name}: {e}")
        return None


def compare_dataframes(df1, df2, name1, name2, key='order_id'):
    print(f"\n=== COMPARING {name1} vs {name2} ===")

    # Ensure key is string for consistent merging
    df1[key] = df1[key].astype(str)
    df2[key] = df2[key].astype(str)

    common_ids = set(df1[key]).intersection(set(df2[key]))
    print(f"Common {key}s: {len(common_ids)}")
    print(f"Unique to {name1}: {len(set(df1[key]) - set(df2[key]))}")
    print(f"Unique to {name2}: {len(set(df2[key]) - set(df1[key]))}")

    if len(common_ids) == 0:
        print("[WARN] No common IDs found. Skipping detailed comparison.")
        return

    # Filter to common rows
    df1_common = df1[df1[key].isin(common_ids)].set_index(key).sort_index()
    df2_common = df2[df2[key].isin(common_ids)].set_index(key).sort_index()

    # Find common columns
    common_cols = list(set(df1_common.columns).intersection(set(df2_common.columns)))
    print(f"Common columns: {common_cols}")

    for col in common_cols:
        try:
            # Convert to numeric if possible for comparison
            val1 = pd.to_numeric(df1_common[col], errors='coerce').fillna(0)
            val2 = pd.to_numeric(df2_common[col], errors='coerce').fillna(0)

            diff = val1 != val2
            diff_count = diff.sum()

            if diff_count > 0:
                print(f"  [DIFF] Column '{col}': {diff_count} mismatches")
                sample_diff = df1_common[diff].join(df2_common[diff], lsuffix=f'_{name1}', rsuffix=f'_{name2}')[
                    [f'{col}_{name1}', f'{col}_{name2}']].head(3)
                print(sample_diff)
            else:
                print(f"  [OK] Column '{col}': Match")
        except Exception as e:
            print(f"  [SKIP] Could not compare '{col}': {e}")


def compare_specific_column(df_target, df_ref, target_name, ref_name, target_col, ref_col, key='order_id'):
    print(f"\n=== CHECKING {target_col} in {target_name} vs {ref_col} in {ref_name} ===")

    if target_col not in df_target.columns:
        print(f"[ERROR] Column {target_col} missing in {target_name}")
        return
    if ref_col not in df_ref.columns:
        print(f"[ERROR] Column {ref_col} missing in {ref_name}")
        return

    # Check if key exists in both
    if key not in df_target.columns or key not in df_ref.columns:
        print(f"[WARN] Key '{key}' missing in one of the dataframes. Falling back to SORT-BASED alignment.")

        # Try to find common sort columns
        # Target (NEW DB1): create_date, total_amount
        # Ref (CLEANEST): create_date, order_amount

        sort_cols_target = []
        sort_cols_ref = []

        if 'create_date' in df_target.columns and 'create_date' in df_ref.columns:
            sort_cols_target.append('create_date')
            sort_cols_ref.append('create_date')

        # Amount might have different names
        amt_target = 'total_amount' if 'total_amount' in df_target.columns else 'order_amount'
        amt_ref = 'order_amount' if 'order_amount' in df_ref.columns else 'total_amount'

        if amt_target in df_target.columns and amt_ref in df_ref.columns:
            sort_cols_target.append(amt_target)
            sort_cols_ref.append(amt_ref)

        if not sort_cols_target:
            print("[ERROR] No common columns found for sorting. Cannot align.")
            return

        print(f"[INFO] Sorting by: {sort_cols_target} vs {sort_cols_ref} to align rows...")

        # Sort and reset index
        df_t_sorted = df_target.sort_values(by=sort_cols_target).reset_index(drop=True)
        df_r_sorted = df_ref.sort_values(by=sort_cols_ref).reset_index(drop=True)

        if len(df_t_sorted) != len(df_r_sorted):
            print(f"[ERROR] Row counts differ ({len(df_t_sorted)} vs {len(df_r_sorted)}). Cannot compare.")
            return

        val_target = pd.to_numeric(df_t_sorted[target_col], errors='coerce').fillna(0)
        val_ref = pd.to_numeric(df_r_sorted[ref_col], errors='coerce').fillna(0)

        diff = val_target != val_ref
        diff_count = diff.sum()

        if diff_count == 0:
            print(f"[SUCCESS] {target_col} matches {ref_col} exactly (after sorting)!")
        else:
            print(f"[FAIL] {diff_count} mismatches found (after sorting).")
            # Show sample mismatches
            mismatches = pd.DataFrame({
                f'SortKey_Date': df_t_sorted[sort_cols_target[0]],
                f'{target_name}_{target_col}': val_target[diff],
                f'{ref_name}_{ref_col}': val_ref[diff]
            })
            print("Sample mismatches:")
            print(mismatches.head(10))
        return

    # Merge on key
    df_target[key] = df_target[key].astype(str)
    df_ref[key] = df_ref[key].astype(str)

    merged = pd.merge(df_target[[key, target_col]], df_ref[[key, ref_col]], on=key, how='inner',
                      suffixes=(f'_{target_name}', f'_{ref_name}'))

    print(f"Matched rows: {len(merged)}")

    val_target = pd.to_numeric(merged[target_col], errors='coerce').fillna(0)
    val_ref = pd.to_numeric(merged[ref_col], errors='coerce').fillna(0)

    diff = val_target != val_ref
    diff_count = diff.sum()

    if diff_count == 0:
        print(f"[SUCCESS] {target_col} matches {ref_col} exactly!")
    else:
        print(f"[FAIL] {diff_count} mismatches found.")
        print("Sample mismatches:")
        print(merged[diff].head(5))


def main():
    df_old = load_csv(OLD_DB1_PATH, "OLD_DB1")
    df_new = load_csv(NEW_DB1_PATH, "NEW_DB1")
    df_clean = load_csv(CLEANEST_DATA_PATH, "CLEANEST")

    if df_old is not None and df_new is not None:
        compare_dataframes(df_old, df_new, "OLD", "NEW")

    if df_new is not None and df_clean is not None:
        # Check order_lines specifically
        # Note: In cleanest_data it might be 'order_lines_count', in new db1 it should be 'order_lines' or 'lines_count' depending on SQL

        # Try to find the lines column in NEW DB1
        new_lines_col = 'order_lines' if 'order_lines' in df_new.columns else 'lines_count' if 'lines_count' in df_new.columns else None

        # Try to find the lines column in CLEANEST
        clean_lines_col = 'order_lines_count' if 'order_lines_count' in df_clean.columns else 'order_lines' if 'order_lines' in df_clean.columns else None

        if new_lines_col and clean_lines_col:
            compare_specific_column(df_new, df_clean, "NEW", "CLEANEST", new_lines_col, clean_lines_col)
        else:
            print(f"\n[WARN] Could not identify order lines columns. New: {new_lines_col}, Cleanest: {clean_lines_col}")


if __name__ == "__main__":
    main()
