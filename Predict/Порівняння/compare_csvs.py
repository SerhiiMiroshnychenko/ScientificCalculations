import pandas as pd
import sys

FILE1_PATH = r"extended_customer_data_2025-01-17.csv"
FILE2_PATH = r"new2.csv"


def compare_csvs():
    print(f"Loading File 1: {FILE1_PATH}")
    try:
        df1 = pd.read_csv(FILE1_PATH)
    except Exception as e:
        print(f"Error loading File 1: {e}")
        return

    print(f"Loading File 2: {FILE2_PATH}")
    try:
        df2 = pd.read_csv(FILE2_PATH)
    except Exception as e:
        print(f"Error loading File 2: {e}")
        return

    print("\n--- Column Analysis ---")
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)

    common_cols = cols1.intersection(cols2)
    unique_to_1 = cols1 - cols2
    unique_to_2 = cols2 - cols1

    print(f"Columns in File 1: {len(cols1)}")
    print(f"Columns in File 2: {len(cols2)}")
    print(f"Common Columns ({len(common_cols)}): {sorted(list(common_cols))}")
    print(f"Unique to File 1 ({len(unique_to_1)}): {sorted(list(unique_to_1))}")
    print(f"Unique to File 2 ({len(unique_to_2)}): {sorted(list(unique_to_2))}")

    print("\n--- Row Analysis ---")
    print(f"Rows in File 1: {len(df1)}")
    print(f"Rows in File 2: {len(df2)}")

    if 'order_id' not in df1.columns or 'order_id' not in df2.columns:
        print("Error: 'order_id' column missing in one or both files. Cannot perform ID-based comparison.")
        return

    ids1 = set(df1['order_id'])
    ids2 = set(df2['order_id'])

    common_ids = ids1.intersection(ids2)
    unique_ids_1 = ids1 - ids2
    unique_ids_2 = ids2 - ids1

    print(f"Unique order_ids in File 1: {len(ids1)}")
    print(f"Unique order_ids in File 2: {len(ids2)}")
    print(f"Common order_ids ({len(common_ids)})")
    print(f"order_ids unique to File 1 ({len(unique_ids_1)})")
    print(f"order_ids unique to File 2 ({len(unique_ids_2)})")

    print("\n--- Value Comparison (Common Rows & Columns) ---")
    if not common_ids:
        print("No common order_ids found.")
        return

    # Filter to common IDs and columns
    # Use string type for order_id to ensure matching works if one is int and other is str
    df1['order_id'] = df1['order_id'].astype(str)
    df2['order_id'] = df2['order_id'].astype(str)
    common_ids_str = set([str(x) for x in common_ids])

    cols_to_compare = sorted([c for c in common_cols if c != 'order_id'])

    df1_common = df1[df1['order_id'].isin(common_ids_str)].set_index('order_id')[cols_to_compare]
    df2_common = df2[df2['order_id'].isin(common_ids_str)].set_index('order_id')[cols_to_compare]

    # Sort to ensure alignment
    df1_common.sort_index(inplace=True)
    df2_common.sort_index(inplace=True)

    # Compare
    for col in cols_to_compare:

        s1 = df1_common[col]
        s2 = df2_common[col]

        # Fill NaNs with a placeholder for comparison to treat NaN == NaN as True
        s1_filled = s1.fillna('__NAN__')
        s2_filled = s2.fillna('__NAN__')

        # Simple equality check
        matches = (s1_filled == s2_filled)
        match_count = matches.sum()
        total_count = len(matches)
        match_pct = (match_count / total_count) * 100

        print(f"Column '{col}': {match_pct:.2f}% match ({match_count}/{total_count})")
        if match_pct < 100:
            # Show a few examples of mismatches
            mismatches = df1_common[~matches][[col]].rename(columns={col: 'File1'}).join(
                df2_common[~matches][[col]].rename(columns={col: 'File2'})
            )
            print(f"  Sample mismatches (first 5):")
            print(mismatches.head(5).to_string())


if __name__ == "__main__":
    compare_csvs()
