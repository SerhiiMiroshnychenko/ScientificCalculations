import pandas as pd
from pathlib import Path

INPUT_FILE = Path(r"db8_for_ml.csv")
OUTPUT_REPORT = Path(r"db8_stats_report.csv")


def main():
    print(f"[INFO] Loading {INPUT_FILE}...")
    if not INPUT_FILE.exists():
        print(f"[ERROR] File {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"[INFO] Shape: {df.shape}")

    # Basic describe
    stats = df.describe(include='all').transpose()

    # Add additional metrics
    stats['missing_count'] = df.isnull().sum()
    stats['missing_pct'] = (df.isnull().sum() / len(df)) * 100
    stats['unique_count'] = df.nunique()
    stats['dtype'] = df.dtypes

    # Reorder columns for better readability
    cols_order = [
        'dtype', 'count', 'missing_count', 'missing_pct', 'unique_count',
        'mean', 'std', 'min', '25%', '50%', '75%', 'max',
        'top', 'freq'
    ]
    # Filter only existing columns (describe might not return all if mixed types)
    cols_order = [c for c in cols_order if c in stats.columns]

    stats = stats[cols_order]

    print("\n=== STATISTICS SUMMARY ===")
    # Force pandas to show all columns and rows
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    print(stats)

    # Save to CSV for detailed analysis
    stats.to_csv(OUTPUT_REPORT)
    print(f"\n[INFO] Detailed report saved to {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
