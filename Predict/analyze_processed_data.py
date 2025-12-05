import pandas as pd
from pathlib import Path
import glob
import datetime

# Configuration
DATA_DIR = Path(r"D:\WINDSURF\DATABASES\Еталони\Дані\Оброблені")

def analyze_file(file_path):
    print(f"\n{'='*50}")
    print(f"ANALYZING: {file_path.name}")
    print(f"{'='*50}")
    
    try:
        df = pd.read_csv(file_path)
        
        # Basic counts
        total_orders = len(df)
        
        # Date analysis
        if 'create_date' in df.columns:
            df['create_date'] = pd.to_datetime(df['create_date'], errors='coerce')
            min_date = df['create_date'].min()
            max_date = df['create_date'].max()
            duration = max_date - min_date
            duration_days = duration.days
            duration_years = duration_days / 365.25
        else:
            min_date = max_date = duration_days = duration_years = None

        # Partner analysis
        if 'partner_id' in df.columns:
            unique_partners = df['partner_id'].nunique()
            avg_orders_per_partner = total_orders / unique_partners if unique_partners else 0
        else:
            unique_partners = avg_orders_per_partner = None

        # Financials
        if 'order_amount' in df.columns:
            total_revenue = df['order_amount'].sum()
            avg_check = df['order_amount'].mean()
            max_check = df['order_amount'].max()
        else:
            total_revenue = avg_check = max_check = None

        # Success Rate
        if 'is_successful' in df.columns:
            success_rate = df['is_successful'].mean() * 100
            successful_orders = df['is_successful'].sum()
        else:
            success_rate = successful_orders = None

        # Order Lines (if available)
        if 'order_lines' in df.columns:
            avg_lines = df['order_lines'].mean()
        elif 'order_lines_count' in df.columns:
            avg_lines = df['order_lines_count'].mean()
        else:
            avg_lines = None

        # --- REPORTING ---
        print(f"1. SCALE & SCOPE")
        print(f"   - Total Orders:      {total_orders:,}")
        print(f"   - Unique Clients:    {unique_partners:,}")
        print(f"   - Avg Orders/Client: {avg_orders_per_partner:.2f}")
        
        print(f"\n2. TIMELINE")
        print(f"   - First Order:       {min_date}")
        print(f"   - Last Order:        {max_date}")
        print(f"   - Business Duration: {duration_days:,} days ({duration_years:.1f} years)")
        if duration_days > 0:
            print(f"   - Orders per Day:    {total_orders / duration_days:.2f}")

        print(f"\n3. FINANCIALS")
        if total_revenue is not None:
            print(f"   - Total Revenue:     {total_revenue:,.2f}")
            print(f"   - Avg Check:         {avg_check:,.2f}")
            print(f"   - Max Check:         {max_check:,.2f}")
        
        print(f"\n4. PERFORMANCE")
        if success_rate is not None:
            print(f"   - Success Rate:      {success_rate:.2f}% ({successful_orders} orders)")
        if avg_lines is not None:
            print(f"   - Avg Order Lines:   {avg_lines:.2f}")

    except Exception as e:
        print(f"[ERROR] Failed to analyze {file_path.name}: {e}")

def main():
    files = glob.glob(str(DATA_DIR / "*_for_ml.csv"))
    if not files:
        print(f"No *_for_ml.csv files found in {DATA_DIR}")
        return

    # Sort files for consistent output
    files.sort()

    for file_path in files:
        analyze_file(Path(file_path))

if __name__ == "__main__":
    main()
