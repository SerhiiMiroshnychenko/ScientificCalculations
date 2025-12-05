import pandas as pd
from pathlib import Path
import glob

STATS_DIR = Path(r"D:\WINDSURF\DATABASES\Еталони\Дані\Статистика")

def main():
    print(f"[INFO] Analyzing stats in {STATS_DIR}...")
    
    files = sorted(glob.glob(str(STATS_DIR / "db*_stats_report.csv")))
    if not files:
        print("[ERROR] No stats files found.")
        return

    summary_data = []

    for f_path in files:
        db_name = Path(f_path).stem.replace("_stats_report", "")
        try:
            df = pd.read_csv(f_path, index_col=0)
            
            # Extract key metrics
            # We expect rows to be features (order_id, is_successful, etc.) and columns to be stats (count, mean, etc.)
            # Based on previous calculate_stats.py output format
            
            row = {'DB': db_name}
            
            # Count
            if 'order_id' in df.index:
                row['Orders'] = df.loc['order_id', 'count']
            
            # Success Rate
            if 'is_successful' in df.index:
                row['Success Rate'] = df.loc['is_successful', 'mean']
            
            # Amount
            if 'order_amount' in df.index:
                row['Avg Amount'] = df.loc['order_amount', 'mean']
                row['Max Amount'] = df.loc['order_amount', 'max']
            
            # Messages
            if 'order_messages' in df.index:
                row['Avg Messages'] = df.loc['order_messages', 'mean']
            elif 'messages_count' in df.index: # Fallback if name differs
                 row['Avg Messages'] = df.loc['messages_count', 'mean']

            # Changes
            if 'order_changes' in df.index:
                row['Avg Changes'] = df.loc['order_changes', 'mean']
            
            # Partner Stats
            if 'partner_total_orders' in df.index:
                row['Avg Partner Orders'] = df.loc['partner_total_orders', 'mean']
            
            summary_data.append(row)

        except Exception as e:
            print(f"[WARN] Error reading {db_name}: {e}")

    summary_df = pd.DataFrame(summary_data)
    
    # Formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.2f}'.format)
    
    print("\n=== COMPARATIVE ANALYSIS OF DATABASES ===")
    print(summary_df)
    
    # Additional insights
    print("\n=== INSIGHTS ===")
    if not summary_df.empty:
        max_orders = summary_df.loc[summary_df['Orders'].idxmax()]
        print(f"Largest DB: {max_orders['DB']} ({max_orders['Orders']:.0f} orders)")
        
        best_success = summary_df.loc[summary_df['Success Rate'].idxmax()]
        print(f"Highest Success Rate: {best_success['DB']} ({best_success['Success Rate']*100:.1f}%)")
        
        rich_db = summary_df.loc[summary_df['Avg Amount'].idxmax()]
        print(f"Highest Avg Check: {rich_db['DB']} ({rich_db['Avg Amount']:.2f})")

if __name__ == "__main__":
    main()
