import pandas as pd
import numpy as np
import os
from datetime import datetime

def generate_violin_stats(csv_path, output_md):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Визначаємо успішність
    df['is_success'] = df['state'] == 'sale'
    
    # Для дат конвертуємо
    if 'date_order' in df.columns:
        df['date_order_dt'] = pd.to_datetime(df['date_order'], errors='coerce')
        # Фільтруємо старі помилкові дати (наприклад до 2017)
        df_dates = df[(df['date_order_dt'] >= '2017-01-01') | (df['date_order_dt'].isna())].copy()
        df_dates['date_order_ts'] = df_dates['date_order_dt'].view('int64') // 10**9
        df_dates.loc[df_dates['date_order_dt'].isna(), 'date_order_ts'] = np.nan
        df['date_order_ts'] = df_dates['date_order_ts']

    features_to_describe = [
        ('messages_count', 'Messages Count'),
        ('customer_relationship_days', 'Customer Relationship (Days)'),
        ('previous_orders_count', 'Previous Orders Count'),
        ('hour_of_day', 'Hour of Day'),
        ('date_order_ts', 'Order Date')
    ]

    df_u = df[df['is_success'] == False]
    df_s = df[df['is_success'] == True]

    md_lines = []
    md_lines.append("### Quantitative Analytics for Violin Plots (Fig. 1-5)")
    md_lines.append("")
    md_lines.append("The density distribution illustrated in the violin plots is underpinned by the following key statistical indicators, which demonstrate a distinct quantitative variance between successfully closed and lost orders.")
    md_lines.append("")
    
    # Table format
    md_lines.append("| Feature | Status | Median | Mean | Q1 (25th) | Q3 (75th) | Std Dev | N |")
    md_lines.append("|---|---|---|---|---|---|---|---|")

    def format_num(val, is_date=False):
        if pd.isna(val): return "N/A"
        if is_date:
            try:
                dt = datetime.fromtimestamp(val)
                return dt.strftime('%Y-%m-%d')
            except: return "N/A"
        
        if abs(val) >= 100: return f"{val:,.1f}"
        return f"{val:.2f}"

    def format_std(val, is_date=False):
        if pd.isna(val): return "N/A"
        if is_date:
            days = val / 86400
            return f"{days:.1f} days"
        if abs(val) >= 100: return f"{val:,.1f}"
        return f"{val:.2f}"

    text_summaries = []

    for col, display_name in features_to_describe:
        if col not in df.columns:
            continue
            
        is_date = (col == 'date_order_ts')
        
        desc_u = df_u[col].describe()
        desc_s = df_s[col].describe()
        
        count_u, count_s = int(desc_u['count']), int(desc_s['count'])
        med_u, med_s = desc_u['50%'], desc_s['50%']
        mean_u, mean_s = desc_u['mean'], desc_s['mean']
        q1_u, q1_s = desc_u['25%'], desc_s['25%']
        q3_u, q3_s = desc_u['75%'], desc_s['75%']
        std_u, std_s = desc_u['std'], desc_s['std']

        # Add Unsuccessful row
        md_lines.append(
            f"| **{display_name}** | Unsuccessful | **{format_num(med_u, is_date)}** | {format_num(mean_u, is_date)} | {format_num(q1_u, is_date)} | {format_num(q3_u, is_date)} | {format_std(std_u, is_date)} | {count_u:,} |"
        )
        # Add Successful row
        md_lines.append(
            f"| | Successful | **{format_num(med_s, is_date)}** | {format_num(mean_s, is_date)} | {format_num(q1_s, is_date)} | {format_num(q3_s, is_date)} | {format_std(std_s, is_date)} | {count_s:,} |"
        )
        
        # Build text summary
        text_summaries.append(
            f"**{display_name}**: The median value for successful interactions is higher/lower ({format_num(med_s, is_date)}) compared to unsuccessful ones ({format_num(med_u, is_date)}), with corresponding means of {format_num(mean_s, is_date)} vs {format_num(mean_u, is_date)}."
        )

    md_lines.append("")
    md_lines.append("### Extract for Text Description:")
    md_lines.append("")
    for ts in text_summaries:
        md_lines.append("- " + ts)

    with open(output_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
        
    print(f"Violin statistics generated at: {output_md}")

if __name__ == '__main__':
    target_csv = "data_collector_extended.csv"
    output_md = "Violin_Plots_Statistics.md"
    
    if os.path.exists(target_csv):
        generate_violin_stats(target_csv, output_md)
    else:
        print(f"Error: {target_csv} not found.")
