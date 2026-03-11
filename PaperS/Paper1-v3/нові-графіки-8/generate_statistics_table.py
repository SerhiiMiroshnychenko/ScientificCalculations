import pandas as pd
import numpy as np
import os
from datetime import datetime

def generate_full_statistics_table(dataset_path, feature_importance_path, output_md):
    print(f"Loading feature importance from {feature_importance_path}...")
    fi_df = pd.read_csv(feature_importance_path)
    
    # Отримуємо точний список фіч із теплової карти
    target_features = fi_df['Ознака'].dropna().tolist()
    print(f"Features targeted (exactly as in heatmap): {target_features}")
    
    print(f"Loading data from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    # У dataset-1.csv колонка-таргет називається 'is_successful' (вже як 1 та 0)
    df['is_success'] = df['is_successful'] == 1
    
    # Відтворюємо create_date_months з create_date
    if 'create_date' in df.columns:
        df['create_date'] = pd.to_datetime(df['create_date'], errors='coerce')
        # Обчислюємо скільки місяців пройшло від першого замовлення для кожної дати
        min_date = df['create_date'].min()
        df['create_date_months'] = (df['create_date'] - min_date).dt.days / 30.44

    md_lines = []
    md_lines.append("### Comprehensive Descriptive Statistics (Heatmap Features)")
    md_lines.append("")
    md_lines.append("This table provides the exact quantitative values ONLY for the specific features highlighted in the Feature Importance Heatmap.")
    md_lines.append("")
    
    md_lines.append("| Feature | Type | Mean / Top Freq (U) | Mean / Top Freq (S) | Q1 (U) | Q1 (S) | Median / Mode (U) | Median / Mode (S) | Q3 (U) | Q3 (S) | Std Dev (U) | Std Dev (S) | Count (U) | Count (S) |")
    md_lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    df_u = df[df['is_success'] == False]
    df_s = df[df['is_success'] == True]

    def format_num(val):
        if pd.isna(val): return "N/A"
        if abs(val) >= 100: return f"{val:,.1f}"
        return f"{val:.2f}"

    for feat in target_features:
        if feat not in df.columns:
            print(f"Warning: Feature '{feat}' not found in dataset. Skipping.")
            continue
            
        is_numeric = pd.api.types.is_numeric_dtype(df[feat])
        count_u = df_u[feat].count()
        count_s = df_s[feat].count()
        
        if count_u < 5 and count_s < 5:
            continue
            
        display_name = feat

        if is_numeric:
            feat_type = "Numeric"
            desc_u = df_u[feat].describe()
            desc_s = df_s[feat].describe()
            
            mean_u, mean_s = format_num(desc_u['mean']), format_num(desc_s['mean'])
            q1_u, q1_s = format_num(desc_u['25%']), format_num(desc_s['25%'])
            med_u, med_s = f"**{format_num(desc_u['50%'])}**", f"**{format_num(desc_s['50%'])}**"
            q3_u, q3_s = format_num(desc_u['75%']), format_num(desc_s['75%'])
            std_u, std_s = format_num(desc_u['std']), format_num(desc_s['std'])
            
        else:
            feat_type = "Categorical"
            mode_u_val = df_u[feat].mode().iloc[0] if not df_u[feat].mode().empty else "N/A"
            mode_s_val = df_s[feat].mode().iloc[0] if not df_s[feat].mode().empty else "N/A"
            
            freq_u = (df_u[feat] == mode_u_val).mean() * 100 if mode_u_val != "N/A" else 0
            freq_s = (df_s[feat] == mode_s_val).mean() * 100 if mode_s_val != "N/A" else 0

            mean_u, mean_s = f"{freq_u:.1f}%", f"{freq_s:.1f}%"
            q1_u, q1_s = "-", "-"
            med_u, med_s = f"**{mode_u_val}**", f"**{mode_s_val}**"
            q3_u, q3_s = "-", "-"
            std_u, std_s = "-", "-"

        row_str = (
            f"| `{display_name}` | {feat_type} | "
            f"{mean_u} | {mean_s} | "
            f"{q1_u} | {q1_s} | "
            f"{med_u} | {med_s} | "
            f"{q3_u} | {q3_s} | "
            f"{std_u} | {std_s} | "
            f"{count_u:,} | {count_s:,} |"
        )
        md_lines.append(row_str)

    with open(output_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
        
    print(f"Table successfully generated at: {output_md}")

if __name__ == '__main__':
    dataset_path = r"D:\WINDSURF\ARTICLEs\BJMC\Paper1-v3\dataset-1.csv"
    fi_path = r"D:\WINDSURF\ARTICLEs\BJMC\Paper1-v3\complete\feature_importance_summary.csv"
    output_md = "Descriptive_Statistics_Table_Complete.md"
    
    if os.path.exists(dataset_path) and os.path.exists(fi_path):
        generate_full_statistics_table(dataset_path, fi_path, output_md)
    else:
        print(f"Error: One of the required files not found:\n{dataset_path}\n{fi_path}")
