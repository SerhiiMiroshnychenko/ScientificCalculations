import pandas as pd
from pathlib import Path

# Paths
EXT_PATH = Path(r"extended_customer_data_2025-01-17.csv")
DB1_PATH = Path(r"db1.csv")
ML_PATH = Path(r"b2b_for_ml.csv")

def normalize_id(val):
    s = str(val).strip().replace('"', '').replace("'", "")
    if s.upper().startswith('SO'): s = s[2:]
    try:
        i = int(s)
        if i > 1000000: i -= 1000000
        return i
    except:
        return s

def load_col(path, col_name, id_col='order_id'):
    print(f"Loading {col_name} from {path.name}...")
    df = pd.read_csv(path, usecols=[id_col, col_name])
    df[id_col] = df[id_col].apply(normalize_id)
    return df.set_index(id_col)[col_name]

def main():
    # Load data
    try:
        ext_cust = load_col(EXT_PATH, 'customer_id')
        db1_part = load_col(DB1_PATH, 'partner_id')
        db1_cust = load_col(DB1_PATH, 'customer_id') # Check this too
        ml_part = load_col(ML_PATH, 'partner_id')
    except Exception as e:
        print(f"Error loading: {e}")
        return

    # Find common order IDs
    common_ids = set(ext_cust.index) & set(db1_part.index) & set(ml_part.index)
    print(f"\nCommon Order IDs: {len(common_ids)}")
    
    if len(common_ids) == 0:
        print("No common IDs found (check normalization?)")
        return

    # Filter to common
    ext = ext_cust.loc[list(common_ids)]
    db1_p = db1_part.loc[list(common_ids)]
    db1_c = db1_cust.loc[list(common_ids)]
    ml = ml_part.loc[list(common_ids)]

    # Compare
    print("\n--- Comparison on Common Orders ---")
    
    # 1. Extended(customer_id) vs DB1(partner_id)
    match = (ext == db1_p).mean()
    print(f"Extended['customer_id'] == DB1['partner_id']: {match:.2%}")
    
    # 2. Extended(customer_id) vs DB1(customer_id)
    match = (ext == db1_c).mean()
    print(f"Extended['customer_id'] == DB1['customer_id']: {match:.2%}")

    # 3. Extended(customer_id) vs ML(partner_id)
    match = (ext == ml).mean()
    print(f"Extended['customer_id'] == ML['partner_id']:  {match:.2%}")

    # 4. DB1(partner_id) vs ML(partner_id)
    match = (db1_p == ml).mean()
    print(f"DB1['partner_id']       == ML['partner_id']:  {match:.2%}")

    print("\n--- Sample Values (First 5 Common) ---")
    df_sample = pd.DataFrame({
        'Ext_Cust': ext.head(5),
        'DB1_Part': db1_p.head(5),
        'DB1_Cust': db1_c.head(5),
        'ML_Part': ml.head(5)
    })
    print(df_sample)

if __name__ == "__main__":
    main()
