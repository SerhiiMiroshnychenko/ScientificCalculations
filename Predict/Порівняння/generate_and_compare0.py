import pandas as pd
import numpy as np
from pathlib import Path
import re
import sys

# --- CONFIGURATION ---
INPUT_DB1 = Path(r"db1.csv")
TARGET_ML = Path(r"b2b_for_ml.csv")
OUTPUT_GEN = Path(r"b2b_generated_now.csv")

# ==========================================
# USER PROVIDED BUILD LOGIC
# ==========================================

def load_orders(path: Path) -> pd.DataFrame:
    print(f"[INFO] Reading DB1: {path}...")
    return pd.read_csv(path)

def get_db_offset(path: Path) -> int:
    # Hardcoded logic for db1 if regex fails, or use user logic
    name = path.stem
    m = re.search(r"(\d+)", name)
    if not m:
        # Fallback if filename is just 'db1' or similar
        if 'db1' in str(path).lower(): return 1_000_000
        return 0
    return int(m.group(1)) * 1_000_000

def add_order_level_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = ["order_id", "partner_id", "state", "total_amount", "messages_count", "changes_count", "create_date", "date_order"]
    # Check missing
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[WARN] Missing columns: {missing}")
        # Try to proceed if possible or error out? User logic raises KeyError.
        # Let's try to be robust or fail as user logic does.
        pass 

    df = df.copy()
    successful_states = {"sale", "done"}
    df["is_successful"] = df["state"].astype(str).str.lower().isin(successful_states).astype(int)
    df["create_date"] = pd.to_datetime(df["create_date"], errors="coerce")
    df["date_order"] = pd.to_datetime(df["date_order"], errors="coerce")
    df["order_amount"] = df["total_amount"].astype(float)
    df["order_messages"] = pd.to_numeric(df["messages_count"], errors="coerce").fillna(0).astype(float)
    df["order_changes"] = pd.to_numeric(df["changes_count"], errors="coerce").fillna(0).astype(float)
    return df

def build_historical_partner_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Sort: partner, date, order_id
    df.sort_values(["partner_id", "create_date", "order_id"], inplace=True)
    
    grp = df.groupby("partner_id", group_keys=False)
    
    df["partner_total_orders"] = grp.cumcount()
    
    cum_amount_all = grp["order_amount"].cumsum()
    cum_changes_all = grp["order_changes"].cumsum()
    cum_messages_all = grp["order_messages"].cumsum()
    
    df["partner_total_messages"] = (cum_messages_all - df["order_messages"]).astype(float)
    df["_amount_prev_all"] = (cum_amount_all - df["order_amount"]).astype(float)
    df["_changes_prev_all"] = (cum_changes_all - df["order_changes"]).astype(float)
    
    df["partner_avg_amount"] = (df["_amount_prev_all"] / df["partner_total_orders"].replace(0, np.nan)).fillna(0.0)
    df["partner_avg_changes"] = (df["_changes_prev_all"] / df["partner_total_orders"].replace(0, np.nan)).fillna(0.0)
    
    success_mask = df["is_successful"] == 1
    fail_mask = ~success_mask
    
    df["_success_flag"] = success_mask.astype(int)
    df["_fail_flag"] = fail_mask.astype(int)
    
    cum_success_orders = grp["_success_flag"].cumsum()
    cum_fail_orders = grp["_fail_flag"].cumsum()
    
    df["partner_success_orders"] = (cum_success_orders - df["_success_flag"]).astype(float)
    df["partner_fail_orders"] = (cum_fail_orders - df["_fail_flag"]).astype(float)
    
    df["_success_amount"] = df["order_amount"].where(success_mask, 0.0)
    df["_fail_amount"] = df["order_amount"].where(fail_mask, 0.0)
    df["_success_messages"] = df["order_messages"].where(success_mask, 0.0)
    df["_fail_messages"] = df["order_messages"].where(fail_mask, 0.0)
    df["_success_changes"] = df["order_changes"].where(success_mask, 0.0)
    df["_fail_changes"] = df["order_changes"].where(fail_mask, 0.0)
    
    # Cumulative sums
    cum_success_amount = grp["_success_amount"].cumsum()
    cum_fail_amount = grp["_fail_amount"].cumsum()
    cum_success_messages = grp["_success_messages"].cumsum()
    cum_fail_messages = grp["_fail_messages"].cumsum()
    cum_success_changes = grp["_success_changes"].cumsum()
    cum_fail_changes = grp["_fail_changes"].cumsum()
    
    # Subtract current to get previous
    df["_amount_prev_success"] = (cum_success_amount - df["_success_amount"]).astype(float)
    df["_amount_prev_fail"] = (cum_fail_amount - df["_fail_amount"]).astype(float)
    df["_messages_prev_success"] = (cum_success_messages - df["_success_messages"]).astype(float)
    df["_messages_prev_fail"] = (cum_fail_messages - df["_fail_messages"]).astype(float)
    df["_changes_prev_success"] = (cum_success_changes - df["_success_changes"]).astype(float)
    df["_changes_prev_fail"] = (cum_fail_changes - df["_fail_changes"]).astype(float)
    
    # Averages
    df["partner_success_avg_amount"] = (df["_amount_prev_success"] / df["partner_success_orders"].replace(0, np.nan)).fillna(0.0)
    df["partner_fail_avg_amount"] = (df["_amount_prev_fail"] / df["partner_fail_orders"].replace(0, np.nan)).fillna(0.0)
    
    df["partner_success_avg_messages"] = (df["_messages_prev_success"] / df["partner_success_orders"].replace(0, np.nan)).fillna(0.0)
    df["partner_fail_avg_messages"] = (df["_messages_prev_fail"] / df["partner_fail_orders"].replace(0, np.nan)).fillna(0.0)
    
    df["partner_success_avg_changes"] = (df["_changes_prev_success"] / df["partner_success_orders"].replace(0, np.nan)).fillna(0.0)
    df["partner_fail_avg_changes"] = (df["_changes_prev_fail"] / df["partner_fail_orders"].replace(0, np.nan)).fillna(0.0)
    
    df["partner_success_rate"] = (df["partner_success_orders"] / df["partner_total_orders"].replace(0, np.nan)).fillna(0.0) * 100.0
    
    first_order_date = grp["create_date"].transform("min")
    df["partner_order_age_days"] = ((df["create_date"] - first_order_date).dt.days.fillna(0).astype(int))
    
    # Cleanup
    drop_cols = ["_amount_prev_all", "_changes_prev_all", "_success_flag", "_fail_flag", 
                 "_success_amount", "_fail_amount", "_success_messages", "_fail_messages", 
                 "_success_changes", "_fail_changes", "_amount_prev_success", "_amount_prev_fail", 
                 "_messages_prev_success", "_messages_prev_fail", "_changes_prev_success", "_changes_prev_fail"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    return df

def apply_id_offset(df: pd.DataFrame, id_offset: int) -> pd.DataFrame:
    df = df.copy()
    if id_offset == 0: return df
    
    if "partner_id" in df.columns:
        df["partner_id"] = pd.to_numeric(df["partner_id"], errors="coerce").fillna(0).astype("int64") + id_offset
        
    if "order_id" in df.columns:
        # Extract numeric part from SOxxx
        order_str = df["order_id"].astype(str)
        numeric_part = order_str.str.extract(r"(\d+)", expand=False)
        mask = numeric_part.notna()
        if mask.any():
            numeric_part = numeric_part[mask].astype(int) + id_offset
            df.loc[mask, "order_id"] = numeric_part.astype("int64")
    return df

def build_b2b(df_orders: pd.DataFrame, id_offset: int) -> pd.DataFrame:
    df = add_order_level_columns(df_orders)
    df = build_historical_partner_features(df)
    df = apply_id_offset(df, id_offset)
    
    cols = [
        "order_id", "is_successful", "create_date", "partner_id", "order_amount", 
        "order_messages", "order_changes", "partner_success_rate", "partner_total_orders", 
        "partner_order_age_days", "partner_avg_amount", "partner_success_avg_amount", 
        "partner_fail_avg_amount", "partner_total_messages", "partner_success_avg_messages", 
        "partner_fail_avg_messages", "partner_avg_changes", "partner_success_avg_changes", 
        "partner_fail_avg_changes"
    ]
    # Return only existing cols
    return df[[c for c in cols if c in df.columns]].copy()

# ==========================================
# COMPARISON LOGIC
# ==========================================

def compare_generated_vs_target(gen_df, target_df):
    print("\n[INFO] Comparing GENERATED vs TARGET...")
    
    # Align by order_id
    common_ids = set(gen_df['order_id']).intersection(set(target_df['order_id']))
    print(f"Common Order IDs: {len(common_ids)}")
    
    if len(common_ids) == 0:
        print("[ERROR] No common IDs! Check offset logic.")
        print(f"Generated Sample ID: {gen_df['order_id'].iloc[0]}")
        print(f"Target Sample ID:    {target_df['order_id'].iloc[0]}")
        return

    g = gen_df[gen_df['order_id'].isin(common_ids)].sort_values('order_id').reset_index(drop=True)
    t = target_df[target_df['order_id'].isin(common_ids)].sort_values('order_id').reset_index(drop=True)
    
    # Compare columns
    common_cols = sorted(list(set(g.columns).intersection(set(t.columns))))
    print(f"Common Columns: {len(common_cols)}")
    
    for col in common_cols:
        if col == 'order_id': continue
        
        s_gen = g[col]
        s_tgt = t[col]
        
        if pd.api.types.is_numeric_dtype(s_gen) and pd.api.types.is_numeric_dtype(s_tgt):
            is_close = np.isclose(s_gen.fillna(-9999), s_tgt.fillna(-9999), rtol=1e-05, atol=1e-08)
            match_pct = (is_close.sum() / len(is_close)) * 100
        else:
            s_gen_str = s_gen.astype(str).str.strip().fillna("")
            s_tgt_str = s_tgt.astype(str).str.strip().fillna("")
            match_pct = ((s_gen_str == s_tgt_str).sum() / len(s_gen_str)) * 100
            
        print(f"  {col}: {match_pct:.2f}% match")
        if match_pct < 100 and match_pct > 90:
             # Show a sample diff
             mask = ~np.isclose(s_gen.fillna(-9999), s_tgt.fillna(-9999), rtol=1e-05, atol=1e-08) if pd.api.types.is_numeric_dtype(s_gen) else (s_gen.astype(str) != s_tgt.astype(str))
             diffs = pd.DataFrame({'ID': g.loc[mask, 'order_id'], 'Gen': s_gen[mask], 'Tgt': s_tgt[mask]})
             print(f"    Sample Diff:\n{diffs.head(2).to_string(index=False)}")

def main():
    # 1. Load DB1
    if not INPUT_DB1.exists():
        print(f"File not found: {INPUT_DB1}")
        return
    df_db1 = load_orders(INPUT_DB1)
    
    # 2. Build B2B
    print("[INFO] Building B2B from DB1...")
    offset = 1_000_000 # Force 1M for db1 as per user context
    b2b_gen = build_b2b(df_db1, offset)
    print(f"Generated {len(b2b_gen)} rows.")
    
    # 3. Save Generated
    b2b_gen.to_csv(OUTPUT_GEN, index=False)
    print(f"Saved generated file to {OUTPUT_GEN}")
    
    # 4. Load Target
    if not TARGET_ML.exists():
        print(f"Target file not found: {TARGET_ML}")
        return
    print(f"[INFO] Loading Target: {TARGET_ML}")
    b2b_target = pd.read_csv(TARGET_ML)
    
    # 5. Compare
    compare_generated_vs_target(b2b_gen, b2b_target)

if __name__ == "__main__":
    main()
