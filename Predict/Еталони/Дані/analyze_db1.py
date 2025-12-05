import pandas as pd
import os

file_path = r"db1_for_ml.csv"

try:
    # Attempt to read with default settings, then try common separators if needed
    df = pd.read_csv(file_path)
    
    print(f"--- Аналіз файлу: {os.path.basename(file_path)} ---")
    print(f"Розмірність: {df.shape}")
    print("\n--- Інформація про типи даних ---")
    print(df.info())
    
    print("\n--- Статистика для числових колонок ---")
    print(df.describe())
    
    print("\n--- Детальний аналіз кожної колонки ---")
    for col in df.columns:
        print(f"\nКолонка: '{col}'")
        print(f"  Тип: {df[col].dtype}")
        print(f"  Кількість унікальних значень: {df[col].nunique()}")
        print(f"  Кількість пропущених (null): {df[col].isnull().sum()}")
        
        # Show sample values
        print(f"  Топ 5 найчастіших значень:")
        print(df[col].value_counts().head(5).to_string())
        
except Exception as e:
    print(f"Помилка при обробці файлу: {e}")
