"""
Аналіз фракційного складу шихти на агломашині №2
Розрахунок середніх значень для кожної фракції по горизонтах
"""

import numpy as np
import pandas as pd

# Вхідні дані
# Структура даних: [права сторона, середина, ліва сторона]
data = {
    'horizon_01': {
        '+10': [3.77, 3.52, 3.48],
        '+8-10': [1.89, 1.32, 1.79],
        '+5-8': [8.14, 10.28, 10.27],
        '+3-5': [17.57, 30.54, 23.61],
        '-3': [68.63, 54.33, 61.63],
        'diameter': [3.74, 3.61, 4.63]
    },
    'horizon_02': {
        '+10': [4.26, 5.96, 7.05],
        '+8-10': [6.04, 4.09, 3.48],
        '+5-8': [15.81, 16.99, 16.99],
        '+3-5': [20.43, 33.99, 24.38],
        '-3': [52.96, 39.00, 48.09],
        'diameter': [4.16, 4.34, 4.49]
    },
    'horizon_03': {
        '+10': [5.08, 7.96, 6.19],
        '+8-10': [3.88, 4.73, 4.67],
        '+5-8': [15.84, 19.34, 19.09],
        '+3-5': [21.61, 29.46, 24.63],
        '-3': [53.59, 38.29, 45.41],
        'diameter': [4.59, 4.72, 5.09]
    },
    'horizon_04': {
        '+10': [19.56, 12.02, 13.04],
        '+8-10': [7.25, 9.92, 7.84],
        '+5-8': [17.28, 23.52, 20.64],
        '+3-5': [16.46, 23.12, 22.04],
        '-3': [39.45, 31.42, 36.44],
        'diameter': [5.37, 5.35, 6.24]
    }
}

# Розрахунок середніх значень
def calculate_means(data):
    means = {}
    for horizon, values in data.items():
        means[horizon] = {
            fraction: np.mean(val) for fraction, val in values.items()
        }
    return means

# Форматування результатів
def format_results(means):
    print("\nСередні значення фракційного складу шихти по горизонтах:")
    print("-" * 70)
    print("| Горизонт | +10  | +8-10 | +5-8  | +3-5  | -3    | Діаметр |")
    print("|----------|------|-------|-------|--------|--------|---------|")
    
    for horizon, values in means.items():
        h = horizon.replace('horizon_0', '')
        print(f"| {h}        | {values['+10']:5.2f} | {values['+8-10']:5.2f} | "
              f"{values['+5-8']:5.2f} | {values['+3-5']:6.2f} | {values['-3']:6.2f} | "
              f"{values['diameter']:7.2f} |")
    print("-" * 70)

# Створення DataFrame для додаткового аналізу
def create_dataframe(means):
    df = pd.DataFrame.from_dict(means, orient='index')
    return df

# Основний код
if __name__ == "__main__":
    # Розрахунок середніх значень
    means = calculate_means(data)
    
    # Виведення результатів у вигляді таблиці
    format_results(means)
    
    # Створення DataFrame
    df = create_dataframe(means)
    
    # Додатковий аналіз
    print("\nСтатистичний аналіз:")
    print("\nСередні значення по всіх горизонтах:")
    print(df.mean())
    
    print("\nСтандартне відхилення:")
    print(df.std())
    
    # Збереження результатів у файл
    with open('fraction_analysis_results.md', 'w', encoding='utf-8') as f:
        f.write("# Результати аналізу фракційного складу шихти\n\n")
        f.write("## Середні значення по горизонтах\n\n")
        f.write(df.to_markdown(floatfmt='.2f'))
        f.write("\n\n## Статистичний аналіз\n\n")
        f.write("### Середні значення по всіх горизонтах\n\n")
        f.write(df.mean().to_markdown(floatfmt='.2f'))
        f.write("\n\n### Стандартне відхилення\n\n")
        f.write(df.std().to_markdown(floatfmt='.2f'))
