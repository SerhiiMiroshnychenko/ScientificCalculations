"""
Розрахунок сумарного вмісту вуглецю по горизонтах агломераційної шихти
"""

import numpy as np
import pandas as pd

# Вміст вуглецю для кожної фракції
carbon_content = {
    '11.0': 1.79,
    '8.0': 2.88,
    '6.5': 3.11,
    '4.0': 3.66,
    '2.0': 4.35,
    '0.5': 4.42
}

# Дані по горизонтах (відсотковий вміст фракцій)
horizons_data = {
    'horizon_01': {
        '11.0': 3.59,
        '8.0': 1.67,
        '6.5': 9.56,
        '4.0': 23.91,
        '2.0': 61.53
    },
    'horizon_02': {
        '11.0': 5.76,
        '8.0': 4.54,
        '6.5': 16.60,
        '4.0': 26.27,
        '2.0': 46.68
    },
    'horizon_03': {
        '11.0': 6.41,
        '8.0': 4.43,
        '6.5': 18.09,
        '4.0': 25.23,
        '2.0': 45.76
    },
    'horizon_04': {
        '11.0': 14.87,
        '8.0': 8.34,
        '6.5': 20.48,
        '4.0': 20.54,
        '2.0': 35.77
    }
}

def calculate_total_carbon(horizon_data):
    """
    Розрахунок сумарного вмісту вуглецю для одного горизонту
    """
    total_carbon = 0
    for fraction, percentage in horizon_data.items():
        # Переводимо відсотки в частки
        fraction_part = percentage / 100
        # Множимо на вміст вуглецю у фракції
        carbon_part = carbon_content[fraction]
        # Додаємо до загальної суми
        total_carbon += fraction_part * carbon_part
    return total_carbon

# Розрахунок для всіх горизонтів
results = {}
for horizon, data in horizons_data.items():
    total_carbon = calculate_total_carbon(data)
    results[horizon] = total_carbon

# Виведення результатів
print("\nСумарний вміст вуглецю по горизонтах:")
print("-" * 50)
print("| Горизонт | Вміст вуглецю, % |")
print("|----------|-----------------|")
for horizon, carbon in results.items():
    h = horizon.replace('horizon_0', '')
    print(f"| {h}        | {carbon:14.2f} |")
print("-" * 50)

# Збереження результатів у markdown файл
with open('carbon_total_results.md', 'w', encoding='utf-8') as f:
    f.write("# Результати розрахунку сумарного вмісту вуглецю\n\n")
    f.write("## Сумарний вміст вуглецю по горизонтах\n\n")
    f.write("| Горизонт | Вміст вуглецю, % |\n")
    f.write("|----------|------------------|\n")
    for horizon, carbon in results.items():
        h = horizon.replace('horizon_0', '')
        f.write(f"| {h}        | {carbon:14.2f} |\n")
    
    # Додаткова статистика
    values = list(results.values())
    f.write("\n## Статистичний аналіз\n\n")
    f.write(f"- Середнє значення: {np.mean(values):.2f}%\n")
    f.write(f"- Мінімальне значення: {np.min(values):.2f}%\n")
    f.write(f"- Максимальне значення: {np.max(values):.2f}%\n")
    f.write(f"- Стандартне відхилення: {np.std(values):.2f}%\n")

# Додатковий аналіз
print("\nСтатистичний аналіз:")
print(f"Середнє значення: {np.mean(values):.2f}%")
print(f"Мінімальне значення: {np.min(values):.2f}%")
print(f"Максимальне значення: {np.max(values):.2f}%")
print(f"Стандартне відхилення: {np.std(values):.2f}%")
