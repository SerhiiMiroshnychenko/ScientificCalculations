"""
Візуалізація залежності вмісту вуглецю та температури від висоти/глибини шару
"""

import numpy as np
import matplotlib.pyplot as plt

# Дані по температурі (глибина шару)
depth_temp = np.array([400, 375, 350, 325, 300, 275, 250, 225, 200, 175, 150, 125, 100, 75, 50, 25, 0])
temperature = np.array([1000, 950, 900, 920, 980, 1100, 1250, 1320, 1350, 1360, 1370, 1380, 1385, 1390, 1395, 1400, 1410])

# Дані по вуглецю (висота шару)
height_carbon = np.array([100, 200, 300, 400])  # висота в мм
carbon_content = np.array([3.92, 3.84, 3.77, 3.58])  # приклад даних, замініть на реальні

# Створення фігури з двома y-осями
fig, ax1 = plt.subplots(figsize=(12, 8))
ax2 = ax1.twinx()

# Налаштування кольорів та стилю
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Побудова графіків
line1 = ax1.plot(depth_temp, temperature, 'b-', linewidth=2, label='Температура')
line2 = ax2.plot(height_carbon, carbon_content, 'r-', linewidth=2, label='Вміст вуглецю')

# Додавання точок даних
ax1.scatter(depth_temp, temperature, color='blue', s=50)
ax2.scatter(height_carbon, carbon_content, color='red', s=100)

# Налаштування осей
ax1.set_xlabel('Висота/глибина шару, мм', fontsize=12)
ax1.set_ylabel('Температура, °C', color='b', fontsize=12)
ax2.set_ylabel('Вміст вуглецю, %', color='r', fontsize=12)

# Налаштування кольору міток осей
ax1.tick_params(axis='y', labelcolor='b')
ax2.tick_params(axis='y', labelcolor='r')

# Додавання сітки
ax1.grid(True, linestyle='--', alpha=0.7)

# Об'єднання легенд
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', fontsize=10)

# Заголовок
plt.title('Залежність температури та вмісту вуглецю від висоти/глибини шару', fontsize=14, pad=20)

# Налаштування меж осей
ax1.set_xlim(-25, 425)
ax1.set_ylim(850, 1450)

# Додавання додаткової інформації
plt.text(0.02, 0.98, 'Агломераційна шихта\nЕксперимент №1\n\nТемпература - від глибини\nВуглець - від висоти', 
         transform=ax1.transAxes, fontsize=10, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Збереження графіка
plt.savefig('carbon_temperature_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Додатково збережемо опис графіка в markdown файл
with open('visualization_description.md', 'w', encoding='utf-8') as f:
    f.write("""# Аналіз розподілу температури та вмісту вуглецю по висоті/глибині шару

## Опис графіка

На графіку представлені дві залежності:
1. Температура шару (синя лінія, ліва вісь) - залежність від глибини шару
2. Вміст вуглецю (червона лінія, права вісь) - залежність від висоти шару

### Особливості розподілу температури (по глибині):
- Початкова температура близько 1000°C на глибині 400 мм
- Мінімум температури на глибині 350 мм
- Різке підвищення в діапазоні глибин 300-225 мм
- Стабілізація після глибини 200 мм
- Максимальна температура 1410°C на поверхні (глибина 0 мм)

### Особливості розподілу вуглецю (по висоті):
- Максимальний вміст на висоті 100 мм
- Поступове зменшення з висотою
- Мінімальне значення на висоті 400 мм

### Взаємозв'язок параметрів:
- Зменшення вмісту вуглецю з висотою корелює з підвищенням температури
- Найбільші зміни обох параметрів спостерігаються в нижній частині шару
""")
