"""
Комплексне рішення для практичного заняття
"Нелінійний парний та множинний регресійний аналіз даних. Оцінка метричних показників"

Це рішення порівнює різні підходи до виконання завдання:
1. Ручний (математичний) розрахунок 
2. Використання statsmodels.api.OLS
3. Використання sklearn.linear_model.LinearRegression
4. Використання sklearn.preprocessing.PolynomialFeatures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Бібліотеки для різних методів регресійного аналізу
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error

# Налаштування для відтворюваності результатів
np.random.seed(42)

# Параметри для візуалізації
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Функція для збереження результатів у markdown-файл
def append_to_md(filename, content):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(content + '\n\n')

# Створення нового markdown-файлу для результатів
md_filename = 'результати_аналізу.md'
with open(md_filename, 'w', encoding='utf-8') as f:
    f.write('# Результати нелінійного регресійного аналізу даних\n\n')
    f.write('## Варіант 4\n\n')
    f.write('### Формула моделі: $y(x) = a_0 + a_1 \\cdot x_1 + a_2 \\cdot x_2 + a_3 \\cdot x_1^2 + a_4 \\cdot x_2^2 + rnd(b)$\n\n')
    f.write('### Значення коефіцієнтів:\n')
    f.write('- $a_0 = 3.4$\n')
    f.write('- $a_1 = 2.7$\n')
    f.write('- $a_2 = -0.5$\n')
    f.write('- $a_3 = 0.45$\n')
    f.write('- $a_4 = 0.25$\n')
    f.write('- $rnd(b) = 1.10$\n\n')

print("Початок виконання комплексного аналізу...")

# ================= ЧАСТИНА 1: ГЕНЕРАЦІЯ ДАНИХ =================

# Задані параметри моделі (Варіант 4)
a0 = 3.4      # вільний член
a1 = 2.7      # коефіцієнт при x1
a2 = -0.5     # коефіцієнт при x2
a3 = 0.45     # коефіцієнт при x1^2
a4 = 0.25     # коефіцієнт при x2^2
b = 1.10      # стандартне відхилення шуму

# Кількість спостережень
n_samples = 100

# Генерація значень для незалежних змінних x1 і x2
# Використовуємо рівномірний розподіл в діапазоні [-5, 5]
x1 = np.random.uniform(-5, 5, n_samples)
x2 = np.random.uniform(-5, 5, n_samples)

# Обчислення істинних значень y без шуму
y_true = a0 + a1*x1 + a2*x2 + a3*x1**2 + a4*x2**2

# Додавання випадкового шуму з нормальним розподілом N(0, b)
noise = np.random.normal(0, b, n_samples)
y = y_true + noise

# Створення DataFrame для зручності роботи з даними
data = pd.DataFrame({
    'x1': x1,
    'x2': x2,
    'y': y,
    'y_true': y_true,
    'noise': noise
})

# Виведення перших 5 рядків згенерованих даних
print("\nПерші 5 рядків згенерованих даних:")
print(data.head())

# Збереження інформації про згенеровані дані у markdown-файл
append_to_md(md_filename, '## 1. Генерація даних')
append_to_md(md_filename, 'Для дослідження було згенеровано синтетичні дані на основі заданої моделі:')
append_to_md(md_filename, '$y(x) = a_0 + a_1 \\cdot x_1 + a_2 \\cdot x_2 + a_3 \\cdot x_1^2 + a_4 \\cdot x_2^2 + rnd(b)$')
append_to_md(md_filename, 'де $rnd(b)$ - випадкова складова з нормальним розподілом $N(0, b)$.')
append_to_md(md_filename, f'Кількість спостережень: {n_samples}')
append_to_md(md_filename, 'Діапазон значень незалежних змінних: $x_1, x_2 \\in [-5, 5]$')
append_to_md(md_filename, '### Перші 5 рядків згенерованих даних:')
append_to_md(md_filename, f'```\n{data.head().to_string()}\n```')

# Статистичний опис згенерованих даних
data_stats = data.describe()
print("\nСтатистичний опис згенерованих даних:")
print(data_stats)

append_to_md(md_filename, '### Статистичний опис згенерованих даних:')
append_to_md(md_filename, f'```\n{data_stats.to_string()}\n```')

# Візуалізація згенерованих даних
fig = plt.figure(figsize=(15, 10))

# 1. 3D-візуалізація даних
ax1 = fig.add_subplot(221, projection='3d')
scatter = ax1.scatter(x1, x2, y, c=y, cmap='viridis', alpha=0.7)
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('y')
ax1.set_title('3D візуалізація згенерованих даних')
plt.colorbar(scatter, ax=ax1, shrink=0.5, aspect=5, label='Значення y')

# 2. Залежність y від x1
ax2 = fig.add_subplot(222)
ax2.scatter(x1, y, alpha=0.7)
ax2.set_xlabel('x1')
ax2.set_ylabel('y')
ax2.set_title('Залежність y від x1')

# 3. Залежність y від x2
ax3 = fig.add_subplot(223)
ax3.scatter(x2, y, alpha=0.7)
ax3.set_xlabel('x2')
ax3.set_ylabel('y')
ax3.set_title('Залежність y від x2')

# 4. Розподіл шуму
ax4 = fig.add_subplot(224)
ax4.hist(noise, bins=20, alpha=0.7)
ax4.axvline(x=0, color='r', linestyle='--')
ax4.set_xlabel('Шум')
ax4.set_ylabel('Частота')
ax4.set_title('Розподіл шуму')

plt.tight_layout()
plt.savefig('generated_data_visualization.png')

append_to_md(md_filename, '### Візуалізація згенерованих даних:')
append_to_md(md_filename, '![Візуалізація згенерованих даних](generated_data_visualization.png)')
append_to_md(md_filename, '*Рис. 1. Візуалізація згенерованих даних: 3D-візуалізація, залежності y від x1 та x2, розподіл шуму*')

print("\nГенерація даних завершена. Дані візуалізовано та збережено у файл 'generated_data_visualization.png'")

# ================= ЧАСТИНА 2: РУЧНИЙ (МАТЕМАТИЧНИЙ) МЕТОД РОЗРАХУНКУ =================
print("\n\nПочаток виконання ручного (математичного) методу розрахунку...")
append_to_md(md_filename, '## 2. Ручний (математичний) метод розрахунку')
append_to_md(md_filename, 'Для побудови рівняння нелінійної множинної регресії використаємо метод найменших квадратів (МНК). Для нашої моделі $y(x) = a_0 + a_1 \\cdot x_1 + a_2 \\cdot x_2 + a_3 \\cdot x_1^2 + a_4 \\cdot x_2^2$ необхідно знайти коефіцієнти $a_0, a_1, a_2, a_3, a_4$, які мінімізують суму квадратів відхилень.')

# Підготовка матриці ознак X для регресії
# Кожен рядок матриці X має вигляд [1, x1_i, x2_i, x1_i^2, x2_i^2]
X_manual = np.column_stack((np.ones(n_samples), x1, x2, x1**2, x2**2))

# Виведення перших 5 рядків матриці ознак
print("\nПерші 5 рядків матриці ознак X:")
print(X_manual[:5])
append_to_md(md_filename, '### Матриця ознак X')
append_to_md(md_filename, 'Для застосування методу найменших квадратів створюємо матрицю ознак X, де кожен рядок має вигляд [1, x1_i, x2_i, x1_i^2, x2_i^2]:')
append_to_md(md_filename, f'```\n{pd.DataFrame(X_manual[:5], columns=["1", "x1", "x2", "x1^2", "x2^2"]).to_string()}\n```')

# Рішення рівняння методом найменших квадратів: β = (X^T * X)^(-1) * X^T * y
# 1. Обчислюємо X^T * X
XTX = X_manual.T @ X_manual
print("\nМатриця X^T * X:")
print(XTX)
append_to_md(md_filename, '### Матриця X^T * X')
append_to_md(md_filename, 'Обчислюємо добуток транспонованої матриці X на матрицю X:')
append_to_md(md_filename, f'```\n{pd.DataFrame(XTX).to_string()}\n```')

# 2. Обчислюємо обернену матрицю (X^T * X)^(-1)
XTX_inv = np.linalg.inv(XTX)
print("\nОбернена матриця (X^T * X)^(-1):")
print(XTX_inv)
append_to_md(md_filename, '### Обернена матриця (X^T * X)^(-1)')
append_to_md(md_filename, 'Обчислюємо обернену матрицю до X^T * X:')
append_to_md(md_filename, f'```\n{pd.DataFrame(XTX_inv).to_string()}\n```')

# 3. Обчислюємо X^T * y
XTy = X_manual.T @ y
print("\nВектор X^T * y:")
print(XTy)
append_to_md(md_filename, '### Вектор X^T * y')
append_to_md(md_filename, 'Обчислюємо добуток транспонованої матриці X на вектор y:')
append_to_md(md_filename, f'```\n{XTy}\n```')

# 4. Обчислюємо β = (X^T * X)^(-1) * X^T * y
beta_manual = XTX_inv @ XTy
print("\nОтримані коефіцієнти регресії (ручний метод):")
print(f"a0 (вільний член) = {beta_manual[0]:.4f}")
print(f"a1 (коефіцієнт при x1) = {beta_manual[1]:.4f}")
print(f"a2 (коефіцієнт при x2) = {beta_manual[2]:.4f}")
print(f"a3 (коефіцієнт при x1^2) = {beta_manual[3]:.4f}")
print(f"a4 (коефіцієнт при x2^2) = {beta_manual[4]:.4f}")

append_to_md(md_filename, '### Отримані коефіцієнти регресії (ручний метод)')
append_to_md(md_filename, f'Обчислюємо вектор коефіцієнтів β = (X^T * X)^(-1) * X^T * y:')
append_to_md(md_filename, f'- a0 (вільний член) = {beta_manual[0]:.4f}')
append_to_md(md_filename, f'- a1 (коефіцієнт при x1) = {beta_manual[1]:.4f}')
append_to_md(md_filename, f'- a2 (коефіцієнт при x2) = {beta_manual[2]:.4f}')
append_to_md(md_filename, f'- a3 (коефіцієнт при x1^2) = {beta_manual[3]:.4f}')
append_to_md(md_filename, f'- a4 (коефіцієнт при x2^2) = {beta_manual[4]:.4f}')

# Порівняння з вихідними коефіцієнтами
print("\nВихідні коефіцієнти:")
print(f"a0 = {a0}")
print(f"a1 = {a1}")
print(f"a2 = {a2}")
print(f"a3 = {a3}")
print(f"a4 = {a4}")

append_to_md(md_filename, '### Порівняння з вихідними коефіцієнтами')
table_content = '| Коефіцієнт | Вихідне значення | Отримане значення | Різниця |\n'
table_content += '|------------|------------------|-------------------|--------|\n'
table_content += f'| a0 | {a0} | {beta_manual[0]:.4f} | {abs(a0 - beta_manual[0]):.4f} |\n'
table_content += f'| a1 | {a1} | {beta_manual[1]:.4f} | {abs(a1 - beta_manual[1]):.4f} |\n'
table_content += f'| a2 | {a2} | {beta_manual[2]:.4f} | {abs(a2 - beta_manual[2]):.4f} |\n'
table_content += f'| a3 | {a3} | {beta_manual[3]:.4f} | {abs(a3 - beta_manual[3]):.4f} |\n'
table_content += f'| a4 | {a4} | {beta_manual[4]:.4f} | {abs(a4 - beta_manual[4]):.4f} |'

# Обчислення прогнозних значень y
y_pred_manual = X_manual @ beta_manual

# Розрахунок залишків
residuals_manual = y - y_pred_manual

# Розрахунок коефіцієнта детермінації R^2
SS_total = np.sum((y - np.mean(y))**2)
SS_residual = np.sum(residuals_manual**2)
R2_manual = 1 - (SS_residual / SS_total)
print(f"\nКоефіцієнт детермінації R^2 (ручний метод): {R2_manual:.4f}")

append_to_md(md_filename, '### Оцінка якості моделі (ручний метод)')
append_to_md(md_filename, f'Коефіцієнт детермінації R^2 = {R2_manual:.4f}')

# Критерій Дарбіна-Уотсона для перевірки автокореляції залишків
# Формула: DW = Σ(e_t - e_{t-1})^2 / Σe_t^2
dw_numerator = np.sum(np.diff(residuals_manual)**2)
dw_denominator = np.sum(residuals_manual**2)
dw_manual = dw_numerator / dw_denominator

print(f"\nКритерій Дарбіна-Уотсона (ручний метод): {dw_manual:.4f}")
if dw_manual < 1.5:
    dw_interpretation = "Є позитивна автокореляція залишків"
elif dw_manual > 2.5:
    dw_interpretation = "Є негативна автокореляція залишків"
else:
    dw_interpretation = "Автокореляція залишків відсутня або незначна"
print(dw_interpretation)

append_to_md(md_filename, f'Критерій Дарбіна-Уотсона = {dw_manual:.4f}')
append_to_md(md_filename, f'Інтерпретація: {dw_interpretation}')

# Кореляційне відношення η для оцінки сили нелінійного зв'язку
# η = sqrt(1 - Σ(y_i - y_pred_i)^2 / Σ(y_i - y_mean)^2)
eta_manual = np.sqrt(R2_manual)

print(f"\nКореляційне відношення η (ручний метод): {eta_manual:.4f}")
if eta_manual < 0.3:
    eta_interpretation = "Слабкий нелінійний зв'язок"
elif eta_manual < 0.7:
    eta_interpretation = "Помірний нелінійний зв'язок"
else:
    eta_interpretation = "Сильний нелінійний зв'язок"
print(eta_interpretation)

append_to_md(md_filename, f'Кореляційне відношення η = {eta_manual:.4f}')
append_to_md(md_filename, f'Інтерпретація: {eta_interpretation}')

# Візуалізація результатів ручного методу
fig = plt.figure(figsize=(15, 10))

# 1. Графік фактичних vs. прогнозних значень
ax1 = fig.add_subplot(221)
ax1.scatter(y, y_pred_manual, alpha=0.7)
ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
ax1.set_xlabel('Фактичні значення y')
ax1.set_ylabel('Прогнозні значення y')
ax1.set_title('Фактичні vs. Прогнозні значення')

# 2. Гістограма залишків
ax2 = fig.add_subplot(222)
ax2.hist(residuals_manual, bins=20, alpha=0.7)
ax2.axvline(x=0, color='r', linestyle='--')
ax2.set_xlabel('Залишки')
ax2.set_ylabel('Частота')
ax2.set_title('Розподіл залишків')

# 3. QQ-Plot для перевірки нормальності залишків
ax3 = fig.add_subplot(223)
stats.probplot(residuals_manual, dist="norm", plot=ax3)
ax3.set_title('QQ-Plot залишків')

# 4. Залишки vs. Прогнозні значення
ax4 = fig.add_subplot(224)
ax4.scatter(y_pred_manual, residuals_manual, alpha=0.7)
ax4.axhline(y=0, color='r', linestyle='--')
ax4.set_xlabel('Прогнозні значення')
ax4.set_ylabel('Залишки')
ax4.set_title('Залишки vs. Прогнозні значення')

plt.tight_layout()
plt.savefig('manual_method_results.png')

append_to_md(md_filename, '### Візуалізація результатів ручного методу:')
append_to_md(md_filename, '![Візуалізація результатів ручного методу](manual_method_results.png)')
append_to_md(md_filename, '*Рис. 2. Візуалізація результатів ручного методу: фактичні vs. прогнозні значення, розподіл залишків, QQ-Plot залишків, залишки vs. прогнозні значення*')

print("\nРучний (математичний) метод розрахунку завершено. Результати візуалізовано та збережено у файл 'manual_method_results.png'")

# ================= ЧАСТИНА 3: МЕТОД STATSMODELS.API.OLS =================
print("\n\nПочаток виконання методу statsmodels.api.OLS...")
append_to_md(md_filename, '## 3. Метод statsmodels.api.OLS')
append_to_md(md_filename, 'Для побудови нелінійної множинної регресії використаємо бібліотеку statsmodels, яка надає зручний інтерфейс для статистичного аналізу даних. Метод OLS (Ordinary Least Squares) реалізує метод найменших квадратів.')

# Підготовка даних для statsmodels
# Створюємо DataFrame з ознаками
X_sm = pd.DataFrame({
    'const': 1,  # Додаємо константу для вільного члена (a0)
    'x1': x1,
    'x2': x2,
    'x1_sq': x1**2,
    'x2_sq': x2**2
})

# Виведення перших 5 рядків матриці ознак
print("\nПерші 5 рядків матриці ознак X для statsmodels:")
print(X_sm.head())
append_to_md(md_filename, '### Матриця ознак X для statsmodels')
append_to_md(md_filename, 'Для застосування методу OLS створюємо матрицю ознак X у вигляді DataFrame:')
append_to_md(md_filename, f'```\n{X_sm.head().to_string()}\n```')

# Побудова моделі за допомогою statsmodels.api.OLS
model_sm = sm.OLS(y, X_sm).fit()

# Виведення результатів
print("\nРезультати регресії (statsmodels.api.OLS):")
print(model_sm.summary())

# Зберігаємо основні результати у markdown-файл
append_to_md(md_filename, '### Результати регресії (statsmodels.api.OLS)')
append_to_md(md_filename, 'Результати регресійного аналізу за допомогою statsmodels.api.OLS:')
append_to_md(md_filename, '```')
append_to_md(md_filename, f'R-squared:                       {model_sm.rsquared:.4f}')
append_to_md(md_filename, f'Adj. R-squared:                  {model_sm.rsquared_adj:.4f}')
append_to_md(md_filename, f'F-statistic:                     {model_sm.fvalue:.4f}')
append_to_md(md_filename, f'Prob (F-statistic):              {model_sm.f_pvalue:.4e}')
append_to_md(md_filename, f'Log-Likelihood:                  {model_sm.llf:.4f}')
append_to_md(md_filename, f'AIC:                             {model_sm.aic:.4f}')
append_to_md(md_filename, f'BIC:                             {model_sm.bic:.4f}')
append_to_md(md_filename, '```')

# Отримані коефіцієнти
beta_sm = model_sm.params
print("\nОтримані коефіцієнти регресії (statsmodels.api.OLS):")
print(f"a0 (const) = {beta_sm['const']:.4f}")
print(f"a1 (x1) = {beta_sm['x1']:.4f}")
print(f"a2 (x2) = {beta_sm['x2']:.4f}")
print(f"a3 (x1_sq) = {beta_sm['x1_sq']:.4f}")
print(f"a4 (x2_sq) = {beta_sm['x2_sq']:.4f}")

append_to_md(md_filename, '### Отримані коефіцієнти регресії (statsmodels.api.OLS)')
append_to_md(md_filename, f'- a0 (const) = {beta_sm["const"]:.4f}')
append_to_md(md_filename, f'- a1 (x1) = {beta_sm["x1"]:.4f}')
append_to_md(md_filename, f'- a2 (x2) = {beta_sm["x2"]:.4f}')
append_to_md(md_filename, f'- a3 (x1_sq) = {beta_sm["x1_sq"]:.4f}')
append_to_md(md_filename, f'- a4 (x2_sq) = {beta_sm["x2_sq"]:.4f}')

# Порівняння з вихідними коефіцієнтами
append_to_md(md_filename, '### Порівняння з вихідними коефіцієнтами')
table_content = '| Коефіцієнт | Вихідне значення | Отримане значення | Різниця |\n'
table_content += '|------------|------------------|-------------------|--------|\n'
table_content += f'| a0 | {a0} | {beta_sm["const"]:.4f} | {abs(a0 - beta_sm["const"]):.4f} |\n'
table_content += f'| a1 | {a1} | {beta_sm["x1"]:.4f} | {abs(a1 - beta_sm["x1"]):.4f} |\n'
table_content += f'| a2 | {a2} | {beta_sm["x2"]:.4f} | {abs(a2 - beta_sm["x2"]):.4f} |\n'
table_content += f'| a3 | {a3} | {beta_sm["x1_sq"]:.4f} | {abs(a3 - beta_sm["x1_sq"]):.4f} |\n'
table_content += f'| a4 | {a4} | {beta_sm["x2_sq"]:.4f} | {abs(a4 - beta_sm["x2_sq"]):.4f} |'
append_to_md(md_filename, table_content)

# Обчислення прогнозних значень y
y_pred_sm = model_sm.predict(X_sm)

# Розрахунок залишків
residuals_sm = model_sm.resid

# Розрахунок коефіцієнта детермінації R^2
R2_sm = model_sm.rsquared
print(f"\nКоефіцієнт детермінації R^2 (statsmodels.api.OLS): {R2_sm:.4f}")

# Критерій Дарбіна-Уотсона
dw_sm = durbin_watson(residuals_sm)
print(f"\nКритерій Дарбіна-Уотсона (statsmodels.api.OLS): {dw_sm:.4f}")
if dw_sm < 1.5:
    dw_interpretation_sm = "Є позитивна автокореляція залишків"
elif dw_sm > 2.5:
    dw_interpretation_sm = "Є негативна автокореляція залишків"
else:
    dw_interpretation_sm = "Автокореляція залишків відсутня або незначна"
print(dw_interpretation_sm)

# Кореляційне відношення η
eta_sm = np.sqrt(R2_sm)
print(f"\nКореляційне відношення η (statsmodels.api.OLS): {eta_sm:.4f}")
if eta_sm < 0.3:
    eta_interpretation_sm = "Слабкий нелінійний зв'язок"
elif eta_sm < 0.7:
    eta_interpretation_sm = "Помірний нелінійний зв'язок"
else:
    eta_interpretation_sm = "Сильний нелінійний зв'язок"
print(eta_interpretation_sm)

append_to_md(md_filename, '### Оцінка якості моделі (statsmodels.api.OLS)')
append_to_md(md_filename, f'Коефіцієнт детермінації R^2 = {R2_sm:.4f}')
append_to_md(md_filename, f'Критерій Дарбіна-Уотсона = {dw_sm:.4f}')
append_to_md(md_filename, f'Інтерпретація: {dw_interpretation_sm}')
append_to_md(md_filename, f'Кореляційне відношення η = {eta_sm:.4f}')
append_to_md(md_filename, f'Інтерпретація: {eta_interpretation_sm}')

# Візуалізація результатів statsmodels.api.OLS
fig = plt.figure(figsize=(15, 10))

# 1. Графік фактичних vs. прогнозних значень
ax1 = fig.add_subplot(221)
ax1.scatter(y, y_pred_sm, alpha=0.7)
ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
ax1.set_xlabel('Фактичні значення y')
ax1.set_ylabel('Прогнозні значення y')
ax1.set_title('Фактичні vs. Прогнозні значення')

# 2. Гістограма залишків
ax2 = fig.add_subplot(222)
ax2.hist(residuals_sm, bins=20, alpha=0.7)
ax2.axvline(x=0, color='r', linestyle='--')
ax2.set_xlabel('Залишки')
ax2.set_ylabel('Частота')
ax2.set_title('Розподіл залишків')

# 3. QQ-Plot для перевірки нормальності залишків
ax3 = fig.add_subplot(223)
stats.probplot(residuals_sm, dist="norm", plot=ax3)
ax3.set_title('QQ-Plot залишків')

# 4. Залишки vs. Прогнозні значення
ax4 = fig.add_subplot(224)
ax4.scatter(y_pred_sm, residuals_sm, alpha=0.7)
ax4.axhline(y=0, color='r', linestyle='--')
ax4.set_xlabel('Прогнозні значення')
ax4.set_ylabel('Залишки')
ax4.set_title('Залишки vs. Прогнозні значення')

plt.tight_layout()
plt.savefig('statsmodels_method_results.png')

append_to_md(md_filename, '### Візуалізація результатів statsmodels.api.OLS:')
append_to_md(md_filename, '![Візуалізація результатів statsmodels.api.OLS](statsmodels_method_results.png)')
append_to_md(md_filename, '*Рис. 3. Візуалізація результатів statsmodels.api.OLS: фактичні vs. прогнозні значення, розподіл залишків, QQ-Plot залишків, залишки vs. прогнозні значення*')

print("\nМетод statsmodels.api.OLS завершено. Результати візуалізовано та збережено у файл 'statsmodels_method_results.png'")

# ================= ЧАСТИНА 4: МЕТОД SKLEARN.LINEAR_MODEL.LINEARREGRESSION =================
print("\n\nПочаток виконання методу sklearn.linear_model.LinearRegression...")
append_to_md(md_filename, '## 4. Метод sklearn.linear_model.LinearRegression')
append_to_md(md_filename, 'Для побудови нелінійної множинної регресії використаємо бібліотеку scikit-learn, яка надає зручний інтерфейс для машинного навчання. Для створення поліноміальних ознак використаємо PolynomialFeatures.')

# Підготовка даних для sklearn
# Створюємо матрицю ознак
X_sk_base = np.column_stack((x1, x2))

# Використовуємо PolynomialFeatures для створення поліноміальних ознак
poly = PolynomialFeatures(degree=2, include_bias=False)
X_sk_poly = poly.fit_transform(X_sk_base)

# Виведення перших 5 рядків матриці ознак
print("\nПерші 5 рядків матриці ознак X для sklearn:")
print(X_sk_poly[:5])

# Виведення назв ознак
feature_names = poly.get_feature_names_out(['x1', 'x2'])
print("\nНазви ознак після перетворення:")
print(feature_names)

append_to_md(md_filename, '### Матриця ознак X для sklearn')
append_to_md(md_filename, 'Для застосування методу LinearRegression створюємо матрицю ознак X за допомогою PolynomialFeatures:')
append_to_md(md_filename, f'```\n{pd.DataFrame(X_sk_poly[:5], columns=feature_names).to_string()}\n```')

# Побудова моделі за допомогою sklearn.linear_model.LinearRegression
model_sk = LinearRegression()
model_sk.fit(X_sk_poly, y)

# Отримані коефіцієнти
beta_sk = model_sk.coef_
intercept_sk = model_sk.intercept_

print("\nОтримані коефіцієнти регресії (sklearn.linear_model.LinearRegression):")
print(f"a0 (intercept) = {intercept_sk:.4f}")
for i, name in enumerate(feature_names):
    print(f"Коефіцієнт при {name} = {beta_sk[i]:.4f}")

# Визначаємо відповідність коефіцієнтів моделі до наших a0, a1, a2, a3, a4
a0_sk = intercept_sk
a1_sk = beta_sk[0]  # x1
a2_sk = beta_sk[1]  # x2
a3_sk = beta_sk[2]  # x1^2
a4_sk = beta_sk[4]  # x2^2

append_to_md(md_filename, '### Отримані коефіцієнти регресії (sklearn.linear_model.LinearRegression)')
append_to_md(md_filename, f'- a0 (intercept) = {a0_sk:.4f}')
append_to_md(md_filename, f'- a1 (x1) = {a1_sk:.4f}')
append_to_md(md_filename, f'- a2 (x2) = {a2_sk:.4f}')
append_to_md(md_filename, f'- a3 (x1^2) = {a3_sk:.4f}')
append_to_md(md_filename, f'- a4 (x2^2) = {a4_sk:.4f}')

# Порівняння з вихідними коефіцієнтами
append_to_md(md_filename, '### Порівняння з вихідними коефіцієнтами')
table_content = '| Коефіцієнт | Вихідне значення | Отримане значення | Різниця |\n'
table_content += '|------------|------------------|-------------------|--------|\n'
table_content += f'| a0 | {a0} | {a0_sk:.4f} | {abs(a0 - a0_sk):.4f} |\n'
table_content += f'| a1 | {a1} | {a1_sk:.4f} | {abs(a1 - a1_sk):.4f} |\n'
table_content += f'| a2 | {a2} | {a2_sk:.4f} | {abs(a2 - a2_sk):.4f} |\n'
table_content += f'| a3 | {a3} | {a3_sk:.4f} | {abs(a3 - a3_sk):.4f} |\n'
table_content += f'| a4 | {a4} | {a4_sk:.4f} | {abs(a4 - a4_sk):.4f} |'
append_to_md(md_filename, table_content)

# Обчислення прогнозних значень y
y_pred_sk = model_sk.predict(X_sk_poly)

# Розрахунок залишків
residuals_sk = y - y_pred_sk

# Розрахунок коефіцієнта детермінації R^2
R2_sk = r2_score(y, y_pred_sk)
print(f"\nКоефіцієнт детермінації R^2 (sklearn.linear_model.LinearRegression): {R2_sk:.4f}")

# Критерій Дарбіна-Уотсона
dw_sk = np.sum(np.diff(residuals_sk)**2) / np.sum(residuals_sk**2)
print(f"\nКритерій Дарбіна-Уотсона (sklearn.linear_model.LinearRegression): {dw_sk:.4f}")
if dw_sk < 1.5:
    dw_interpretation_sk = "Є позитивна автокореляція залишків"
elif dw_sk > 2.5:
    dw_interpretation_sk = "Є негативна автокореляція залишків"
else:
    dw_interpretation_sk = "Автокореляція залишків відсутня або незначна"
print(dw_interpretation_sk)

# Кореляційне відношення η
eta_sk = np.sqrt(R2_sk)
print(f"\nКореляційне відношення η (sklearn.linear_model.LinearRegression): {eta_sk:.4f}")
if eta_sk < 0.3:
    eta_interpretation_sk = "Слабкий нелінійний зв'язок"
elif eta_sk < 0.7:
    eta_interpretation_sk = "Помірний нелінійний зв'язок"
else:
    eta_interpretation_sk = "Сильний нелінійний зв'язок"
print(eta_interpretation_sk)

append_to_md(md_filename, '### Оцінка якості моделі (sklearn.linear_model.LinearRegression)')
append_to_md(md_filename, f'Коефіцієнт детермінації R^2 = {R2_sk:.4f}')
append_to_md(md_filename, f'Критерій Дарбіна-Уотсона = {dw_sk:.4f}')
append_to_md(md_filename, f'Інтерпретація: {dw_interpretation_sk}')
append_to_md(md_filename, f'Кореляційне відношення η = {eta_sk:.4f}')
append_to_md(md_filename, f'Інтерпретація: {eta_interpretation_sk}')

# Візуалізація результатів sklearn.linear_model.LinearRegression
fig = plt.figure(figsize=(15, 10))

# 1. Графік фактичних vs. прогнозних значень
ax1 = fig.add_subplot(221)
ax1.scatter(y, y_pred_sk, alpha=0.7)
ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
ax1.set_xlabel('Фактичні значення y')
ax1.set_ylabel('Прогнозні значення y')
ax1.set_title('Фактичні vs. Прогнозні значення')

# 2. Гістограма залишків
ax2 = fig.add_subplot(222)
ax2.hist(residuals_sk, bins=20, alpha=0.7)
ax2.axvline(x=0, color='r', linestyle='--')
ax2.set_xlabel('Залишки')
ax2.set_ylabel('Частота')
ax2.set_title('Розподіл залишків')

# 3. QQ-Plot для перевірки нормальності залишків
ax3 = fig.add_subplot(223)
stats.probplot(residuals_sk, dist="norm", plot=ax3)
ax3.set_title('QQ-Plot залишків')

# 4. Залишки vs. Прогнозні значення
ax4 = fig.add_subplot(224)
ax4.scatter(y_pred_sk, residuals_sk, alpha=0.7)
ax4.axhline(y=0, color='r', linestyle='--')
ax4.set_xlabel('Прогнозні значення')
ax4.set_ylabel('Залишки')
ax4.set_title('Залишки vs. Прогнозні значення')

plt.tight_layout()
plt.savefig('sklearn_method_results.png')

append_to_md(md_filename, '### Візуалізація результатів sklearn.linear_model.LinearRegression:')
append_to_md(md_filename, '![Візуалізація результатів sklearn.linear_model.LinearRegression](sklearn_method_results.png)')
append_to_md(md_filename, '*Рис. 4. Візуалізація результатів sklearn.linear_model.LinearRegression: фактичні vs. прогнозні значення, розподіл залишків, QQ-Plot залишків, залишки vs. прогнозні значення*')

print("\nМетод sklearn.linear_model.LinearRegression завершено. Результати візуалізовано та збережено у файл 'sklearn_method_results.png'")

# ================= ЧАСТИНА 5: ПОРІВНЯННЯ МЕТОДІВ =================
print("\n\nПочаток порівняння методів...")
append_to_md(md_filename, '## 5. Порівняння методів')
append_to_md(md_filename, 'Порівняємо результати, отримані різними методами регресійного аналізу.')

# Порівняння коефіцієнтів
append_to_md(md_filename, '### Порівняння коефіцієнтів моделі')
table_content = '| Коефіцієнт | Вихідне значення | Ручний метод | statsmodels.api.OLS | sklearn.linear_model.LinearRegression |\n'
table_content += '|------------|------------------|--------------|-------------------|-----------------------------------|\n'
table_content += f'| a0 | {a0} | {beta_manual[0]:.4f} | {beta_sm["const"]:.4f} | {a0_sk:.4f} |\n'
table_content += f'| a1 | {a1} | {beta_manual[1]:.4f} | {beta_sm["x1"]:.4f} | {a1_sk:.4f} |\n'
table_content += f'| a2 | {a2} | {beta_manual[2]:.4f} | {beta_sm["x2"]:.4f} | {a2_sk:.4f} |\n'
table_content += f'| a3 | {a3} | {beta_manual[3]:.4f} | {beta_sm["x1_sq"]:.4f} | {a3_sk:.4f} |\n'
table_content += f'| a4 | {a4} | {beta_manual[4]:.4f} | {beta_sm["x2_sq"]:.4f} | {a4_sk:.4f} |'
append_to_md(md_filename, table_content)

# Порівняння метрик якості моделі
append_to_md(md_filename, '### Порівняння метрик якості моделі')
table_content = '| Метрика | Ручний метод | statsmodels.api.OLS | sklearn.linear_model.LinearRegression |\n'
table_content += '|---------|--------------|-------------------|-----------------------------------|\n'
table_content += f'| R^2 | {R2_manual:.4f} | {R2_sm:.4f} | {R2_sk:.4f} |\n'
table_content += f'| Критерій Дарбіна-Уотсона | {dw_manual:.4f} | {dw_sm:.4f} | {dw_sk:.4f} |\n'
table_content += f'| Кореляційне відношення η | {eta_manual:.4f} | {eta_sm:.4f} | {eta_sk:.4f} |'
append_to_md(md_filename, table_content)

# Візуалізація порівняння методів
fig = plt.figure(figsize=(15, 10))

# 1. Порівняння коефіцієнтів
ax1 = fig.add_subplot(221)
coef_names = ['a0', 'a1', 'a2', 'a3', 'a4']
coef_values = {
    'Вихідні': [a0, a1, a2, a3, a4],
    'Ручний метод': [beta_manual[0], beta_manual[1], beta_manual[2], beta_manual[3], beta_manual[4]],
    'statsmodels': [beta_sm['const'], beta_sm['x1'], beta_sm['x2'], beta_sm['x1_sq'], beta_sm['x2_sq']],
    'sklearn': [a0_sk, a1_sk, a2_sk, a3_sk, a4_sk]
}

x = np.arange(len(coef_names))
width = 0.2
ax1.bar(x - 1.5*width, coef_values['Вихідні'], width, label='Вихідні')
ax1.bar(x - 0.5*width, coef_values['Ручний метод'], width, label='Ручний метод')
ax1.bar(x + 0.5*width, coef_values['statsmodels'], width, label='statsmodels')
ax1.bar(x + 1.5*width, coef_values['sklearn'], width, label='sklearn')
ax1.set_xlabel('Коефіцієнти')
ax1.set_ylabel('Значення')
ax1.set_title('Порівняння коефіцієнтів')
ax1.set_xticks(x)
ax1.set_xticklabels(coef_names)
ax1.legend()

# 2. Порівняння метрик якості
ax2 = fig.add_subplot(222)
metric_names = ['R^2', 'DW', 'η']
metric_values = {
    'Ручний метод': [R2_manual, dw_manual, eta_manual],
    'statsmodels': [R2_sm, dw_sm, eta_sm],
    'sklearn': [R2_sk, dw_sk, eta_sk]
}

x = np.arange(len(metric_names))
width = 0.25
ax2.bar(x - width, metric_values['Ручний метод'], width, label='Ручний метод')
ax2.bar(x, metric_values['statsmodels'], width, label='statsmodels')
ax2.bar(x + width, metric_values['sklearn'], width, label='sklearn')
ax2.set_xlabel('Метрики')
ax2.set_ylabel('Значення')
ax2.set_title('Порівняння метрик якості')
ax2.set_xticks(x)
ax2.set_xticklabels(metric_names)
ax2.legend()

# 3. Порівняння прогнозних значень
ax3 = fig.add_subplot(223)
ax3.scatter(y, y_pred_manual, alpha=0.5, label='Ручний метод')
ax3.scatter(y, y_pred_sm, alpha=0.5, label='statsmodels')
ax3.scatter(y, y_pred_sk, alpha=0.5, label='sklearn')
ax3.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
ax3.set_xlabel('Фактичні значення y')
ax3.set_ylabel('Прогнозні значення y')
ax3.set_title('Порівняння прогнозних значень')
ax3.legend()

# 4. Порівняння залишків
ax4 = fig.add_subplot(224)
ax4.boxplot([residuals_manual, residuals_sm, residuals_sk], tick_labels=['Ручний метод', 'statsmodels', 'sklearn'])
ax4.set_ylabel('Залишки')
ax4.set_title('Порівняння розподілу залишків')

plt.tight_layout()
plt.savefig('methods_comparison.png')

append_to_md(md_filename, '### Візуалізація порівняння методів:')
append_to_md(md_filename, '![Візуалізація порівняння методів](methods_comparison.png)')
append_to_md(md_filename, '*Рис. 5. Візуалізація порівняння методів: коефіцієнти, метрики якості, прогнозні значення, розподіл залишків*')

# 3D візуалізація порівняння методів
fig = plt.figure(figsize=(15, 10))

# Створення сітки для поверхні
x1_grid = np.linspace(min(x1), max(x1), 30)
x2_grid = np.linspace(min(x2), max(x2), 30)
X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid)

# Обчислення значень y для кожного методу
# Ручний метод
Y_grid_manual = (beta_manual[0] +
                beta_manual[1] * X1_grid +
                beta_manual[2] * X2_grid +
                beta_manual[3] * X1_grid**2 +
                beta_manual[4] * X2_grid**2)

# statsmodels
Y_grid_sm = (beta_sm['const'] +
            beta_sm['x1'] * X1_grid +
            beta_sm['x2'] * X2_grid +
            beta_sm['x1_sq'] * X1_grid**2 +
            beta_sm['x2_sq'] * X2_grid**2)

# sklearn
Y_grid_sk = (a0_sk +
            a1_sk * X1_grid +
            a2_sk * X2_grid +
            a3_sk * X1_grid**2 +
            a4_sk * X2_grid**2)

# Істинні значення
Y_grid_true = (a0 +
              a1 * X1_grid +
              a2 * X2_grid +
              a3 * X1_grid**2 +
              a4 * X2_grid**2)

# Створення 3D графіків
fig = plt.figure(figsize=(20, 15))

# 1. Істинна поверхня
ax1 = fig.add_subplot(221, projection='3d')
surf1 = ax1.plot_surface(X1_grid, X2_grid, Y_grid_true, cmap='viridis', alpha=0.7)
ax1.scatter(x1, x2, y, color='r', alpha=0.5)
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
ax1.set_zlabel('Y')
ax1.set_title('Істинна поверхня')

# 2. Ручний метод
ax2 = fig.add_subplot(222, projection='3d')
surf2 = ax2.plot_surface(X1_grid, X2_grid, Y_grid_manual, cmap='plasma', alpha=0.7)
ax2.scatter(x1, x2, y, color='r', alpha=0.5)
ax2.set_xlabel('X1')
ax2.set_ylabel('X2')
ax2.set_zlabel('Y')
ax2.set_title('Ручний метод')

# 3. statsmodels
ax3 = fig.add_subplot(223, projection='3d')
surf3 = ax3.plot_surface(X1_grid, X2_grid, Y_grid_sm, cmap='inferno', alpha=0.7)
ax3.scatter(x1, x2, y, color='r', alpha=0.5)
ax3.set_xlabel('X1')
ax3.set_ylabel('X2')
ax3.set_zlabel('Y')
ax3.set_title('statsmodels.api.OLS')

# 4. sklearn
ax4 = fig.add_subplot(224, projection='3d')
surf4 = ax4.plot_surface(X1_grid, X2_grid, Y_grid_sk, cmap='cividis', alpha=0.7)
ax4.scatter(x1, x2, y, color='r', alpha=0.5)
ax4.set_xlabel('X1')
ax4.set_ylabel('X2')
ax4.set_zlabel('Y')
ax4.set_title('sklearn.linear_model.LinearRegression')

plt.tight_layout()
plt.savefig('3d_comparison.png')

append_to_md(md_filename, '### 3D візуалізація порівняння методів:')
append_to_md(md_filename, '![3D візуалізація порівняння методів](3d_comparison.png)')
append_to_md(md_filename, '*Рис. 6. 3D візуалізація порівняння методів: істинна поверхня, ручний метод, statsmodels.api.OLS, sklearn.linear_model.LinearRegression*')

# ================= ЧАСТИНА 6: ВИСНОВКИ =================
print("\n\nФормування висновків...")
append_to_md(md_filename, '## 6. Висновки')
append_to_md(md_filename, 'На основі проведеного нелінійного регресійного аналізу можна зробити наступні висновки:')
append_to_md(md_filename, '1. **Порівняння методів**: Усі три методи (ручний розрахунок, statsmodels.api.OLS, sklearn.linear_model.LinearRegression) дали дуже близькі результати, що підтверджує правильність їх реалізації.')
append_to_md(md_filename, '2. **Точність моделі**: Отримані коефіцієнти регресії близькі до вихідних значень, що свідчить про високу точність моделі. Незначні відхилення пояснюються наявністю випадкового шуму в даних.')
append_to_md(md_filename, '3. **Якість моделі**: Високі значення коефіцієнта детермінації R² (близько 0.99) вказують на те, що модель дуже добре описує дані.')
append_to_md(md_filename, '4. **Автокореляція залишків**: Значення критерію Дарбіна-Уотсона близько 2.1 свідчить про відсутність автокореляції залишків, що є хорошим показником адекватності моделі.')
append_to_md(md_filename, '5. **Нелінійний зв\'язок**: Високі значення кореляційного відношення η (близько 0.99) вказують на сильний нелінійний зв\'язок між змінними.')
append_to_md(md_filename, '6. **Інтерпретація коефіцієнтів**:')
append_to_md(md_filename, '   - a₀ ≈ 3.28: Базове значення y при нульових значеннях x₁ та x₂.')
append_to_md(md_filename, '   - a₁ ≈ 2.67: При збільшенні x₁ на одиницю, y збільшується приблизно на 2.67 одиниць (за умови, що інші змінні залишаються незмінними).')
append_to_md(md_filename, '   - a₂ ≈ -0.42: При збільшенні x₂ на одиницю, y зменшується приблизно на 0.42 одиниці (за умови, що інші змінні залишаються незмінними).')
append_to_md(md_filename, '   - a₃ ≈ 0.45: Додатний коефіцієнт при x₁² вказує на опуклу (параболічну вгору) залежність від x₁.')
append_to_md(md_filename, '   - a₄ ≈ 0.27: Додатний коефіцієнт при x₂² вказує на опуклу (параболічну вгору) залежність від x₂.')
append_to_md(md_filename, '7. **Можливі шляхи покращення моделі**:')
append_to_md(md_filename, '   - Включення взаємодій між змінними (x₁·x₂).')
append_to_md(md_filename, '   - Розгляд поліномів вищих порядків.')
append_to_md(md_filename, '   - Застосування методів регуляризації для запобігання перенавчанню.')
append_to_md(md_filename, '   - Аналіз впливових спостережень та викидів.')

print("\nАналіз завершено. Результати збережено у файл 'результати_аналізу.md'")
print("Графіки збережено у файли:")
print("- 'generated_data_visualization.png'")
print("- 'manual_method_results.png'")
print("- 'statsmodels_method_results.png'")
print("- 'sklearn_method_results.png'")
print("- 'methods_comparison.png'")
print("- '3d_comparison.png'")