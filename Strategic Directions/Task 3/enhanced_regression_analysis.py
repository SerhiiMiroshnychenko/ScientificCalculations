import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns

# Налаштування для відтворюваності результатів
np.random.seed(42)

# Налаштування для відображення графіків з українськими символами
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set(style="whitegrid")

# Генерація даних
n = 150
i = np.arange(n + 1)
X1 = -0.7 + i * 0.01
X2 = 0.5 + i * 0.01
Y = 3.4 + 2.7 * X1 - 0.5 * X2 + np.random.uniform(-1.1, 1.1, size=n + 1)

# Створення DataFrame для зручної роботи з даними
data = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})
print("Перші 5 рядків згенерованих даних:")
print(data.head())

# 1. Побудова множинної регресії
X = data[['X1', 'X2']]
y = data['Y']
X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const).fit()
print("\nРезультати регресійного аналізу:")
print(model.summary())

# Виведення рівняння регресії
b0, b1, b2 = model.params
print(f"\nРівняння множинної регресії: Y = {b0:.4f} + {b1:.4f}*X1 + {b2:.4f}*X2")

# 2. Парні коефіцієнти кореляції
corr_matrix = data.corr()
print("\nМатриця парних коефіцієнтів кореляції Пірсона:")
print(corr_matrix)


# Оцінка значущості парних коефіцієнтів
def corr_significance(r, n, alpha=0.05):
    # Попередня обробка для уникнення помилок
    r_squared = min(r ** 2, 0.9999)  # обмежуємо значення r^2 щоб уникнути ділення на 0

    t_stat = r * np.sqrt(n - 2) / np.sqrt(1 - r_squared)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
    t_crit = stats.t.ppf(1 - alpha / 2, n - 2)
    return {"t_stat": t_stat, "p_value": p_value, "t_crit": t_crit, "significant": abs(t_stat) > t_crit}


print("\nОцінка значущості парних коефіцієнтів кореляції:")
for pair in [('X1', 'Y'), ('X2', 'Y'), ('X1', 'X2')]:
    r = corr_matrix.loc[pair[0], pair[1]]
    result = corr_significance(r, n + 1)
    print(
        f"{pair[0]}-{pair[1]}: r = {r:.4f}, t = {result['t_stat']:.4f}, p = {result['p_value']:.4f}, Значущий: {result['significant']}")

# Рангова кореляція Спірмена
spearman_corr = data.corr(method='spearman')
print("\nМатриця рангових коефіцієнтів кореляції Спірмена:")
print(spearman_corr)


# 3. Часткові коефіцієнти кореляції
def partial_corr(df, x, y, control):
    # Регресії для залишків
    x_resid = sm.OLS(df[x], sm.add_constant(df[control])).fit().resid
    y_resid = sm.OLS(df[y], sm.add_constant(df[control])).fit().resid

    # Кореляція між залишками
    partial_r = np.corrcoef(x_resid, y_resid)[0, 1]
    return partial_r


print("\nЧасткові коефіцієнти кореляції:")
r_y_x1_x2 = partial_corr(data, 'Y', 'X1', ['X2'])
r_y_x2_x1 = partial_corr(data, 'Y', 'X2', ['X1'])
print(f"r_y_x1.x2 = {r_y_x1_x2:.4f}")
print(f"r_y_x2.x1 = {r_y_x2_x1:.4f}")


# Оцінка значущості часткових коефіцієнтів
def partial_corr_significance(r, n, k, alpha=0.05):
    t_stat = r * np.sqrt(n - k - 1) / np.sqrt(1 - r ** 2)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - k - 1))
    t_crit = stats.t.ppf(1 - alpha / 2, n - k - 1)
    return {"t_stat": t_stat, "p_value": p_value, "t_crit": t_crit, "significant": abs(t_stat) > t_crit}


print("\nОцінка значущості часткових коефіцієнтів кореляції:")
r_y_x1_x2_sig = partial_corr_significance(r_y_x1_x2, n + 1, 1)
r_y_x2_x1_sig = partial_corr_significance(r_y_x2_x1, n + 1, 1)
print(
    f"r_y_x1.x2: t = {r_y_x1_x2_sig['t_stat']:.4f}, p = {r_y_x1_x2_sig['p_value']:.4f}, Значущий: {r_y_x1_x2_sig['significant']}")
print(
    f"r_y_x2.x1: t = {r_y_x2_x1_sig['t_stat']:.4f}, p = {r_y_x2_x1_sig['p_value']:.4f}, Значущий: {r_y_x2_x1_sig['significant']}")

# 4. Коефіцієнт множинної кореляції і детермінації
R2 = model.rsquared
R = np.sqrt(R2)
print(f"\nКоефіцієнт множинної кореляції R = {R:.4f}")
print(f"Коефіцієнт детермінації R² = {R2:.4f}")

# 5. Скоригований коефіцієнт детермінації
R2_adj = model.rsquared_adj
print(f"Скоригований коефіцієнт детермінації R²_adj = {R2_adj:.4f}")

# 6. Оцінка значущості моделі (F-тест)
f_stat = model.fvalue
f_pvalue = model.f_pvalue
print(f"\nF-статистика: {f_stat:.4f}")
print(f"p-значення (F): {f_pvalue:.8f}")
alpha = 0.05
if f_pvalue < alpha:
    print("Модель є статистично значущою")
else:
    print("Модель не є статистично значущою")

# 7. Оцінка значущості коефіцієнтів регресії (t-тест)
print("\nОцінка значущості коефіцієнтів множинної регресії:")
for var, t_stat, p_val in zip(['const', 'X1', 'X2'], model.tvalues, model.pvalues):
    print(f"{var}: t = {t_stat:.4f}, p = {p_val:.8f}, {'Значущий' if p_val < alpha else 'Не значущий'}")

# 8. Коефіцієнти еластичності
elasticity_X1 = b1 * np.mean(X1) / np.mean(Y)
elasticity_X2 = b2 * np.mean(X2) / np.mean(Y)
print(f"\nКоефіцієнти еластичності:")
print(f"E_X1 = {elasticity_X1:.4f}")
print(f"E_X2 = {elasticity_X2:.4f}")

# 9. β-коефіцієнти
beta_X1 = b1 * np.std(X1) / np.std(Y)
beta_X2 = b2 * np.std(X2) / np.std(Y)
print(f"\nβ-коефіцієнти:")
print(f"β_X1 = {beta_X1:.4f}")
print(f"β_X2 = {beta_X2:.4f}")

# 10. Δ-коефіцієнти
delta_X1 = beta_X1 * corr_matrix.loc['X1', 'Y']
delta_X2 = beta_X2 * corr_matrix.loc['X2', 'Y']
print(f"\nΔ-коефіцієнти:")
print(f"Δ_X1 = {delta_X1:.4f}")
print(f"Δ_X2 = {delta_X2:.4f}")
print(f"Сума Δ-коефіцієнтів: {delta_X1 + delta_X2:.4f}")

# 11. Перевірка наявності мультиколінеарності за допомогою VIF
vif_data = pd.DataFrame()
vif_data["Фактор"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\nПеревірка на мультиколінеарність (VIF):")
print(vif_data)

# 12. Аналіз залишків
y_pred = model.predict(X_with_const)
residuals = Y - y_pred

print("\nСтатистика залишків:")
residual_stats = pd.Series({
    'mean': np.mean(residuals),
    'std': np.std(residuals),
    'min': np.min(residuals),
    'max': np.max(residuals)
})
print(residual_stats)

# Перевірка нормальності залишків
shapiro_test = stats.shapiro(residuals)
print(f"\nПеревірка нормальності залишків (тест Шапіро-Вілка):")
print(f"Статистика W: {shapiro_test[0]:.4f}")
print(f"p-значення: {shapiro_test[1]:.8f}")
print(f"Висновок: залишки {'мають' if shapiro_test[1] > alpha else 'не мають'} нормальний розподіл")

# 13. Розширені візуалізації
# Створюємо фігуру з підграфіками
plt.figure(figsize=(18, 12))

# 1. 3D візуалізація даних та поверхні регресії
ax1 = plt.subplot(2, 3, 1, projection='3d')
ax1.scatter(X1, X2, Y, c='blue', marker='o', alpha=0.6)

# Створюємо сітку для поверхні
x1_grid, x2_grid = np.meshgrid(
    np.linspace(min(X1), max(X1), 20),
    np.linspace(min(X2), max(X2), 20)
)

# Обчислюємо передбачені значення
y_pred_grid = b0 + b1 * x1_grid + b2 * x2_grid

# Будуємо поверхню регресії
ax1.plot_surface(x1_grid, x2_grid, y_pred_grid, alpha=0.3, color='red')
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
ax1.set_zlabel('Y')
ax1.set_title('3D візуалізація даних та поверхні регресії')

# 2. Залежність Y від X1 (2D)
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(X1, Y, alpha=0.6)

# Сортуємо дані для побудови лінії тренду
sorted_idx = np.argsort(X1)
x1_sorted = X1[sorted_idx]

# Обчислюємо передбачені значення для X1 (при середньому X2)
X2_mean = np.mean(X2)
y_pred_x1 = b0 + b1 * x1_sorted + b2 * X2_mean

ax2.plot(x1_sorted, y_pred_x1, 'r-', linewidth=2)
ax2.set_xlabel('X1')
ax2.set_ylabel('Y')
ax2.set_title('Залежність Y від X1 (при середньому X2)')

# 3. Залежність Y від X2 (2D)
ax3 = plt.subplot(2, 3, 3)
ax3.scatter(X2, Y, alpha=0.6)

# Сортуємо дані для побудови лінії тренду
sorted_idx = np.argsort(X2)
x2_sorted = X2[sorted_idx]

# Обчислюємо передбачені значення для X2 (при середньому X1)
X1_mean = np.mean(X1)
y_pred_x2 = b0 + b1 * X1_mean + b2 * x2_sorted

ax3.plot(x2_sorted, y_pred_x2, 'r-', linewidth=2)
ax3.set_xlabel('X2')
ax3.set_ylabel('Y')
ax3.set_title('Залежність Y від X2 (при середньому X1)')

# 4. Кореляційна матриця
ax4 = plt.subplot(2, 3, 4)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax4)
ax4.set_title('Кореляційна матриця')

# 5. Фактичні vs Передбачені значення
ax5 = plt.subplot(2, 3, 5)
ax5.scatter(Y, y_pred, alpha=0.6)
ax5.plot([min(Y), max(Y)], [min(Y), max(Y)], 'r--', linewidth=2)
ax5.set_xlabel('Фактичні значення')
ax5.set_ylabel('Передбачені значення')
ax5.set_title('Фактичні vs Передбачені значення')

# 6. Залишки
ax6 = plt.subplot(2, 3, 6)
ax6.scatter(y_pred, residuals, alpha=0.6)
ax6.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax6.set_xlabel('Передбачені значення')
ax6.set_ylabel('Залишки')
ax6.set_title('Графік залишків')

plt.tight_layout()
plt.savefig('D:\\PROJECTs\\MY\\ScientificCalculations\\SC\\ScientificCalculations\\Strategic Directions\\Task 3\\enhanced_regression_visualization.png', dpi=300)

# Додаткові діагностичні графіки
plt.figure(figsize=(15, 10))

# 1. QQ-графік для перевірки нормальності залишків
plt.subplot(2, 2, 1)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('QQ-графік залишків')

# 2. Гістограма залишків
plt.subplot(2, 2, 2)
plt.hist(residuals, bins=20, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--')
plt.xlabel('Залишки')
plt.ylabel('Частота')
plt.title('Гістограма залишків')

# 3. Залишки vs X1
plt.subplot(2, 2, 3)
plt.scatter(X1, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('X1')
plt.ylabel('Залишки')
plt.title('Залишки vs X1')

# 4. Залишки vs X2
plt.subplot(2, 2, 4)
plt.scatter(X2, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('X2')
plt.ylabel('Залишки')
plt.title('Залишки vs X2')

plt.tight_layout()
plt.savefig('D:\\PROJECTs\\MY\\ScientificCalculations\\SC\\ScientificCalculations\\Strategic Directions\\Task 3\\residuals_diagnostics.png', dpi=300)

# Зберігаємо дані у CSV-файл
data.to_csv('D:\\PROJECTs\\MY\\ScientificCalculations\\SC\\ScientificCalculations\\Strategic Directions\\Task 3\\enhanced_regression_data.csv', index=False)
print("\nВізуалізації збережено у файлах 'enhanced_regression_visualization.png' та 'residuals_diagnostics.png'")
print("Дані збережено у файлі 'enhanced_regression_data.csv'")

plt.show()
