import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Налаштування для відтворюваності результатів
np.random.seed(42)

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

# 12. Графічне представлення
# 3D-графік: фактичні дані та регресійна площина
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Фактичні точки
ax.scatter(X1, X2, Y, color='blue', label='Спостережувані дані')

# Створення сітки для регресійної площини
x1_grid, x2_grid = np.meshgrid(np.linspace(min(X1), max(X1), 20),
                               np.linspace(min(X2), max(X2), 20))
y_pred_grid = b0 + b1 * x1_grid + b2 * x2_grid

# Побудова регресійної площини
ax.plot_surface(x1_grid, x2_grid, y_pred_grid, alpha=0.5, color='green')

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.set_title('Множинна регресія: фактичні дані та регресійна площина')
ax.legend()

# Графік залишків
y_pred = model.predict(X_with_const)
residuals = Y - y_pred

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Прогнозовані значення')
plt.ylabel('Залишки')
plt.title('Залишки vs Прогнозовані значення')

# QQ-графік для перевірки нормальності залишків
plt.subplot(122)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('QQ-графік залишків')

plt.tight_layout()
plt.show()

# Вивід даних для залишків
residual_stats = {'mean': np.mean(residuals),
                  'std': np.std(residuals),
                  'min': np.min(residuals),
                  'max': np.max(residuals)}
print('\nСтатистика залишків:')
print(pd.Series(residual_stats))

# Перевірка нормальності залишків (тест Шапіро-Вілка)
shapiro_test = stats.shapiro(residuals)
print('\nПеревірка нормальності залишків (тест Шапіро-Вілка):')
print(f'Статистика W: {shapiro_test[0]:.4f}')
print(f'p-значення: {shapiro_test[1]:.8f}')
print(f'Висновок: залишки {"мають" if shapiro_test[1] > alpha else "не мають"} нормальний розподіл')