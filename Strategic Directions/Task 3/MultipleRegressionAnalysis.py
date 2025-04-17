import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Генерація даних
n = 150
np.random.seed(42)
X1 = np.random.uniform(-0.7, 0.8, n+1)
X2 = np.random.uniform(0.5, 2.0, n+1)

# Генерація залежної змінної з випадковим шумом
Y = 3.4 + 2.7 * X1 - 0.5 * X2 + np.random.uniform(-1.10, 1.10, size=len(X1))

# Створення датафрейму
data = pd.DataFrame({
    'X1': X1,
    'X2': X2,
    'Y': Y
})

# Виведення початкової статистики
print("Статистика вхідних даних:")
print("\nКореляція між X1 та X2:")
print(f"{data['X1'].corr(data['X2']):.4f}")
print("\nОсновні статистичні показники:")
print(data.describe())

# Візуалізація даних
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(X1, Y, alpha=0.5)
plt.xlabel('X1')
plt.ylabel('Y')
plt.title('Залежність Y від X1')

plt.subplot(122)
plt.scatter(X2, Y, alpha=0.5)
plt.xlabel('X2')
plt.ylabel('Y')
plt.title('Залежність Y від X2')
plt.tight_layout()
plt.show()

# Побудова регресії
X = data[['X1', 'X2']]
y = data['Y']

model = LinearRegression()
model.fit(X, y)

# Виведення результатів та порівняння
print('\nКоефіцієнти регресії:')
print(f'b0 (константа) = {model.intercept_:.4f}')
print(f'b1 (X1) = {model.coef_[0]:.4f}')
print(f'b2 (X2) = {model.coef_[1]:.4f}')

print('\nПорівняння рівнянь:')
print('Задане рівняння:')
print('y(x1, x2) = 3.4 + 2.7 * x1 - 0.5 * x2 + rnd(1.10)')
print('Отримане рівняння регресії:')
print(f'y = {model.intercept_:.4f} + {model.coef_[0]:.4f} * x1 + {model.coef_[1]:.4f} * x2')

# Статистичні показники
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
n = len(y)
p = X.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

correlation_matrix = data.corr()

# Розрахунок стандартних помилок
X_with_const = np.column_stack([np.ones(len(X)), X])
mse = np.sum((y - y_pred) ** 2) / (len(y) - p - 1)
var_covar_matrix = mse * np.linalg.inv(X_with_const.T @ X_with_const)
std_errors = np.sqrt(np.diag(var_covar_matrix))

print('\nСтатистичні показники:')
print(f'R-квадрат: {r2:.4f}')
print(f'Скоригований R-квадрат: {adj_r2:.4f}')
print('\nКореляційна матриця:')
print(correlation_matrix.round(4))

# Візуалізація кореляційної матриці
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Кореляційна матриця')
plt.show()

# VIF аналіз
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

print("\nVIF (Variance Inflation Factor) для перевірки мультиколінеарності:")
print(calculate_vif(X))

# 3D візуалізація
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

x1_grid, x2_grid = np.meshgrid(np.linspace(X1.min(), X1.max(), 100),
                              np.linspace(X2.min(), X2.max(), 100))

X_grid = pd.DataFrame({
    'X1': x1_grid.ravel(),
    'X2': x2_grid.ravel()
})

y_grid = model.predict(X_grid)
y_grid = y_grid.reshape(x1_grid.shape)

ax.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.3)
ax.scatter(X1, X2, Y, c='r', marker='o')

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.set_title('Множинна регресія')
plt.show()

# Розрахунок додаткових коефіцієнтів
elasticity_x1 = model.coef_[0] * X1.mean() / Y.mean()
elasticity_x2 = model.coef_[1] * X2.mean() / Y.mean()

beta_x1 = model.coef_[0] * X1.std() / Y.std()
beta_x2 = model.coef_[1] * X2.std() / Y.std()

print('\nКоефіцієнти еластичності:')
print(f'E1 = {elasticity_x1:.4f}')
print(f'E2 = {elasticity_x2:.4f}')

print('\nБета-коефіцієнти:')
print(f'β1 = {beta_x1:.4f}')
print(f'β2 = {beta_x2:.4f}')

# Аналіз результатів
print("\nАналіз виявлених проблем:")
print("\n1. Порівняння коефіцієнтів:")
print(f"{'Параметр':<15} {'Заданий':<10} {'Отриманий':<10} {'Різниця':<10}")
print("-" * 45)
print(f"{'Вільний член':<15} {3.4:<10.4f} {model.intercept_:<10.4f} {abs(3.4-model.intercept_):<10.4f}")
print(f"{'X1':<15} {2.7:<10.4f} {model.coef_[0]:<10.4f} {abs(2.7-model.coef_[0]):<10.4f}")
print(f"{'X2':<15} {-0.5:<10.4f} {model.coef_[1]:<10.4f} {abs(-0.5-model.coef_[1]):<10.4f}")

correlation_x1_x2 = data['X1'].corr(data['X2'])
print(f"\n2. Кореляція між X1 та X2: {correlation_x1_x2:.4f}")
if abs(correlation_x1_x2) > 0.8:
    print("Виявлено сильну мультиколінеарність!")