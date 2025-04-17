import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Налаштування для відображення графіків з українськими символами
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set(style="whitegrid")

# Налаштування для відтворюваності результатів
np.random.seed(42)

# Функція для генерації даних послідовним методом
def generate_sequential_data(n=150):
    """
    Генерує дані послідовним методом (як у enhanced_regression_analysis.py)
    X1 та X2 генеруються послідовно з фіксованим кроком
    """
    i = np.arange(n + 1)
    X1 = -0.7 + i * 0.01
    X2 = 0.5 + i * 0.01
    Y = 3.4 + 2.7 * X1 - 0.5 * X2 + np.random.uniform(-1.1, 1.1, size=n + 1)
    
    return pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})

# Функція для генерації даних випадковим методом
def generate_random_data(n=150):
    """
    Генерує дані випадковим методом (як у MultipleRegressionAnalysis.py)
    X1 та X2 генеруються випадково з рівномірного розподілу
    """
    X1 = np.random.uniform(-0.7, 0.8, n+1)
    X2 = np.random.uniform(0.5, 2.0, n+1)
    Y = 3.4 + 2.7 * X1 - 0.5 * X2 + np.random.uniform(-1.10, 1.10, size=len(X1))
    
    return pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})

# Функція для побудови регресійної моделі
def build_regression_model(data, method_name):
    """
    Будує регресійну модель та виводить результати
    """
    X = data[['X1', 'X2']]
    y = data['Y']
    
    # Модель statsmodels (для детальної статистики)
    X_with_const = sm.add_constant(X)
    model_sm = sm.OLS(y, X_with_const).fit()
    
    # Модель sklearn (для прогнозування)
    model_sk = LinearRegression()
    model_sk.fit(X, y)
    
    # Виведення результатів
    print(f"\n{'='*50}")
    print(f"Результати для методу: {method_name}")
    print(f"{'='*50}")
    
    print("\nПерші 5 рядків даних:")
    print(data.head())
    
    print("\nОписова статистика:")
    print(data.describe())
    
    print("\nКореляція між X1 та X2:")
    print(f"{data['X1'].corr(data['X2']):.4f}")
    
    print("\nРезультати регресійного аналізу:")
    print(model_sm.summary())
    
    # Виведення рівняння регресії
    b0, b1, b2 = model_sm.params
    print(f"\nРівняння множинної регресії: Y = {b0:.4f} + {b1:.4f}*X1 + {b2:.4f}*X2")
    
    # Порівняння з заданим рівнянням
    print("\nПорівняння з заданим рівнянням:")
    print("Задане рівняння: Y = 3.4000 + 2.7000*X1 - 0.5000*X2 + rnd(-1.10, 1.10)")
    print(f"Отримане рівняння: Y = {b0:.4f} + {b1:.4f}*X1 + {b2:.4f}*X2")
    print(f"Різниця: {abs(3.4-b0):.4f}, {abs(2.7-b1):.4f}, {abs(-0.5-b2):.4f}")
    
    # Кореляційна матриця
    corr_matrix = data.corr()
    print("\nКореляційна матриця:")
    print(corr_matrix.round(4))
    
    # VIF аналіз
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print("\nVIF (Variance Inflation Factor) для перевірки мультиколінеарності:")
    print(vif_data)
    
    # Коефіцієнти еластичності
    elasticity_x1 = b1 * X['X1'].mean() / y.mean()
    elasticity_x2 = b2 * X['X2'].mean() / y.mean()
    print("\nКоефіцієнти еластичності:")
    print(f"E1 = {elasticity_x1:.4f}")
    print(f"E2 = {elasticity_x2:.4f}")
    
    # β-коефіцієнти
    beta_x1 = b1 * X['X1'].std() / y.std()
    beta_x2 = b2 * X['X2'].std() / y.std()
    print("\nβ-коефіцієнти:")
    print(f"β1 = {beta_x1:.4f}")
    print(f"β2 = {beta_x2:.4f}")
    
    # Δ-коефіцієнти
    delta_x1 = beta_x1 * corr_matrix.loc['X1', 'Y']
    delta_x2 = beta_x2 * corr_matrix.loc['X2', 'Y']
    print("\nΔ-коефіцієнти:")
    print(f"Δ1 = {delta_x1:.4f}")
    print(f"Δ2 = {delta_x2:.4f}")
    print(f"Сума Δ-коефіцієнтів: {delta_x1 + delta_x2:.4f}")
    
    # Прогнозування та аналіз залишків
    y_pred = model_sm.predict(X_with_const)
    residuals = y - y_pred
    
    # Статистика залишків
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
    alpha = 0.05
    print(f"Висновок: залишки {'мають' if shapiro_test[1] > alpha else 'не мають'} нормальний розподіл")
    
    return {
        'data': data,
        'model_sm': model_sm,
        'model_sk': model_sk,
        'X': X,
        'y': y,
        'residuals': residuals,
        'y_pred': y_pred,
        'corr_matrix': corr_matrix,
        'method_name': method_name
    }

# Функція для візуалізації результатів
def visualize_results(results_seq, results_rand):
    """
    Візуалізує результати обох методів для порівняння
    """
    # Створюємо фігуру для порівняння регресійних моделей
    plt.figure(figsize=(15, 10))
    
    # 1. Порівняння рівнянь регресії (3D)
    ax1 = plt.subplot(2, 2, 1, projection='3d')
    
    # Створюємо сітку для поверхні
    x1_min = min(results_seq['data']['X1'].min(), results_rand['data']['X1'].min())
    x1_max = max(results_seq['data']['X1'].max(), results_rand['data']['X1'].max())
    x2_min = min(results_seq['data']['X2'].min(), results_rand['data']['X2'].min())
    x2_max = max(results_seq['data']['X2'].max(), results_rand['data']['X2'].max())
    
    x1_grid, x2_grid = np.meshgrid(
        np.linspace(x1_min, x1_max, 20),
        np.linspace(x2_min, x2_max, 20)
    )
    
    # Обчислюємо передбачені значення для послідовного методу
    b0_seq, b1_seq, b2_seq = results_seq['model_sm'].params
    y_pred_grid_seq = b0_seq + b1_seq * x1_grid + b2_seq * x2_grid
    
    # Обчислюємо передбачені значення для випадкового методу
    b0_rand, b1_rand, b2_rand = results_rand['model_sm'].params
    y_pred_grid_rand = b0_rand + b1_rand * x1_grid + b2_rand * x2_grid
    
    # Будуємо поверхні регресії
    ax1.plot_surface(x1_grid, x2_grid, y_pred_grid_seq, alpha=0.3, color='blue', label='Послідовний метод')
    ax1.plot_surface(x1_grid, x2_grid, y_pred_grid_rand, alpha=0.3, color='red', label='Випадковий метод')
    
    # Додаємо точки даних
    ax1.scatter(results_seq['data']['X1'], results_seq['data']['X2'], results_seq['data']['Y'], 
                c='blue', marker='o', alpha=0.2, label='Дані (послідовний)')
    ax1.scatter(results_rand['data']['X1'], results_rand['data']['X2'], results_rand['data']['Y'], 
                c='red', marker='^', alpha=0.2, label='Дані (випадковий)')
    
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.set_zlabel('Y')
    ax1.set_title('Порівняння регресійних моделей (3D)')
    
    # 2. Порівняння кореляційних матриць
    ax2 = plt.subplot(2, 2, 2)
    
    # Створюємо об'єднану кореляційну матрицю
    corr_comparison = pd.DataFrame({
        'X1-X2 (послід.)': [results_seq['corr_matrix'].loc['X1', 'X2']],
        'X1-Y (послід.)': [results_seq['corr_matrix'].loc['X1', 'Y']],
        'X2-Y (послід.)': [results_seq['corr_matrix'].loc['X2', 'Y']],
        'X1-X2 (випадк.)': [results_rand['corr_matrix'].loc['X1', 'X2']],
        'X1-Y (випадк.)': [results_rand['corr_matrix'].loc['X1', 'Y']],
        'X2-Y (випадк.)': [results_rand['corr_matrix'].loc['X2', 'Y']]
    })
    
    sns.heatmap(corr_comparison, annot=True, cmap='coolwarm', center=0, ax=ax2)
    ax2.set_title('Порівняння кореляцій')
    
    # 3. Порівняння фактичних і передбачених значень
    ax3 = plt.subplot(2, 2, 3)
    ax3.scatter(results_seq['y'], results_seq['y_pred'], alpha=0.5, label='Послідовний метод', color='blue')
    ax3.scatter(results_rand['y'], results_rand['y_pred'], alpha=0.5, label='Випадковий метод', color='red')
    
    # Додаємо лінію ідеального прогнозу
    y_min = min(results_seq['y'].min(), results_rand['y'].min())
    y_max = max(results_seq['y'].max(), results_rand['y'].max())
    ax3.plot([y_min, y_max], [y_min, y_max], 'k--', linewidth=2)
    
    ax3.set_xlabel('Фактичні значення')
    ax3.set_ylabel('Передбачені значення')
    ax3.set_title('Порівняння точності прогнозів')
    ax3.legend()
    
    # 4. Порівняння розподілу залишків
    ax4 = plt.subplot(2, 2, 4)
    sns.kdeplot(results_seq['residuals'], label='Послідовний метод', color='blue', ax=ax4)
    sns.kdeplot(results_rand['residuals'], label='Випадковий метод', color='red', ax=ax4)
    ax4.axvline(x=0, color='k', linestyle='--', linewidth=2)
    ax4.set_xlabel('Залишки')
    ax4.set_ylabel('Щільність')
    ax4.set_title('Порівняння розподілу залишків')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('D:\\PROJECTs\\MY\\ScientificCalculations\\SC\\ScientificCalculations\\Strategic Directions\\Task 3\\comparison_visualization.png', dpi=300)
    
    # Додаткові діагностичні графіки
    plt.figure(figsize=(15, 10))
    
    # 1. QQ-графіки для залишків
    plt.subplot(2, 2, 1)
    stats.probplot(results_seq['residuals'], dist="norm", plot=plt)
    plt.title('QQ-графік залишків (послідовний метод)')
    
    plt.subplot(2, 2, 2)
    stats.probplot(results_rand['residuals'], dist="norm", plot=plt)
    plt.title('QQ-графік залишків (випадковий метод)')
    
    # 2. Залишки vs Прогнозовані значення
    plt.subplot(2, 2, 3)
    plt.scatter(results_seq['y_pred'], results_seq['residuals'], alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Прогнозовані значення')
    plt.ylabel('Залишки')
    plt.title('Залишки vs Прогнозовані (послідовний метод)')
    
    plt.subplot(2, 2, 4)
    plt.scatter(results_rand['y_pred'], results_rand['residuals'], alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Прогнозовані значення')
    plt.ylabel('Залишки')
    plt.title('Залишки vs Прогнозовані (випадковий метод)')
    
    plt.tight_layout()
    plt.savefig('D:\\PROJECTs\\MY\\ScientificCalculations\\SC\\ScientificCalculations\\Strategic Directions\\Task 3\\residuals_comparison.png', dpi=300)
    
    # Порівняння коефіцієнтів моделей
    plt.figure(figsize=(12, 6))
    
    # Створюємо датафрейм для порівняння коефіцієнтів
    coef_comparison = pd.DataFrame({
        'Коефіцієнт': ['Вільний член', 'X1', 'X2'],
        'Задані': [3.4, 2.7, -0.5],
        'Послідовний метод': [b0_seq, b1_seq, b2_seq],
        'Випадковий метод': [b0_rand, b1_rand, b2_rand]
    })
    
    # Обчислюємо абсолютні відхилення
    coef_comparison['Відхилення (послід.)'] = abs(coef_comparison['Задані'] - coef_comparison['Послідовний метод'])
    coef_comparison['Відхилення (випадк.)'] = abs(coef_comparison['Задані'] - coef_comparison['Випадковий метод'])
    
    # Візуалізуємо порівняння коефіцієнтів
    ax = plt.subplot(1, 1, 1)
    
    x = np.arange(len(coef_comparison['Коефіцієнт']))
    width = 0.2
    
    ax.bar(x - width, coef_comparison['Задані'], width, label='Задані коефіцієнти')
    ax.bar(x, coef_comparison['Послідовний метод'], width, label='Послідовний метод')
    ax.bar(x + width, coef_comparison['Випадковий метод'], width, label='Випадковий метод')
    
    ax.set_xticks(x)
    ax.set_xticklabels(coef_comparison['Коефіцієнт'])
    ax.set_ylabel('Значення коефіцієнтів')
    ax.set_title('Порівняння коефіцієнтів регресії')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('D:\\PROJECTs\\MY\\ScientificCalculations\\SC\\ScientificCalculations\\Strategic Directions\\Task 3\\coefficients_comparison.png', dpi=300)
    
    # Виведення таблиці порівняння
    print("\n" + "="*80)
    print("ПОРІВНЯННЯ РЕЗУЛЬТАТІВ ОБОХ МЕТОДІВ")
    print("="*80)
    
    comparison_table = pd.DataFrame({
        'Показник': [
            'R²', 
            'Скоригований R²', 
            'Кореляція X1-X2',
            'Коефіцієнт при X1',
            'Коефіцієнт при X2',
            'Вільний член',
            'β1',
            'β2',
            'VIF',
            'Нормальність залишків (p)'
        ],
        'Послідовний метод': [
            results_seq['model_sm'].rsquared,
            results_seq['model_sm'].rsquared_adj,
            results_seq['corr_matrix'].loc['X1', 'X2'],
            b1_seq,
            b2_seq,
            b0_seq,
            beta_x1,
            beta_x2,
            vif_data['VIF'].mean(),
            stats.shapiro(results_seq['residuals'])[1]
        ],
        'Випадковий метод': [
            results_rand['model_sm'].rsquared,
            results_rand['model_sm'].rsquared_adj,
            results_rand['corr_matrix'].loc['X1', 'X2'],
            b1_rand,
            b2_rand,
            b0_rand,
            beta_x1,
            beta_x2,
            vif_data['VIF'].mean(),
            stats.shapiro(results_rand['residuals'])[1]
        ],
        'Задані значення': [
            'Невідомо',
            'Невідомо',
            'Незалежні',
            2.7,
            -0.5,
            3.4,
            'Невідомо',
            'Невідомо',
            1.0,
            '> 0.05'
        ]
    })
    
    # Форматування таблиці
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(comparison_table)
    
    # Зберігаємо таблицю порівняння
    comparison_table.to_csv('D:\\PROJECTs\\MY\\ScientificCalculations\\SC\\ScientificCalculations\\Strategic Directions\\Task 3\\methods_comparison.csv', index=False)
    
    print("\nВізуалізації збережено у файлах:")
    print("- comparison_visualization.png")
    print("- residuals_comparison.png")
    print("- coefficients_comparison.png")
    print("Таблиця порівняння збережена у файлі methods_comparison.csv")

# Головна функція
def main():
    print("Порівняльний аналіз методів генерації даних для множинної регресії")
    print("="*70)
    
    # Генеруємо дані обома методами
    data_seq = generate_sequential_data()
    data_rand = generate_random_data()
    
    # Будуємо регресійні моделі
    results_seq = build_regression_model(data_seq, "Послідовний метод")
    results_rand = build_regression_model(data_rand, "Випадковий метод")
    
    # Візуалізуємо результати
    visualize_results(results_seq, results_rand)

if __name__ == "__main__":
    main()
