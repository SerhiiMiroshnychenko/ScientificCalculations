"""
ANFIS система для вилучення керуючих правил з інформації про процес управління

Програма виконує:
1. Трансформацію даних з файлу data2.xls
2. Побудову та навчання ANFIS системи
3. Візуалізацію результатів (структура мережі, помилки, функції приналежності, правила)
4. Інтерактивне введення параметрів для отримання виходу

Автор: Система ANFIS
Дата: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrow
import seaborn as sns
from scipy.optimize import least_squares
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Встановлюємо шрифти для підтримки української мови
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class GaussianMembershipFunction:
    """Гаусівська функція приналежності"""
    
    def __init__(self, mean, sigma):
        """
        Ініціалізація гаусівської функції приналежності
        
        Args:
            mean: Середнє значення
            sigma: Стандартне відхилення
        """
        self.mean = mean
        self.sigma = sigma
    
    def calculate(self, x):
        """Обчислення значення функції приналежності"""
        return np.exp(-((x - self.mean) ** 2) / (2 * self.sigma ** 2))
    
    def __str__(self):
        return f"Gaussian(μ={self.mean:.4f}, σ={self.sigma:.4f})"


class ANFISSystem:
    """
    Adaptive Neuro-Fuzzy Inference System (ANFIS)
    
    Система адаптивного нейро-нечіткого висновку з гібридним методом навчання
    """
    
    def __init__(self, n_mfs=5, mf_type='gaussmf'):
        """
        Ініціалізація ANFIS системи
        
        Args:
            n_mfs: Кількість функцій приналежності для кожного входу
            mf_type: Тип функцій приналежності ('gaussmf')
        """
        self.n_mfs = n_mfs
        self.mf_type = mf_type
        self.mf_params_input1 = []
        self.mf_params_input2 = []
        self.consequent_params = []
        self.n_rules = n_mfs * n_mfs
        self.training_errors = []
        self.is_trained = False
        
    def _initialize_membership_functions(self, X):
        """Ініціалізація функцій приналежності для входів"""
        x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
        x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
        
        # Ініціалізація функцій приналежності для першого входу
        x1_centers = np.linspace(x1_min, x1_max, self.n_mfs)
        x1_sigma = (x1_max - x1_min) / (2 * self.n_mfs)
        
        self.mf_params_input1 = []
        for center in x1_centers:
            self.mf_params_input1.append(GaussianMembershipFunction(center, x1_sigma))
        
        # Ініціалізація функцій приналежності для другого входу
        x2_centers = np.linspace(x2_min, x2_max, self.n_mfs)
        x2_sigma = (x2_max - x2_min) / (2 * self.n_mfs)
        
        self.mf_params_input2 = []
        for center in x2_centers:
            self.mf_params_input2.append(GaussianMembershipFunction(center, x2_sigma))
        
        # Ініціалізація параметрів консеквентів (p, q, r для кожного правила)
        self.consequent_params = np.random.randn(self.n_rules, 3) * 0.1
    
    def _fuzzification(self, X):
        """Фазифікація - обчислення ступенів приналежності"""
        n_samples = X.shape[0]
        
        # Обчислення ступенів приналежності для входу 1
        mu1 = np.zeros((n_samples, self.n_mfs))
        for i, mf in enumerate(self.mf_params_input1):
            mu1[:, i] = mf.calculate(X[:, 0])
        
        # Обчислення ступенів приналежності для входу 2
        mu2 = np.zeros((n_samples, self.n_mfs))
        for i, mf in enumerate(self.mf_params_input2):
            mu2[:, i] = mf.calculate(X[:, 1])
        
        return mu1, mu2
    
    def _firing_strengths(self, mu1, mu2):
        """Обчислення сили спрацювання правил"""
        n_samples = mu1.shape[0]
        w = np.zeros((n_samples, self.n_rules))
        
        rule_idx = 0
        for i in range(self.n_mfs):
            for j in range(self.n_mfs):
                w[:, rule_idx] = mu1[:, i] * mu2[:, j]
                rule_idx += 1
        
        return w
    
    def _normalize_firing_strengths(self, w):
        """Нормалізація сили спрацювання правил"""
        w_sum = np.sum(w, axis=1, keepdims=True)
        w_sum = np.where(w_sum == 0, 1e-10, w_sum)  # Уникнення ділення на нуль
        w_normalized = w / w_sum
        return w_normalized
    
    def predict(self, X):
        """
        Прогнозування виходу для заданих входів
        
        Args:
            X: Вхідні дані (n_samples, 2)
            
        Returns:
            y_pred: Прогнозовані значення виходу
        """
        # Фазифікація
        mu1, mu2 = self._fuzzification(X)
        
        # Обчислення сили спрацювання правил
        w = self._firing_strengths(mu1, mu2)
        
        # Нормалізація
        w_normalized = self._normalize_firing_strengths(w)
        
        # Обчислення виходу за правилами Takagi-Sugeno
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)
        
        for i in range(self.n_rules):
            # Лінійна комбінація: f_i = p_i * x1 + q_i * x2 + r_i
            f_i = (self.consequent_params[i, 0] * X[:, 0] + 
                   self.consequent_params[i, 1] * X[:, 1] + 
                   self.consequent_params[i, 2])
            y_pred += w_normalized[:, i] * f_i
        
        return y_pred
    
    def _hybrid_learning_step(self, X, y):
        """
        Один крок гібридного навчання
        
        Args:
            X: Вхідні дані
            y: Цільові значення
            
        Returns:
            error: Середньоквадратична помилка
        """
        # Фазифікація
        mu1, mu2 = self._fuzzification(X)
        
        # Обчислення сили спрацювання правил
        w = self._firing_strengths(mu1, mu2)
        
        # Нормалізація
        w_normalized = self._normalize_firing_strengths(w)
        
        # Forward pass - оновлення консеквентних параметрів (метод найменших квадратів)
        n_samples = X.shape[0]
        A = np.zeros((n_samples, self.n_rules * 3))
        
        for i in range(self.n_rules):
            A[:, i*3] = w_normalized[:, i] * X[:, 0]
            A[:, i*3 + 1] = w_normalized[:, i] * X[:, 1]
            A[:, i*3 + 2] = w_normalized[:, i]
        
        # Розв'язання системи методом найменших квадратів
        params_flat, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        self.consequent_params = params_flat.reshape(self.n_rules, 3)
        
        # Backward pass - оновлення параметрів функцій приналежності (градієнтний спуск)
        y_pred = self.predict(X)
        error = np.mean((y - y_pred) ** 2)
        
        # Градієнтний спуск для оновлення параметрів MF
        learning_rate = 0.01
        
        for mf_idx, mf in enumerate(self.mf_params_input1):
            grad_mean = 0
            grad_sigma = 0
            
            for sample_idx in range(n_samples):
                x1 = X[sample_idx, 0]
                
                # Обчислення градієнтів
                dmu_dmean = mf.calculate(x1) * (x1 - mf.mean) / (mf.sigma ** 2)
                dmu_dsigma = mf.calculate(x1) * ((x1 - mf.mean) ** 2) / (mf.sigma ** 3)
                
                # Накопичення градієнтів
                grad_mean += 2 * (y_pred[sample_idx] - y[sample_idx]) * dmu_dmean
                grad_sigma += 2 * (y_pred[sample_idx] - y[sample_idx]) * dmu_dsigma
            
            # Оновлення параметрів
            mf.mean -= learning_rate * grad_mean / n_samples
            mf.sigma -= learning_rate * grad_sigma / n_samples
            mf.sigma = max(mf.sigma, 1e-6)  # Запобігання від'ємних або нульових sigma
        
        for mf_idx, mf in enumerate(self.mf_params_input2):
            grad_mean = 0
            grad_sigma = 0
            
            for sample_idx in range(n_samples):
                x2 = X[sample_idx, 1]
                
                # Обчислення градієнтів
                dmu_dmean = mf.calculate(x2) * (x2 - mf.mean) / (mf.sigma ** 2)
                dmu_dsigma = mf.calculate(x2) * ((x2 - mf.mean) ** 2) / (mf.sigma ** 3)
                
                # Накопичення градієнтів
                grad_mean += 2 * (y_pred[sample_idx] - y[sample_idx]) * dmu_dmean
                grad_sigma += 2 * (y_pred[sample_idx] - y[sample_idx]) * dmu_dsigma
            
            # Оновлення параметрів
            mf.mean -= learning_rate * grad_mean / n_samples
            mf.sigma -= learning_rate * grad_sigma / n_samples
            mf.sigma = max(mf.sigma, 1e-6)
        
        return error
    
    def train(self, X, y, epochs=10, error_tolerance=0, verbose=True):
        """
        Навчання ANFIS системи гібридним методом
        
        Args:
            X: Вхідні дані (n_samples, 2)
            y: Цільові значення (n_samples,)
            epochs: Кількість епох навчання
            error_tolerance: Допустима помилка (якщо 0, то тренування до кінця епох)
            verbose: Виводити проміжні результати
        """
        # Ініціалізація функцій приналежності
        self._initialize_membership_functions(X)
        
        self.training_errors = []
        
        print("=" * 60)
        print("ПОЧАТОК НАВЧАННЯ ANFIS СИСТЕМИ")
        print("=" * 60)
        print(f"Кількість правил: {self.n_rules}")
        print(f"Кількість епох: {epochs}")
        print(f"Метод оптимізації: Гібридний (Hybrid)")
        print(f"Допустима помилка: {error_tolerance}")
        print("=" * 60)
        
        for epoch in range(epochs):
            error = self._hybrid_learning_step(X, y)
            self.training_errors.append(error)
            
            if verbose:
                print(f"Епоха {epoch + 1}/{epochs}, Помилка навчання (RMSE): {np.sqrt(error):.6f}")
            
            if error_tolerance > 0 and error < error_tolerance:
                print(f"Досягнуто допустиму помилку на епосі {epoch + 1}")
                break
        
        self.is_trained = True
        print("=" * 60)
        print("НАВЧАННЯ ЗАВЕРШЕНО")
        print(f"Фінальна помилка навчання (RMSE): {np.sqrt(self.training_errors[-1]):.6f}")
        print("=" * 60)
    
    def test(self, X, y):
        """
        Тестування ANFIS системи
        
        Args:
            X: Вхідні дані для тестування
            y: Цільові значення для тестування
            
        Returns:
            y_pred: Прогнозовані значення
            rmse: Середньоквадратична помилка
        """
        y_pred = self.predict(X)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        
        print("=" * 60)
        print("РЕЗУЛЬТАТИ ТЕСТУВАННЯ")
        print("=" * 60)
        print(f"Помилка тестування (RMSE): {rmse:.6f}")
        print(f"Середнє абсолютне відхилення (MAE): {np.mean(np.abs(y - y_pred)):.6f}")
        print(f"Коефіцієнт детермінації (R²): {1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2):.6f}")
        print("=" * 60)
        
        return y_pred, rmse
    
    def get_rules(self):
        """Отримання сформульованих правил"""
        rules = []
        rule_idx = 0
        
        for i in range(self.n_mfs):
            for j in range(self.n_mfs):
                rule = {
                    'number': rule_idx + 1,
                    'input1_mf': i + 1,
                    'input2_mf': j + 1,
                    'consequent': self.consequent_params[rule_idx]
                }
                rules.append(rule)
                rule_idx += 1
        
        return rules


class ANFISVisualizer:
    """Клас для візуалізації результатів ANFIS системи"""
    
    def __init__(self, anfis_system):
        """
        Ініціалізація візуалізатора
        
        Args:
            anfis_system: Об'єкт ANFISSystem
        """
        self.anfis = anfis_system
    
    def plot_structure(self):
        """Візуалізація структури нейро-нечіткої мережі"""
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Визначення позицій шарів
        layer_x = [1, 2.5, 5, 7.5, 9]
        
        # Шар 1: Входи
        input_y = [5.5, 4.5]
        for i, y in enumerate(input_y):
            circle = plt.Circle((layer_x[0], y), 0.2, color='lightblue', ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(layer_x[0] - 0.8, y, f'x{i+1}', fontsize=12, ha='center', va='center', weight='bold')
        
        # Шар 2: Функції приналежності
        mf_spacing = 8 / self.anfis.n_mfs
        mf_y_start = 1 + mf_spacing / 2
        
        input1_mf_positions = []
        input2_mf_positions = []
        
        for i in range(self.anfis.n_mfs):
            y = mf_y_start + i * mf_spacing
            # MF для входу 1
            circle = plt.Circle((layer_x[1], y), 0.15, color='lightgreen', ec='black', linewidth=1.5)
            ax.add_patch(circle)
            ax.text(layer_x[1], y, f'A{i+1}', fontsize=8, ha='center', va='center')
            input1_mf_positions.append(y)
            
            # Лінія від входу 1 до MF
            ax.plot([layer_x[0] + 0.2, layer_x[1] - 0.15], [input_y[0], y], 'k-', linewidth=0.5, alpha=0.3)
        
        for i in range(self.anfis.n_mfs):
            y = mf_y_start + i * mf_spacing
            # MF для входу 2
            circle = plt.Circle((layer_x[1] + 0.5, y), 0.15, color='lightgreen', ec='black', linewidth=1.5)
            ax.add_patch(circle)
            ax.text(layer_x[1] + 0.5, y, f'B{i+1}', fontsize=8, ha='center', va='center')
            input2_mf_positions.append(y)
            
            # Лінія від входу 2 до MF
            ax.plot([layer_x[0] + 0.2, layer_x[1] + 0.5 - 0.15], [input_y[1], y], 'k-', linewidth=0.5, alpha=0.3)
        
        # Шар 3 і 4: Правила (показуємо тільки деякі для зручності)
        n_rules_to_show = min(self.anfis.n_rules, 25)
        rule_spacing = 8 / n_rules_to_show
        rule_y_start = 1 + rule_spacing / 2
        
        rule_positions = []
        for i in range(n_rules_to_show):
            y = rule_y_start + i * rule_spacing
            rule_positions.append(y)
            
            # Шар 3: Множення (П)
            circle = plt.Circle((layer_x[2], y), 0.12, color='yellow', ec='black', linewidth=1.5)
            ax.add_patch(circle)
            ax.text(layer_x[2], y, 'П', fontsize=8, ha='center', va='center', weight='bold')
            
            # Шар 4: Нормалізація (N)
            circle = plt.Circle((layer_x[3], y), 0.12, color='orange', ec='black', linewidth=1.5)
            ax.add_patch(circle)
            ax.text(layer_x[3], y, 'N', fontsize=8, ha='center', va='center', weight='bold')
            
            # Лінії від правил до нормалізації
            ax.plot([layer_x[2] + 0.12, layer_x[3] - 0.12], [y, y], 'k-', linewidth=1, alpha=0.5)
        
        # Лінії від MF до правил (тільки для перших кількох)
        for i in range(min(3, n_rules_to_show)):
            rule_y = rule_positions[i]
            # З'єднуємо з деякими MF
            ax.plot([layer_x[1] + 0.15, layer_x[2] - 0.12], 
                   [input1_mf_positions[0], rule_y], 'k-', linewidth=0.5, alpha=0.2)
            ax.plot([layer_x[1] + 0.5 + 0.15, layer_x[2] - 0.12], 
                   [input2_mf_positions[0], rule_y], 'k-', linewidth=0.5, alpha=0.2)
        
        # Шар 5: Вихід
        output_y = 5
        circle = plt.Circle((layer_x[4], output_y), 0.2, color='lightcoral', ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(layer_x[4] + 0.6, output_y, 'y', fontsize=12, ha='center', va='center', weight='bold')
        
        # Лінії від нормалізації до виходу
        for y in rule_positions:
            ax.plot([layer_x[3] + 0.12, layer_x[4] - 0.2], [y, output_y], 'k-', linewidth=0.5, alpha=0.3)
        
        # Підписи шарів
        layer_names = ['Шар 1\n(Входи)', 'Шар 2\n(Фазифікація)', 
                       'Шар 3\n(Правила)', 'Шар 4\n(Нормалізація)', 
                       'Шар 5\n(Вихід)']
        
        for i, name in enumerate(layer_names):
            ax.text(layer_x[i], 0.3, name, fontsize=9, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_title('Структура нейро-нечіткої мережі ANFIS', fontsize=14, weight='bold', pad=20)
        
        # Додаткова інформація
        info_text = f'Кількість входів: 2\nКількість правил: {self.anfis.n_rules}\nКількість MF на вхід: {self.anfis.n_mfs}'
        ax.text(5, 9.5, info_text, fontsize=10, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('D:\\WINDSURF\\SCRIPTs\\FUZZY-LOGIC\\Інд1\\Рішення\\anfis_structure.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Структура мережі збережена: anfis_structure.png")
    
    def plot_training_error(self):
        """Візуалізація помилки навчання"""
        if not self.anfis.training_errors:
            print("⚠ Немає даних про помилки навчання. Спочатку навчіть модель.")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(self.anfis.training_errors) + 1)
        rmse_errors = [np.sqrt(err) for err in self.anfis.training_errors]
        
        ax.plot(epochs, rmse_errors, 'b-', linewidth=2, marker='o', markersize=6)
        ax.set_xlabel('Епоха', fontsize=12, weight='bold')
        ax.set_ylabel('Помилка навчання (RMSE)', fontsize=12, weight='bold')
        ax.set_title('Крива навчання ANFIS системи', fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        
        # Додаємо текст з фінальною помилкою
        final_rmse = rmse_errors[-1]
        ax.text(0.95, 0.95, f'Фінальна RMSE: {final_rmse:.6f}',
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('D:\\WINDSURF\\SCRIPTs\\FUZZY-LOGIC\\Інд1\\Рішення\\training_error.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Графік помилки навчання збережено: training_error.png")
    
    def plot_membership_functions(self, X):
        """Візуалізація функцій приналежності"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Функції приналежності для першого входу
        x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
        ax1 = axes[0]
        
        for i, mf in enumerate(self.anfis.mf_params_input1):
            mu = mf.calculate(x1_range)
            ax1.plot(x1_range, mu, linewidth=2, label=f'MF{i+1}')
        
        ax1.set_xlabel('Вхід 1 (x1)', fontsize=12, weight='bold')
        ax1.set_ylabel('Ступінь приналежності', fontsize=12, weight='bold')
        ax1.set_title('Функції приналежності для входу x1', fontsize=13, weight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([-0.05, 1.05])
        
        # Функції приналежності для другого входу
        x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 200)
        ax2 = axes[1]
        
        for i, mf in enumerate(self.anfis.mf_params_input2):
            mu = mf.calculate(x2_range)
            ax2.plot(x2_range, mu, linewidth=2, label=f'MF{i+1}')
        
        ax2.set_xlabel('Вхід 2 (x2)', fontsize=12, weight='bold')
        ax2.set_ylabel('Ступінь приналежності', fontsize=12, weight='bold')
        ax2.set_title('Функції приналежності для входу x2', fontsize=13, weight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([-0.05, 1.05])
        
        plt.tight_layout()
        plt.savefig('D:\\WINDSURF\\SCRIPTs\\FUZZY-LOGIC\\Інд1\\Рішення\\membership_functions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Функції приналежності збережено: membership_functions.png")
    
    def plot_test_results(self, X, y_true, y_pred):
        """Візуалізація результатів тестування"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Графік порівняння справжніх та прогнозованих значень
        ax1 = axes[0]
        indices = range(len(y_true))
        ax1.plot(indices, y_true, 'b-', linewidth=2, label='Справжні значення', marker='o', markersize=4)
        ax1.plot(indices, y_pred, 'r--', linewidth=2, label='Прогнозовані значення', marker='x', markersize=4)
        ax1.set_xlabel('Номер зразка', fontsize=12, weight='bold')
        ax1.set_ylabel('Значення виходу', fontsize=12, weight='bold')
        ax1.set_title('Порівняння справжніх та прогнозованих значень', fontsize=13, weight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Графік розсіювання
        ax2 = axes[1]
        ax2.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
        
        # Лінія ідеального прогнозу
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ідеальний прогноз')
        
        ax2.set_xlabel('Справжні значення', fontsize=12, weight='bold')
        ax2.set_ylabel('Прогнозовані значення', fontsize=12, weight='bold')
        ax2.set_title('Кореляція між справжніми та прогнозованими значеннями', fontsize=13, weight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # R² score
        r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
        ax2.text(0.05, 0.95, f'R² = {r2:.4f}',
                transform=ax2.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('D:\\WINDSURF\\SCRIPTs\\FUZZY-LOGIC\\Інд1\\Рішення\\test_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Результати тестування збережено: test_results.png")
    
    def plot_surface(self, X, resolution=50):
        """Візуалізація поверхні виходу (Surface Viewer)"""
        x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
        x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
        
        x1_grid = np.linspace(x1_min, x1_max, resolution)
        x2_grid = np.linspace(x2_min, x2_max, resolution)
        
        X1, X2 = np.meshgrid(x1_grid, x2_grid)
        
        # Обчислення виходу для кожної точки сітки
        Z = np.zeros_like(X1)
        for i in range(resolution):
            for j in range(resolution):
                X_point = np.array([[X1[i, j], X2[i, j]]])
                Z[i, j] = self.anfis.predict(X_point)[0]
        
        # 3D графік
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8, edgecolor='none')
        
        ax.set_xlabel('Вхід 1 (x1)', fontsize=11, weight='bold')
        ax.set_ylabel('Вхід 2 (x2)', fontsize=11, weight='bold')
        ax.set_zlabel('Вихід (y)', fontsize=11, weight='bold')
        ax.set_title('Поверхня виходу ANFIS системи (Surface Viewer)', fontsize=14, weight='bold', pad=20)
        
        # Колірна шкала
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        plt.tight_layout()
        plt.savefig('D:\\WINDSURF\\SCRIPTs\\FUZZY-LOGIC\\Інд1\\Рішення\\surface_viewer.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Поверхня виходу збережена: surface_viewer.png")
    
    def plot_rule_viewer(self, x1_input, x2_input, X):
        """
        Візуалізація роботи правил для конкретних входів (Rule Viewer)
        
        Args:
            x1_input: Значення першого входу
            x2_input: Значення другого входу
            X: Діапазон вхідних даних для побудови функцій приналежності
        """
        # Обчислення виходу
        X_input = np.array([[x1_input, x2_input]])
        y_output = self.anfis.predict(X_input)[0]
        
        # Фазифікація для даного входу
        mu1, mu2 = self.anfis._fuzzification(X_input)
        w = self.anfis._firing_strengths(mu1, mu2)
        w_normalized = self.anfis._normalize_firing_strengths(w)
        
        # Створення фігури
        n_rules = min(self.anfis.n_rules, 9)  # Показуємо до 9 правил
        n_cols = 3
        n_rows = (n_rules + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(15, 3 * n_rows + 2))
        
        # Діапазони для візуалізації
        x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
        x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 200)
        
        for rule_idx in range(n_rules):
            # Визначаємо які MF використовуються в цьому правилі
            mf1_idx = rule_idx // self.anfis.n_mfs
            mf2_idx = rule_idx % self.anfis.n_mfs
            
            ax = plt.subplot(n_rows, n_cols, rule_idx + 1)
            
            # Малюємо функції приналежності для входів
            mu1_values = self.anfis.mf_params_input1[mf1_idx].calculate(x1_range)
            mu2_values = self.anfis.mf_params_input2[mf2_idx].calculate(x2_range)
            
            # Нормалізуємо для відображення на одному графіку
            ax.plot(x1_range, mu1_values, 'b-', linewidth=2, label=f'x1 (MF{mf1_idx+1})')
            ax.plot(x2_range, mu2_values, 'g-', linewidth=2, label=f'x2 (MF{mf2_idx+1})')
            
            # Відмічаємо вхідні значення
            mu1_val = mu1[0, mf1_idx]
            mu2_val = mu2[0, mf2_idx]
            
            ax.axvline(x1_input, color='b', linestyle='--', alpha=0.5, linewidth=1)
            ax.axvline(x2_input, color='g', linestyle='--', alpha=0.5, linewidth=1)
            
            ax.axhline(mu1_val, color='b', linestyle=':', alpha=0.3, linewidth=1)
            ax.axhline(mu2_val, color='g', linestyle=':', alpha=0.3, linewidth=1)
            
            ax.set_title(f'Правило {rule_idx+1}: w={w_normalized[0, rule_idx]:.3f}', fontsize=10, weight='bold')
            ax.set_ylabel('μ', fontsize=9)
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([-0.05, 1.05])
        
        plt.suptitle(f'Rule Viewer: x1={x1_input:.3f}, x2={x2_input:.3f} → y={y_output:.3f}',
                    fontsize=14, weight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig('D:\\WINDSURF\\SCRIPTs\\FUZZY-LOGIC\\Інд1\\Рішення\\rule_viewer.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Rule Viewer збережено: rule_viewer.png")
        
        return y_output


def load_and_transform_data(file_path):
    """
    Завантаження та трансформація даних з файлу Excel
    
    Трансформація аналогічна Matlab коду:
    p=data2(2:end,1);
    M=data2(1,2:end);
    tabT=data2(2:end,2:end);
    xxT - трансформована матриця
    
    Args:
        file_path: Шлях до файлу data2.xls
        
    Returns:
        xxT: Трансформовані дані (n_samples, 3) - [x1, x2, y]
    """
    print("=" * 60)
    print("ЗАВАНТАЖЕННЯ ТА ТРАНСФОРМАЦІЯ ДАНИХ")
    print("=" * 60)
    
    # Завантаження даних
    try:
        data2 = pd.read_excel(file_path, header=None)
        print(f"✓ Файл успішно завантажено: {file_path}")
        print(f"  Розмір матриці: {data2.shape}")
    except Exception as e:
        print(f"✗ Помилка завантаження файлу: {e}")
        return None
    
    # Трансформація даних (аналогічно Matlab коду)
    p = data2.iloc[1:, 0].values  # data2(2:end,1) - перший стовбець (без заголовка)
    M = data2.iloc[0, 1:].values  # data2(1,2:end) - перша строка (без першого елемента)
    tabT = data2.iloc[1:, 1:].values  # data2(2:end,2:end) - таблиця даних
    
    max_i, max_j = tabT.shape
    
    print(f"  Кількість значень x1 (p): {len(p)}")
    print(f"  Кількість значень x2 (M): {len(M)}")
    print(f"  Розмір таблиці: {max_i} x {max_j}")
    
    # Створення трансформованої матриці xxT
    xxT = np.zeros((max_j * max_i, 3))
    
    for i in range(max_i):
        for j in range(max_j):
            idx = i + j * max_i
            xxT[idx, 0] = p[i]      # x1
            xxT[idx, 1] = M[j]      # x2
            xxT[idx, 2] = tabT[i, j]  # y
    
    print(f"✓ Дані успішно трансформовано")
    print(f"  Результуюча матриця xxT: {xxT.shape}")
    print(f"  Діапазон x1: [{xxT[:, 0].min():.3f}, {xxT[:, 0].max():.3f}]")
    print(f"  Діапазон x2: [{xxT[:, 1].min():.3f}, {xxT[:, 1].max():.3f}]")
    print(f"  Діапазон y: [{xxT[:, 2].min():.3f}, {xxT[:, 2].max():.3f}]")
    print("=" * 60)
    
    return xxT


def interactive_prediction(anfis_system, X_range):
    """
    Інтерактивне введення параметрів для отримання виходу
    
    Args:
        anfis_system: Навчена ANFIS система
        X_range: Діапазон вхідних даних для валідації
    """
    print("\n" + "=" * 60)
    print("ІНТЕРАКТИВНИЙ РЕЖИМ ПРОГНОЗУВАННЯ")
    print("=" * 60)
    print("Введіть значення вхідних параметрів для отримання виходу.")
    print("Для виходу введіть 'q' або 'quit'")
    print("=" * 60)
    
    x1_min, x1_max = X_range[:, 0].min(), X_range[:, 0].max()
    x2_min, x2_max = X_range[:, 1].min(), X_range[:, 1].max()
    
    while True:
        print(f"\nДіапазон x1: [{x1_min:.3f}, {x1_max:.3f}]")
        try:
            x1_input = input("Введіть значення x1: ").strip()
            if x1_input.lower() in ['q', 'quit', 'вихід']:
                print("Вихід з інтерактивного режиму.")
                break
            x1 = float(x1_input)
        except ValueError:
            print("⚠ Помилка: введіть числове значення або 'q' для виходу")
            continue
        
        print(f"Діапазон x2: [{x2_min:.3f}, {x2_max:.3f}]")
        try:
            x2_input = input("Введіть значення x2: ").strip()
            if x2_input.lower() in ['q', 'quit', 'вихід']:
                print("Вихід з інтерактивного режиму.")
                break
            x2 = float(x2_input)
        except ValueError:
            print("⚠ Помилка: введіть числове значення або 'q' для виходу")
            continue
        
        # Перевірка діапазону (попередження, але не блокування)
        if not (x1_min <= x1 <= x1_max):
            print(f"⚠ Попередження: x1={x1:.3f} поза діапазоном навчальних даних")
        
        if not (x2_min <= x2 <= x2_max):
            print(f"⚠ Попередження: x2={x2:.3f} поза діапазоном навчальних даних")
        
        # Прогнозування
        X_input = np.array([[x1, x2]])
        y_pred = anfis_system.predict(X_input)[0]
        
        print("-" * 60)
        print(f"РЕЗУЛЬТАТ ПРОГНОЗУВАННЯ:")
        print(f"  x1 = {x1:.6f}")
        print(f"  x2 = {x2:.6f}")
        print(f"  y (прогноз) = {y_pred:.6f}")
        print("-" * 60)
        
        # Запит на продовження
        continue_input = input("\nПродовжити? (y/n): ").strip().lower()
        if continue_input in ['n', 'no', 'ні']:
            break


def main():
    """Головна функція програми"""
    print("\n" + "=" * 70)
    print(" " * 15 + "ANFIS СИСТЕМА ДЛЯ ВИЛУЧЕННЯ КЕРУЮЧИХ ПРАВИЛ")
    print(" " * 20 + "З ІНФОРМАЦІЇ ПРО ПРОЦЕС УПРАВЛІННЯ")
    print("=" * 70)
    
    # 1. Завантаження та трансформація даних
    data_file = 'D:\\WINDSURF\\SCRIPTs\\FUZZY-LOGIC\\Інд1\\data2.xls'
    xxT = load_and_transform_data(data_file)
    
    if xxT is None:
        print("Помилка: не вдалося завантажити дані. Завершення програми.")
        return
    
    # Розділення на входи та виходи
    X = xxT[:, :2]  # Перші два стовбці - входи x1 та x2
    y = xxT[:, 2]   # Третій стовбець - вихід y
    
    # 2. Створення та налаштування ANFIS системи
    print("\n" + "=" * 60)
    print("СТВОРЕННЯ ANFIS СИСТЕМИ")
    print("=" * 60)
    print("Параметри системи:")
    print("  - Кількість функцій приналежності (MF) на вхід: 5")
    print("  - Тип функцій приналежності: gaussmf (Гаусівська)")
    print("  - Метод генерації правил: Grid partition (Метод решітки)")
    print("  - Кількість правил: 5 x 5 = 25")
    print("=" * 60)
    
    anfis = ANFISSystem(n_mfs=5, mf_type='gaussmf')
    
    # 3. Навчання ANFIS системи
    anfis.train(X, y, epochs=10, error_tolerance=0, verbose=True)
    
    # 4. Тестування ANFIS системи
    y_pred, rmse = anfis.test(X, y)
    
    # 5. Візуалізація результатів
    print("\n" + "=" * 60)
    print("СТВОРЕННЯ ВІЗУАЛІЗАЦІЙ")
    print("=" * 60)
    
    visualizer = ANFISVisualizer(anfis)
    
    print("\n1. Структура нейро-нечіткої мережі...")
    visualizer.plot_structure()
    
    print("\n2. Крива навчання...")
    visualizer.plot_training_error()
    
    print("\n3. Функції приналежності...")
    visualizer.plot_membership_functions(X)
    
    print("\n4. Результати тестування...")
    visualizer.plot_test_results(X, y, y_pred)
    
    print("\n5. Поверхня виходу (Surface Viewer)...")
    visualizer.plot_surface(X, resolution=50)
    
    print("\n6. Rule Viewer (приклад для середніх значень)...")
    x1_mid = (X[:, 0].min() + X[:, 0].max()) / 2
    x2_mid = (X[:, 1].min() + X[:, 1].max()) / 2
    visualizer.plot_rule_viewer(x1_mid, x2_mid, X)
    
    # 6. Виведення правил
    print("\n" + "=" * 60)
    print("СФОРМУЛЬОВАНІ ПРАВИЛА")
    print("=" * 60)
    rules = anfis.get_rules()
    print(f"Загальна кількість правил: {len(rules)}")
    print("\nПриклади правил (перші 10):")
    for i, rule in enumerate(rules[:10]):
        p, q, r = rule['consequent']
        print(f"Правило {rule['number']}: ЯКЩО x1 є MF{rule['input1_mf']} ТА x2 є MF{rule['input2_mf']} "
              f"ТО y = {p:.4f}*x1 + {q:.4f}*x2 + {r:.4f}")
    if len(rules) > 10:
        print(f"... та ще {len(rules) - 10} правил")
    print("=" * 60)
    
    # 7. Інтерактивний режим
    interactive_prediction(anfis, X)
    
    print("\n" + "=" * 70)
    print(" " * 25 + "ПРОГРАМА ЗАВЕРШЕНА")
    print("=" * 70)
    print("\nВсі результати збережено в директорію:")
    print("D:\\WINDSURF\\SCRIPTs\\FUZZY-LOGIC\\Інд1\\Рішення\\")
    print("\nЗбережені файли:")
    print("  - anfis_structure.png     - Структура нейро-нечіткої мережі")
    print("  - training_error.png      - Крива навчання")
    print("  - membership_functions.png - Функції приналежності")
    print("  - test_results.png        - Результати тестування")
    print("  - surface_viewer.png      - Поверхня виходу")
    print("  - rule_viewer.png         - Візуалізація правил")
    print("=" * 70)


if __name__ == "__main__":
    main()

