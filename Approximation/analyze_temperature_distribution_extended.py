"""
Розширений аналіз розподілу температури по глибині шару агломераційної шихти
з детальним логуванням всіх розрахунків та проміжних результатів.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

def create_log_file():
    """Створення файлу для логування"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return open(f'temperature_analysis_log_{timestamp}.txt', 'w', encoding='utf-8')

def log_message(log_file, message):
    """Запис повідомлення в лог-файл та виведення в консоль"""
    print(message)
    log_file.write(message + '\n')
    log_file.flush()

# Функції для нелінійної апроксимації
def exp_func(x, a, b):
    """Експоненціальна функція: a*exp(b*x)"""
    return a * np.exp(b * x)

def log_func(x, a, b):
    """Логарифмічна функція: a*log(x) + b"""
    return a * np.log(x + 1) + b

def power_func(x, a, b):
    """Степенева функція: a*x^b"""
    return a * np.power(x + 1, b)

def calculate_metrics(y_true, y_pred):
    """Розрахунок метрик якості апроксимації"""
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Середня абсолютна похибка (MAE)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Середньоквадратична похибка (RMSE)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    return r2, mae, rmse

def main():
    # Відкриваємо файл для логування
    log_file = create_log_file()
    
    log_message(log_file, "=== Початок аналізу ===\n")
    
    # Вхідні дані
    depth = np.array([400, 375, 350, 325, 300, 275, 250, 225, 200, 175, 150, 
                      125, 100, 75, 50, 25, 0])
    temp = np.array([1000, 950, 900, 920, 980, 1100, 1250, 1320, 1350, 1360, 
                     1370, 1380, 1385, 1390, 1395, 1400, 1410])
    
    log_message(log_file, "Вхідні дані:")
    for d, t in zip(depth, temp):
        log_message(log_file, f"Глибина: {d:3d} мм, Температура: {t:4d}°C")
    log_message(log_file, "")
    
    # Створюємо точки для побудови кривих апроксимації
    depth_fit = np.linspace(min(depth), max(depth), 100)
    
    # Поліноміальні апроксимації
    models = {
        'Лінійна': {'degree': 1},
        'Квадратична': {'degree': 2},
        'Кубічна': {'degree': 3},
        '4-го ступеню': {'degree': 4}
    }
    
    log_message(log_file, "=== Поліноміальні апроксимації ===")
    
    for name, params in models.items():
        degree = params['degree']
        p = np.polyfit(depth, temp, degree)
        y_pred = np.polyval(p, depth)
        r2, mae, rmse = calculate_metrics(temp, y_pred)
        
        models[name].update({
            'coefficients': p,
            'predictions': y_pred,
            'r2': r2,
            'mae': mae,
            'rmse': rmse
        })
        
        log_message(log_file, f"\n{name} модель:")
        log_message(log_file, "Коефіцієнти:")
        for i, coef in enumerate(p):
            log_message(log_file, f"  a{degree-i} = {coef:.6f}")
        log_message(log_file, f"R² = {r2:.6f}")
        log_message(log_file, f"MAE = {mae:.6f}°C")
        log_message(log_file, f"RMSE = {rmse:.6f}°C")
        
        log_message(log_file, "\nПередбачені значення:")
        for d, t, p in zip(depth, temp, y_pred):
            log_message(log_file, f"Глибина: {d:3d} мм, Реальна т-ра: {t:4d}°C, "
                                f"Передбачена т-ра: {p:.1f}°C, "
                                f"Похибка: {abs(t-p):.1f}°C")
    
    # Нелінійні апроксимації
    log_message(log_file, "\n=== Нелінійні апроксимації ===")
    
    # Експоненціальна
    popt_exp, _ = curve_fit(exp_func, depth, temp, p0=[1400, -0.001])
    y_pred_exp = exp_func(depth, *popt_exp)
    r2_exp, mae_exp, rmse_exp = calculate_metrics(temp, y_pred_exp)
    
    log_message(log_file, "\nЕкспоненціальна модель:")
    log_message(log_file, f"y = {popt_exp[0]:.6f} * exp({popt_exp[1]:.6f}x)")
    log_message(log_file, f"R² = {r2_exp:.6f}")
    log_message(log_file, f"MAE = {mae_exp:.6f}°C")
    log_message(log_file, f"RMSE = {rmse_exp:.6f}°C")
    
    # Логарифмічна
    popt_log, _ = curve_fit(log_func, depth, temp, p0=[1400, -100])
    y_pred_log = log_func(depth, *popt_log)
    r2_log, mae_log, rmse_log = calculate_metrics(temp, y_pred_log)
    
    log_message(log_file, "\nЛогарифмічна модель:")
    log_message(log_file, f"y = {popt_log[0]:.6f} * log(x+1) + {popt_log[1]:.6f}")
    log_message(log_file, f"R² = {r2_log:.6f}")
    log_message(log_file, f"MAE = {mae_log:.6f}°C")
    log_message(log_file, f"RMSE = {rmse_log:.6f}°C")
    
    # Степенева
    popt_pow, _ = curve_fit(power_func, depth, temp, p0=[1400, -0.1])
    y_pred_pow = power_func(depth, *popt_pow)
    r2_pow, mae_pow, rmse_pow = calculate_metrics(temp, y_pred_pow)
    
    log_message(log_file, "\nСтепенева модель:")
    log_message(log_file, f"y = {popt_pow[0]:.6f} * (x+1)^{popt_pow[1]:.6f}")
    log_message(log_file, f"R² = {r2_pow:.6f}")
    log_message(log_file, f"MAE = {mae_pow:.6f}°C")
    log_message(log_file, f"RMSE = {rmse_pow:.6f}°C")
    
    # Створення графіку
    plt.figure(figsize=(12, 8))
    plt.scatter(depth, temp, color='red', label='Експериментальні дані')
    
    # Побудова кривих апроксимації
    for name, params in models.items():
        plt.plot(depth_fit, np.polyval(params['coefficients'], depth_fit), '--', 
                label=f'{name} (R² = {params["r2"]:.4f})')
    
    plt.plot(depth_fit, exp_func(depth_fit, *popt_exp), '--', 
            label=f'Експоненціальна (R² = {r2_exp:.4f})')
    plt.plot(depth_fit, log_func(depth_fit, *popt_log), '--', 
            label=f'Логарифмічна (R² = {r2_log:.4f})')
    plt.plot(depth_fit, power_func(depth_fit, *popt_pow), '--', 
            label=f'Степенева (R² = {r2_pow:.4f})')
    
    plt.xlabel('Глибина шару (мм)')
    plt.ylabel('Температура (°C)')
    plt.title('Розподіл температури по глибині шару агломераційної шихти')
    plt.legend()
    plt.grid(True)
    
    # Зберігаємо графік
    plt.savefig('temperature_distribution_extended.png')
    plt.close()
    
    log_message(log_file, "\n=== Аналіз завершено ===")
    log_message(log_file, "Графік збережено як 'temperature_distribution_extended.png'")
    log_file.close()

if __name__ == "__main__":
    main()
