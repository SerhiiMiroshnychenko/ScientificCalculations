#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

"""
Тема: Виявлення та ідентифікації грубих промахів у рядах спостереження.

Мета: ознайомлення здобувачів вищої освіти з методами виявлення та
ідентифікації грубих промахів у рядах спостережень, що
використовуються під час аналізу великих даних.
"""

# Вхідні дані з слайдів
# Три серії спостережень x1, x2, x3
x1 = np.array([1.239, 1.206, 1.231, 1.226, 1.211, 1.234, 1.279, 1.236, 1.251, 1.325, 1.331, 1.209, 1.213])
x2 = np.array([1.244, 1.244, 1.263, 1.262, 1.221, 1.232, 1.223, 1.234, 1.245, 1.226, 1.327, 1.248, 1.209])
x3 = np.array([1.223, 1.252, 1.267, 1.262, 1.237, 1.239, 1.253, 1.232, 1.217, 1.202, 1.327, 1.266, 1.214])

def print_header(text):
    """Функція виведення заголовка секції"""
    print("\n" + "="*50)
    print(text)
    print("="*50)

def calculate_statistics(data):
    """Розрахунок статистичних характеристик для серії даних"""
    n = len(data)
    x_mean = np.mean(data)  # середнє значення
    D = np.sum((data - x_mean)**2) / (n - 1)  # дисперсія
    sigma = np.sqrt(D)  # середньоквадратичне відхилення
    
    return n, x_mean, D, sigma

def calculate_gamma(data, x_mean, sigma):
    """Розрахунок коефіцієнтів γ1 та γ2 для виявлення викидів"""
    gamma1 = (np.max(data) - x_mean) / sigma
    gamma2 = (x_mean - np.min(data)) / sigma
    
    return gamma1, gamma2

def check_outliers(data, gamma_p=2.18):
    """Перевірка наявності викидів у серії даних"""
    n, x_mean, D, sigma = calculate_statistics(data)
    gamma1, gamma2 = calculate_gamma(data, x_mean, sigma)
    
    has_outliers = False
    outlier_indices = []
    
    print(f"γ1 = {gamma1:.3f}, γ2 = {gamma2:.3f}, γp = {gamma_p}")
    
    if gamma1 > gamma_p:
        has_outliers = True
        max_index = np.argmax(data)
        outlier_indices.append(max_index)
        print(f"γ1 > γp: Викид у максимальному значенні {data[max_index]}")
    else:
        print(f"γ1 <= γp: Немає викиду у максимальному значенні")
        
    if gamma2 > gamma_p:
        has_outliers = True
        min_index = np.argmin(data)
        outlier_indices.append(min_index)
        print(f"γ2 > γp: Викид у мінімальному значенні {data[min_index]}")
    else:
        print(f"γ2 <= γp: Немає викиду у мінімальному значенні")
    
    return has_outliers, outlier_indices

def remove_outliers(data, outlier_indices):
    """Видалення викидів з серії даних"""
    return np.delete(data, outlier_indices)

def weighted_average(means, sigmas, ns):
    """Розрахунок середньозваженого значення"""
    weights = 1 / (sigmas**2)
    weighted_mean = np.sum(means * weights) / np.sum(weights)
    return weighted_mean, weights

def confidence_interval(weighted_mean, sigmas, ns, confidence=0.95, df=None):
    """Розрахунок довірчого інтервалу"""
    if df is None:
        # Визначення числа ступенів свободи як m(n-1), де m - кількість серій, n - найменша кількість спостережень
        m = len(sigmas)
        n_min = np.min(ns)
        df = m * (n_min - 1)
    
    # Коефіцієнт Стьюдента для заданої довірчої ймовірності
    t_p = 2.04  # P=0.95, f=33 згідно зі слайдами
    
    # Розрахунок стандартного відхилення середньозваженого
    sigma_weighted = np.sqrt(np.sum((1/sigmas)**2)**(-1))
    
    # Довірчий інтервал
    delta = t_p * sigma_weighted
    lower_bound = weighted_mean - delta
    upper_bound = weighted_mean + delta
    
    return lower_bound, upper_bound, delta, sigma_weighted, df

def main():
    """Основна функція програми"""
    print_header("ПОЧАТКОВІ ДАНІ")
    print(f"Серія x1: {x1}")
    print(f"Серія x2: {x2}")
    print(f"Серія x3: {x3}")
    
    # 1. Перевірка наявності грубих викидів
    print_header("ПЕРЕВІРКА НАЯВНОСТІ ГРУБИХ ВИКИДІВ")
    
    series = {"x1": x1, "x2": x2, "x3": x3}
    cleaned_series = {}
    stats = {}
    
    for name, data in series.items():
        print(f"\nАналіз серії {name}:")
        n, x_mean, D, sigma = calculate_statistics(data)
        stats[name] = {"n": n, "mean": x_mean, "D": D, "sigma": sigma}
        
        print(f"Середнє: {x_mean:.3f}")
        print(f"Дисперсія: {D:.6f}")
        print(f"С.к.в.: {sigma:.3f}")
        
        has_outliers, outlier_indices = check_outliers(data)
        
        if has_outliers:
            cleaned_data = remove_outliers(data, outlier_indices)
            print(f"Видалені значення: {data[outlier_indices]}")
            print(f"Очищена серія {name}: {cleaned_data}")
            
            # Перерахунок статистик для очищених даних
            n, x_mean, D, sigma = calculate_statistics(cleaned_data)
            stats[name] = {"n": n, "mean": x_mean, "D": D, "sigma": sigma}
            
            print(f"Нове середнє: {x_mean:.3f}")
            print(f"Нова дисперсія: {D:.6f}")
            print(f"Нове с.к.в.: {sigma:.3f}")
            
            # Повторна перевірка на викиди
            has_outliers_again, outlier_indices_again = check_outliers(cleaned_data)
            
            if has_outliers_again:
                cleaned_data = remove_outliers(cleaned_data, outlier_indices_again)
                print(f"Видалені значення: {cleaned_data[outlier_indices_again]}")
                print(f"Очищена серія {name} (2 ітерація): {cleaned_data}")
                
                # Перерахунок статистик для очищених даних
                n, x_mean, D, sigma = calculate_statistics(cleaned_data)
                stats[name] = {"n": n, "mean": x_mean, "D": D, "sigma": sigma}
                
                print(f"Нове середнє (2 ітерація): {x_mean:.3f}")
                print(f"Нова дисперсія (2 ітерація): {D:.6f}")
                print(f"Нове с.к.в. (2 ітерація): {sigma:.3f}")
            
            cleaned_series[name] = cleaned_data
        else:
            print(f"Серія {name} не містить викидів")
            cleaned_series[name] = data
    
    # 2. Визначення результату нерівноточного спостереження
    print_header("ВИЗНАЧЕННЯ РЕЗУЛЬТАТУ НЕРІВНОТОЧНОГО СПОСТЕРЕЖЕННЯ")
    
    means = np.array([stats["x1"]["mean"], stats["x2"]["mean"], stats["x3"]["mean"]])
    sigmas = np.array([stats["x1"]["sigma"], stats["x2"]["sigma"], stats["x3"]["sigma"]])
    ns = np.array([stats["x1"]["n"], stats["x2"]["n"], stats["x3"]["n"]])
    
    print(f"Середні значення серій: {means}")
    print(f"С.к.в. результатів спостережень серій: {sigmas}")
    print(f"Число результатів спостережень серій: {ns}")
    
    # Розрахунок ваг серій
    weights = 1 / sigmas**2
    print(f"Ваги результатів серій: {weights}")
    
    # Розрахунок середньозваженого
    weighted_mean, _ = weighted_average(means, sigmas, ns)
    print(f"Середнє зважене результатів в серіях: {weighted_mean:.3f}")
    
    # Розрахунок c.к.в. результатів у серіях
    sigma_x1 = sigmas[0] / np.sqrt(ns[0])
    sigma_x2 = sigmas[1] / np.sqrt(ns[1])
    sigma_x3 = sigmas[2] / np.sqrt(ns[2])
    
    print(f"С.к.в. результатів у серіях:")
    print(f"σ_x1 = {sigma_x1:.3f}")
    print(f"σ_x2 = {sigma_x2:.3f}")
    print(f"σ_x3 = {sigma_x3:.3f}")
    
    # С.к.в. середнього зваженого
    sigma_weighted = np.sqrt(1 / np.sum(1 / np.array([sigma_x1**2, sigma_x2**2, sigma_x3**2])))
    print(f"С.к.в. середнього зваженого: σ_x = {sigma_weighted:.3f}")
    
    # Інтервальна оцінка нерівноточного результату
    # Число ступенів свободи
    m = 3  # кількість серій
    n_min = np.min(ns)
    f = m * (n_min - 1)
    
    print(f"Число ступенів свободи: f = {f}")
    
    # Коефіцієнт Стьюдента
    t_p = 2.04  # P=0.95, f=33
    print(f"Коефіцієнт Стьюдента: t_p,f = {t_p}")
    
    # Довірча межа випадкової величини
    delta = t_p * sigma_weighted
    print(f"Довірча межа випадкової величини: Δ = {delta:.3f}")
    
    # Довірчий інтервал
    lower_bound = weighted_mean - delta
    upper_bound = weighted_mean + delta
    
    print(f"Довірчий інтервал: x - Δ = {lower_bound:.3f}, x + Δ = {upper_bound:.3f}")
    print(f"Інтервальна оцінка нерівноточного результату: x ∈ [{lower_bound:.3f} : {upper_bound:.3f}], P=0.95, f={f}")

if __name__ == "__main__":
    main()
