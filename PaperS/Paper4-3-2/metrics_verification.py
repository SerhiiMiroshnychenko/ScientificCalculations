#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для перевірки достовірності метрик класифікації
Обраховує accuracy, precision, recall, F1-score за формулами
та порівнює з наданими показниками
"""

import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

def calculate_metrics_from_confusion_matrix(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    """
    Обраховує метрики класифікації з матриці плутанини
    
    Args:
        tp: True Positives (правильно класифіковані позитивні)
        fp: False Positives (помилково класифіковані як позитивні)
        fn: False Negatives (пропущені позитивні)
        tn: True Negatives (правильно класифіковані негативні)
    
    Returns:
        Dict з метриками: accuracy, precision, recall, f1_score
    """
    
    # Accuracy = (TP + TN) / (TP + TN + FP + FN)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1-score = 2 * (Precision * Recall) / (Precision + Recall)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

def reverse_engineer_confusion_matrix(accuracy: float, precision: float, recall: float, 
                                    total_samples: int = 10000, positive_ratio: float = 0.3) -> Tuple[int, int, int, int]:
    """
    Зворотна інженерія матриці плутанини з відомих метрик
    
    Args:
        accuracy: Точність моделі
        precision: Прецизійність
        recall: Повнота
        total_samples: Загальна кількість зразків (припущення)
        positive_ratio: Частка позитивних зразків (припущення)
    
    Returns:
        Tuple (tp, fp, fn, tn)
    """
    
    # Припускаємо кількість позитивних та негативних зразків
    total_positives = int(total_samples * positive_ratio)
    total_negatives = total_samples - total_positives
    
    # З recall: TP = recall * (TP + FN) = recall * total_positives
    tp = int(recall * total_positives)
    fn = total_positives - tp
    
    # З precision: TP = precision * (TP + FP), тому FP = TP/precision - TP
    if precision > 0:
        fp = int(tp / precision - tp)
    else:
        fp = 0
    
    # TN з accuracy: accuracy = (TP + TN) / total_samples
    tn = int(accuracy * total_samples - tp)
    
    # Коригуємо, якщо TN виходить за межі
    if tn > total_negatives:
        tn = total_negatives
        fp = total_negatives - tn
    elif tn < 0:
        tn = 0
        fp = total_negatives
    
    return tp, fp, fn, tn

def verify_metrics_consistency(given_metrics: Dict[str, float], tolerance: float = 0.001) -> Dict[str, bool]:
    """
    Перевіряє узгодженість наданих метрик
    
    Args:
        given_metrics: Словник з наданими метриками
        tolerance: Допустима похибка
    
    Returns:
        Словник з результатами перевірки
    """
    
    # Отримуємо матрицю плутанини
    tp, fp, fn, tn = reverse_engineer_confusion_matrix(
        given_metrics['accuracy'], 
        given_metrics['precision'], 
        given_metrics['recall']
    )
    
    # Обраховуємо метрики з матриці плутанини
    calculated_metrics = calculate_metrics_from_confusion_matrix(tp, fp, fn, tn)
    
    # Перевіряємо відповідність
    results = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        if metric in given_metrics:
            diff = abs(calculated_metrics[metric] - given_metrics[metric])
            results[f'{metric}_consistent'] = diff <= tolerance
            results[f'{metric}_difference'] = diff
    
    # Додаємо інформацію про матрицю плутанини
    results['confusion_matrix'] = {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}
    results['calculated_metrics'] = calculated_metrics
    
    return results

def print_verification_results(model_name: str, given_metrics: Dict[str, float], 
                             verification_results: Dict[str, any]):
    """
    Виводить результати перевірки у зрозумілому форматі
    """
    print(f"\n{'='*60}")
    print(f"ПЕРЕВІРКА МЕТРИК: {model_name}")
    print(f"{'='*60}")
    
    # Виводимо матрицю плутанини
    cm = verification_results['confusion_matrix']
    print(f"\nМатриця плутанини (припущення):")
    print(f"TP: {cm['tp']:>6} | FP: {cm['fp']:>6}")
    print(f"FN: {cm['fn']:>6} | TN: {cm['tn']:>6}")
    
    # Порівнюємо метрики
    print(f"\nПорівняння метрик:")
    print(f"{'Метрика':<12} | {'Надано':<8} | {'Обраховано':<10} | {'Різниця':<8} | {'Узгоджено'}")
    print(f"{'-'*65}")
    
    calc_metrics = verification_results['calculated_metrics']
    
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        if metric in given_metrics:
            given = given_metrics[metric]
            calculated = calc_metrics[metric]
            difference = verification_results[f'{metric}_difference']
            consistent = verification_results[f'{metric}_consistent']
            status = "✓" if consistent else "✗"
            
            print(f"{metric:<12} | {given:<8.4f} | {calculated:<10.4f} | {difference:<8.6f} | {status}")

def main():
    """
    Основна функція для перевірки всіх наданих показників
    """
    
    # Дані для Датасету 1
    dataset1_models = {
        "Датасет 1 - Базова модель": {
            'accuracy': 0.9290,
            'precision': 0.9218,
            'recall': 0.9698,
            'f1_score': 0.9452
        },
        "Датасет 1 - Оптимізована модель": {
            'accuracy': 0.9311,
            'precision': 0.9310,
            'recall': 0.9621,
            'f1_score': 0.9463
        },
        "Датасет 1 - Найкраща модель": {
            'accuracy': 0.9317,
            'precision': 0.9315,
            'recall': 0.9625,
            'f1_score': 0.9467
        }
    }
    
    # Дані для Датасету 2
    dataset2_models = {
        "Датасет 2 - Базова модель": {
            'accuracy': 0.9293,
            'precision': 0.9212,
            'recall': 0.9710,
            'f1_score': 0.9455
        },
        "Датасет 2 - Оптимізована модель": {
            'accuracy': 0.9305,
            'precision': 0.9227,
            'recall': 0.9713,
            'f1_score': 0.9464
        },
        "Датасет 2 - Найкраща модель": {
            'accuracy': 0.9313,
            'precision': 0.9229,
            'recall': 0.9722,
            'f1_score': 0.9469
        }
    }
    
    # Об'єднуємо всі моделі
    all_models = {**dataset1_models, **dataset2_models}
    
    # Перевіряємо кожну модель
    overall_consistent = True
    
    for model_name, metrics in all_models.items():
        verification_results = verify_metrics_consistency(metrics, tolerance=0.002)
        print_verification_results(model_name, metrics, verification_results)
        
        # Перевіряємо загальну узгодженість
        model_consistent = all([
            verification_results.get(f'{metric}_consistent', True) 
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']
            if metric in metrics
        ])
        
        if not model_consistent:
            overall_consistent = False
    
    # Загальний висновок
    print(f"\n{'='*60}")
    print("ЗАГАЛЬНИЙ ВИСНОВОК")
    print(f"{'='*60}")
    
    if overall_consistent:
        print("✓ Всі надані метрики математично узгоджені")
        print("✓ F1-score правильно обраховані як гармонійне середнє precision та recall")
        print("✓ Показники виглядають достовірними")
    else:
        print("✗ Виявлено невідповідності у деяких метриках")
        print("✗ Рекомендується перевірити вихідні дані")
    
    print(f"\nПримітка: Перевірка базується на припущеннях щодо розподілу класів")
    print("та загальної кількості зразків. Невеликі розбіжності можуть бути")
    print("пов'язані з округленням або різними методами обрахунку.")

if __name__ == "__main__":
    main()
