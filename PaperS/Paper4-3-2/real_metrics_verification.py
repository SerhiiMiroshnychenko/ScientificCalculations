#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для точної перевірки метрик класифікації з реальними матрицями плутанини
Використовує справжні дані з confusion matrix для обрахунку всіх метрик
"""

import numpy as np
from typing import Dict, Tuple

def calculate_metrics_from_real_confusion_matrix(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    """
    Обраховує всі метрики класифікації з реальної матриці плутанини
    
    Args:
        tp: True Positives
        fp: False Positives  
        fn: False Negatives
        tn: True Negatives
    
    Returns:
        Dict з усіма метриками
    """
    
    # Основні метрики
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Додаткові метрики
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'specificity': specificity,
        'npv': npv,
        'total_samples': total
    }

def verify_with_real_data(model_name: str, tp: int, fp: int, fn: int, tn: int, 
                         given_metrics: Dict[str, float]) -> Dict[str, any]:
    """
    Перевіряє метрики з реальними даними матриці плутанини
    """
    
    # Обраховуємо метрики з реальної матриці
    calculated_metrics = calculate_metrics_from_real_confusion_matrix(tp, fp, fn, tn)
    
    # Порівнюємо з наданими метриками
    results = {
        'model_name': model_name,
        'confusion_matrix': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn},
        'calculated_metrics': calculated_metrics,
        'given_metrics': given_metrics,
        'differences': {},
        'consistent': {}
    }
    
    # Обраховуємо різниці та перевіряємо узгодженість
    tolerance = 0.0001  # Дуже мала похибка для точних обрахунків
    
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        if metric in given_metrics:
            calc_value = calculated_metrics[metric]
            given_value = given_metrics[metric]
            difference = abs(calc_value - given_value)
            
            results['differences'][metric] = difference
            results['consistent'][metric] = difference <= tolerance
    
    return results

def print_detailed_verification(results: Dict[str, any]):
    """
    Виводить детальні результати перевірки
    """
    model_name = results['model_name']
    cm = results['confusion_matrix']
    calc_metrics = results['calculated_metrics']
    given_metrics = results['given_metrics']
    
    print(f"\n{'='*70}")
    print(f"ТОЧНА ПЕРЕВІРКА: {model_name}")
    print(f"{'='*70}")
    
    # Матриця плутанини
    print(f"\nРеальна матриця плутанини:")
    print(f"                 Predicted")
    print(f"                 0      1")
    print(f"Actual    0   {cm['tn']:>5}  {cm['fp']:>5}")
    print(f"          1   {cm['fn']:>5}  {cm['tp']:>5}")
    print(f"\nЗагальна кількість зразків: {calc_metrics['total_samples']}")
    
    # Детальне порівняння метрик
    print(f"\nДетальне порівняння метрик:")
    print(f"{'Метрика':<12} | {'Надано':<10} | {'Обраховано':<12} | {'Різниця':<12} | {'Статус'}")
    print(f"{'-'*75}")
    
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        if metric in given_metrics:
            given = given_metrics[metric]
            calculated = calc_metrics[metric]
            difference = results['differences'][metric]
            consistent = results['consistent'][metric]
            status = "✓ ТОЧНО" if consistent else "✗ ПОМИЛКА"
            
            print(f"{metric:<12} | {given:<10.6f} | {calculated:<12.6f} | {difference:<12.8f} | {status}")
    
    # Додаткові обраховані метрики
    print(f"\nДодаткові метрики:")
    print(f"Specificity (TNR): {calc_metrics['specificity']:.6f}")
    print(f"NPV:               {calc_metrics['npv']:.6f}")

def main():
    """
    Основна функція з реальними даними матриць плутанини
    """
    
    # Реальні дані з матриць плутанини
    real_data = {
        # Dataset 1
        "Dataset 1 - Базова модель": {
            'tp': 15933, 'fp': 496, 'fn': 1352, 'tn': 8258,
            'given_metrics': {
                'accuracy': 0.9290,
                'precision': 0.9218,
                'recall': 0.9698,
                'f1_score': 0.9452
            }
        },
        "Dataset 1 - Оптимізована модель": {
            'tp': 15806, 'fp': 623, 'fn': 1171, 'tn': 8439,
            'given_metrics': {
                'accuracy': 0.9311,
                'precision': 0.9310,
                'recall': 0.9621,
                'f1_score': 0.9463
            }
        },
        "Dataset 1 - Найкраща модель": {
            'tp': 15813, 'fp': 616, 'fn': 1163, 'tn': 8417,
            'given_metrics': {
                'accuracy': 0.9317,
                'precision': 0.9315,
                'recall': 0.9625,
                'f1_score': 0.9467
            }
        },
        
        # Dataset 2
        "Dataset 2 - Базова модель": {
            'tp': 15953, 'fp': 476, 'fn': 1364, 'tn': 8246,
            'given_metrics': {
                'accuracy': 0.9293,
                'precision': 0.9212,
                'recall': 0.9710,
                'f1_score': 0.9455
            }
        },
        "Dataset 2 - Оптимізована модель": {
            'tp': 15957, 'fp': 472, 'fn': 1337, 'tn': 8273,
            'given_metrics': {
                'accuracy': 0.9305,
                'precision': 0.9227,
                'recall': 0.9713,
                'f1_score': 0.9464
            }
        },
        "Dataset 2 - Найкраща модель": {
            'tp': 15973, 'fp': 456, 'fn': 1334, 'tn': 8276,
            'given_metrics': {
                'accuracy': 0.9313,
                'precision': 0.9229,
                'recall': 0.9722,
                'f1_score': 0.9469
            }
        }
    }
    
    # Перевіряємо кожну модель
    all_results = []
    overall_consistent = True
    
    for model_name, data in real_data.items():
        results = verify_with_real_data(
            model_name,
            data['tp'], data['fp'], data['fn'], data['tn'],
            data['given_metrics']
        )
        
        all_results.append(results)
        print_detailed_verification(results)
        
        # Перевіряємо загальну узгодженість
        model_consistent = all(results['consistent'].values())
        if not model_consistent:
            overall_consistent = False
    
    # Загальний аналіз
    print(f"\n{'='*70}")
    print("ЗАГАЛЬНИЙ АНАЛІЗ РЕЗУЛЬТАТІВ")
    print(f"{'='*70}")
    
    if overall_consistent:
        print("✓ ВСІ МЕТРИКИ МАТЕМАТИЧНО ТОЧНІ")
        print("✓ Надані показники повністю відповідають реальним матрицям плутанини")
        print("✓ Обрахунки F1-score, precision, recall та accuracy коректні")
    else:
        print("✗ Виявлено розбіжності між наданими та обрахованими метриками")
        
        # Детальний аналіз помилок
        print("\nДетальний аналіз розбіжностей:")
        for results in all_results:
            inconsistent_metrics = [metric for metric, consistent in results['consistent'].items() if not consistent]
            if inconsistent_metrics:
                print(f"- {results['model_name']}: {', '.join(inconsistent_metrics)}")
    
    # Статистика по датасетах
    print(f"\nСтатистика по датасетах:")
    
    dataset1_results = [r for r in all_results if "Dataset 1" in r['model_name']]
    dataset2_results = [r for r in all_results if "Dataset 2" in r['model_name']]
    
    for dataset_name, dataset_results in [("Dataset 1", dataset1_results), ("Dataset 2", dataset2_results)]:
        print(f"\n{dataset_name}:")
        for results in dataset_results:
            cm = results['confusion_matrix']
            calc = results['calculated_metrics']
            total = calc['total_samples']
            pos_ratio = (cm['tp'] + cm['fn']) / total
            print(f"  {results['model_name'].split(' - ')[1]}: {total} зразків, {pos_ratio:.1%} позитивних")

if __name__ == "__main__":
    main()
