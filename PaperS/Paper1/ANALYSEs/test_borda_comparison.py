"""
Скрипт для порівняння двох реалізацій модифікованого методу Борда.

Порівнюються:
1. Оригінальна реалізація (pandas rank з method="max")
2. Модифікована реалізація (точно за формулою (1))

Формула (1): Rⱼ(fᵢ) = |{fₗ∈F | Sⱼ(fₗ) ≥ Sⱼ(fᵢ)}|
Формула (3): R̄(fᵢ) = (1/k) Σⱼ₌₁ᵏ Rⱼ(fᵢ)
"""

import pandas as pd
import numpy as np


# === ОРИГІНАЛЬНА РЕАЛІЗАЦІЯ ===
def modified_borda_mean_rank_original(method_scores: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Оригінальна реалізація з використанням pandas rank.
    """
    df = pd.DataFrame(method_scores)
    ranks = df.rank(ascending=False, method="max")
    ranks["mean_rank"] = ranks.mean(axis=1)
    ranks.insert(0, "feature", ranks.index)
    return ranks.sort_values("mean_rank").reset_index(drop=True)


# === МОДИФІКОВАНА РЕАЛІЗАЦІЯ (точно за формулою) ===
def modified_borda_mean_rank_formula(method_scores: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Модифікована реалізація, що точно відповідає формулі (1).
    
    Формула (1): Rⱼ(fᵢ) = |{fₗ∈F | Sⱼ(fₗ) ≥ Sⱼ(fᵢ)}|
    """
    df = pd.DataFrame(method_scores)
    # Формула (1): кількість елементів >= поточного значення
    ranks = df.apply(lambda col: col.apply(lambda x: (col >= x).sum()))
    # Формула (3): середнє арифметичне рангів
    ranks["mean_rank"] = ranks.mean(axis=1)
    ranks.insert(0, "feature", ranks.index)
    return ranks.sort_values("mean_rank").reset_index(drop=True)


def create_test_data() -> dict[str, pd.Series]:
    """
    Створює тестові дані, що імітують результати різних методів оцінки значущості ознак.
    """
    features = [
        'order_messages',      # Кількість повідомлень
        'partner_success_rate', # Успішність партнера
        'order_amount',        # Сума замовлення
        'delivery_time',       # Час доставки
        'product_count',       # Кількість товарів
        'discount_rate',       # Розмір знижки
    ]
    
    # Симуляція результатів різних методів (AUC, MI, dCor, LogReg, DecTree)
    test_data = {
        'AUC': pd.Series({
            'order_messages': 0.92,
            'partner_success_rate': 0.88,
            'order_amount': 0.75,
            'delivery_time': 0.65,
            'product_count': 0.55,
            'discount_rate': 0.45,
        }),
        'MI': pd.Series({
            'order_messages': 0.85,
            'partner_success_rate': 0.90,
            'order_amount': 0.70,
            'delivery_time': 0.60,
            'product_count': 0.50,
            'discount_rate': 0.40,
        }),
        'dCor': pd.Series({
            'order_messages': 0.88,
            'partner_success_rate': 0.82,
            'order_amount': 0.78,
            'delivery_time': 0.55,
            'product_count': 0.48,
            'discount_rate': 0.42,
        }),
        'LogReg': pd.Series({
            'order_messages': 0.95,
            'partner_success_rate': 0.80,
            'order_amount': 0.72,
            'delivery_time': 0.58,
            'product_count': 0.52,
            'discount_rate': 0.38,
        }),
        'DecTree': pd.Series({
            'order_messages': 0.90,
            'partner_success_rate': 0.85,
            'order_amount': 0.68,
            'delivery_time': 0.62,
            'product_count': 0.45,
            'discount_rate': 0.35,
        }),
    }
    return test_data


def create_test_data_with_ties() -> dict[str, pd.Series]:
    """
    Створює тестові дані з однаковими значеннями (ties) для перевірки edge cases.
    """
    test_data = {
        'Method_A': pd.Series({
            'feature_1': 0.90,
            'feature_2': 0.90,  # tie з feature_1
            'feature_3': 0.70,
            'feature_4': 0.70,  # tie з feature_3
            'feature_5': 0.50,
        }),
        'Method_B': pd.Series({
            'feature_1': 0.85,
            'feature_2': 0.80,
            'feature_3': 0.80,  # tie з feature_2
            'feature_4': 0.60,
            'feature_5': 0.60,  # tie з feature_4
        }),
    }
    return test_data


def compare_implementations(test_data: dict[str, pd.Series], test_name: str):
    """
    Порівнює результати двох реалізацій на заданих тестових даних.
    """
    print(f"\n{'='*60}")
    print(f"ТЕСТ: {test_name}")
    print('='*60)
    
    # Вхідні дані
    print("\n📊 Вхідні дані (значущість ознак за методами):")
    df_input = pd.DataFrame(test_data)
    print(df_input.to_string())
    
    # Оригінальна реалізація
    result_original = modified_borda_mean_rank_original(test_data)
    print("\n📈 Результат ОРИГІНАЛЬНОЇ реалізації (pandas rank):")
    print(result_original.to_string(index=False))
    
    # Модифікована реалізація
    result_formula = modified_borda_mean_rank_formula(test_data)
    print("\n📉 Результат МОДИФІКОВАНОЇ реалізації (за формулою):")
    print(result_formula.to_string(index=False))
    
    # Порівняння
    print("\n🔍 ПОРІВНЯННЯ РЕЗУЛЬТАТІВ:")
    
    # Перевірка чи ранжування однакове
    ranking_same = list(result_original['feature']) == list(result_formula['feature'])
    print(f"   Порядок ознак однаковий: {'✅ ТАК' if ranking_same else '❌ НІ'}")
    
    # Перевірка чи mean_rank однаковий
    mean_rank_diff = abs(result_original['mean_rank'].values - result_formula['mean_rank'].values)
    max_diff = mean_rank_diff.max()
    print(f"   Максимальна різниця mean_rank: {max_diff:.6f}")
    
    if max_diff < 0.0001:
        print("   ✅ Результати ІДЕНТИЧНІ")
    else:
        print("   ⚠️ Результати ВІДРІЗНЯЮТЬСЯ")
        print("\n   Детальне порівняння:")
        comparison = pd.DataFrame({
            'feature': result_original['feature'],
            'mean_rank_original': result_original['mean_rank'],
            'mean_rank_formula': result_formula['mean_rank'],
            'difference': mean_rank_diff
        })
        print(comparison.to_string(index=False))
    
    return ranking_same, max_diff


def demonstrate_formula_step_by_step():
    """
    Покрокова демонстрація обчислення за формулою (1).
    """
    print("\n" + "="*60)
    print("ПОКРОКОВА ДЕМОНСТРАЦІЯ ФОРМУЛИ (1)")
    print("="*60)
    
    # Простий приклад
    values = pd.Series({
        'A': 0.9,
        'B': 0.7,
        'C': 0.7,
        'D': 0.5,
    })
    
    print("\n📊 Значення: ", dict(values))
    print("\n🔢 Обчислення рангів за формулою (1):")
    print("   Rⱼ(fᵢ) = |{fₗ∈F | Sⱼ(fₗ) ≥ Sⱼ(fᵢ)}|")
    print("   (кількість елементів >= поточного значення)\n")
    
    for feature, value in values.items():
        count = (values >= value).sum()
        elements_ge = [f for f, v in values.items() if v >= value]
        print(f"   {feature} (значення={value}): count(x >= {value}) = {count}")
        print(f"      Елементи >= {value}: {elements_ge}")
    
    print("\n📈 Результуючі ранги:")
    ranks = values.apply(lambda x: (values >= x).sum())
    for feature, rank in ranks.items():
        print(f"   {feature}: ранг = {rank}")


def main():
    """
    Головна функція для запуску всіх тестів.
    """
    print("╔════════════════════════════════════════════════════════════╗")
    print("║  ПОРІВНЯННЯ РЕАЛІЗАЦІЙ МОДИФІКОВАНОГО МЕТОДУ БОРДА        ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    # Тест 1: Базові дані без ties
    test_data_basic = create_test_data()
    same1, diff1 = compare_implementations(test_data_basic, "Базові дані (без ties)")
    
    # Тест 2: Дані з ties
    test_data_ties = create_test_data_with_ties()
    same2, diff2 = compare_implementations(test_data_ties, "Дані з однаковими значеннями (ties)")
    
    # Покрокова демонстрація
    demonstrate_formula_step_by_step()
    
    # Підсумок
    print("\n" + "="*60)
    print("ПІДСУМОК")
    print("="*60)
    print(f"\n✅ Тест 1 (базові дані): {'Пройдено' if same1 else 'Не пройдено'}")
    print(f"✅ Тест 2 (дані з ties): {'Пройдено' if same2 else 'Не пройдено'}")
    
    if same1 and same2:
        print("\n🎉 Обидві реалізації дають ОДНАКОВІ результати!")
        print("   Модифікована версія точніше відповідає математичній формулі,")
        print("   але практичний результат ідентичний.")
    else:
        print("\n⚠️ Реалізації дають РІЗНІ результати в деяких випадках.")
        print("   Рекомендується використовувати модифіковану версію")
        print("   для точної відповідності формулі (1).")


if __name__ == "__main__":
    main()
