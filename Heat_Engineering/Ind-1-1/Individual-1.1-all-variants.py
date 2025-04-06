#!/usr/bin/env python
# coding: utf-8

"""
Тепловий і гідравлічний розрахунок водоохолоджуваної панелі ДСП
Розрахунок всіх варіантів завдання з таблиці 1.1
"""
import math

# Масиви даних для всіх варіантів із таблиці 1.1
# Формат: [тепловий потік (кВт/м²), зовнішній діаметр (мм), внутрішній діаметр (мм), 
#          кількість поворотів на 90°, кількість поворотів на 180°, 
#          температура води на вході (°С), тиск води в цеху (МПа), матеріал труби]

variants_data = [
    # Номер варіанту 1
    [155, 76, 56, 2, 10, 20, 0.45, "Сталь Ст20"],
    # Номер варіанту 2
    [280, 89, 65, 2, 6, 25, 0.35, "Мідь"],
    # Номер варіанту 3
    [145, 73, 53, 2, 10, 25, 0.27, "Сталь Ст20"],
    # Номер варіанту 4
    [170, 76, 56, 2, 8, 20, 0.34, "Сталь Ст20"],
    # Номер варіанту 5
    [160, 89, 65, 3, 10, 15, 0.40, "Сталь Ст20"],
    # Номер варіанту 6
    [250, 73, 53, 3, 12, 18, 0.26, "Мідь"],
    # Номер варіанту 7
    [305, 76, 52, 2, 5, 22, 0.36, "Мідь"],
    # Номер варіанту 8
    [120, 89, 69, 2, 8, 24, 0.32, "Сталь Ст20"],
    # Номер варіанту 9
    [175, 73, 49, 4, 12, 23, 0.28, "Сталь Ст20"],
    # Номер варіанту 10
    [230, 76, 56, 2, 8, 20, 0.30, "Мідь"],
    # Номер варіанту 11
    [165, 89, 65, 2, 6, 19, 0.25, "Сталь Ст20"],
    # Номер варіанту 12
    [150, 73, 53, 3, 12, 18, 0.35, "Сталь Ст20"],
    # Номер варіанту 13
    [140, 76, 54, 2, 10, 25, 0.27, "Сталь Ст20"],
    # Номер варіанту 14
    [235, 89, 69, 4, 12, 25, 0.34, "Мідь"],
    # Номер варіанту 15
    [255, 73, 49, 2, 8, 20, 0.40, "Мідь"],
    # Номер варіанту 16
    [200, 76, 56, 2, 6, 20, 0.26, "Сталь Ст20"],
    # Номер варіанту 17
    [185, 89, 65, 3, 10, 15, 0.35, "Сталь Ст20"],
    # Номер варіанту 18
    [285, 73, 53, 2, 8, 18, 0.32, "Мідь"],
    # Номер варіанту 19
    [140, 60, 44, 4, 10, 22, 0.28, "Сталь Ст20"],
    # Номер варіанту 20
    [210, 73, 49, 3, 12, 24, 0.30, "Сталь Ст20"],
    # Номер варіанту 21
    [170, 76, 56, 2, 5, 23, 0.25, "Сталь Ст20"],
    # Номер варіанту 22
    [260, 89, 65, 2, 8, 20, 0.35, "Мідь"],
    # Номер варіанту 23
    [155, 73, 53, 3, 12, 19, 0.32, "Сталь Ст20"],
    # Номер варіанту 24
    [295, 60, 44, 2, 8, 18, 0.34, "Мідь"],
    # Номер варіанту 25
    [200, 89, 65, 4, 6, 25, 0.39, "Сталь Ст20"],
]

# Константи з умов розрахунку
t1 = 75  # температура стінки водяного каналу панелі, °С
t_out = 55  # температура води на виході, °С
dP_g = 0  # статичні втрати тиску, Па

# Фізичні константи
lambda_st20 = 39  # теплопровідність сталі Ст20, Вт/(м·К)
lambda_cu = 370  # теплопровідність міді, Вт/(м·К)
rho = 1000  # щільність води, кг/м³
C = 4200  # теплоємність води, Дж/(кг·К)
nu = 1e-6  # коефіцієнт кінематичної в'язкості води, м²/с
mu_fr = 0.045  # коефіцієнт тертя води в трубі
xi_lr1 = 0.22  # коефіцієнт місцевого опору поворот на 90°
xi_lr2 = 0.31  # коефіцієнт місцевого опору поворот на 180°

# Функція для розрахунку всіх параметрів для варіанту
def calculate_variant(variant_num, variant_data):
    # Розпаковуємо дані варіанту
    q, d_mm, d1_mm, n_90, n_180, t_in, P_wat, material = variant_data
    
    # Конвертуємо діаметри з мм в м
    d = d_mm / 1000
    d1 = d1_mm / 1000
    
    # Вибір теплопровідності матеріалу
    lambda_pipe = lambda_st20 if material == "Сталь Ст20" else lambda_cu
    
    # Максимально допустима температура для матеріалу
    t2_max = 450 if material == "Сталь Ст20" else 260
    
    # Початкова швидкість води
    w = 0.5  # м/с
    
    # Розрахунок довжини змійовика за формулою (1.7)
    L = (w * rho * math.pi * (d1**2/4) * C * (t_out - t_in)) / (q * 1000 * math.pi * (d/2))
    
    # Перевірка умови довжини (10-30 м) та корекція швидкості якщо потрібно
    iterations = 0
    max_iterations = 100  # Запобігання нескінченному циклу
    
    while (L < 10 or L > 30) and iterations < max_iterations:
        if L < 10:
            w += 0.1
        else:
            w -= 0.1
        L = (w * rho * math.pi * (d1**2/4) * C * (t_out - t_in)) / (q * 1000 * math.pi * (d/2))
        iterations += 1
    
    # Розрахунок температури робочої поверхні за формулою (1.8)
    t2 = t1 + (q * 1000 * math.log(d/d1)) / (2 * math.pi * lambda_pipe)
    
    # Розрахунок втрат тиску
    # Втрати на тертя за формулою (1.9)
    dP_fr = mu_fr * (L/d1) * rho * (w**2/2)
    
    # Місцеві втрати за формулою (1.10)
    dP_lr = (n_90 * xi_lr1 + n_180 * xi_lr2) * rho * (w**2/2)
    
    # Сумарні втрати за формулою (1.12)
    dP_loss = dP_fr + dP_lr + dP_g
    
    # Перевірка достатності тиску води
    P_wat_min = (dP_loss + 1e5)/1e6  # МПа
    
    # Визначення результатів розрахунку
    results = {
        "variant_num": variant_num,
        "material": material,
        "L": L,
        "w": w,
        "t2": t2,
        "t2_max": t2_max,
        "dP_fr": dP_fr,
        "dP_lr": dP_lr,
        "dP_loss": dP_loss,
        "P_wat_min": P_wat_min,
        "P_wat": P_wat,
        "check_t1": t1 <= 75,
        "check_t_out": t_out <= 55,
        "check_t2": t2 <= t2_max,
        "check_L": 10 <= L <= 30,
        "check_P_wat": P_wat >= P_wat_min
    }
    
    return results

# Функція для виведення результатів розрахунку
def print_variant_results(results):
    print(f"\n{'=' * 80}")
    print(f"ВАРІАНТ {results['variant_num']}")
    print(f"{'=' * 80}")
    
    print(f"\n1. Розрахована довжина змійовика: {results['L']:.2f} м")
    print(f"   Швидкість води: {results['w']:.2f} м/с")
    
    print(f"\n2. Температура зовнішньої поверхні змійовика: {results['t2']:.2f} °С")
    print(f"   Максимально допустима температура ({results['material']}): {results['t2_max']} °С")
    if results['t2'] > results['t2_max']:
        print(f"   УВАГА! Перевищення допустимої температури на {results['t2'] - results['t2_max']:.2f} °С")
    
    print(f"\n3. Втрати тиску:")
    print(f"   - на тертя: {results['dP_fr']/1000:.2f} кПа")
    print(f"   - місцеві: {results['dP_lr']/1000:.2f} кПа")
    print(f"   - сумарні: {results['dP_loss']/1000:.2f} кПа")
    print(f"   Мінімально необхідний тиск води: {results['P_wat_min']:.2f} МПа")
    print(f"   Наявний тиск води: {results['P_wat']:.2f} МПа")
    if results['P_wat'] < results['P_wat_min']:
        print(f"   УВАГА! Недостатній тиск води. Потрібно збільшити на {(results['P_wat_min'] - results['P_wat']):.2f} МПа")
    
    print("\n4. Перевірка обмежень:")
    print(f"   1) Температура стінки водяного каналу: {t1} °С {'≤' if results['check_t1'] else '>'} 75 °С")
    print(f"   2) Температура води на виході: {t_out} °С {'≤' if results['check_t_out'] else '>'} 55 °С")
    print(f"   3) Температура зовнішньої поверхні: {results['t2']:.1f} °С {'≤' if results['check_t2'] else '>'} {results['t2_max']} °С")
    print(f"   4) Довжина змійовика: {results['L']:.1f} м {'в межах' if results['check_L'] else 'поза межами'} 10-30 м")
    print(f"   5) Тиск води: {results['P_wat']} МПа {'≥' if results['check_P_wat'] else '<'} {results['P_wat_min']:.2f} МПа")
    
    # Підсумкова оцінка працездатності системи
    all_checks = (results['check_t1'] and results['check_t_out'] and 
                 results['check_t2'] and results['check_L'] and results['check_P_wat'])
    
    print(f"\n5. Висновок: конструкція водоохолоджуваної панелі ДСП для варіанту {results['variant_num']} є " + 
          ("ПРАЦЕЗДАТНОЮ" if all_checks else "НЕПРАЦЕЗДАТНОЮ"))
    
    if not all_checks:
        print("   Необхідно внести зміни в конструкцію для забезпечення працездатності.")

# Функція для зібрання статистики по всіх варіантах
def collect_statistics(all_results):
    valid_variants = [r['variant_num'] for r in all_results if 
                      r['check_t1'] and r['check_t_out'] and 
                      r['check_t2'] and r['check_L'] and r['check_P_wat']]
    
    invalid_variants = [r['variant_num'] for r in all_results if 
                        not (r['check_t1'] and r['check_t_out'] and 
                            r['check_t2'] and r['check_L'] and r['check_P_wat'])]
    
    temp_exceed_variants = [r['variant_num'] for r in all_results if not r['check_t2']]
    pressure_low_variants = [r['variant_num'] for r in all_results if not r['check_P_wat']]
    length_wrong_variants = [r['variant_num'] for r in all_results if not r['check_L']]
    
    steel_variants = [r for r in all_results if r['material'] == "Сталь Ст20"]
    copper_variants = [r for r in all_results if r['material'] == "Мідь"]
    
    avg_steel_temp = sum(r['t2'] for r in steel_variants) / len(steel_variants) if steel_variants else 0
    avg_copper_temp = sum(r['t2'] for r in copper_variants) / len(copper_variants) if copper_variants else 0
    
    print(f"\n{'=' * 80}")
    print(f"ЗАГАЛЬНА СТАТИСТИКА ПО ВСІХ ВАРІАНТАХ")
    print(f"{'=' * 80}")
    
    print(f"\nВсього варіантів: {len(all_results)}")
    print(f"Працездатних варіантів: {len(valid_variants)} ({len(valid_variants)/len(all_results)*100:.1f}%)")
    print(f"Непрацездатних варіантів: {len(invalid_variants)} ({len(invalid_variants)/len(all_results)*100:.1f}%)")
    
    if invalid_variants:
        print(f"\nНепрацездатні варіанти: {', '.join(map(str, invalid_variants))}")
        
        if temp_exceed_variants:
            print(f"Варіанти з перевищенням температури: {', '.join(map(str, temp_exceed_variants))}")
        if pressure_low_variants:
            print(f"Варіанти з недостатнім тиском води: {', '.join(map(str, pressure_low_variants))}")
        if length_wrong_variants:
            print(f"Варіанти з неприйнятною довжиною змійовика: {', '.join(map(str, length_wrong_variants))}")
    
    print(f"\nСередня температура зовнішньої поверхні для сталевих труб: {avg_steel_temp:.2f} °С")
    print(f"Середня температура зовнішньої поверхні для мідних труб: {avg_copper_temp:.2f} °С")

# Головний блок коду
def main():
    print("Тепловий і гідравлічний розрахунок водоохолоджуваної панелі ДСП")
    print("Розрахунок всіх 25 варіантів завдання з таблиці 1.1")
    
    all_results = []
    
    # Розрахунок для кожного варіанту
    for i, variant_data in enumerate(variants_data, 1):
        results = calculate_variant(i, variant_data)
        all_results.append(results)
        print_variant_results(results)
    
    # Виведення загальної статистики
    collect_statistics(all_results)

if __name__ == "__main__":
    main()
