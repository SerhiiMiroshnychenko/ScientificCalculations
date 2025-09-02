# Розрахунок теплового балансу киснево-конвертерної плавки (Частина 5)

## Головна функція розрахунку теплового балансу

Нижче наведена функція для виконання повного розрахунку теплового балансу киснево-конвертерної плавки.

```python
def main():
    """
    Головна функція для виконання розрахунку теплового балансу
    """
    print("Розрахунок теплового балансу конвертерної плавки")
    n = int(input("Введіть номер варіанту (1-15): "))
    v = variants[n - 1]

    # Для прикладу: склад чавуну (можна розширити для кожного варіанту)
    C_metsh = 4.3
    Si_metsh = 0.4
    Mn_metsh = 1.1
    P_metsh = 0.18
    C_pov = 0.2
    Mn_pov = 0.2
    P_pov = 0.02

    # Розрахунок надходження тепла
    Q_chav_val = Q_chav(v["M_чав"], v["t_чав"])
    Q_msh_val = Q_msh(v["M_м.ш"], v["t_чав"])
    Q_dom_val = Q_dom(C_metsh, v["M_ст"], C_pov, Si_metsh, Mn_metsh, Mn_pov, P_metsh, P_pov)
    Q_Fe_val = Q_Fe(v["M_шл"], v["M_пил"])
    Q_shl_utv_val = Q_shl_utv(v["M_шл"])
    Q_nadh = Q_chav_val + Q_msh_val + Q_dom_val + Q_Fe_val + Q_shl_utv_val

    # Розрахунок витрати тепла - базові статті
    Q_st_val = Q_st(v["M_ст"], v["t_ст"])
    Q_shl_val = Q_shl(v["M_шл"])
    Q_pil_val = Q_pil(v["M_пил"], v["t_чав"], v["t_ст"])
    Q_vkv_val = Q_vkv(v["Fe_vtr"], v["M_пил"], v["t_ст"])
    Q_vtr_val = Q_vtr(Q_nadh)

    # Об'єми газів - приблизні значення
    V_gi = {
        'CO': 0.85,  # м³
        'CO2': 0.11,  # м³
        'N2': 0.02,  # м³
        'O2': 0.02   # м³
    }

    # Коефіцієнти теплоємності газів з Таблиці І.2
    gas_coefs = {
        'CO': {'a': 6.79, 'b': 0.98, 'c': -0.11},
        'CO2': {'a': 10.55, 'b': 2.16, 'c': -2.04},
        'N2': {'a': 6.66, 'b': 1.02, 'c': 0},
        'O2': {'a': 7.16, 'b': 1.0, 'c': -0.40}
    }

    # Розрахунок додаткових статей витрат тепла
    T_st = v["t_ст"] + 273  # Температура в К
    T_chav = v["t_чав"] + 273  # Температура в К

    Q_g_val = Q_g(V_gi, T_st, T_chav, gas_coefs)
    Q_okys_Fe_val = Q_okys_Fe(v["M_чав"], v["M_м.бр"])
    # Маса вапна (припустимо, 5% від маси шлаку)
    M_vap = 0.05 * v["M_шл"]
    Q_okys_CaCO3_val = Q_okys_CaCO3(M_vap)

    # Оновлення загальної витрати тепла з урахуванням додаткових статей
    Q_vytr = Q_st_val + Q_shl_val + Q_g_val + Q_okys_Fe_val + Q_okys_CaCO3_val + Q_pil_val + Q_vkv_val + Q_vtr_val

    # Розрахунок надлишку/дефіциту тепла
    delta_Q = Q_nadh - Q_vytr
    delta_Q_percent = delta_Q / Q_nadh * 100

    # Виведення результатів
    print("\n--- Результати розрахунку ---")
    print("\n-- НАДХОДЖЕННЯ ТЕПЛА --")
    print(f"Q_чав = {Q_chav_val:.2f} кДж ({Q_chav_val / Q_nadh * 100:.1f}%)")
    print(f"Q_м.ш = {Q_msh_val:.2f} кДж ({Q_msh_val / Q_nadh * 100:.1f}%)")
    print(f"Q_дом = {Q_dom_val:.2f} кДж ({Q_dom_val / Q_nadh * 100:.1f}%)")
    print(f"Q_Fe = {Q_Fe_val:.2f} кДж ({Q_Fe_val / Q_nadh * 100:.1f}%)")
    print(f"Q_шл.утв = {Q_shl_utv_val:.2f} кДж ({Q_shl_utv_val / Q_nadh * 100:.1f}%)")
    print(f"Q_надх = {Q_nadh:.2f} кДж (100.0%)")

    print("\n-- ВИТРАТА ТЕПЛА --")
    print(f"Q_ст = {Q_st_val:.2f} кДж ({Q_st_val / Q_vytr * 100:.1f}%)")
    print(f"Q_шл = {Q_shl_val:.2f} кДж ({Q_shl_val / Q_vytr * 100:.1f}%)")
    print(f"Q_г = {Q_g_val:.2f} кДж ({Q_g_val / Q_vytr * 100:.1f}%)")
    print(f"Q_окис^Fe = {Q_okys_Fe_val:.2f} кДж ({Q_okys_Fe_val / Q_vytr * 100:.1f}%)")
    print(f"Q_окис^CaCO3 = {Q_okys_CaCO3_val:.2f} кДж ({Q_okys_CaCO3_val / Q_vytr * 100:.1f}%)")
    print(f"Q_пил = {Q_pil_val:.2f} кДж ({Q_pil_val / Q_vytr * 100:.1f}%)")
    print(f"Q_в.к.в = {Q_vkv_val:.2f} кДж ({Q_vkv_val / Q_vytr * 100:.1f}%)")
    print(f"Q_втр = {Q_vtr_val:.2f} кДж ({Q_vtr_val / Q_vytr * 100:.1f}%)")
    print(f"Q_витр = {Q_vytr:.2f} кДж (100.0%)")

    print(f"\nΔQ = {delta_Q:.2f} кДж ({delta_Q_percent:.2f}% від надходження)")

    # Розрахунок коригуючих добавок
    if delta_Q > 0:
        print("Надлишок тепла. Розрахунок додаткової маси металобрухту:")
        dM_mbr = delta_Q / (ST_COEFFICIENT_2 * v["t_ст"] + ST_COEFFICIENT_1)
        print(f"ΔM_м.бр = {dM_mbr:.2f} кг")
    else:
        print("Дефіцит тепла. Розрахунок необхідної маси палива:")
        print("Варіанти палива:")
        fuels = {
            "Карбід кремнію": SIC_HEAT,
            "Карбід кальцію": CAC2_HEAT,
            "Антрацит": COAL_HEAT,
            "Піролізована біомаса": BIOMASS_HEAT
        }
        for fuel_name, Q_pal in fuels.items():
            dM_pal = abs(delta_Q) / Q_pal
            print(f"ΔM_пал = {dM_pal:.2f} кг ({fuel_name}, Q_пал = {Q_pal} кДж/кг)")


if __name__ == "__main__":
    main()
```

## Висновки

Представлений розрахунок теплового балансу киснево-конвертерної плавки дозволяє:

1. Визначити кількість тепла, що надходить і витрачається під час плавки
2. Розрахувати надлишок або дефіцит тепла
3. Визначити необхідну кількість коригуючих добавок для забезпечення заданої температури сталі

Реалізований алгоритм ґрунтується на методиці, наведеній у "Тепловий-баланс-методика.md", а константи взяті з довідкових даних, наведених у "Додатки.md".

Для використання цього розрахунку в jupyter notebook необхідно скопіювати всі частини коду в один notebook, забезпечивши правильний порядок визначення функцій та констант перед їх використанням. 