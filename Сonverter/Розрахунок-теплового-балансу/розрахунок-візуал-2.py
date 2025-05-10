# -*- coding: utf-8 -*-
# Автоматичний розрахунок теплового балансу для обраного варіанту

import sys
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey
import numpy as np
import matplotlib.gridspec as gridspec

# Дані з "Таблиця-варантів.md" (скопійовано у вигляді списку словників)
variants = [
    {"M_чав":71.8,"M_ст":87.5,"M_шл":16.0,"M_м.бр":17.7,"M_м.ш":1.3,"M_окат":0.1,"M_вап":8.7,"M_фут":0.13,"M_пил":0.9,"Fe_vtr":4.5,"t_чав":1350,"t_ст":1595,"t_шл":1600,"FeO":10,"Fe2O3":1,"SiO2":10,"P2O5":1,"alpha":0.1,"a":100,"c":100},
    {"M_чав":70.0,"M_ст":85.0,"M_шл":15.0,"M_м.бр":5.0,"M_м.ш":1.2,"M_окат":0.1,"M_вап":8.5,"M_фут":0.13,"M_пил":0.85,"Fe_vtr":4.2,"t_чав":1335,"t_ст":1630,"t_шл":1600,"FeO":10,"Fe2O3":1,"SiO2":10,"P2O5":1,"alpha":0.1,"a":100,"c":100},
    {"M_чав":72.0,"M_ст":88.0,"M_шл":16.5,"M_м.бр":10.8,"M_м.ш":1.3,"M_окат":0.1,"M_вап":8.8,"M_фут":0.13,"M_пил":0.92,"Fe_vtr":4.6,"t_чав":1380,"t_ст":1620,"t_шл":1600,"FeO":10,"Fe2O3":1,"SiO2":10,"P2O5":1,"alpha":0.1,"a":100,"c":100},
    {"M_чав":71.0,"M_ст":86.5,"M_шл":15.8,"M_м.бр":7.9,"M_м.ш":1.2,"M_окат":0.1,"M_вап":8.6,"M_фут":0.13,"M_пил":0.87,"Fe_vtr":4.3,"t_чав":1360,"t_ст":1620,"t_шл":1600,"FeO":10,"Fe2O3":1,"SiO2":10,"P2O5":1,"alpha":0.1,"a":100,"c":100},
    {"M_чав":70.5,"M_ст":85.8,"M_шл":15.5,"M_м.бр":5.7,"M_м.ш":1.2,"M_окат":0.1,"M_вап":8.5,"M_фут":0.13,"M_пил":0.85,"Fe_vtr":4.2,"t_чав":1340,"t_ст":1590,"t_шл":1600,"FeO":10,"Fe2O3":1,"SiO2":10,"P2O5":1,"alpha":0.1,"a":100,"c":100},
    {"M_чав":72.2,"M_ст":88.2,"M_шл":16.6,"M_м.бр":8.1,"M_м.ш":1.3,"M_окат":0.1,"M_вап":8.9,"M_фут":0.13,"M_пил":0.93,"Fe_vtr":4.7,"t_чав":1390,"t_ст":1620,"t_шл":1600,"FeO":10,"Fe2O3":1,"SiO2":10,"P2O5":1,"alpha":0.1,"a":100,"c":100},
    {"M_чав":70.8,"M_ст":86.2,"M_шл":15.7,"M_м.бр":8.5,"M_м.ш":1.2,"M_окат":0.1,"M_вап":8.6,"M_фут":0.13,"M_пил":0.86,"Fe_vtr":4.3,"t_чав":1320,"t_ст":1620,"t_шл":1600,"FeO":10,"Fe2O3":1,"SiO2":10,"P2O5":1,"alpha":0.1,"a":100,"c":100},
    {"M_чав":71.5,"M_ст":87.0,"M_шл":15.9,"M_м.бр":7.2,"M_м.ш":1.2,"M_окат":0.1,"M_вап":8.7,"M_фут":0.13,"M_пил":0.88,"Fe_vtr":4.4,"t_чав":1370,"t_ст":1600,"t_шл":1600,"FeO":10,"Fe2O3":1,"SiO2":10,"P2O5":1,"alpha":0.1,"a":100,"c":100},
    {"M_чав":71.2,"M_ст":86.7,"M_шл":15.8,"M_м.бр":9.3,"M_м.ш":1.2,"M_окат":0.1,"M_вап":8.6,"M_фут":0.13,"M_пил":0.87,"Fe_vtr":4.3,"t_чав":1380,"t_ст":1620,"t_шл":1600,"FeO":10,"Fe2O3":1,"SiO2":10,"P2O5":1,"alpha":0.1,"a":100,"c":100},
    {"M_чав":70.9,"M_ст":86.3,"M_шл":15.6,"M_м.бр":9.5,"M_м.ш":1.2,"M_окат":0.1,"M_вап":8.5,"M_фут":0.13,"M_пил":0.86,"Fe_vtr":4.3,"t_чав":1380,"t_ст":1590,"t_шл":1600,"FeO":10,"Fe2O3":1,"SiO2":10,"P2O5":1,"alpha":0.1,"a":100,"c":100},
    {"M_чав":70.0,"M_ст":85.0,"M_шл":15.0,"M_м.бр":1.8,"M_м.ш":1.2,"M_окат":0.1,"M_вап":8.5,"M_фут":0.13,"M_пил":0.85,"Fe_vtr":4.2,"t_чав":1300,"t_ст":1590,"t_шл":1600,"FeO":10,"Fe2O3":1,"SiO2":10,"P2O5":1,"alpha":0.1,"a":100,"c":100},
    {"M_чав":71.7,"M_ст":87.7,"M_шл":16.1,"M_м.бр":7.7,"M_м.ш":1.3,"M_окат":0.1,"M_вап":8.8,"M_фут":0.13,"M_пил":0.91,"Fe_vtr":4.6,"t_чав":1370,"t_ст":1610,"t_шл":1600,"FeO":10,"Fe2O3":1,"SiO2":10,"P2O5":1,"alpha":0.1,"a":100,"c":100},
    {"M_чав":70.2,"M_ст":85.2,"M_шл":15.1,"M_м.бр":0.9,"M_м.ш":1.2,"M_окат":0.1,"M_вап":8.5,"M_фут":0.13,"M_пил":0.85,"Fe_vtr":4.2,"t_чав":1350,"t_ст":1625,"t_шл":1600,"FeO":10,"Fe2O3":1,"SiO2":10,"P2O5":1,"alpha":0.1,"a":100,"c":100},
    {"M_чав":70.6,"M_ст":85.6,"M_шл":15.3,"M_м.бр":1.3,"M_м.ш":1.2,"M_окат":0.1,"M_вап":8.5,"M_фут":0.13,"M_пил":0.85,"Fe_vtr":4.2,"t_чав":1300,"t_ст":1600,"t_шл":1600,"FeO":10,"Fe2O3":1,"SiO2":10,"P2O5":1,"alpha":0.1,"a":100,"c":100},
    {"M_чав":71.3,"M_ст":86.8,"M_шл":15.9,"M_м.бр":9.6,"M_м.ш":1.2,"M_окат":0.1,"M_вап":8.6,"M_фут":0.13,"M_пил":0.87,"Fe_vtr":4.3,"t_чав":1380,"t_ст":1620,"t_шл":1600,"FeO":10,"Fe2O3":1,"SiO2":10,"P2O5":1,"alpha":0.1,"a":100,"c":100},
]

def Q_chav(M_chav, t_chav):
    return M_chav * (61.9 + 0.88 * t_chav)

def Q_msh(M_msh, a, t_msh):
    return M_msh * (a / 100) * (1.53 * t_msh - 710)

def Q_dom(alpha, C_metsh, M_st, C_pov, Si_metsh, Mn_metsh, Mn_pov, P_metsh, P_pov):
    return (
        (11680 * (1 - alpha) + 35300 * alpha) * (C_metsh - 0.01 * M_st * C_pov)
        + 26930 * Si_metsh
        + 7035 * (Mn_metsh - 0.01 * M_st * Mn_pov)
        + 19755 * (P_metsh - 0.01 * M_st * P_pov)
    )

def Q_Fe(M_shl, FeO, Fe2O3, chi):
    return 0.01 * M_shl * (3600 * FeO + 5110 * Fe2O3 + 5110 * chi)

def Q_shl_utv(M_shl, SiO2, P2O5):
    return 0.01 * M_shl * (2300 * SiO2 + 4886 * P2O5)

def Q_st(M_st, t_st):
    return M_st * (54.8 + 0.84 * t_st)

def Q_shl(M_shl, t_shl):
    return M_shl * (2.09 * t_shl - 1380)

def Q_pil(chi, t_chav, t_st):
    return chi * (23.05 + 0.69 * (t_chav + t_st) / 2)

def Q_vkv(Delta_Fe_vtr, chi, t_st):
    return (Delta_Fe_vtr - chi) * (54.9 + 0.838 * t_st)

def Q_vtr(Q_nadh):
    return 0.01 * Q_nadh * 5

def main():
    print("Розрахунок теплового балансу конвертерної плавки")
    n = int(input("Введіть номер варіанту (1-15): "))
    v = variants[n-1]

    # Для прикладу: склад чавуну (можна розширити для кожного варіанту)
    C_metsh = 4.3
    Si_metsh = 0.4
    Mn_metsh = 1.1
    P_metsh = 0.18
    C_pov = 0.2
    Mn_pov = 0.2
    P_pov = 0.02

    Q_chav_val = Q_chav(v["M_чав"], v["t_чав"])
    Q_msh_val = Q_msh(v["M_м.ш"], v["a"], v["t_чав"])
    Q_dom_val = Q_dom(v["alpha"], C_metsh, v["M_ст"], C_pov, Si_metsh, Mn_metsh, Mn_pov, P_metsh, P_pov)
    Q_Fe_val = Q_Fe(v["M_шл"], v["FeO"], v["Fe2O3"], v["M_пил"])
    Q_shl_utv_val = Q_shl_utv(v["M_шл"], v["SiO2"], v["P2O5"])
    Q_nadh = Q_chav_val + Q_msh_val + Q_dom_val + Q_Fe_val + Q_shl_utv_val

    Q_st_val = Q_st(v["M_ст"], v["t_ст"])
    Q_shl_val = Q_shl(v["M_шл"], v["t_шл"])
    Q_pil_val = Q_pil(v["M_пил"], v["t_чав"], v["t_ст"])
    Q_vkv_val = Q_vkv(v["Fe_vtr"], v["M_пил"], v["t_ст"])
    Q_vtr_val = Q_vtr(Q_nadh)
    Q_vytr = Q_st_val + Q_shl_val + Q_pil_val + Q_vkv_val + Q_vtr_val

    delta_Q = Q_nadh - Q_vytr
    delta_Q_percent = delta_Q / Q_nadh * 100

    print("\n--- Результати розрахунку ---")
    print(f"Q_чав = {Q_chav_val:.2f} кДж")
    print(f"Q_м.ш = {Q_msh_val:.2f} кДж")
    print(f"Q_дом = {Q_dom_val:.2f} кДж")
    print(f"Q_Fe = {Q_Fe_val:.2f} кДж")
    print(f"Q_шл.утв = {Q_shl_utv_val:.2f} кДж")
    print(f"Q_надх = {Q_nadh:.2f} кДж")
    print(f"Q_ст = {Q_st_val:.2f} кДж")
    print(f"Q_шл = {Q_shl_val:.2f} кДж")
    print(f"Q_пил = {Q_pil_val:.2f} кДж")
    print(f"Q_в.к.в = {Q_vkv_val:.2f} кДж")
    print(f"Q_втр = {Q_vtr_val:.2f} кДж")
    print(f"Q_витр = {Q_vytr:.2f} кДж")
    print(f"ΔQ = {delta_Q:.2f} кДж ({delta_Q_percent:.2f} % від надходження)")

    if delta_Q > 0:
        print("Надлишок тепла. Розрахуйте додаткову масу металобрухту:")
        dM_mbr = delta_Q / (0.84 * v["t_ст"] + 54.9)
        print(f"ΔM_м.бр = {dM_mbr:.2f} кг")
    else:
        print("Дефіцит тепла. Розрахуйте необхідну масу палива:")
        Q_pal = 17000  # наприклад, карбід кремнію
        dM_pal = abs(delta_Q) / Q_pal
        print(f"ΔM_пал = {dM_pal:.2f} кг (для карбіду кремнію, Q_пал = 17000 кДж/кг)")

    # --- Візуалізація результатів: кругові діаграми з відсотками та номерами у кружечках зовні ---
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 2, height_ratios=[4, 1])

    # Діаграми
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    # Легенда
    ax_legend0 = fig.add_subplot(gs[1, 0])
    ax_legend1 = fig.add_subplot(gs[1, 1])
    ax_legend0.axis('off')
    ax_legend1.axis('off')

    # Дані для приходу
    labels_in = ["1", "2", "3", "4", "5"]
    values_in = [Q_chav_val, Q_msh_val, Q_dom_val, Q_Fe_val, Q_shl_utv_val]
    names_in = [
        "1 — Тепло рідкого чавуну",
        "2 — Тепло міксерного шлаку",
        "3 — Тепло окислення домішок",
        "4 — Тепло окислення заліза",
        "5 — Тепло шлакоутворення"
    ]
    colors_in = plt.cm.Blues(range(50, 250, 40))

    # Дані для витрати
    labels_out = ["1", "2", "3", "4", "5"]
    values_out = [Q_st_val, Q_shl_val, Q_pil_val, Q_vkv_val, Q_vtr_val]
    names_out = [
        "1 — Тепло рідкої сталі",
        "2 — Тепло шлаку",
        "3 — Тепло пилу",
        "4 — Тепло викидів Fe",
        "5 — Теплові втрати"
    ]
    colors_out = plt.cm.Oranges(range(50, 250, 40))

    # Кругова діаграма приходу тепла
    wedges_in, texts_in, autotexts_in = ax0.pie(
        values_in, labels=None, autopct='%1.1f%%', startangle=90, colors=colors_in
    )
    ax0.set_title('Структура приходу тепла', pad=20)
    for i, w in enumerate(wedges_in):
        ang = (w.theta2 + w.theta1) / 2
        x = 1.1 * np.cos(np.deg2rad(ang))
        y = 1.1 * np.sin(np.deg2rad(ang))
        ax0.text(
            x, y, labels_in[i], ha='center', va='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle="circle,pad=0.3", fc=colors_in[i], ec="black", lw=1)
        )

    # Кругова діаграма витрати тепла
    wedges_out, texts_out, autotexts_out = ax1.pie(
        values_out, labels=None, autopct='%1.1f%%', startangle=90, colors=colors_out
    )
    ax1.set_title('Структура витрати тепла', pad=20)
    for i, w in enumerate(wedges_out):
        ang = (w.theta2 + w.theta1) / 2
        x = 1.1 * np.cos(np.deg2rad(ang))
        y = 1.1 * np.sin(np.deg2rad(ang))
        ax1.text(
            x, y, labels_out[i], ha='center', va='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle="circle,pad=0.3", fc=colors_out[i], ec="black", lw=1)
        )

    # Легенда під діаграмами
    ax_legend0.text(0, 0.5, '\n'.join(names_in), fontsize=12, ha='left', va='center')
    ax_legend1.text(0, 0.5, '\n'.join(names_out), fontsize=12, ha='left', va='center')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()