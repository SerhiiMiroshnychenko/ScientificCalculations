import pandas as pd
import numpy as np
import datetime
from tabulate import tabulate
from scipy import stats
import os
import json

def load_data(file_path, group_column='is_successful'):
    """
    Завантажує дані з файлу

    Args:
        file_path (str): Шлях до файлу даних
        group_column (str): Назва колонки для групування

    Returns:
        pd.DataFrame: Завантажений датафрейм
        list: Список колонок для аналізу
    """
    print(f"Завантаження даних з {file_path}...")

    # Завантаження даних
    df = pd.read_csv(file_path)

    # Перетворення стовпця групування на числовий тип (0 або 1), якщо це ще не зроблено
    df[group_column] = df[group_column].astype(int)

    # Визначення колонок для аналізу (всі числові колонки)
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    # Видаляємо колонку групування зі списку, якщо вона числова
    if group_column in numeric_columns:
        numeric_columns.remove(group_column)

    # Замінюємо від'ємні значення на 0 в числових колонках
    for col in numeric_columns:
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            print(f"Знайдено {negative_count} від'ємних значень у колонці {col}. Замінюємо їх на 0.")
            df[col] = df[col].apply(lambda x: max(0, x) if not pd.isna(x) else x)

    print(f"Завантажено {len(df)} записів.")
    print(f"Для аналізу обрано {len(numeric_columns)} числових колонок.")

    return df, numeric_columns

def calculate_basic_stats(df, column_name, group_column='is_successful', group_names=['Неуспішні', 'Успішні']):
    """
    Розраховує базові статистичні показники для двох груп даних

    Args:
        df (pd.DataFrame): Датафрейм з даними
        column_name (str): Назва колонки для аналізу
        group_column (str): Назва колонки для групування
        group_names (list): Назви груп

    Returns:
        pd.DataFrame: Датафрейм із статистичними показниками
    """
    # Розділення даних на групи
    group_0 = df[df[group_column] == 0][column_name].dropna()
    group_1 = df[df[group_column] == 1][column_name].dropna()

    # Пропускаємо, якщо даних недостатньо
    if len(group_0) < 2 or len(group_1) < 2:
        return None

    # Створення словника з даними для кожної групи
    data_dict = {
        group_names[0]: group_0,
        group_names[1]: group_1
    }

    # Створення датафрейму
    stats_df = pd.DataFrame()

    # Розрахунок базових статистик для кожної групи
    for group_name, data in data_dict.items():
        group_stats = {
            'Кількість': len(data),
            'Середнє': data.mean(),
            'Медіана': data.median(),
            'Стандартне відхилення': data.std(),
            'Коефіцієнт варіації': data.std() / data.mean() if data.mean() != 0 else np.nan,
            'Мінімум': data.min(),
            'Максимум': data.max(),
            'Квартиль 25%': data.quantile(0.25),
            'Квартиль 75%': data.quantile(0.75),
            'Коефіцієнт асиметрії': stats.skew(data),
            'Ексцес': stats.kurtosis(data)
        }

        # Додавання статистик до датафрейму
        if stats_df.empty:
            stats_df = pd.DataFrame(group_stats, index=[group_name])
        else:
            stats_df.loc[group_name] = group_stats

    return stats_df

def perform_statistical_tests(df, column_name, group_column='is_successful', alpha=0.05):
    """
    Проводить статистичні тести для порівняння двох груп даних

    Args:
        df (pd.DataFrame): Датафрейм з даними
        column_name (str): Назва колонки для аналізу
        group_column (str): Назва колонки для групування
        alpha (float): Рівень значущості

    Returns:
        pd.DataFrame: Датафрейм із результатами тестів
        pd.DataFrame: Датафрейм із довірчими інтервалами
    """
    # Розділення даних на групи
    group_0 = df[df[group_column] == 0][column_name].dropna()
    group_1 = df[df[group_column] == 1][column_name].dropna()

    # Пропускаємо, якщо даних недостатньо
    if len(group_0) < 2 or len(group_1) < 2:
        return None, None

    # t-тест для незалежних вибірок
    try:
        t_stat, t_pvalue = stats.ttest_ind(group_0, group_1, equal_var=False)
    except:
        t_stat, t_pvalue = np.nan, np.nan

    # Тест Манна-Уітні
    try:
        mw_stat, mw_pvalue = stats.mannwhitneyu(group_0, group_1)
    except:
        mw_stat, mw_pvalue = np.nan, np.nan

    # Розрахунок довірчих інтервалів
    try:
        ci_0 = stats.t.interval(1-alpha, len(group_0)-1, loc=group_0.mean(), scale=stats.sem(group_0))
        ci_1 = stats.t.interval(1-alpha, len(group_1)-1, loc=group_1.mean(), scale=stats.sem(group_1))
    except:
        ci_0 = (np.nan, np.nan)
        ci_1 = (np.nan, np.nan)

    # Створення датафрейму із результатами
    tests_results = pd.DataFrame({
        'Тест': ['t-тест (Welch)', 'Тест Манна-Уітні'],
        'Статистика': [t_stat, mw_stat],
        'p-значення': [t_pvalue, mw_pvalue],
        'Значущість': [
            "Значуща різниця" if p < alpha and not np.isnan(p) else "Немає значущої різниці"
            for p in [t_pvalue, mw_pvalue]
        ]
    })

    # Результати довірчих інтервалів
    ci_results = pd.DataFrame({
        'Група': ['Неуспішні', 'Успішні'],
        'Середнє': [group_0.mean(), group_1.mean()],
        'Нижня межа CI': [ci_0[0], ci_1[0]],
        'Верхня межа CI': [ci_0[1], ci_1[1]]
    })

    return tests_results, ci_results

def comparative_analysis(basic_stats):
    """
    Проводить порівняльний аналіз між групами

    Args:
        basic_stats (pd.DataFrame): Базові статистичні показники

    Returns:
        pd.DataFrame: Датафрейм з результатами порівняння
    """
    if basic_stats is None or basic_stats.empty:
        return None

    # Вибір статистик для порівняння
    stats_to_compare = ['Середнє', 'Медіана', 'Стандартне відхилення',
                        'Квартиль 25%', 'Квартиль 75%', 'Коефіцієнт варіації',
                        'Коефіцієнт асиметрії', 'Ексцес']

    comparison_data = []
    for stat in stats_to_compare:
        if stat not in basic_stats.columns:
            continue

        group_0_value = basic_stats.iloc[0][stat]
        group_1_value = basic_stats.iloc[1][stat]

        if pd.isna(group_0_value) or pd.isna(group_1_value) or group_1_value == 0:
            ratio = np.nan
            diff = np.nan if pd.isna(group_0_value) or pd.isna(group_1_value) else group_0_value - group_1_value
        else:
            ratio = group_0_value / group_1_value
            diff = group_0_value - group_1_value

        comparison_data.append({
            'Статистика': stat,
            'Неуспішні': group_0_value,
            'Успішні': group_1_value,
            'Різниця': diff,
            'Відношення': ratio
        })

    if not comparison_data:
        return None

    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df

def is_column_significant(tests_results):
    """
    Визначає, чи є статистично значуща різниця для колонки

    Args:
        tests_results (pd.DataFrame): Результати статистичних тестів

    Returns:
        bool: True якщо колонка має значущу різницю, False інакше
    """
    if tests_results is None or tests_results.empty:
        return False

    # Перевіряємо p-значення для t-тесту та тесту Манна-Уітні
    t_test_pvalue = tests_results.loc[tests_results['Тест'] == 't-тест (Welch)', 'p-значення'].values[0]
    mw_test_pvalue = tests_results.loc[tests_results['Тест'] == 'Тест Манна-Уітні', 'p-значення'].values[0]

    # Якщо хоча б один тест показує значущу різницю (p < 0.05), вважаємо колонку значущою
    return (t_test_pvalue < 0.05 and not np.isnan(t_test_pvalue)) or (mw_test_pvalue < 0.05 and not np.isnan(mw_test_pvalue))

def is_practical_difference(mean_0, mean_1, threshold=0.05):
    """
    Визначає, чи є практична різниця між середніми значеннями

    Args:
        mean_0 (float): Середнє значення для першої групи
        mean_1 (float): Середнє значення для другої групи
        threshold (float): Поріг мінімальної відносної різниці (0.05 = 5%)

    Returns:
        bool: True якщо є практична різниця, False інакше
    """
    if pd.isna(mean_0) or pd.isna(mean_1) or mean_0 == 0 or mean_1 == 0:
        return False

    # Обчислюємо відносну різницю
    relative_diff = abs(mean_0 - mean_1) / max(mean_0, mean_1)
    return relative_diff > threshold

def format_value(value):
    """Форматує числове значення для відображення"""
    if pd.isna(value):
        return "N/A"
    elif isinstance(value, (int, np.integer)):
        return f"{value:,d}"
    elif abs(value) < 0.0001 or abs(value) >= 10000:
        return f"{value:.4e}"
    else:
        return f"{value:.4f}"

def calculate_cohen_d(group_0_values, group_1_values):
    """
    Розраховує розмір ефекту d Коена

    Args:
        group_0_values (pd.Series): Значення для першої групи
        group_1_values (pd.Series): Значення для другої групи

    Returns:
        float: Значення d Коена
    """
    # Перевірка на достатню кількість даних
    if len(group_0_values) < 2 or len(group_1_values) < 2:
        return np.nan

    # Розрахунок середніх та стандартних відхилень
    mean_0 = group_0_values.mean()
    mean_1 = group_1_values.mean()
    std_0 = group_0_values.std()
    std_1 = group_1_values.std()

    # Перевірка на валідні значення
    if np.isnan(mean_0) or np.isnan(mean_1) or np.isnan(std_0) or np.isnan(std_1):
        return np.nan

    # Розрахунок об'єднаного стандартного відхилення
    pooled_std = np.sqrt((std_0**2 + std_1**2) / 2)

    # Уникнення ділення на нуль
    if pooled_std == 0:
        return np.nan

    # Розрахунок d Коена
    cohen_d = abs(mean_0 - mean_1) / pooled_std

    return cohen_d

def calculate_auc(df, column, target='is_successful'):
    """
    Розраховує AUC для окремої колонки як предиктора

    Args:
        df (pd.DataFrame): Датафрейм з даними
        column (str): Назва колонки-предиктора
        target (str): Назва цільової змінної

    Returns:
        float: Значення AUC
    """
    from sklearn.metrics import roc_auc_score

    # Видаляємо рядки з пропущеними значеннями
    valid_data = df[[column, target]].dropna()

    # Пропускаємо колонки з константними значеннями
    if len(valid_data[column].unique()) <= 1:
        return 0.5

    try:
        # Розрахунок AUC
        auc = roc_auc_score(valid_data[target], valid_data[column])

        # Якщо AUC < 0.5, це означає негативну кореляцію
        # Інвертуємо значення для отримання позитивної метрики
        return max(auc, 1 - auc)
    except:
        return 0.5  # У випадку помилки повертаємо нейтральне значення

def calculate_iv(df, column, target='is_successful', bins=10):
    """
    Розраховує Information Value для числової колонки

    Args:
        df (pd.DataFrame): Датафрейм з даними
        column (str): Назва колонки для аналізу
        target (str): Назва цільової змінної
        bins (int): Кількість бінів для дискретизації

    Returns:
        float: Значення IV
    """
    # Видаляємо рядки з пропущеними значеннями
    valid_data = df[[column, target]].dropna()

    # Перевірка на порожні значення
    if valid_data.empty or len(valid_data[column].unique()) <= 1:
        return 0

    # Створюємо копію для уникнення попереджень
    df_temp = valid_data.copy()

    try:
        # Дискретизація числової колонки
        df_temp['bin'] = pd.qcut(df_temp[column], bins, duplicates='drop')
    except:
        # Якщо дискретизація не вдалася, використовуємо унікальні значення
        df_temp['bin'] = df_temp[column]

    # Розрахунок WoE та IV
    grouped = df_temp.groupby('bin')[target].agg(['count', 'sum'])
    grouped['non_target'] = grouped['count'] - grouped['sum']

    # Уникаємо ділення на нуль
    if grouped['sum'].sum() == 0 or grouped['non_target'].sum() == 0:
        return 0

    grouped['target_rate'] = grouped['sum'] / grouped['sum'].sum()
    grouped['non_target_rate'] = grouped['non_target'] / grouped['non_target'].sum()

    # Заміна нульових значень на малі числа для уникнення log(0)
    grouped['target_rate'] = grouped['target_rate'].replace(0, 0.0001)
    grouped['non_target_rate'] = grouped['non_target_rate'].replace(0, 0.0001)

    grouped['woe'] = np.log(grouped['target_rate'] / grouped['non_target_rate'])
    grouped['iv'] = (grouped['target_rate'] - grouped['non_target_rate']) * grouped['woe']

    return grouped['iv'].sum()

def create_rankings(all_column_results, df, group_column='is_successful'):
    """
    Створює рейтингові таблиці за різними показниками

    Args:
        all_column_results (dict): Словник з результатами аналізу колонок
        df (pd.DataFrame): Датафрейм з даними
        group_column (str): Назва колонки для групування

    Returns:
        dict: Словник з рейтинговими таблицями
    """
    # Підготовка даних для рейтингування
    ranking_data = []

    for column_name, result in all_column_results.items():
        if result is None:
            continue

        # Розділення даних на групи
        group_0 = df[df[group_column] == 0][column_name].dropna()
        group_1 = df[df[group_column] == 1][column_name].dropna()

        # Отримання основних показників
        t_pvalue = result['p_values']['t_test']
        mw_pvalue = result['p_values']['mann_whitney']

        # Отримання середніх значень
        mean_0 = result['basic_stats'].get('Неуспішні', {}).get('Середнє', np.nan)
        mean_1 = result['basic_stats'].get('Успішні', {}).get('Середнє', np.nan)

        # Розрахунок відносної різниці у відсотках
        if not pd.isna(mean_0) and not pd.isna(mean_1) and max(abs(mean_0), abs(mean_1)) > 0:
            relative_diff = abs(mean_0 - mean_1) / max(abs(mean_0), abs(mean_1)) * 100
        else:
            relative_diff = 0

        # Розрахунок d Коена
        cohen_d = calculate_cohen_d(group_0, group_1)

        # Розрахунок AUC
        auc = calculate_auc(df, column_name, group_column)

        # Розрахунок Information Value
        iv = calculate_iv(df, column_name, group_column)

        # Додавання даних до списку
        ranking_data.append({
            'column': column_name,
            't_pvalue': t_pvalue if not pd.isna(t_pvalue) else 1.0,  # Менше значення = краще
            'mw_pvalue': mw_pvalue if not pd.isna(mw_pvalue) else 1.0,  # Менше значення = краще
            'relative_diff': relative_diff,  # Більше значення = краще
            'cohen_d': cohen_d if not pd.isna(cohen_d) else 0,  # Більше значення = краще
            'auc': auc,  # Більше значення = краще
            'iv': iv  # Більше значення = краще
        })

    # Створення датафрейму
    ranking_df = pd.DataFrame(ranking_data)

    # Створення рейтингів за різними показниками
    rankings = {}

    # 1. Рейтинг за p-значенням t-тесту
    t_pvalue_ranking = ranking_df.sort_values('t_pvalue', ascending=True).copy()
    t_pvalue_ranking['rank'] = range(1, len(t_pvalue_ranking) + 1)
    t_pvalue_ranking = t_pvalue_ranking[['rank', 'column', 't_pvalue']]
    t_pvalue_ranking.columns = ['Ранг', 'Колонка', 'p-значення t-тесту']
    rankings['t_pvalue'] = t_pvalue_ranking

    # 2. Рейтинг за p-значенням тесту Манна-Уітні
    mw_pvalue_ranking = ranking_df.sort_values('mw_pvalue', ascending=True).copy()
    mw_pvalue_ranking['rank'] = range(1, len(mw_pvalue_ranking) + 1)
    mw_pvalue_ranking = mw_pvalue_ranking[['rank', 'column', 'mw_pvalue']]
    mw_pvalue_ranking.columns = ['Ранг', 'Колонка', 'p-значення тесту Манна-Уітні']
    rankings['mw_pvalue'] = mw_pvalue_ranking

    # 3. Рейтинг за відносною різницею
    rel_diff_ranking = ranking_df.sort_values('relative_diff', ascending=False).copy()
    rel_diff_ranking['rank'] = range(1, len(rel_diff_ranking) + 1)
    rel_diff_ranking = rel_diff_ranking[['rank', 'column', 'relative_diff']]
    rel_diff_ranking.columns = ['Ранг', 'Колонка', 'Відносна різниця (%)']
    rankings['relative_diff'] = rel_diff_ranking

    # 4. Рейтинг за d Коена
    cohen_d_ranking = ranking_df.sort_values('cohen_d', ascending=False).copy()
    cohen_d_ranking['rank'] = range(1, len(cohen_d_ranking) + 1)
    cohen_d_ranking = cohen_d_ranking[['rank', 'column', 'cohen_d']]
    cohen_d_ranking.columns = ['Ранг', 'Колонка', 'd Коена']
    rankings['cohen_d'] = cohen_d_ranking

    # 5. Рейтинг за AUC
    auc_ranking = ranking_df.sort_values('auc', ascending=False).copy()
    auc_ranking['rank'] = range(1, len(auc_ranking) + 1)
    auc_ranking = auc_ranking[['rank', 'column', 'auc']]
    auc_ranking.columns = ['Ранг', 'Колонка', 'AUC']
    rankings['auc'] = auc_ranking

    # 6. Рейтинг за Information Value
    iv_ranking = ranking_df.sort_values('iv', ascending=False).copy()
    iv_ranking['rank'] = range(1, len(iv_ranking) + 1)
    iv_ranking = iv_ranking[['rank', 'column', 'iv']]
    iv_ranking.columns = ['Ранг', 'Колонка', 'Information Value']
    rankings['iv'] = iv_ranking

    # 7. Загальний рейтинг (комбінований)
    # Нормалізація показників для створення комбінованого рейтингу
    ranking_df['t_pvalue_norm'] = 1 - (ranking_df['t_pvalue'] / max(ranking_df['t_pvalue'].max(), 1))
    ranking_df['mw_pvalue_norm'] = 1 - (ranking_df['mw_pvalue'] / max(ranking_df['mw_pvalue'].max(), 1))
    ranking_df['relative_diff_norm'] = ranking_df['relative_diff'] / max(ranking_df['relative_diff'].max(), 1)
    ranking_df['cohen_d_norm'] = ranking_df['cohen_d'] / max(ranking_df['cohen_d'].max(), 1)
    ranking_df['auc_norm'] = (ranking_df['auc'] - 0.5) / 0.5  # Нормалізація від 0 до 1
    ranking_df['iv_norm'] = ranking_df['iv'] / max(ranking_df['iv'].max(), 1)

    # Розрахунок комбінованого показника
    ranking_df['combined_score'] = (
                                           ranking_df['t_pvalue_norm'] +
                                           ranking_df['mw_pvalue_norm'] +
                                           ranking_df['relative_diff_norm'] +
                                           ranking_df['cohen_d_norm'] +
                                           ranking_df['auc_norm'] +
                                           ranking_df['iv_norm']
                                   ) / 6

    combined_ranking = ranking_df.sort_values('combined_score', ascending=False).copy()
    combined_ranking['rank'] = range(1, len(combined_ranking) + 1)
    combined_ranking = combined_ranking[['rank', 'column', 'combined_score']]
    combined_ranking.columns = ['Ранг', 'Колонка', 'Комбінований показник']
    rankings['combined'] = combined_ranking

    return rankings

def display_rankings(rankings, title="Рейтинг колонок"):
    """
    Виводить рейтингові таблиці в консоль

    Args:
        rankings (dict): Словник з рейтинговими таблицями
        title (str): Заголовок для виведення
    """
    print(f"\n\n=== {title} ===\n")

    # Виведення рейтингу за p-значенням t-тесту
    print("\n1. Рейтинг за p-значенням t-тесту (менше значення = сильніший зв'язок)")
    print(tabulate(rankings['t_pvalue'].head(20), headers='keys', tablefmt='pretty', showindex=False, floatfmt=".4f"))

    # Виведення рейтингу за p-значенням тесту Манна-Уітні
    print("\n2. Рейтинг за p-значенням тесту Манна-Уітні (менше значення = сильніший зв'язок)")
    print(tabulate(rankings['mw_pvalue'].head(20), headers='keys', tablefmt='pretty', showindex=False, floatfmt=".4f"))

    # Виведення рейтингу за відносною різницею
    print("\n3. Рейтинг за відносною різницею середніх значень (більше значення = сильніший зв'язок)")
    print(tabulate(rankings['relative_diff'].head(20), headers='keys', tablefmt='pretty', showindex=False, floatfmt=".2f"))

    # Виведення рейтингу за d Коена
    print("\n4. Рейтинг за розміром ефекту d Коена (більше значення = сильніший зв'язок)")
    print(tabulate(rankings['cohen_d'].head(20), headers='keys', tablefmt='pretty', showindex=False, floatfmt=".4f"))

    # Виведення рейтингу за AUC
    print("\n5. Рейтинг за AUC (більше значення = сильніший зв'язок)")
    print(tabulate(rankings['auc'].head(20), headers='keys', tablefmt='pretty', showindex=False, floatfmt=".4f"))

    # Виведення рейтингу за Information Value
    print("\n6. Рейтинг за Information Value (більше значення = сильніший зв'язок)")
    print(tabulate(rankings['iv'].head(20), headers='keys', tablefmt='pretty', showindex=False, floatfmt=".4f"))

    # Виведення комбінованого рейтингу
    print("\n7. Комбінований рейтинг (враховує всі показники)")
    print(tabulate(rankings['combined'].head(20), headers='keys', tablefmt='pretty', showindex=False, floatfmt=".4f"))

def display_column_analysis(column_name, basic_stats, tests_results, ci_results, comparison_df, is_significant):
    """
    Виводить результати аналізу колонки в консоль у форматованому вигляді

    Args:
        column_name (str): Назва колонки
        basic_stats (pd.DataFrame): Базові статистичні показники
        tests_results (pd.DataFrame): Результати статистичних тестів
        ci_results (pd.DataFrame): Довірчі інтервали
        comparison_df (pd.DataFrame): Результати порівняльного аналізу
        is_significant (bool): Чи є значуща різниця
    """
    print("\n" + "="*80)
    print(f"СТАТИСТИЧНИЙ АНАЛІЗ КОЛОНКИ: {column_name}".center(80))
    print("="*80)

    # Форматування таблиці базових статистик
    if basic_stats is not None:
        basic_stats_formatted = basic_stats.copy()
        for col in basic_stats_formatted.columns:
            if col == 'Кількість':
                continue
            basic_stats_formatted[col] = basic_stats_formatted[col].map(lambda x: f"{x:,.4f}" if not pd.isna(x) else "N/A")

        print("\nОсновні статистичні показники за групами:")
        print(tabulate(basic_stats_formatted, headers='keys', tablefmt='grid', showindex=True))

    # Форматування таблиці результатів тестів
    if tests_results is not None:
        tests_formatted = tests_results.copy()
        tests_formatted['p-значення'] = tests_formatted['p-значення'].map(lambda x: f"{x:.8f}" if not pd.isna(x) else "N/A")
        tests_formatted['Статистика'] = tests_formatted['Статистика'].map(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")

        print("\nРезультати статистичних тестів:")
        print(tabulate(tests_formatted, headers='keys', tablefmt='grid', showindex=False))

    # Форматування таблиці довірчих інтервалів
    if ci_results is not None:
        ci_formatted = ci_results.copy()
        numeric_cols = ['Середнє', 'Нижня межа CI', 'Верхня межа CI']
        for col in numeric_cols:
            ci_formatted[col] = ci_formatted[col].map(lambda x: f"{x:,.4f}" if not pd.isna(x) else "N/A")

        print("\nДовірчі інтервали для середніх значень (95%):")
        print(tabulate(ci_formatted, headers='keys', tablefmt='grid', showindex=False))

    # Форматування порівняльної таблиці
    if comparison_df is not None:
        comparison_formatted = comparison_df.copy()
        for col in ['Неуспішні', 'Успішні', 'Різниця', 'Відношення']:
            comparison_formatted[col] = comparison_formatted[col].map(lambda x: f"{x:,.4f}" if not pd.isna(x) else "N/A")

        print("\nПорівняльний аналіз статистичних показників:")
        print(tabulate(comparison_formatted, headers='keys', tablefmt='grid', showindex=False))

    # Виведення висновку
    print("\nВисновок:")
    if is_significant:
        # Отримання середніх значень
        mean_0 = basic_stats.loc['Неуспішні', 'Середнє'] if 'Неуспішні' in basic_stats.index else np.nan
        mean_1 = basic_stats.loc['Успішні', 'Середнє'] if 'Успішні' in basic_stats.index else np.nan

        # Перевірка на наявність практичної різниці
        if is_practical_difference(mean_0, mean_1):
            print(f"⚠ Виявлено СТАТИСТИЧНО ЗНАЧУЩУ різницю в значеннях колонки '{column_name}' між успішними та неуспішними замовленнями.")

            if not pd.isna(mean_0) and not pd.isna(mean_1) and mean_1 != 0 and mean_0 != 0:
                if mean_0 > mean_1:
                    print(f"   • Неуспішні замовлення мають більше значення '{column_name}' (в {mean_0/mean_1:.6f} рази)")
                else:
                    print(f"   • Успішні замовлення мають більше значення '{column_name}' (в {mean_1/mean_0:.6f} рази)")

                # Відображення абсолютної різниці
                abs_diff = abs(mean_0 - mean_1)
                print(f"   • Абсолютна різниця: {abs_diff:.6f}")

                # Відносна різниця у відсотках
                rel_diff = abs_diff / max(mean_0, mean_1) * 100
                print(f"   • Відносна різниця: {rel_diff:.4f}%")

                if abs(rel_diff) < 5:
                    print(f"   • Відносна різниця близька до нуля ({rel_diff:.4f}%), що вказує на відсутність практичної різниці.")
        else:
            print(f"⚠ Виявлено СТАТИСТИЧНО ЗНАЧУЩУ, але ПРАКТИЧНО НЕЗНАЧНУ різницю в значеннях колонки '{column_name}'.")
            print(f"   • p-значення показують статистичну значущість, але абсолютні значення майже однакові:")
            print(f"   • Неуспішні: {mean_0:.6f}, Успішні: {mean_1:.6f}")
            print(f"   • Відносна різниця: {abs(mean_0 - mean_1) / max(mean_0, mean_1) * 100:.4f}% (менше порогового значення 5%)")
    else:
        print(f"ℹ Не виявлено статистично значущої різниці в значеннях колонки '{column_name}' між групами.")

    print("="*80)

def analyze_column(df, column_name, group_column='is_successful', output_dir='.'):
    """
    Проводить повний аналіз для однієї колонки

    Args:
        df (pd.DataFrame): Датафрейм з даними
        column_name (str): Назва колонки для аналізу
        group_column (str): Назва колонки для групування
        output_dir (str): Директорія для збереження результатів

    Returns:
        dict: Результати аналізу
    """
    print(f"\nАналіз колонки: {column_name}")

    # Базові статистичні показники
    basic_stats = calculate_basic_stats(df, column_name, group_column)
    if basic_stats is None:
        print(f"Недостатньо даних для аналізу колонки {column_name}")
        return None

    # Проведення статистичних тестів
    tests_results, ci_results = perform_statistical_tests(df, column_name, group_column)
    if tests_results is None:
        print(f"Недостатньо даних для статистичних тестів колонки {column_name}")
        return None

    # Порівняльний аналіз
    comparison_df = comparative_analysis(basic_stats)

    # Збереження результатів у CSV файл
    date_time = datetime.datetime.now().strftime("%Y%m%d")
    os.makedirs(f'{output_dir}/column_stats', exist_ok=True)

    # Перевіряємо значущість колонки
    is_significant = is_column_significant(tests_results)
    significance_marker = "✓" if is_significant else "✗"

    # Виведення результатів аналізу в консоль
    display_column_analysis(column_name, basic_stats, tests_results, ci_results, comparison_df, is_significant)

    # Отримання середніх значень для кожної групи
    mean_0 = basic_stats.loc['Неуспішні', 'Середнє'] if 'Неуспішні' in basic_stats.index else np.nan
    mean_1 = basic_stats.loc['Успішні', 'Середнє'] if 'Успішні' in basic_stats.index else np.nan

    # Обчислення різниці та співвідношення середніх
    if not pd.isna(mean_0) and not pd.isna(mean_1) and mean_1 != 0:
        mean_diff = mean_0 - mean_1
        mean_ratio = mean_0 / mean_1
    else:
        mean_diff = np.nan
        mean_ratio = np.nan

    # Перевірка на практичну значущість
    practical_significance = is_practical_difference(mean_0, mean_1)

    # Збереження результатів у файли
    basic_stats.to_csv(f'{output_dir}/column_stats/{column_name}_basic_stats.csv')
    if tests_results is not None:
        tests_results.to_csv(f'{output_dir}/column_stats/{column_name}_tests_results.csv')
    if ci_results is not None:
        ci_results.to_csv(f'{output_dir}/column_stats/{column_name}_ci_results.csv')
    if comparison_df is not None:
        comparison_df.to_csv(f'{output_dir}/column_stats/{column_name}_comparison.csv')

    # Підготовка результатів для повернення (безпечне перетворення DataFrame на словник)
    basic_stats_dict = {}
    for idx in basic_stats.index:
        basic_stats_dict[idx] = {}
        for col in basic_stats.columns:
            basic_stats_dict[idx][col] = basic_stats.loc[idx, col]

    summary = {
        'column': column_name,
        'is_significant': is_significant,
        'practical_significance': practical_significance,
        'significance_marker': significance_marker,
        'mean_diff': mean_diff,
        'mean_ratio': mean_ratio,
        'relative_diff_percent': abs(mean_diff) / max(abs(mean_0), abs(mean_1)) * 100 if not pd.isna(mean_diff) and max(abs(mean_0), abs(mean_1)) > 0 else np.nan,
        'p_values': {
            't_test': tests_results.iloc[0]['p-значення'] if tests_results is not None else np.nan,
            'mann_whitney': tests_results.iloc[1]['p-значення'] if tests_results is not None else np.nan
        },
        'basic_stats': basic_stats_dict
    }

    return summary

def analyze_all_columns(file_path, group_column='is_successful', output_dir='.'):
    """
    Проводить аналіз всіх числових колонок у файлі

    Args:
        file_path (str): Шлях до файлу даних
        group_column (str): Назва колонки для групування
        output_dir (str): Директорія для збереження результатів
    """
    print("\n" + "="*80)
    print("КОМПЛЕКСНИЙ СТАТИСТИЧНИЙ АНАЛІЗ ВСІХ КОЛОНОК".center(80))
    print("="*80)

    # Завантаження даних
    df, numeric_columns = load_data(file_path, group_column)

    # Створюємо директорію для результатів
    os.makedirs(output_dir, exist_ok=True)

    # Аналіз кожної колонки
    all_results = []
    significant_columns = []

    for column in numeric_columns:
        column_result = analyze_column(df, column, group_column, output_dir)
        if column_result:
            all_results.append(column_result)
            if column_result['is_significant'] and column_result['practical_significance']:
                significant_columns.append((column, column_result['p_values']['t_test']))

    # Сортуємо значущі колонки за p-значенням (від найменшого до найбільшого)
    significant_columns.sort(key=lambda x: x[1] if not pd.isna(x[1]) else float('inf'))

    # Створюємо зведену таблицю значущих колонок
    significant_summary = []
    for column, p_value in significant_columns:
        column_data = next((r for r in all_results if r['column'] == column), None)
        if column_data:
            significant_summary.append({
                'Колонка': column,
                'p-значення (t-test)': format_value(p_value),
                'Середнє (Неуспішні)': format_value(column_data['basic_stats']['Неуспішні']['Середнє']),
                'Середнє (Успішні)': format_value(column_data['basic_stats']['Успішні']['Середнє']),
                'Різниця середніх': format_value(column_data['mean_diff']),
                'Відношення середніх': format_value(column_data['mean_ratio']),
                'Відносна різниця, %': format_value(column_data['relative_diff_percent'])
            })

    significant_df = pd.DataFrame(significant_summary)

    # Виведення підсумкової таблиці
    print("\n" + "="*80)
    print("КОЛОНКИ ЗІ СТАТИСТИЧНО ТА ПРАКТИЧНО ЗНАЧУЩОЮ РІЗНИЦЕЮ".center(80))
    print("="*80)

    if significant_df.empty:
        print("\nНе знайдено колонок зі статистично та практично значущою різницею.")
    else:
        print("\nКолонки зі статистично та практично значущою різницею (відсортовані за p-значенням):")
        print(tabulate(significant_df, headers='keys', tablefmt='grid', showindex=False))

        # Зберігаємо підсумкову таблицю у CSV
        significant_df.to_csv(f'{output_dir}/significant_columns_summary.csv', index=False)

    # Створення загального звіту у json
    date_time = datetime.datetime.now().strftime("%Y%m%d")
    with open(f'{output_dir}/analysis_report_{date_time}.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nДетальні результати аналізу збережено в директорії: {output_dir}/column_stats/")
    print(f"Загальний звіт збережено у файлі: {output_dir}/analysis_report_{date_time}.json")
    print(f"Підсумкова таблиця значущих колонок: {output_dir}/significant_columns_summary.csv")

    # Створення рейтингів колонок
    rankings = create_rankings({r['column']: r for r in all_results}, df, group_column)

    # Виведення рейтингів
    display_rankings(rankings, "Рейтинг колонок за зв'язком з is_successful")

    # Збереження рейтингів у файли
    for name, ranking in rankings.items():
        ranking.to_csv(f'{output_dir}/ranking_{name}.csv', index=False)

    # Підготовка зведеного звіту
    significant_columns = []
    for column_name, summary in {r['column']: r for r in all_results}.items():
        if summary is not None and summary.get('is_significant', False):
            significant_columns.append({
                'column': column_name,
                'p_value_t': summary['p_values']['t_test'],
                'p_value_mw': summary['p_values']['mann_whitney'],
                'mean_diff': summary.get('mean_diff', np.nan),
                'mean_ratio': summary.get('mean_ratio', np.nan),
                'relative_diff_percent': summary.get('relative_diff_percent', np.nan)
            })

    # Сортування значущих колонок за p-значенням
    significant_columns.sort(key=lambda x: x['p_value_t'])

    # Виведення зведеного звіту
    if significant_columns:
        print("\n\n=== Зведений звіт про статистично значущі колонки ===")

        report_df = pd.DataFrame(significant_columns)
        report_df.columns = [
            'Колонка',
            'p-значення t-тесту',
            'p-значення Манна-Уітні',
            'Різниця середніх',
            'Співвідношення середніх',
            'Відносна різниця (%)'
        ]

        print(tabulate(report_df, headers='keys', tablefmt='pretty', showindex=False, floatfmt=".4f"))

        # Збереження звіту у файл
        report_df.to_csv(f'{output_dir}/significant_columns_report.csv', index=False)

        print(f"\nВиявлено {len(significant_columns)} колонок зі статистично значущими відмінностями.")
    else:
        print("\nНе виявлено колонок із статистично значущими відмінностями.")

    # Зведений звіт у форматі JSON
    summary_report = {
        'file_name': os.path.basename(file_path),
        'total_columns_analyzed': len(numeric_columns),
        'significant_columns_count': len(significant_columns),
        'analysis_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'columns': {r['column']: r for r in all_results}
    }

    with open(f'{output_dir}/analysis_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, ensure_ascii=False, indent=4, default=str)

    print(f"\nАналіз завершено. Результати збережено в директорії {output_dir}")

    return df, all_results, rankings

# Запуск програми
if __name__ == "__main__":
    # Параметри для аналізу
    FILE_PATH = 'cleaned_result.csv'  # Шлях до файлу
    GROUP_COLUMN = 'is_successful'    # Колонка для групування
    OUTPUT_DIR = 'all_columns_analysis'  # Директорія для збереження результатів

    # Виконання аналізу
    analyze_all_columns(FILE_PATH, GROUP_COLUMN, OUTPUT_DIR)
