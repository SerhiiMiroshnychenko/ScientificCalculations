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

    # Тест Колмогорова-Смирнова
    try:
        ks_stat, ks_pvalue = stats.ks_2samp(group_0, group_1)
    except:
        ks_stat, ks_pvalue = np.nan, np.nan

    # Розрахунок довірчих інтервалів
    try:
        ci_0 = stats.t.interval(1-alpha, len(group_0)-1, loc=group_0.mean(), scale=stats.sem(group_0))
        ci_1 = stats.t.interval(1-alpha, len(group_1)-1, loc=group_1.mean(), scale=stats.sem(group_1))
    except:
        ci_0 = (np.nan, np.nan)
        ci_1 = (np.nan, np.nan)

    # Створення датафрейму із результатами
    tests_results = pd.DataFrame({
        'Тест': ['t-тест (Welch)', 'Тест Манна-Уітні', 'Тест Колмогорова-Смирнова'],
        'Статистика': [t_stat, mw_stat, ks_stat],
        'p-значення': [t_pvalue, mw_pvalue, ks_pvalue],
        'Значущість': [
            "Значуща різниця" if p < alpha and not np.isnan(p) else "Немає значущої різниці"
            for p in [t_pvalue, mw_pvalue, ks_pvalue]
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

    # Перевіряємо, чи хоча б один тест показує значущу різницю
    significant_tests = tests_results['p-значення'].apply(lambda p: p < 0.05 if not np.isnan(p) else False)
    return significant_tests.any()

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

    # Форматування результатів
    formatted_basic_stats = basic_stats.copy()
    for col in formatted_basic_stats.columns:
        if col == 'Кількість':
            continue
        formatted_basic_stats[col] = formatted_basic_stats[col].map(format_value)

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
        'significance_marker': significance_marker,
        'mean_diff': mean_diff,
        'mean_ratio': mean_ratio,
        'p_values': {
            't_test': tests_results.iloc[0]['p-значення'] if tests_results is not None else np.nan,
            'mann_whitney': tests_results.iloc[1]['p-значення'] if tests_results is not None else np.nan,
            'ks_test': tests_results.iloc[2]['p-значення'] if tests_results is not None else np.nan
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
            if column_result['is_significant']:
                significant_columns.append((column, column_result['p_values']['t_test']))

    # Сортуємо значущі колонки за p-значенням (від найменшого до найбільшого)
    significant_columns.sort(key=lambda x: x[1] if not np.isnan(x[1]) else float('inf'))

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
                'Відношення середніх': format_value(column_data['mean_ratio'])
            })

    significant_df = pd.DataFrame(significant_summary)

    # Виведення підсумкової таблиці
    print("\n" + "="*80)
    print("КОЛОНКИ ЗІ СТАТИСТИЧНО ЗНАЧУЩОЮ РІЗНИЦЕЮ".center(80))
    print("="*80)

    if significant_df.empty:
        print("\nНе знайдено колонок зі статистично значущою різницею.")
    else:
        print("\nКолонки зі статистично значущою різницею (відсортовані за p-значенням):")
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

# Запуск програми
if __name__ == "__main__":
    # Параметри для аналізу
    FILE_PATH = 'cleaned_result.csv'  # Шлях до файлу
    GROUP_COLUMN = 'is_successful'    # Колонка для групування
    OUTPUT_DIR = 'all_columns_analysis'  # Директорія для збереження результатів

    # Виконання аналізу
    analyze_all_columns(FILE_PATH, GROUP_COLUMN, OUTPUT_DIR)
