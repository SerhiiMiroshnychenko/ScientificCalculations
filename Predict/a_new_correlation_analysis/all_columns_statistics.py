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

# Запуск програми
if __name__ == "__main__":
    # Параметри для аналізу
    FILE_PATH = 'cleaned_result.csv'  # Шлях до файлу
    GROUP_COLUMN = 'is_successful'    # Колонка для групування
    OUTPUT_DIR = 'all_columns_analysis'  # Директорія для збереження результатів

    # Виконання аналізу
    analyze_all_columns(FILE_PATH, GROUP_COLUMN, OUTPUT_DIR)
