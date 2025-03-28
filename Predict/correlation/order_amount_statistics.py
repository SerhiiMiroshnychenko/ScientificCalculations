import pandas as pd
import numpy as np
import datetime
from tabulate import tabulate
from scipy import stats
import statsmodels.api as sm

# Функція для завантаження та підготовки даних
def load_data(file_path, column_name, group_column='is_successful'):
    """
    Завантажує дані та готує їх до аналізу

    Args:
        file_path (str): Шлях до файлу даних
        column_name (str): Назва колонки для аналізу
        group_column (str): Назва колонки для групування

    Returns:
        pd.DataFrame: Завантажений датафрейм
        pd.Series: Дані групи 0
        pd.Series: Дані групи 1
    """
    print(f"Завантаження даних з {file_path}...")

    # Завантаження даних
    df = pd.read_csv(file_path)

    # Заміна від'ємних значень на 0
    negative_count = (df[column_name] < 0).sum()
    if negative_count > 0:
        print(f"Знайдено {negative_count} від'ємних значень у колонці {column_name}. Замінюємо їх на 0.")
        df[column_name] = df[column_name].apply(lambda x: max(0, x))

    # Перетворення стовпця групування на числовий тип (0 або 1), якщо це ще не зроблено
    df[group_column] = df[group_column].astype(int)

    # Розділення даних на групи
    group_0 = df[df[group_column] == 0][column_name]
    group_1 = df[df[group_column] == 1][column_name]

    print(f"Завантажено {len(df)} записів.")
    print(f"Група 0 (Неуспішні): {len(group_0)} записів")
    print(f"Група 1 (Успішні): {len(group_1)} записів")

    return df, group_0, group_1

# Функція для розрахунку базових статистичних показників
def calculate_basic_stats(data_0, data_1, group_names=['Неуспішні', 'Успішні']):
    """
    Розраховує базові статистичні показники для двох груп даних

    Args:
        data_0 (pd.Series): Дані першої групи
        data_1 (pd.Series): Дані другої групи
        group_names (list): Назви груп

    Returns:
        pd.DataFrame: Датафрейм із статистичними показниками
    """
    print("Обчислення базових статистичних показників...")

    # Створення словника з даними для кожної групи
    data_dict = {
        group_names[0]: data_0,
        group_names[1]: data_1
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
            'Мінімум': data.min(),
            'Максимум': data.max(),
            'Квартиль 25%': data.quantile(0.25),
            'Квартиль 75%': data.quantile(0.75),
            'Коефіцієнт варіації': data.std() / data.mean() if data.mean() != 0 else np.nan,
            'Коефіцієнт асиметрії': stats.skew(data),
            'Ексцес': stats.kurtosis(data)
        }

        # Додавання статистик до датафрейму
        if stats_df.empty:
            stats_df = pd.DataFrame(group_stats, index=[group_name])
        else:
            stats_df.loc[group_name] = group_stats

    return stats_df

# Функція для проведення статистичних тестів
def perform_statistical_tests(data_0, data_1, alpha=0.05):
    """
    Проводить статистичні тести для порівняння двох груп даних

    Args:
        data_0 (pd.Series): Дані першої групи
        data_1 (pd.Series): Дані другої групи
        alpha (float): Рівень значущості

    Returns:
        pd.DataFrame: Датафрейм із результатами тестів
    """
    print("Проведення статистичних тестів...")

    # t-тест для незалежних вибірок
    t_stat, t_pvalue = stats.ttest_ind(data_0, data_1, equal_var=False)

    # Тест Манна-Уітні
    mw_stat, mw_pvalue = stats.mannwhitneyu(data_0, data_1)

    # Тест Колмогорова-Смирнова
    ks_stat, ks_pvalue = stats.ks_2samp(data_0, data_1)

    # Розрахунок довірчих інтервалів
    ci_0 = stats.t.interval(1-alpha, len(data_0)-1, loc=data_0.mean(), scale=stats.sem(data_0))
    ci_1 = stats.t.interval(1-alpha, len(data_1)-1, loc=data_1.mean(), scale=stats.sem(data_1))

    # Створення датафрейму із результатами
    tests_results = pd.DataFrame({
        'Тест': ['t-тест (Welch)', 'Тест Манна-Уітні', 'Тест Колмогорова-Смирнова'],
        'Статистика': [t_stat, mw_stat, ks_stat],
        'p-значення': [t_pvalue, mw_pvalue, ks_pvalue],
        'Значущість': [
            "Значуща різниця" if p < alpha else "Немає значущої різниці"
            for p in [t_pvalue, mw_pvalue, ks_pvalue]
        ],
        'Інтерпретація': [
            "Середні значення відрізняються" if t_pvalue < alpha else "Середні значення статистично не відрізняються",
            "Розподіли відрізняються" if mw_pvalue < alpha else "Розподіли статистично не відрізняються",
            "Розподіли відрізняються" if ks_pvalue < alpha else "Розподіли статистично не відрізняються"
        ]
    })

    # Результати довірчих інтервалів
    ci_results = pd.DataFrame({
        'Група': ['Неуспішні', 'Успішні'],
        'Середнє': [data_0.mean(), data_1.mean()],
        'Нижня межа CI': [ci_0[0], ci_1[0]],
        'Верхня межа CI': [ci_0[1], ci_1[1]],
        'Ширина CI': [ci_0[1] - ci_0[0], ci_1[1] - ci_1[0]]
    })

    return tests_results, ci_results

# Функція для форматування і виведення таблиць
def display_tables(basic_stats, tests_results, ci_results):
    """
    Виводить таблиці із статистичними показниками та результатами тестів

    Args:
        basic_stats (pd.DataFrame): Базові статистичні показники
        tests_results (pd.DataFrame): Результати статистичних тестів
        ci_results (pd.DataFrame): Довірчі інтервали
    """
    print("\n" + "="*80)
    print("СТАТИСТИЧНИЙ АНАЛІЗ ДАНИХ".center(80))
    print("="*80)

    # Форматування таблиці базових статистик
    basic_stats_formatted = basic_stats.copy()
    for col in basic_stats_formatted.columns:
        if col == 'Кількість':
            continue
        basic_stats_formatted[col] = basic_stats_formatted[col].map(lambda x: f"{x:,.4f}" if not pd.isna(x) else "N/A")

    print("\nОсновні статистичні показники за групами:")
    print(tabulate(basic_stats_formatted, headers='keys', tablefmt='grid', showindex=True))

    # Форматування таблиці результатів тестів
    tests_formatted = tests_results.copy()
    tests_formatted['p-значення'] = tests_formatted['p-значення'].map(lambda x: f"{x:.8f}")
    tests_formatted['Статистика'] = tests_formatted['Статистика'].map(lambda x: f"{x:.4f}")

    print("\nРезультати статистичних тестів:")
    print(tabulate(tests_formatted, headers='keys', tablefmt='grid', showindex=False))

    # Форматування таблиці довірчих інтервалів
    ci_formatted = ci_results.copy()
    numeric_cols = ['Середнє', 'Нижня межа CI', 'Верхня межа CI', 'Ширина CI']
    for col in numeric_cols:
        ci_formatted[col] = ci_formatted[col].map(lambda x: f"{x:,.4f}")

    print("\nДовірчі інтервали для середніх значень (95%):")
    print(tabulate(ci_formatted, headers='keys', tablefmt='grid', showindex=False))

# Функція для порівняльного аналізу
def comparative_analysis(basic_stats):
    """
    Проводить порівняльний аналіз між групами

    Args:
        basic_stats (pd.DataFrame): Базові статистичні показники

    Returns:
        pd.DataFrame: Датафрейм з результатами порівняння
    """
    print("\nПорівняльний аналіз між групами...")

    # Вибір статистик для порівняння
    stats_to_compare = ['Середнє', 'Медіана', 'Стандартне відхилення',
                        'Квартиль 25%', 'Квартиль 75%', 'Коефіцієнт варіації',
                        'Коефіцієнт асиметрії', 'Ексцес']

    comparison_data = []
    for stat in stats_to_compare:
        group_0_value = basic_stats.iloc[0][stat]
        group_1_value = basic_stats.iloc[1][stat]

        if pd.isna(group_0_value) or pd.isna(group_1_value):
            ratio = "N/A"
            diff = "N/A"
        else:
            ratio = group_0_value / group_1_value if group_1_value != 0 else float('inf')
            diff = group_0_value - group_1_value

        comparison_data.append({
            'Статистика': stat,
            'Неуспішні': f"{group_0_value:,.4f}" if not pd.isna(group_0_value) else "N/A",
            'Успішні': f"{group_1_value:,.4f}" if not pd.isna(group_1_value) else "N/A",
            'Різниця': f"{diff:,.4f}" if not isinstance(diff, str) else diff,
            'Відношення': f"{ratio:.4f}" if not isinstance(ratio, str) else ratio
        })

    comparison_df = pd.DataFrame(comparison_data)
    print("\nПорівняльний аналіз статистичних показників:")
    print(tabulate(comparison_df, headers='keys', tablefmt='grid', showindex=False))

    return comparison_df

# Функція для виведення висновків
def print_conclusions(basic_stats, tests_results):
    """
    Виводить висновки на основі статистичного аналізу

    Args:
        basic_stats (pd.DataFrame): Базові статистичні показники
        tests_results (pd.DataFrame): Результати статистичних тестів
    """
    print("\n" + "="*80)
    print("ВИСНОВКИ".center(80))
    print("="*80)

    # Отримання значень для висновків
    mean_0 = basic_stats.iloc[0]['Середнє']
    mean_1 = basic_stats.iloc[1]['Середнє']
    median_0 = basic_stats.iloc[0]['Медіана']
    median_1 = basic_stats.iloc[1]['Медіана']

    # t-тест значущість
    t_test_significant = tests_results.iloc[0]['p-значення'] < 0.05
    mw_test_significant = tests_results.iloc[1]['p-значення'] < 0.05
    ks_test_significant = tests_results.iloc[2]['p-значення'] < 0.05

    print("\n1. Порівняння середніх і медіан:")
    if mean_0 > mean_1:
        print(f"   ⚠ Неуспішні замовлення мають БІЛЬШУ середню суму (в {mean_0/mean_1:.2f} рази)")
    else:
        print(f"   ⚠ Успішні замовлення мають БІЛЬШУ середню суму (в {mean_1/mean_0:.2f} рази)")

    if median_0 > median_1:
        print(f"   ⚠ Неуспішні замовлення мають БІЛЬШУ медіанну суму (в {median_0/median_1:.2f} рази)")
    else:
        print(f"   ⚠ Успішні замовлення мають БІЛЬШУ медіанну суму (в {median_1/median_0:.2f} рази)")

    print("\n2. Результати статистичних тестів:")
    print(f"   • t-тест (Welch): {'Статистично значуща різниця середніх' if t_test_significant else 'Немає статистично значущої різниці середніх'}")
    print(f"   • Тест Манна-Уітні: {'Статистично значуща різниця в розподілах' if mw_test_significant else 'Немає статистично значущої різниці в розподілах'}")
    print(f"   • Тест Колмогорова-Смирнова: {'Статистично значуща різниця в розподілах' if ks_test_significant else 'Немає статистично значущої різниці в розподілах'}")

    print("\n3. Загальний висновок:")
    if t_test_significant or mw_test_significant or ks_test_significant:
        print("   ⚠ Існує статистично значуща різниця між сумами замовлень у успішних і неуспішних групах.")
        if mean_0 > mean_1:
            print("   ⚠ Неуспішні замовлення в середньому мають більші суми, що може свідчити про підвищений ризик невиконання замовлень з великими сумами.")
        else:
            print("   ⚠ Успішні замовлення в середньому мають більші суми, що може свідчити про підвищену увагу до замовлень з великими сумами.")
    else:
        print("   ℹ Не виявлено статистично значущої різниці між сумами замовлень у успішних і неуспішних групах.")

# Головна функція
def main(file_path, column_name='order_amount', group_column='is_successful',
         group_names=['Неуспішні', 'Успішні'], output_dir='.'):
    """
    Головна функція для виконання повного статистичного аналізу

    Args:
        file_path (str): Шлях до файлу даних
        column_name (str): Назва колонки для аналізу
        group_column (str): Назва колонки для групування
        group_names (list): Назви груп
        output_dir (str): Директорія для збереження результатів
    """
    # Завантаження даних
    df, group_0, group_1 = load_data(file_path, column_name, group_column)

    # Базові статистичні показники
    basic_stats = calculate_basic_stats(group_0, group_1, group_names)

    # Проведення статистичних тестів
    tests_results, ci_results = perform_statistical_tests(group_0, group_1)

    # Виведення таблиць
    display_tables(basic_stats, tests_results, ci_results)

    # Порівняльний аналіз
    comparison_df = comparative_analysis(basic_stats)

    # Виведення висновків
    print_conclusions(basic_stats, tests_results)

    # Зберігання результатів у CSV файл
    date_time = datetime.datetime.now().strftime("%Y%m%d")
    basic_stats.to_csv(f'{output_dir}/{column_name}_basic_stats_{date_time}.csv')
    tests_results.to_csv(f'{output_dir}/{column_name}_tests_results_{date_time}.csv')
    ci_results.to_csv(f'{output_dir}/{column_name}_ci_results_{date_time}.csv')
    comparison_df.to_csv(f'{output_dir}/{column_name}_comparison_{date_time}.csv')

    print(f"\nРезультати аналізу збережено в директорії {output_dir}")

# Запуск програми
if __name__ == "__main__":
    # Параметри для аналізу
    FILE_PATH = 'cleaned_result.csv'  # Шлях до файлу
    COLUMN_NAME = 'order_amount'      # Колонка для аналізу
    GROUP_COLUMN = 'is_successful'    # Колонка для групування
    GROUP_NAMES = ['Неуспішні', 'Успішні']  # Назви груп
    OUTPUT_DIR = '.'  # Директорія для збереження результатів

    # Виконання аналізу
    main(FILE_PATH, COLUMN_NAME, GROUP_COLUMN, GROUP_NAMES, OUTPUT_DIR)
