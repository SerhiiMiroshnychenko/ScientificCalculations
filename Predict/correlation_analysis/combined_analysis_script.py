#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')  # Встановлюємо Agg бекенд
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif, f_classif, RFE, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from scipy import stats
from tabulate import tabulate
import os
import json
import joblib
from datetime import datetime
from matplotlib import scale as mscale
from matplotlib.transforms import Transform
import matplotlib.transforms as mtransforms
import logging
import sys

# Клас для одночасного запису в консоль та файл
class TeeOutput:
    """
    Клас для перенаправлення виводу в консоль та файл одночасно
    """
    def __init__(self, file_path, mode='a'):
        """
        Ініціалізує об'єкт перенаправлення виводу

        Args:
            file_path (str): Шлях до файлу для запису
            mode (str): Режим відкриття файлу ('a' - додавання, 'w' - перезапис)
        """
        self.file = open(file_path, mode, encoding='utf-8')
        self.stdout = sys.stdout

    def write(self, data):
        """
        Записує дані в консоль та файл

        Args:
            data (str): Дані для запису
        """
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        """
        Очищує буфери виводу
        """
        self.file.flush()
        self.stdout.flush()

    def close(self):
        """
        Закриває файл виводу
        """
        if self.file:
            self.file.close()

# Створюємо директорії для збереження результатів
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = f"data_analysis_results_{timestamp}"
stats_dir = f"{results_dir}/statistics"
vis_dir = f"{results_dir}/visualization"
feature_dir = f"{results_dir}/feature_selection"

# Створюємо директорії
os.makedirs(results_dir, exist_ok=True)
os.makedirs(stats_dir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)
os.makedirs(feature_dir, exist_ok=True)

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{results_dir}/analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Встановлюємо українську локаль для графіків
import locale
try:
    locale.setlocale(locale.LC_ALL, 'uk_UA.UTF-8')
except:
    logger.warning("Українська локаль не знайдена, використовуємо стандартну")

# Словник для перекладу назв ознак
feature_names_ua = {
    'order_amount': 'Сума замовлення',
    'order_messages': 'Кількість повідомлень',
    'order_changes': 'Кількість змін в замовлені',
    'partner_success_rate': 'Сердній % успішних замовлень клієнта',
    'partner_total_orders': 'Кількість замовлень клієнта',
    'partner_order_age_days': 'Термін співпраці',
    'partner_avg_amount': 'Середня сума замовлень клієнта',
    'partner_success_avg_amount': 'Середня сума успішних замовлень клієнта',
    'partner_fail_avg_amount': 'Середня сума невдалих замовлень клієнта',
    'partner_total_messages': 'Загальна кількість повідомлень клієнта',
    'partner_success_avg_messages': 'Середня кількість повідомлень успішних замовлень',
    'partner_fail_avg_messages': 'Середня кількість повідомлень невдалих замовлень',
    'partner_avg_changes': 'Середня кількість змін в замовленях клієнта',
    'partner_success_avg_changes': 'Середня кількість змін в успішних замовленях клієнта',
    'partner_fail_avg_changes': 'Середня кількість змін в невдалих замовленях клієнта',
    'day_of_week': 'День тижня',
    'month': 'Місяць',
    'quarter': 'Квартал',
    'hour_of_day': 'Година доби',
    'order_lines_count': 'Кількість позицій в замовленні',
    'discount_total': 'Загальна знижка',
    'salesperson': 'Менеджер',
    'source': 'Джерело замовлення'
}

#####################################
# Частина 1: Завантаження та підготовка даних
#####################################

def get_ua_feature_name(feature):
    """
    Повертає українську назву ознаки, якщо вона є в словнику

    Args:
        feature (str): Назва ознаки англійською

    Returns:
        str: Назва ознаки українською або оригінальна назва
    """
    return feature_names_ua.get(feature, feature)

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
    logger.info(f"Завантаження даних з {file_path}...")

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
            logger.info(f"Знайдено {negative_count} від'ємних значень у колонці {col}. Замінюємо їх на 0.")
            df[col] = df[col].apply(lambda x: max(0, x) if not pd.isna(x) else x)

    logger.info(f"Завантажено {len(df)} записів.")
    logger.info(f"Для аналізу обрано {len(numeric_columns)} числових колонок.")

    return df, numeric_columns

def preprocess_data(df):
    """
    Проводить попередню обробку даних

    Args:
        df (pd.DataFrame): Датафрейм з даними

    Returns:
        pd.DataFrame: Оброблений датафрейм
        dict: Словник з енкодерами для категоріальних змінних
    """
    logger.info("Попередня обробка даних...")

    # Перевірка та обробка пропущених значень
    missing_values = df.isnull().sum()
    columns_with_missing = missing_values[missing_values > 0]

    if not columns_with_missing.empty:
        logger.warning(f"Знайдено стовпці з пропущеними значеннями: {columns_with_missing}")

        # Заповнення пропущених значень
        num_columns = df.select_dtypes(include=['number']).columns
        cat_columns = df.select_dtypes(exclude=['number']).columns

        if not num_columns.empty:
            num_imputer = SimpleImputer(strategy='median')
            df[num_columns] = num_imputer.fit_transform(df[num_columns])

        if not cat_columns.empty:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df[cat_columns] = cat_imputer.fit_transform(df[cat_columns])

        logger.info("Пропущені значення заповнено")
    else:
        logger.info("Пропущених значень не виявлено")

    # Обробка категоріальних змінних
    logger.info("Кодуємо категоріальні змінні...")
    cat_columns = df.select_dtypes(include=['object']).columns.tolist()
    ignore_cols = ['order_id', 'state']
    cat_columns = [col for col in cat_columns if col not in ignore_cols]

    df_encoded = df.copy()
    encoders = {}

    for col in cat_columns:
        encoder = LabelEncoder()
        df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
        encoders[col] = encoder

    logger.info("Попередня обробка даних завершена")
    return df_encoded, encoders

def format_value(value):
    """
    Форматує числове значення для відображення

    Args:
        value: Числове значення

    Returns:
        str: Відформатоване значення
    """
    if pd.isna(value):
        return "N/A"
    elif isinstance(value, (int, np.integer)):
        return f"{value:,}".replace(",", " ")
    elif abs(value) < 0.001 or abs(value) > 1000:
        return f"{value:.2e}"
    else:
        return f"{value:.4f}".rstrip('0').rstrip('.') if '.' in f"{value:.4f}" else f"{value:.0f}"

#####################################
# Частина 2: Статистичний аналіз
#####################################

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
            'Співвідношення': ratio
        })

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

    # Перевіряємо, чи є хоча б один значущий результат
    significant_results = tests_results[tests_results['Значущість'] == 'Значуща різниця']
    return len(significant_results) > 0

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
    if pd.isna(mean_0) or pd.isna(mean_1) or mean_1 == 0:
        return False

    relative_diff = abs(mean_0 - mean_1) / abs(mean_1)
    return relative_diff > threshold

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
    # Виводимо заголовок
    stars = "**" if is_significant else ""
    header = f"\n{stars}{'='*50}{stars}\n"
    header += f"{stars}АНАЛІЗ КОЛОНКИ: {column_name}{stars}\n"
    header += f"{stars}{'='*50}{stars}\n"
    print(header)
    logger.info(header)

    # Виводимо базові статистики
    if basic_stats is not None and not basic_stats.empty:
        print("\nБазові статистичні показники:")
        print(tabulate(basic_stats, headers='keys', tablefmt='pipe', floatfmt='.4f'))
        logger.info("\nБазові статистичні показники:")

    # Виводимо результати порівняльного аналізу
    if comparison_df is not None and not comparison_df.empty:
        print("\nПорівняльний аналіз між групами:")
        print(tabulate(comparison_df, headers='keys', tablefmt='pipe', floatfmt='.4f', showindex=False))
        logger.info("\nПорівняльний аналіз між групами:")

    # Виводимо результати тестів
    if tests_results is not None and not tests_results.empty:
        print("\nРезультати статистичних тестів:")
        print(tabulate(tests_results, headers='keys', tablefmt='pipe', floatfmt='.4f', showindex=False))
        logger.info("\nРезультати статистичних тестів:")

    # Виводимо довірчі інтервали
    if ci_results is not None and not ci_results.empty:
        print("\nДовірчі інтервали (95%):")
        print(tabulate(ci_results, headers='keys', tablefmt='pipe', floatfmt='.4f', showindex=False))
        logger.info("\nДовірчі інтервали (95%):")

    # Виводимо висновок
    conclusion = "\nВИСНОВОК: Виявлено статистично значущу різницю між групами" if is_significant else "\nВИСНОВОК: Не виявлено статистично значущої різниці між групами"
    print(conclusion)
    logger.info(conclusion)

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
    # Перевіряємо тип даних
    if df[column_name].dtype not in [np.int64, np.float64]:
        logger.warning(f"Колонка {column_name} не є числовою. Статистичний аналіз неможливий.")
        return None

    results = {}

    # Базові статистики
    basic_stats = calculate_basic_stats(df, column_name, group_column)
    results['basic_stats'] = basic_stats

    # Статистичні тести
    tests_results, ci_results = perform_statistical_tests(df, column_name, group_column)
    results['tests_results'] = tests_results
    results['ci_results'] = ci_results

    # Порівняльний аналіз
    comparison_df = comparative_analysis(basic_stats)
    results['comparison_df'] = comparison_df

    # Визначаємо, чи є значуща різниця
    is_significant = is_column_significant(tests_results)
    results['is_significant'] = is_significant

    # Визначаємо, чи є практична різниця
    if basic_stats is not None and not basic_stats.empty:
        mean_0 = basic_stats.iloc[0]['Середнє']
        mean_1 = basic_stats.iloc[1]['Середнє']
        is_practical = is_practical_difference(mean_0, mean_1)
        results['is_practical'] = is_practical
    else:
        results['is_practical'] = False

    # Виводимо результати
    display_column_analysis(column_name, basic_stats, tests_results, ci_results, comparison_df, is_significant)

    # Логуємо завершення аналізу
    logger.info(f"Аналіз колонки {column_name} завершено")

    return results

def analyze_all_columns(df, numeric_columns, group_column='is_successful', output_dir='.'):
    """
    Проводить аналіз всіх числових колонок у датафреймі

    Args:
        df (pd.DataFrame): Датафрейм з даними
        numeric_columns (list): Список числових колонок для аналізу
        group_column (str): Назва колонки для групування
        output_dir (str): Директорія для збереження результатів

    Returns:
        dict: Зведені результати аналізу
    """
    summary_results = {}
    significant_columns = []

    logger.info(f"Початок аналізу {len(numeric_columns)} числових колонок...")

    for column_name in numeric_columns:
        logger.info(f"\nАналіз колонки: {column_name}")

        # Проводимо аналіз
        results = analyze_column(df, column_name, group_column, output_dir)

        if results is not None:
            summary_results[column_name] = {
                'is_significant': results['is_significant'],
                'is_practical': results.get('is_practical', False)
            }

            # Додаємо в список значущих колонок
            if results['is_significant']:
                significant_columns.append(column_name)

    # Виводимо загальний підсумок
    logger.info("\n" + "="*50)
    logger.info(f"ПІДСУМОК АНАЛІЗУ ({len(numeric_columns)} колонок)")
    logger.info("="*50)
    logger.info(f"Кількість колонок із статистично значущою різницею: {len(significant_columns)}")
    if significant_columns:
        logger.info("Колонки зі значущою різницею:")
        for col in significant_columns:
            logger.info(f"  - {col}")

    logger.info("\nАналіз колонок завершено")

    return summary_results

#####################################
# Частина 3: Візуалізація даних
#####################################

def create_violin_plot(df, column_name, output_dir='.'):
    """
    Створює віоліновий графік для порівняння розподілів

    Args:
        df (pd.DataFrame): Датафрейм з даними
        column_name (str): Назва колонки для аналізу
        output_dir (str): Директорія для збереження результатів
    """
    if df[column_name].dtype not in [np.int64, np.float64]:
        logger.warning(f"Колонка {column_name} не є числовою. Пропускаємо віоліновий графік.")
        return

    logger.info(f"Створюємо віоліновий графік для {column_name}...")
    plt.figure(figsize=(10, 6))

    # Визначаємо межу для 95-го перцентиля
    p95 = df[column_name].quantile(0.95)

    # Фільтруємо дані за 95-м перцентилем для кращої візуалізації
    df_filtered = df[df[column_name] <= p95]

    # Створюємо віоліновий графік
    ax = sns.violinplot(x='is_successful', y=column_name, data=df_filtered,
                        inner='box', cut=0, density_norm='width')

    # Налаштування графіка
    ax.set_title(f'Розподіл {column_name} за успішністю замовлення\n(віоліновий графік)')
    ax.set_xlabel('Успішність замовлення (0 - невдале, 1 - успішне)')
    ax.set_ylabel(column_name)

    # Додаємо медіану та середнє значення як текст
    for i, success in enumerate([0, 1]):
        subset = df[df['is_successful'] == success][column_name]
        median = subset.median()
        mean = subset.mean()
        plt.text(i, median - p95 * 0.1, f'Медіана: {median:.1f}', ha='center')
        plt.text(i, median + p95 * 0.1, f'Середнє: {mean:.1f}', ha='center')

    # Додаємо інформацію про викиди
    plt.text(0.5, p95 * 0.9, f"95-й перцентиль: {p95:.0f}\nМакс. значення: {df[column_name].max():.0f}",
             ha='center', bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/violin_plot_{column_name}.png", dpi=300)
    plt.close()


def create_density_plot(df, column_name, output_dir='.'):
    """
    Створює графік щільності розподілу

    Args:
        df (pd.DataFrame): Датафрейм з даними
        column_name (str): Назва колонки для аналізу
        output_dir (str): Директорія для збереження результатів
    """
    if df[column_name].dtype not in [np.int64, np.float64]:
        logger.warning(f"Колонка {column_name} не є числовою. Пропускаємо графік щільності.")
        return

    logger.info(f"Створюємо графік щільності розподілу для {column_name}...")

    # Створюємо один графік
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # Визначаємо межу для 95-го перцентиля
    p95 = df[column_name].quantile(0.95)

    # Малюємо графік щільності для кожної категорії
    for i, (success, color, label) in enumerate([(0, 'forestgreen', 'Невдалі замовлення'),
                                                 (1, 'crimson', 'Успішні замовлення')]):
        subset = df[df['is_successful'] == success][column_name]
        subset_filtered = subset[subset <= p95]  # Фільтруємо для кращої візуалізації

        if len(subset_filtered) > 0:
            mean_val = subset.mean()
            median_val = subset.median()

            # Малюємо графік щільності
            sns.kdeplot(data=subset_filtered, ax=ax, color=color, fill=True, alpha=0.5, label=f"{label}")

            # Додаємо вертикальні лінії
            plt.axvline(x=mean_val, color=color, linestyle='--', alpha=0.7)
            plt.axvline(x=median_val, color=color, linestyle=':', alpha=0.7)

            # Додаємо текст у верхній правий кут
            plt.text(0.98, 0.85 - i * 0.1,
                     f"{label}:\nСереднє: {mean_val:.1f}\nМедіана: {median_val:.1f}",
                     transform=ax.transAxes,  # Координати відносно графіка (0-1)
                     color=color, ha='right', va='top',
                     bbox=dict(facecolor='white', alpha=0.8))

    # Додаємо інформацію про максимальне значення
    plt.text(0.98, 0.65, f"95-й перцентиль: {p95:.0f}\nМакс. значення: {df[column_name].max():.0f}",
             transform=ax.transAxes, ha='right', va='top',
             bbox=dict(facecolor='white', alpha=0.8))

    plt.xlim(0, p95 * 1.1)  # Показуємо до 95-го перцентиля + 10%
    plt.ylim(0, plt.ylim()[1] * 1.1)  # Додаємо 10% простору зверху
    plt.title(f'Щільність розподілу {column_name} за успішністю замовлення\n(до 95-го перцентиля)')
    plt.xlabel(column_name)
    plt.ylabel('Щільність')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/density_{column_name}_by_success.png", dpi=300)
    plt.close()


def create_success_rate_plot(df, column_name, output_dir='.'):
    """
    Створює графік вірогідності успіху залежно від значення колонки

    Args:
        df (pd.DataFrame): Датафрейм з даними
        column_name (str): Назва колонки для аналізу
        output_dir (str): Директорія для збереження результатів
    """
    if df[column_name].dtype not in [np.int64, np.float64]:
        logger.warning(f"Колонка {column_name} не є числовою. Пропускаємо графік вірогідності успіху.")
        return

    logger.info(f"Створюємо графік вірогідності успіху залежно від {column_name}...")

    # Визначаємо межу для 95-го перцентиля
    p95 = df[column_name].quantile(0.95)

    # Використовуємо біни для групування даних за значенням колонки
    bins = 20
    df_filtered = df[df[column_name] <= p95].copy()

    # Перевіряємо, чи є достатньо даних і варіація
    if df_filtered[column_name].nunique() < 3 or len(df_filtered) < 10:
        logger.warning(f"Недостатньо унікальних значень або записів для {column_name}. Пропускаємо графік вірогідності успіху.")
        return

    df_filtered[f'{column_name}_bin'] = pd.cut(df_filtered[column_name], bins=bins)

    # Розраховуємо вірогідність успіху та середину біна
    success_rates = df_filtered.groupby(f'{column_name}_bin')['is_successful'].mean().reset_index()
    success_rates['bin_mid'] = success_rates[f'{column_name}_bin'].apply(lambda x: x.mid)
    success_rates['counts'] = df_filtered.groupby(f'{column_name}_bin').size().values

    # Створюємо графік
    plt.figure(figsize=(12, 8))

    # Основний графік вірогідності успіху
    ax1 = plt.gca()
    ax1.scatter(success_rates['bin_mid'], success_rates['is_successful'], s=success_rates['counts']/success_rates['counts'].max()*100+10, alpha=0.6, c='blue')
    ax1.set_xlabel(column_name)
    ax1.set_ylabel('Вірогідність успіху')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Додаємо лінію тренду
    try:
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(success_rates['bin_mid'], success_rates['is_successful'])
        x_line = np.linspace(success_rates['bin_mid'].min(), success_rates['bin_mid'].max(), 100)
        y_line = slope * x_line + intercept
        ax1.plot(x_line, y_line, 'r--', alpha=0.7)

        # Додаємо інформацію про тренд
        plt.text(0.02, 0.95, f"Тренд: y = {slope:.4f}x + {intercept:.4f}\nR² = {r_value**2:.4f}, p = {p_value:.4f}",
                 transform=ax1.transAxes, ha='left', va='top',
                 bbox=dict(facecolor='white', alpha=0.8))
    except:
        logger.warning(f"Не вдалося створити лінію тренду для {column_name}")

    # Додаємо другу вісь для гістограми
    ax2 = ax1.twinx()
    counts = df_filtered.groupby(f'{column_name}_bin').size()
    ax2.bar(success_rates['bin_mid'], counts.values, alpha=0.2, width=(success_rates['bin_mid'].max() - success_rates['bin_mid'].min())/bins)
    ax2.set_ylabel('Кількість замовлень')

    # Назва графіка
    plt.title(f'Вірогідність успіху залежно від значення {column_name}\n(до 95-го перцентиля)')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/success_rate_{column_name}.png", dpi=300)
    plt.close()


def create_category_histogram(df, column_name, output_dir='.'):
    """
    Створює гістограму для категоріальних колонок

    Args:
        df (pd.DataFrame): Датафрейм з даними
        column_name (str): Назва колонки для аналізу
        output_dir (str): Директорія для збереження результатів
    """
    if df[column_name].dtype in [np.int64, np.float64]:
        logger.warning(f"Колонка {column_name} є числовою. Пропускаємо гістограму для категорій.")
        return

    logger.info(f"Створюємо гістограму для категоріальної колонки {column_name}...")

    # Отримуємо частоти категорій
    value_counts = df[column_name].value_counts()
    total_values = len(df)

    # Обмежуємо кількість категорій до 15 найбільш частих
    if len(value_counts) > 15:
        value_counts = value_counts.head(15)
        truncated = True
    else:
        truncated = False

    plt.figure(figsize=(12, 8))
    bars = plt.bar(value_counts.index, value_counts.values)

    # Додаємо відсотки на стовпці
    for i, bar in enumerate(bars):
        height = bar.get_height()
        percentage = 100 * height / total_values
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{percentage:.1f}%', ha='center', va='bottom', rotation=0)

    plt.xlabel('Категорії')
    plt.ylabel('Кількість')
    title = f'Розподіл значень для {column_name}'
    if truncated:
        title += f'\n(показано топ-15 з {len(df[column_name].unique())} категорій)'
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/category_hist_{column_name}.png", dpi=300)
    plt.close()


def create_log_violin_plot(df, column_name, output_dir='.'):
    """
    Створює віоліновий графік з логарифмічною шкалою для колонок з великим розкидом значень

    Args:
        df (pd.DataFrame): Датафрейм з даними
        column_name (str): Назва колонки для аналізу
        output_dir (str): Директорія для збереження результатів
    """
    if df[column_name].dtype not in [np.int64, np.float64]:
        logger.warning(f"Колонка {column_name} не є числовою. Пропускаємо логарифмічний віоліновий графік.")
        return

    # Обчислюємо розмах значень
    min_val = df[column_name].min()
    max_val = df[column_name].max()

    # Визначаємо, чи має сенс використовувати логарифмічну шкалу
    if min_val <= 0 or max_val / min_val < 10:
        logger.warning(f"Колонка {column_name} не потребує логарифмічної шкали (min: {min_val}, max: {max_val}).")
        return

    logger.info(f"Створюємо віоліновий графік з логарифмічною шкалою для {column_name}...")

    # Створюємо графік
    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(x='is_successful', y=column_name, data=df, inner='box', cut=0)

    # Встановлюємо логарифмічну шкалу
    ax.set_yscale('log')

    # Налаштування графіка
    ax.set_title(f'Розподіл {column_name} за успішністю замовлення\n(логарифмічна шкала)')
    ax.set_xlabel('Успішність замовлення (0 - невдале, 1 - успішне)')
    ax.set_ylabel(f'{column_name} (log)')

    # Додаємо медіану та середнє значення як текст
    for i, success in enumerate([0, 1]):
        subset = df[df['is_successful'] == success][column_name]
        median = subset.median()
        mean = subset.mean()

        # Логарифмічна шкала потребує обережності при розміщенні тексту
        plt.text(i, median * 0.7, f'Медіана: {median:.1f}', ha='center')
        plt.text(i, median * 1.5, f'Середнє: {mean:.1f}', ha='center')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/log_violin_{column_name}.png", dpi=300)
    plt.close()


def create_log_violin_plot(df, column_name, output_dir='.'):
    """
    Створює віоліновий графік з логарифмічною шкалою для колонок з великим розкидом значень

    Args:
        df (pd.DataFrame): Датафрейм з даними
        column_name (str): Назва колонки для аналізу
        output_dir (str): Директорія для збереження результатів
    """
    if df[column_name].dtype not in [np.int64, np.float64]:
        logger.warning(f"Колонка {column_name} не є числовою. Пропускаємо логарифмічний віоліновий графік.")
        return

    # Обчислюємо розмах значень
    min_val = df[column_name].min()
    max_val = df[column_name].max()

    # Визначаємо, чи має сенс використовувати логарифмічну шкалу
    if min_val <= 0 or max_val / min_val < 10:
        logger.warning(f"Колонка {column_name} не потребує логарифмічної шкали (min: {min_val}, max: {max_val}).")
        return

    logger.info(f"Створюємо віоліновий графік з логарифмічною шкалою для {column_name}...")

    # Створюємо графік
    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(x='is_successful', y=column_name, data=df, inner='box', cut=0)

    # Встановлюємо логарифмічну шкалу
    ax.set_yscale('log')

    # Налаштування графіка
    ax.set_title(f'Розподіл {column_name} за успішністю замовлення\n(логарифмічна шкала)')
    ax.set_xlabel('Успішність замовлення (0 - невдале, 1 - успішне)')
    ax.set_ylabel(f'{column_name} (log)')

    # Додаємо медіану та середнє значення як текст
    for i, success in enumerate([0, 1]):
        subset = df[df['is_successful'] == success][column_name]
        median = subset.median()
        mean = subset.mean()

        # Логарифмічна шкала потребує обережності при розміщенні тексту
        plt.text(i, median * 0.7, f'Медіана: {median:.1f}', ha='center')
        plt.text(i, median * 1.5, f'Середнє: {mean:.1f}', ha='center')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/log_violin_{column_name}.png", dpi=300)
    plt.close()


def create_log_violin_plot(df, column_name, output_dir='.'):
    """
    Створює віоліновий графік з логарифмічною шкалою для колонок з великим розкидом значень

    Args:
        df (pd.DataFrame): Датафрейм з даними
        column_name (str): Назва колонки для аналізу
        output_dir (str): Директорія для збереження результатів
    """
    if df[column_name].dtype not in [np.int64, np.float64]:
        logger.warning(f"Колонка {column_name} не є числовою. Пропускаємо логарифмічний віоліновий графік.")
        return

    # Обчислюємо розмах значень
    min_val = df[column_name].min()
    max_val = df[column_name].max()

    # Визначаємо, чи має сенс використовувати логарифмічну шкалу
    if min_val <= 0 or max_val / min_val < 10:
        logger.warning(f"Колонка {column_name} не потребує логарифмічної шкали (min: {min_val}, max: {max_val}).")
        return

    logger.info(f"Створюємо віоліновий графік з логарифмічною шкалою для {column_name}...")

    # Створюємо графік
    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(x='is_successful', y=column_name, data=df, inner='box', cut=0)

    # Встановлюємо логарифмічну шкалу
    ax.set_yscale('log')

    # Налаштування графіка
    ax.set_title(f'Розподіл {column_name} за успішністю замовлення\n(логарифмічна шкала)')
    ax.set_xlabel('Успішність замовлення (0 - невдале, 1 - успішне)')
    ax.set_ylabel(f'{column_name} (log)')

    # Додаємо медіану та середнє значення як текст
    for i, success in enumerate([0, 1]):
        subset = df[df['is_successful'] == success][column_name]
        median = subset.median()
        mean = subset.mean()

        # Логарифмічна шкала потребує обережності при розміщенні тексту
        plt.text(i, median * 0.7, f'Медіана: {median:.1f}', ha='center')
        plt.text(i, median * 1.5, f'Середнє: {mean:.1f}', ha='center')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/log_violin_{column_name}.png", dpi=300)
    plt.close()


def visualize_all_columns(df, numeric_columns, output_dir='.'):
    """
    Створює візуалізації для всіх колонок у датафреймі

    Args:
        df (pd.DataFrame): Датафрейм з даними
        numeric_columns (list): Список числових колонок для аналізу
        output_dir (str): Директорія для збереження результатів
    """
    logger.info("Починаємо візуалізацію даних...")

    # Обробка та візуалізація усіх колонок
    for column_name in df.columns:
        if column_name == 'is_successful':
            continue

        # Попередня обробка колонки
        df = preprocess_column(df, column_name)

        if column_name in numeric_columns:
            # Для числових колонок створюємо всі типи графіків
            create_violin_plot(df, column_name, output_dir)
            create_density_plot(df, column_name, output_dir)
            create_success_rate_plot(df, column_name, output_dir)
            create_log_violin_plot(df, column_name, output_dir)
        else:
            # Для категоріальних колонок створюємо гістограму
            create_category_histogram(df, column_name, output_dir)

    logger.info(f"Візуалізацію завершено. Результати збережено в {output_dir}")


def preprocess_column(df, column_name):
    """
    Проводить попередню обробку колонки перед візуалізацією

    Args:
        df (pd.DataFrame): Датафрейм з даними
        column_name (str): Назва колонки для обробки

    Returns:
        pd.DataFrame: Оброблений датафрейм
    """
    if df[column_name].dtype in [np.int64, np.float64]:
        # Обробка від'ємних значень для числових колонок
        negative_values = (df[column_name] < 0).sum()
        if negative_values > 0:
            logger.info(f"Виявлено {negative_values} від'ємних значень в {column_name}, замінюємо на нулі")
            df.loc[df[column_name] < 0, column_name] = 0
            logger.info(f"Після обробки від'ємних значень: мін = {df[column_name].min()}, макс = {df[column_name].max()}")

    # Обробка пропущених значень для будь-яких типів колонок
    missing_values = df[column_name].isnull().sum()
    if missing_values > 0:
        logger.info(f"Виявлено {missing_values} пропущених значень в {column_name}")
        if df[column_name].dtype in [np.int64, np.float64]:
            logger.info(f"Заповнюємо медіаною для числової колонки {column_name}")
            df[column_name] = df[column_name].fillna(df[column_name].median())
        else:
            logger.info(f"Заповнюємо найчастішим значенням для категоріальної колонки {column_name}")
            df[column_name] = df[column_name].fillna(df[column_name].mode()[0])

    return df

#####################################
# Частина 4: Відбір найважливіших ознак
#####################################

def evaluate_features(X, y, cv=5):
    """
    Функція для оцінки важливості ознак різними методами

    Args:
        X (pd.DataFrame): Набір ознак
        y (pd.Series): Цільова змінна
        cv (int): Кількість фолдів для крос-валідації

    Returns:
        tuple: Кортеж з DataFrame результатів та словником моделей
    """
    logger.info(f"Оцінюємо важливість {X.shape[1]} ознак...")
    features = X.columns
    n_features = len(features)
    results_dict = {}
    models_dict = {}

    # Додаємо перевірку на мультиколінеарність
    correlation_matrix = X.corr().abs()
    upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    high_corr_pairs = [(upper.columns[i], upper.columns[j]) for i in range(len(upper.columns))
                       for j in range(len(upper.columns)) if i < j and upper.iloc[i, j] > 0.8]

    if high_corr_pairs:
        logger.warning(f"Виявлено {len(high_corr_pairs)} пар ознак з високою кореляцією (>0.8):")
        for pair in high_corr_pairs[:5]:
            logger.warning(f"  {pair[0]} <-> {pair[1]}: {correlation_matrix.loc[pair[0], pair[1]]:.3f}")
        if len(high_corr_pairs) > 5:
            logger.warning(f"  ... та {len(high_corr_pairs)-5} інших пар")

    # 1. Mutual Information
    logger.info("Обчислюємо Mutual Information...")
    mi_scores = mutual_info_classif(X, y, random_state=42, discrete_features='auto')
    results_dict['MI Score'] = dict(zip(features, mi_scores))

    # 2. ANOVA F-test
    logger.info("Обчислюємо ANOVA F-test...")
    f_scores, _ = f_classif(X, y)
    results_dict['F Score'] = dict(zip(features, f_scores))

    # 3. Spearman Correlation
    logger.info("Обчислюємо Spearman Correlation...")
    spearman_corr = X.corrwith(y, method="spearman").abs()
    results_dict['Spearman Score'] = dict(zip(features, spearman_corr))

    # 5. Logistic Regression Coefficients
    logger.info("Обчислюємо Logistic Regression Coefficients...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model_lr = LogisticRegression(max_iter=10000, random_state=42, solver='liblinear',
                                  class_weight='balanced')
    model_lr.fit(X_scaled, y)
    lr_coefficients = model_lr.coef_[0]
    results_dict['LR Coefficient'] = dict(zip(features, lr_coefficients))
    results_dict['Absolute Coefficient'] = dict(zip(features, np.abs(lr_coefficients)))
    models_dict['LogisticRegression'] = model_lr
    models_dict['Scaler'] = scaler

    # 4. Recursive Feature Elimination (RFE) з крос-валідацією
    logger.info("Запускаємо Recursive Feature Elimination (RFE) з крос-валідацією...")
    n_features_to_select = max(10, n_features // 3)
    rfe_selector = RFE(estimator=model_lr, n_features_to_select=n_features_to_select, step=1)
    rfe_selector.fit(X_scaled, y)
    rfe_selected = rfe_selector.support_
    results_dict['RFE Selected'] = dict(zip(features, rfe_selected))
    models_dict['RFE'] = rfe_selector

    # 6. Decision Tree Feature Importance з крос-валідацією
    logger.info("Обчислюємо Decision Tree Feature Importance з крос-валідацією...")
    model_dt = DecisionTreeClassifier(random_state=42)
    dt_importances = []
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    for train_idx, val_idx in cv_obj.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        model_dt.fit(X_train, y_train)
        dt_importances.append(model_dt.feature_importances_)
    dt_importance = np.mean(dt_importances, axis=0)
    results_dict['DT Score'] = dict(zip(features, dt_importance))
    models_dict['DecisionTree'] = model_dt

    # 7. Random Forest Importance з крос-валідацією
    logger.info("Обчислюємо Random Forest Importance з крос-валідацією...")
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    importances = []
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    for train_idx, val_idx in cv_obj.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        model_rf.fit(X_train, y_train)
        importances.append(model_rf.feature_importances_)
    rf_importance = np.mean(importances, axis=0)
    results_dict['RF Score'] = dict(zip(features, rf_importance))
    models_dict['RandomForest'] = model_rf

    # 8. XGBoost Feature Importance з крос-валідацією
    logger.info("Обчислюємо XGBoost Feature Importance з крос-валідацією...")
    model_xgb = xgb.XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1,
                                  scale_pos_weight=(len(y) - sum(y)) / sum(y))
    importances = []
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    for train_idx, val_idx in cv_obj.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        model_xgb.fit(X_train, y_train)
        importances.append(model_xgb.feature_importances_)
    xgb_importance = np.mean(importances, axis=0)
    results_dict['XGBoost Score'] = dict(zip(features, xgb_importance))
    models_dict['XGBoost'] = model_xgb

    # Створюємо DataFrame з результатами
    results_df = pd.DataFrame({col: pd.Series(results_dict[col]) for col in results_dict})
    results_df.index.name = 'Feature'
    results_df.reset_index(inplace=True)

    # Обчислюємо ранги (чим більше значення – тим важливіше)
    rank_columns = ['MI Score', 'F Score', 'Spearman Score', 'Absolute Coefficient', 'DT Score', 'RF Score', 'XGBoost Score']

    for col in rank_columns:
        if col in results_df.columns:
            # Нормалізуємо на максимальне значення
            max_val = results_df[col].max()
            if max_val > 0:
                results_df[f'{col} Normalized'] = results_df[col] / max_val
            else:
                results_df[f'{col} Normalized'] = 0

    # Для колонки RFE Selected використовуємо бінарні значення
    if 'RFE Selected' in results_df.columns:
        results_df['RFE Selected Normalized'] = results_df['RFE Selected'].astype(float)

    # Обчислюємо загальний ранг
    norm_columns = [f'{col} Normalized' for col in rank_columns if f'{col} Normalized' in results_df.columns]

    if 'RFE Selected Normalized' in results_df.columns:
        norm_columns.append('RFE Selected Normalized')

    # Сумуємо нормалізовані значення та упорядковуємо за загальним рангом
    results_df['Total Importance Rank'] = results_df[norm_columns].sum(axis=1)
    results_df = results_df.sort_values('Total Importance Rank', ascending=False).reset_index(drop=True)

    return results_df, models_dict


def evaluate_models_with_features(X, y, results_df, top_n_features=None, cv=5):
    """
    Оцінює моделі з різною кількістю найважливіших ознак

    Args:
        X (pd.DataFrame): Повний набір ознак
        y (pd.Series): Цільова змінна
        results_df (pd.DataFrame): DataFrame з результатами оцінки ознак
        top_n_features (list): Список з кількостями ознак для перевірки. Якщо None, тестує всі ознаки від 1 до max
        cv (int): Кількість фолдів для крос-валідації

    Returns:
        pd.DataFrame: DataFrame з результатами
    """
    logger.info("Оцінюємо моделі з різною кількістю найважливіших ознак...")

    # Якщо не вказано список кількостей ознак, створюємо лінійну послідовність
    if top_n_features is None:
        max_features = min(50, len(results_df))
        # Створюємо список: [1, 2, 3, 4, 5, 10, 15, 20, ..., max_features]
        top_n_features = list(range(1, 6)) + list(range(10, max_features + 1, 5))

    # Нормалізуємо дані для всіх моделей окрім дерев
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Словник моделей для тестування
    models = {
        'LogisticRegression': LogisticRegression(max_iter=10000, random_state=42, solver='liblinear', class_weight='balanced'),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1, scale_pos_weight=(len(y) - sum(y)) / sum(y))
    }

    # Створюємо датафрейм для результатів
    results = []

    # Проходимо по різним кількостям ознак
    for n in top_n_features:
        logger.info(f"Тестуємо з {n} найважливішими ознаками...")

        # Відбираємо топ-N ознак
        top_features = results_df['Feature'].head(n).tolist()
        X_selected = X[top_features]
        X_scaled_selected = X_scaled[:, [list(X.columns).index(f) for f in top_features]]

        # Тестуємо кожну модель
        for model_name, model in models.items():
            logger.info(f"  Оцінюємо модель {model_name}...")

            # Для логістичної регресії використовуємо масштабовані дані
            if model_name == 'LogisticRegression':
                X_model = X_scaled_selected
            else:
                X_model = X_selected

            # Обчислюємо метрики з крос-валідацією
            accuracy = cross_val_score(model, X_model, y, cv=cv, scoring='accuracy').mean()
            precision = cross_val_score(model, X_model, y, cv=cv, scoring='precision').mean()
            recall = cross_val_score(model, X_model, y, cv=cv, scoring='recall').mean()
            f1 = cross_val_score(model, X_model, y, cv=cv, scoring='f1').mean()
            roc_auc = cross_val_score(model, X_model, y, cv=cv, scoring='roc_auc').mean()

            # Додаємо результати
            results.append({
                'Model': model_name,
                'N Features': n,
                'Features': ', '.join(top_features[:5]) + ('...' if n > 5 else ''),
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'ROC AUC': roc_auc
            })

    # Створюємо датафрейм з результатами
    results_df = pd.DataFrame(results)

    # Виводимо таблицю з найкращими результатами по метриці AUC для кожної моделі
    logger.info("\nНайкращі результати за ROC AUC для кожної моделі:")
    best_results = results_df.loc[results_df.groupby('Model')['ROC AUC'].idxmax()]
    print("\nНайкращі результати за ROC AUC для кожної моделі:")
    print(tabulate(best_results[['Model', 'N Features', 'ROC AUC', 'Accuracy', 'F1 Score']],
                   headers='keys', tablefmt='pipe', showindex=False, floatfmt='.4f'))

    return results_df


def feature_selection_analysis(df, output_dir='.'):
    """
    Проводить аналіз та відбір найважливіших ознак

    Args:
        df (pd.DataFrame): Датафрейм з даними
        output_dir (str): Директорія для збереження результатів

    Returns:
        pd.DataFrame: DataFrame з результатами оцінки ознак
    """
    logger.info("Починаємо аналіз та відбір найважливіших ознак...")

    # Перевіряємо наявність цільової змінної
    if 'is_successful' not in df.columns:
        logger.error("Колонка 'is_successful' відсутня в даних. Аналіз неможливий.")
        return None

    # Аналіз дисбалансу класів
    class_counts = df['is_successful'].value_counts()
    logger.info(f"Розподіл класів: {class_counts.to_dict()}")
    if class_counts.min() / class_counts.max() < 0.2:
        logger.warning("Виявлено суттєвий дисбаланс класів")

    # Кодуємо категоріальні змінні
    logger.info("Кодуємо категоріальні змінні...")
    cat_columns = df.select_dtypes(include=['object']).columns.tolist()
    ignore_cols = ['order_id', 'state']
    cat_columns = [col for col in cat_columns if col not in ignore_cols]
    df_encoded = df.copy()
    encoders = {}

    for col in cat_columns:
        encoder = LabelEncoder()
        df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
        encoders[col] = encoder

    # Вибираємо числові змінні
    num_columns = df.select_dtypes(include=['number']).columns.tolist()
    if 'is_successful' in num_columns:
        num_columns.remove('is_successful')

    # Формуємо повний набір ознак
    X_full = df_encoded[num_columns + cat_columns]
    y = df_encoded['is_successful']

    # Оцінюємо важливість ознак
    results_df, models_dict = evaluate_features(X_full, y, cv=5)

    # Оцінюємо моделі з різною кількістю ознак
    performance_df = evaluate_models_with_features(X_full, y, results_df)

    # Візуалізація результатів
    logger.info("Створюємо візуалізації...")

    # 1. Візуалізація ТОП-15 найважливіших ознак
    plt.figure(figsize=(12, 8))
    top_features = results_df[['Feature', 'Total Importance Rank']].head(15)
    # Додаємо українські назви
    top_features['Feature_UA'] = top_features['Feature'].apply(get_ua_feature_name)
    sns.barplot(x='Total Importance Rank', y='Feature_UA', hue='Feature_UA', data=top_features, palette='coolwarm', legend=False)
    plt.xlabel("Total Importance Rank (Чим більший – тим важливіше)")
    plt.ylabel("Ознаки")
    plt.title("ТОП-15 найважливіших ознак")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/top_features.png", dpi=300)
    plt.close()

    # 2. Візуалізація розподілу оцінок для ТОП-10 ознак
    top10_features = results_df['Feature'].head(10).tolist()
    scores_data = []
    methods = ['MI Score', 'F Score', 'Spearman Score', 'RFE Selected',
               'Absolute Coefficient', 'DT Score', 'RF Score', 'XGBoost Score']

    for method in methods:
        for feature in top10_features:
            score = results_df.loc[results_df['Feature'] == feature, method].values[0]
            max_score = results_df[method].max()
            norm_score = score / max_score if max_score > 0 else 0
            # Додаємо українську назву
            feature_ua = get_ua_feature_name(feature)
            scores_data.append({'Feature': feature, 'Feature_UA': feature_ua,
                                'Method': method, 'Normalized Score': norm_score})

    scores_df = pd.DataFrame(scores_data)

    # Визначаємо кольори для методів
    method_colors = {
        'MI Score': 'blue',           # синій
        'F Score': 'red',             # червоний
        'Spearman Score': 'skyblue',  # блакитний
        'RFE Selected': 'purple',     # пурпурний
        'Absolute Coefficient': 'orange',  # помаранчовий
        'DT Score': 'lightgreen',     # світло зелений
        'RF Score': 'forestgreen',    # середньо зелений
        'XGBoost Score': 'darkgreen'  # темно зелений
    }

    plt.figure(figsize=(14, 10))
    palette = [method_colors[method] for method in scores_df['Method'].unique()]
    sns.barplot(x='Feature_UA', y='Normalized Score', hue='Method', data=scores_df, palette=palette)
    plt.xticks(rotation=45, ha='right')
    plt.title("Порівняння нормалізованих оцінок для ТОП-10 ознак за різними методами")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_scores_comparison.png", dpi=300)
    plt.close()

    # 3. Візуалізація продуктивності моделей
    plt.figure(figsize=(12, 8))
    sns.lineplot(x='N Features', y='ROC AUC', hue='Model', data=performance_df, marker='o')
    plt.title("ROC AUC моделей залежно від кількості ознак")
    plt.xlabel("Кількість ознак")
    plt.ylabel("ROC AUC")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_performance_auc.png", dpi=300)
    plt.close()

    # 4. Візуалізація кореляційної матриці для ТОП-15 ознак
    plt.figure(figsize=(14, 12))
    top15_features = results_df['Feature'].head(15).tolist()
    correlation_matrix = X_full[top15_features].corr()

    # Створюємо DataFrame з українськими назвами для кореляційної матриці
    correlation_matrix_ua = correlation_matrix.copy()
    ua_index = [get_ua_feature_name(feature) for feature in correlation_matrix.index]
    ua_columns = [get_ua_feature_name(feature) for feature in correlation_matrix.columns]
    correlation_matrix_ua.index = ua_index
    correlation_matrix_ua.columns = ua_columns

    mask = np.triu(np.ones_like(correlation_matrix_ua, dtype=bool))
    sns.heatmap(correlation_matrix_ua, mask=mask, cmap='coolwarm', annot=True,
                fmt='.2f', linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Кореляційна матриця ТОП-15 ознак")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_matrix.png", dpi=300)
    plt.close()

    # Створюємо окремі таблиці для кожного методу з 5 найважливішими ознаками
    print("\nТаблиці з 5 найважливішими ознаками за кожним методом:\n")

    # Словник з назвами методів та відповідними стовпцями
    methods = {
        'Mutual Information': 'MI Score',
        'ANOVA F-test': 'F Score',
        'Spearman Correlation': 'Spearman Score',
        'Logistic Regression Coefficients': 'LR Coefficient',
        'Decision Tree Feature Importance': 'DT Score',
        'Random Forest Importance': 'RF Score',
        'XGBoost Feature Importance': 'XGBoost Score'
    }

    # Створюємо та виводимо окремі таблиці для кожного методу
    for method_name, method_column in methods.items():
        # Перевіряємо чи є стовпець у DataFrame
        if method_column not in results_df.columns:
            logger.warning(f"Стовпець {method_column} відсутній у результатах. Пропускаємо метод {method_name}.")
            continue

        # Сортуємо дані за відповідним методом (у порядку спадання, тобто найважливіші ознаки будуть спочатку)
        if method_name == 'Logistic Regression Coefficients':
            # Для логістичної регресії сортуємо за абсолютними значеннями, але виводимо оригінальні
            method_df = results_df.copy()
            # Створюємо тимчасовий стовпець з абсолютними значеннями для сортування
            method_df['Abs_LR'] = method_df['LR Coefficient'].abs()
            # Сортуємо за абсолютними значеннями і беремо топ-5
            method_df = method_df.sort_values(by='Abs_LR', ascending=False).head(5)
            # Видаляємо тимчасовий стовпець
            method_df.drop('Abs_LR', axis=1, inplace=True)
        else:
            method_df = results_df.sort_values(by=method_column, ascending=False).head(5)

        # Додаємо українські назви
        method_df['Feature_UA'] = method_df['Feature'].apply(get_ua_feature_name)

        # Вибираємо та форматуємо дані для виведення
        display_method_df = method_df[['Feature', 'Feature_UA', method_column]].round(4)

        # Виводимо назву методу та дані через print
        print(f"\n{method_name} - 5 найважливіших ознак:")
        method_table = tabulate(display_method_df, headers=['Ознака', 'Назва українською', 'Значення'], tablefmt='fancy_grid')
        print(f"\n{method_table}")

    # Виводимо загальну таблицю з топ-20 ознаками
    print("\nПорівняння впливу всіх факторів за рейтингами:\n")

    # Вибираємо і готуємо дані для відображення
    display_columns = ['Feature', 'MI Score', 'F Score', 'Spearman Score',
                       'LR Coefficient', 'DT Score', 'RF Score', 'XGBoost Score', 'Total Importance Rank']

    # Додаємо українські назви ознак
    results_df['Feature_UA'] = results_df['Feature'].apply(get_ua_feature_name)

    # Переставляємо стовпець з українською назвою на друге місце
    display_columns.insert(1, 'Feature_UA')

    # Відфільтровуємо лише ті стовпці, які дійсно є у DataFrame
    display_columns = [col for col in display_columns if col in results_df.columns]
    display_df = results_df[display_columns].head(20).round(4)

    print(tabulate(display_df, headers='keys', tablefmt='fancy_grid'))

    # Виводимо інформацію про найкращу продуктивність моделей
    print(f"\nНайкраща продуктивність моделей:")
    best_performance = performance_df.loc[performance_df.groupby('Model')['Accuracy'].idxmax()]
    print(tabulate(best_performance, headers='keys', tablefmt='fancy_grid'))

    logger.info("Аналіз та відбір найважливіших ознак завершено")

    return results_df

#####################################
# Основна функція для запуску всього аналізу
#####################################

def main(file_path="cleaned_result.csv", output_dir=None):
    """
    Головна функція, яка запускає весь аналіз

    Args:
        file_path (str): Шлях до файлу з даними
        output_dir (str): Директорія для збереження результатів (якщо None, створюється автоматично)
    """
    # Якщо директорія не вказана, використовуємо директорію за замовчуванням
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"data_analysis_results_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        stats_dir = f"{output_dir}/statistics"
        vis_dir = f"{output_dir}/visualization"
        feature_dir = f"{output_dir}/feature_selection"
        os.makedirs(stats_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(feature_dir, exist_ok=True)

    # Перенаправляємо вивід print у файл та консоль одночасно
    console_output_file = f"{output_dir}/console_output.txt"
    original_stdout = sys.stdout
    sys.stdout = TeeOutput(console_output_file, 'w')

    print(f"Починаємо повний аналіз даних з файлу {file_path}")
    print(f"Результати будуть збережені в {output_dir}")
    print(f"Цей вивід також зберігається у файлі: {console_output_file}")
    print("="*80)

    logger.info(f"Починаємо повний аналіз даних з файлу {file_path}")
    logger.info(f"Результати будуть збережені в {output_dir}")

    # 1. Завантаження даних
    df, numeric_columns = load_data(file_path)

    # 2. Статистичний аналіз
    logger.info("\n" + "="*50)
    logger.info("СТАТИСТИЧНИЙ АНАЛІЗ")
    logger.info("="*50)
    analyze_all_columns(df, numeric_columns, output_dir=stats_dir)

    # 3. Візуалізація даних
    logger.info("\n" + "="*50)
    logger.info("ВІЗУАЛІЗАЦІЯ ДАНИХ")
    logger.info("="*50)
    visualize_all_columns(df, numeric_columns, output_dir=vis_dir)

    # 4. Відбір найважливіших ознак
    logger.info("\n" + "="*50)
    logger.info("ВІДБІР НАЙВАЖЛИВІШИХ ОЗНАК")
    logger.info("="*50)
    feature_selection_analysis(df, output_dir=feature_dir)

    logger.info("\n" + "="*50)
    logger.info("АНАЛІЗ ЗАВЕРШЕНО")
    logger.info("="*50)
    logger.info(f"Усі результати збережено в {output_dir}")

    # Повертаємо оригінальний stdout
    sys.stdout.close()
    sys.stdout = original_stdout
    print(f"Аналіз завершено. Результати збережено в {output_dir}")
    print(f"Вивід консолі збережено у файлі: {console_output_file}")


if __name__ == "__main__":
    main()
