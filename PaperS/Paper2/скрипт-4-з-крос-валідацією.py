"""
Statistical prediction model based on comparative statistical tests.
Automatically selects analysis methods and calculates statistical significance of factors.
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import logging
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics  # Додаємо імпорт metrics тут

# Налаштування параметрів обробки викидів
HANDLE_OUTLIERS = True  # Чи потрібно обробляти викиди
OUTLIER_METHOD = 'iqr'  # Метод виявлення викидів: 'iqr', 'zscore', або 'percentile'
OUTLIER_TREATMENT = 'cap'  # Метод обробки викидів: 'cap', 'remove', або 'median'
OUTLIER_THRESHOLD = 1.5  # Поріг для виявлення викидів

# Налаштування логування
def setup_logging(log_file='statistical_model_log.txt'):
    """Set up logging to file and console with detailed formatting."""
    # Створення директорії для логів, якщо вона не існує
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Налаштування логера
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Очистка попередніх обробників
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    # Додавання файлового обробника
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)

    # Додавання консольного обробника
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    # Запис заголовка в лог
    logger.info('=' * 80)
    logger.info('СТАТИСТИЧНА МОДЕЛЬ ПРОГНОЗУВАННЯ')
    logger.info(f'Час запуску: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info('=' * 80 + '\n')

    return logger


# Допоміжна функція для виведення роздільників у логах
def log_section(logger, title, section_level=1):
    """Log section header with appropriate formatting based on importance level."""
    if section_level == 1:
        # Головний розділ
        logger.info('\n' + '=' * 80)
        logger.info(f' {title} '.center(80, '='))
        logger.info('=' * 80)
    elif section_level == 2:
        # Підрозділ
        logger.info('\n' + '-' * 80)
        logger.info(f' {title} '.center(80, '-'))
        logger.info('-' * 80)
    else:
        # Мінорний підрозділ
        logger.info(f"\n--- {title} ---")

# Клас для серіалізації об'єктів numpy в JSON
class NumpyEncoder(json.JSONEncoder):
    """Special encoder for NumPy objects to JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if hasattr(obj, 'to_json'):
            return obj.to_json()
        return super().default(obj)


def load_data(file_path, group_column='is_successful', handle_outliers_flag=True):
    """
    Завантаження даних з CSV-файлу та їх підготовка для аналізу.

    Args:
        file_path: Шлях до файлу з даними
        group_column: Назва цільової колонки

    Returns:
        DataFrame та список числових колонок
    """
    logger = logging.getLogger()
    log_section(logger, "ЗАВАНТАЖЕННЯ ТА ПІДГОТОВКА ДАНИХ")

    logger.info(f"Завантаження даних з файлу: {file_path}")
    df = pd.read_csv(file_path)

    logger.info(f"Завантажено датасет розміром {df.shape[0]} рядків x {df.shape[1]} колонок")

    # Перетворення цільової колонки на цілі числа
    df[group_column] = df[group_column].astype(int)

    # Підрахунок розподілу цільової змінної
    target_dist = df[group_column].value_counts()
    logger.info(f"\nРозподіл цільової змінної '{group_column}':")
    logger.info(f"0 (Неуспішні замовлення): {target_dist.get(0, 0)} ({target_dist.get(0, 0) / len(df) * 100:.2f}%)")
    logger.info(f"1 (Успішні замовлення): {target_dist.get(1, 0)} ({target_dist.get(1, 0) / len(df) * 100:.2f}%)")

    # Вибір числових колонок для аналізу
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if group_column in numeric_columns:
        numeric_columns.remove(group_column)
    logger.info(f"\nВиявлено {len(numeric_columns)} числових колонок для аналізу")

    # Заміна від'ємних значень нулями
    negative_values = {}
    for col in numeric_columns:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            negative_values[col] = neg_count
            df[col] = df[col].apply(lambda x: max(0, x) if not pd.isna(x) else x)

    if negative_values:
        logger.info("\nЗаміна від'ємних значень нулями у наступних колонках:")
        for col, count in negative_values.items():
            logger.info(f"  - {col}: {count} значень замінено")

    # Перевірка пропущених значень
    missing_values = df[numeric_columns].isna().sum()
    if missing_values.sum() > 0:
        logger.info("\nВиявлено пропущені значення у наступних колонках:")
        for col, count in missing_values[missing_values > 0].items():
            logger.info(f"  - {col}: {count} пропущених значень ({count / len(df) * 100:.2f}%)")

    # Додаємо обробку викидів
    if handle_outliers_flag:
        df, outlier_report = handle_all_outliers(
            df, numeric_columns, method='iqr', treatment='cap', threshold=1.5
        )

        # Додаємо візуалізацію для колонок з найбільшою кількістю викидів
        if outlier_report:
            # Перевіряємо структуру outlier_report
            for key, value in list(outlier_report.items())[:1]:
                if 'outliers_found' not in value:
                    # Якщо ключ інший, спробуйте знайти відповідний
                    # або змініть key у сортуванні на правильний
                    logger.warning(
                        f"У звіті про викиди відсутній ключ 'outliers_found'. Наявні ключі: {list(value.keys())}")

            # Спробуйте використовувати безпечний підхід до сортування
            top_outliers = sorted(
                outlier_report.items(),
                key=lambda x: x[1].get('outliers_found', x[1].get('count', 0)),  # Спробуйте альтернативні ключі
                reverse=True
            )[:5]


    return df, numeric_columns


def detect_outliers(df, column_name, method='iqr', threshold=1.5):
    """
    Виявлення викидів у числовій колонці за допомогою різних методів.

    Args:
        df: DataFrame з даними
        column_name: Назва колонки для аналізу
        method: Метод виявлення викидів ('iqr', 'zscore', 'percentile')
        threshold: Поріг для виявлення викидів (для IQR та Z-score)

    Returns:
        Маска логічних значень, де True вказує на викид
    """
    logger = logging.getLogger()
    data = df[column_name].dropna()

    if method == 'iqr':
        # Метод міжквартильного розмаху (IQR)
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outliers = (df[column_name] < lower_bound) | (df[column_name] > upper_bound)

    elif method == 'zscore':
        # Метод Z-score
        mean = data.mean()
        std = data.std()
        z_scores = abs((df[column_name] - mean) / std)
        outliers = z_scores > threshold

    elif method == 'percentile':
        # Метод процентилів
        lower_bound = data.quantile(0.01)  # 1й процентиль
        upper_bound = data.quantile(0.99)  # 99й процентиль
        outliers = (df[column_name] < lower_bound) | (df[column_name] > upper_bound)

    else:
        logger.warning(f"Невідомий метод виявлення викидів: {method}. Використовуємо IQR.")
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outliers = (df[column_name] < lower_bound) | (df[column_name] > upper_bound)

    return outliers


def handle_outliers(df, column_name, method='iqr', treatment='cap', threshold=1.5):
    """
    Виявлення та обробка викидів у числовій колонці.

    Args:
        df: DataFrame з даними
        column_name: Назва колонки для аналізу
        method: Метод виявлення викидів ('iqr', 'zscore', 'percentile')
        treatment: Спосіб обробки викидів ('cap', 'remove', 'median')
        threshold: Поріг для виявлення викидів

    Returns:
        DataFrame з обробленими викидами та кількість виявлених викидів
    """
    logger = logging.getLogger()
    outliers = detect_outliers(df, column_name, method, threshold)
    outlier_count = outliers.sum()

    if outlier_count == 0:
        return df.copy(), 0

    if treatment == 'remove':
        # Видалення рядків з викидами
        df_cleaned = df[~outliers].copy()

    elif treatment == 'cap':
        # Обмеження значень викидів (вінсоризація)
        df_cleaned = df.copy()
        data = df[column_name].dropna()

        if method == 'iqr':
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr

        elif method == 'zscore':
            mean = data.mean()
            std = data.std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std

        elif method == 'percentile':
            lower_bound = data.quantile(0.01)
            upper_bound = data.quantile(0.99)

        # Замінюємо викиди граничними значеннями
        df_cleaned.loc[df_cleaned[column_name] > upper_bound, column_name] = upper_bound
        df_cleaned.loc[df_cleaned[column_name] < lower_bound, column_name] = lower_bound

    elif treatment == 'median':
        # Заміна викидів на медіану
        df_cleaned = df.copy()
        median_value = df[column_name].median()
        df_cleaned.loc[outliers, column_name] = median_value

    else:
        logger.warning(f"Невідомий метод обробки викидів: {treatment}. Використовуємо вінсоризацію.")
        df_cleaned = df.copy()
        data = df[column_name].dropna()
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        df_cleaned.loc[df_cleaned[column_name] > upper_bound, column_name] = upper_bound
        df_cleaned.loc[df_cleaned[column_name] < lower_bound, column_name] = lower_bound

    return df_cleaned, outlier_count


def handle_all_outliers(df, numeric_columns, method='iqr', treatment='cap', threshold=1.5):
    """
    Обробляє викиди у всіх вказаних числових колонках.

    Args:
        df: DataFrame з даними
        numeric_columns: Список числових колонок для обробки
        method: Метод виявлення викидів ('iqr', 'zscore', 'percentile')
        treatment: Спосіб обробки викидів ('cap', 'remove', 'median')
        threshold: Поріг для виявлення викидів

    Returns:
        DataFrame з обробленими викидами та звіт про виявлені викиди
    """
    logger = logging.getLogger()
    log_section(logger, "ОБРОБКА ВИКИДІВ", 2)

    logger.info(f"Метод виявлення викидів: {method}")
    logger.info(f"Метод обробки викидів: {treatment}")
    logger.info(f"Поріг для виявлення: {threshold}\n")

    df_cleaned = df.copy()
    outlier_report = {}

    for column in numeric_columns:
        try:
            temp_df, outlier_count = handle_outliers(
                df_cleaned, column, method=method, treatment=treatment, threshold=threshold
            )

            if treatment == 'remove':
                # Якщо видаляємо викиди, потрібно оновити весь DataFrame
                df_cleaned = temp_df

                if outlier_count > 0:
                    outlier_report[column] = {
                        'outliers_found': outlier_count,
                        'percentage': outlier_count / len(df) * 100,
                        'remaining_rows': len(df_cleaned)
                    }
            else:
                # Якщо обмежуємо або замінюємо, оновлюємо тільки колонку
                if outlier_count > 0:
                    df_cleaned[column] = temp_df[column]
                    outlier_report[column] = {
                        'outliers_found': outlier_count,
                        'percentage': outlier_count / len(df) * 100
                    }
        except Exception as e:
            logger.warning(f"Помилка обробки викидів у колонці {column}: {str(e)}")

    # Логування результатів
    if outlier_report:
        logger.info("Виявлено викиди у наступних колонках:")
        for col, stats in outlier_report.items():
            logger.info(f"  - {col}: {stats['outliers_found']} викидів ({stats['percentage']:.2f}%)")

        if treatment == 'remove':
            logger.info(f"\nПісля видалення рядків з викидами залишилось {len(df_cleaned)} з {len(df)} рядків "
                        f"({len(df_cleaned) / len(df) * 100:.2f}%)")
    else:
        logger.info("Викидів не виявлено в жодній з колонок.")

    return df_cleaned, outlier_report


def plot_outliers(df, column_name, outliers=None, output_dir='.'):
    """
    Створює візуалізацію для виявлення та відображення викидів.

    Args:
        df: DataFrame з даними
        column_name: Назва колонки для аналізу
        outliers: Маска логічних значень, де True вказує на викид (якщо None, буде використано IQR)
        output_dir: Каталог для збереження графіків

    Returns:
        Шлях до збереженого графіка
    """
    if outliers is None:
        outliers = detect_outliers(df, column_name)

    plt.figure(figsize=(12, 6))

    # Створюємо підграфіки
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Boxplot
    sns.boxplot(y=df[column_name], ax=ax1)
    ax1.set_title(f'Boxplot для {column_name}')

    # Histogram with outliers highlighted
    sns.histplot(df.loc[~outliers, column_name], ax=ax2, color='blue',
                 label='Нормальні значення', alpha=0.5)
    if outliers.sum() > 0:
        sns.histplot(df.loc[outliers, column_name], ax=ax2, color='red',
                     label='Викиди', alpha=0.5)
    ax2.set_title(f'Розподіл значень для {column_name}')
    ax2.legend()

    plt.tight_layout()

    # Зберігаємо графік
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'outliers_{column_name}.png')
    plt.savefig(output_path)
    plt.close()

    return output_path




def split_train_test(df, test_size=0.3, random_state=42):
    """
    Розбиває дані на тренувальну та тестову вибірки, зберігаючи
    співвідношення цільової змінної.

    Args:
        df: DataFrame з даними
        test_size: Розмір тестової вибірки (від 0 до 1)
        random_state: Зерно генератора випадкових чисел

    Returns:
        Кортеж (df_train, df_test)
    """
    logger = logging.getLogger()
    log_section(logger, "РОЗБИТТЯ НА ТРЕНУВАЛЬНУ І ТЕСТОВУ ВИБІРКИ", 2)

    logger.info(f"Параметри розбиття: test_size={test_size}, random_state={random_state}")

    # Стратифіковане розбиття за цільовою змінною
    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['is_successful']
    )

    # Логування результатів розбиття
    logger.info(f"Тренувальна вибірка: {len(df_train)} записів ({len(df_train) / len(df) * 100:.2f}%)")
    logger.info(f"Тестова вибірка: {len(df_test)} записів ({len(df_test) / len(df) * 100:.2f}%)")

    # Перевірка розподілу цільової змінної
    train_target_dist = df_train['is_successful'].value_counts(normalize=True) * 100
    test_target_dist = df_test['is_successful'].value_counts(normalize=True) * 100

    logger.info("\nРозподіл цільової змінної після розбиття:")
    logger.info(
        f"Тренувальна вибірка: Неуспішні - {train_target_dist.get(0, 0):.2f}%, Успішні - {train_target_dist.get(1, 0):.2f}%")
    logger.info(
        f"Тестова вибірка: Неуспішні - {test_target_dist.get(0, 0):.2f}%, Успішні - {test_target_dist.get(1, 0):.2f}%")

    return df_train, df_test


def test_normality(data, test_name='shapiro'):
    """
    Проводить тест на нормальність розподілу даних.

    Args:
        data: Серія або масив даних для аналізу
        test_name: Назва тесту (shapiro, dagostino, anderson)

    Returns:
        Кортеж (statistic, p_value, is_normal)
    """
    logger = logging.getLogger()

    # Видалення пропущених значень
    clean_data = data.dropna()

    # Перевірка мінімальної кількості даних
    if len(clean_data) < 3:
        logger.info(f"Недостатньо даних для тесту нормальності: {len(clean_data)} < 3")
        return None, None, False

    # Перевірка наявності варіації в даних
    if len(clean_data.unique()) <= 1:
        logger.info(f"Дані не містять варіації (унікальних значень: {len(clean_data.unique())})")
        return None, None, False

    try:
        if test_name == 'shapiro':
            # Тест Шапіро-Вілка (обмеження: <5000 спостережень)
            if len(clean_data) > 5000:
                sample = clean_data.sample(5000, random_state=42)
                logger.info(
                    f"Використовується вибірка {len(sample)} з {len(clean_data)} спостережень для тесту Шапіро-Вілка")
                stat, p = stats.shapiro(sample)
            else:
                stat, p = stats.shapiro(clean_data)

            logger.info(f"Тест Шапіро-Вілка: статистика={stat:.6f}, p-значення={p:.6e}")

        elif test_name == 'dagostino':
            # Тест Д'Агостіно-Пірсона (вимагає n >= 20)
            if len(clean_data) < 20:
                logger.info(f"Недостатньо даних для тесту Д'Агостіно-Пірсона: {len(clean_data)} < 20")
                return None, None, False

            stat, p = stats.normaltest(clean_data)
            logger.info(f"Тест Д'Агостіно-Пірсона: статистика={stat:.6f}, p-значення={p:.6e}")

        elif test_name == 'anderson':
            # Тест Андерсона-Дарлінга
            result = stats.anderson(clean_data, dist='norm')
            stat = result.statistic

            # Визначення p-значення за критичними значеннями
            significance_levels = [15, 10, 5, 2.5, 1]
            critical_values = result.critical_values

            # Знаходження найближчого рівня значущості
            for i, (sl, cv) in enumerate(zip(significance_levels, critical_values)):
                if stat > cv:
                    # Якщо статистика більша за критичне значення, то p < рівень значущості
                    p = sl / 100  # Перетворення відсотка у частку
                    break
            else:
                # Якщо статистика менша за всі критичні значення
                p = 0.15  # > найбільшого рівня значущості

            logger.info(f"Тест Андерсона-Дарлінга: статистика={stat:.6f}, приблизне p-значення={p:.6f}")
            logger.info(
                f"  Критичні значення: {', '.join([f'{s}%: {v:.6f}' for s, v in zip(significance_levels, critical_values)])}")

        else:
            logger.warning(f"Невідомий тест нормальності: {test_name}")
            return None, None, False

        # Інтерпретація p-значення (при alpha=0.05)
        is_normal = p > 0.05
        conclusion = "нормальний" if is_normal else "не нормальний"
        logger.info(f"Висновок: розподіл {conclusion} (p {'>' if is_normal else '<='} 0.05)")

        return stat, p, is_normal

    except Exception as e:
        logger.error(f"Помилка при виконанні тесту нормальності: {str(e)}")
        return None, None, False


def assess_normality_for_column(df, column_name, group_column='is_successful'):
    """
    Комплексна оцінка нормальності розподілу для груп у колонці.
    Використовує кілька тестів та формує загальний висновок.

    Args:
        df: DataFrame з даними
        column_name: Назва колонки для аналізу
        group_column: Назва колонки групування

    Returns:
        Словник з результатами оцінки та рекомендованим методом аналізу
    """
    logger = logging.getLogger()
    log_section(logger, f"ОЦІНКА НОРМАЛЬНОСТІ ДЛЯ КОЛОНКИ '{column_name}'", 2)

    # Дані для кожної групи
    group_0 = df[df[group_column] == 0][column_name].dropna()
    group_1 = df[df[group_column] == 1][column_name].dropna()

    # Базова статистика
    logger.info(f"Група 0 (Неуспішні замовлення): {len(group_0)} спостережень")
    logger.info(f"Середнє: {group_0.mean():.4f}, Медіана: {group_0.median():.4f}, "
                f"Ст. відхилення: {group_0.std():.4f}, CV: {group_0.std() / group_0.mean() if group_0.mean() != 0 else float('nan'):.4f}")
    logger.info(f"Асиметрія: {stats.skew(group_0):.4f}, Ексцес: {stats.kurtosis(group_0):.4f}")

    logger.info(f"\nГрупа 1 (Успішні замовлення): {len(group_1)} спостережень")
    logger.info(f"Середнє: {group_1.mean():.4f}, Медіана: {group_1.median():.4f}, "
                f"Ст. відхилення: {group_1.std():.4f}, CV: {group_1.std() / group_1.mean() if group_1.mean() != 0 else float('nan'):.4f}")
    logger.info(f"Асиметрія: {stats.skew(group_1):.4f}, Ексцес: {stats.kurtosis(group_1):.4f}")

    # Тести нормальності
    logger.info("\nТести нормальності для Групи 0 (Неуспішні замовлення):")
    g0_shapiro = test_normality(group_0, 'shapiro')
    g0_dagostino = test_normality(group_0, 'dagostino')
    g0_anderson = test_normality(group_0, 'anderson')

    logger.info("\nТести нормальності для Групи 1 (Успішні замовлення):")
    g1_shapiro = test_normality(group_1, 'shapiro')
    g1_dagostino = test_normality(group_1, 'dagostino')
    g1_anderson = test_normality(group_1, 'anderson')

    # Підрахунок кількості тестів, які підтверджують нормальність
    g0_normal_tests = sum([g0_shapiro[2], g0_dagostino[2], g0_anderson[2]])
    g1_normal_tests = sum([g1_shapiro[2], g1_dagostino[2], g1_anderson[2]])

    # Визначення нормальності для кожної групи (якщо хоча б 2 тести підтверджують)
    g0_is_normal = g0_normal_tests >= 2
    g1_is_normal = g1_normal_tests >= 2

    # Загальний висновок щодо нормальності
    both_normal = g0_is_normal and g1_is_normal

    # Визначення рекомендованого методу порівняння
    if both_normal:
        recommended_method = 'parametric'  # t-тест, Cohen's d
        logger.info("\nВИСНОВОК: Обидві групи мають нормальний розподіл")
        logger.info("Рекомендований метод: ПАРАМЕТРИЧНИЙ (t-тест, Cohen's d)")
    else:
        recommended_method = 'nonparametric'  # Mann-Whitney U, AUC
        logger.info("\nВИСНОВОК: Принаймні одна з груп має ненормальний розподіл")
        logger.info("Рекомендований метод: НЕПАРАМЕТРИЧНИЙ (Mann-Whitney U, AUC)")

    return {
        'column': column_name,
        'group_0_normal': g0_is_normal,
        'group_1_normal': g1_is_normal,
        'both_normal': both_normal,
        'recommended_method': recommended_method,
        'group_0_size': len(group_0),
        'group_1_size': len(group_1)
    }


def compare_groups_parametric(group_0, group_1):
    """
    Порівнює дві групи за допомогою параметричного t-тесту та обчислює ефект за Cohen's d.

    Args:
        group_0: Дані першої групи (неуспішні)
        group_1: Дані другої групи (успішні)

    Returns:
        Словник з результатами тесту та розміром ефекту
    """
    logger = logging.getLogger()
    logger.info("Виконується ПАРАМЕТРИЧНИЙ аналіз (t-тест, Cohen's d)")

    # Welch's t-test (не вимагає рівних дисперсій)
    t_stat, p_value = stats.ttest_ind(group_0, group_1, equal_var=False)

    # Cohen's d (розмір ефекту)
    mean_diff = group_1.mean() - group_0.mean()
    pooled_std = np.sqrt(((len(group_0) - 1) * group_0.std() ** 2 + (len(group_1) - 1) * group_1.std() ** 2) /
                         (len(group_0) + len(group_1) - 2))
    cohen_d = mean_diff / pooled_std

    # 95% довірчий інтервал для різниці середніх
    df = (group_0.var() / len(group_0) + group_1.var() / len(group_1)) ** 2 / (
            (group_0.var() / len(group_0)) ** 2 / (len(group_0) - 1) +
            (group_1.var() / len(group_1)) ** 2 / (len(group_1) - 1))
    ci = stats.t.interval(0.95, df, loc=mean_diff,
                          scale=np.sqrt(group_0.var() / len(group_0) + group_1.var() / len(group_1)))

    # Логування
    logger.info(f"t-статистика: {t_stat:.4f}, p-значення: {p_value:.8f}")
    logger.info(f"Cohen's d (розмір ефекту): {cohen_d:.4f}")

    # Різниця середніх
    rel_diff_percent = (mean_diff / group_0.mean() * 100) if group_0.mean() != 0 else float('inf')
    logger.info(f"Різниця середніх: {mean_diff:.4f} ({rel_diff_percent:.2f}%)")
    logger.info(f"95% довірчий інтервал для різниці: [{ci[0]:.4f}, {ci[1]:.4f}]")

    # Висновок про статистичну значущість (при alpha=0.05)
    if p_value <= 0.05:
        significance = "СТАТИСТИЧНО ЗНАЧУЩА"
    else:
        significance = "статистично незначуща"
    logger.info(f"Висновок: різниця між групами {significance} (p {'<=' if p_value <= 0.05 else '>'} 0.05)")

    # Інтерпретація розміру ефекту
    if abs(cohen_d) < 0.2:
        effect_size_interpretation = "незначний"
    elif abs(cohen_d) < 0.5:
        effect_size_interpretation = "малий"
    elif abs(cohen_d) < 0.8:
        effect_size_interpretation = "середній"
    else:
        effect_size_interpretation = "великий"
    logger.info(f"Інтерпретація розміру ефекту: {effect_size_interpretation} (|d| = {abs(cohen_d):.2f})")

    return {
        'test': 't_test',
        'statistic': t_stat,
        'p_value': p_value,
        'effect_size': cohen_d,
        'mean_difference': mean_diff,
        'relative_difference_percent': rel_diff_percent,
        'confidence_interval': ci,
        'is_significant': p_value <= 0.05,
        'effect_size_interpretation': effect_size_interpretation
    }


def compare_groups_nonparametric(group_0, group_1):
    """
    Порівнює дві групи за допомогою непараметричного Mann-Whitney U тесту
    та обчислює AUC як міру розміру ефекту.

    Args:
        group_0: Дані першої групи (неуспішні)
        group_1: Дані другої групи (успішні)

    Returns:
        Словник з результатами тесту та розміром ефекту
    """
    logger = logging.getLogger()
    logger.info("Виконується НЕПАРАМЕТРИЧНИЙ аналіз (Mann-Whitney U, AUC)")

    # Mann-Whitney U тест
    u_stat, p_value = stats.mannwhitneyu(group_0, group_1, alternative='two-sided')

    # AUC (як міра розміру ефекту)
    # AUC = P(X_1 > X_0), де X_1 - випадкова величина з групи 1, X_0 - з групи 0
    n1, n0 = len(group_1), len(group_0)
    auc = u_stat / (n0 * n1)  # Нормалізація U-статистики

    # Перевірка напрямку ефекту (AUC має бути більше 0.5, якщо група_1 > група_0)
    if auc < 0.5:
        auc = 1 - auc  # Перетворюємо, щоб AUC > 0.5 відповідало group_1 > group_0

    # Медіанна різниця та процентна різниця
    median_diff = group_1.median() - group_0.median()
    rel_diff_percent = (median_diff / group_0.median() * 100) if group_0.median() != 0 else float('inf')

    # Логування
    logger.info(f"Mann-Whitney U статистика: {u_stat:.4f}, p-значення: {p_value:.8f}")
    logger.info(f"AUC (розмір ефекту): {auc:.4f}")
    logger.info(f"Різниця медіан: {median_diff:.4f} ({rel_diff_percent:.2f}%)")

    # Висновок про статистичну значущість (при alpha=0.05)
    if p_value <= 0.05:
        significance = "СТАТИСТИЧНО ЗНАЧУЩА"
    else:
        significance = "статистично незначуща"
    logger.info(f"Висновок: різниця між групами {significance} (p {'<=' if p_value <= 0.05 else '>'} 0.05)")

    # Інтерпретація розміру ефекту AUC
    auc_centered = abs(auc - 0.5)  # центруємо AUC навколо 0.5
    if auc_centered < 0.05:
        effect_size_interpretation = "незначний"
    elif auc_centered < 0.1:
        effect_size_interpretation = "малий"
    elif auc_centered < 0.2:
        effect_size_interpretation = "середній"
    else:
        effect_size_interpretation = "великий"
    logger.info(f"Інтерпретація AUC: {effect_size_interpretation} (AUC = {auc:.4f})")

    return {
        'test': 'mann_whitney_u',
        'statistic': u_stat,
        'p_value': p_value,
        'effect_size': auc,
        'median_difference': median_diff,
        'relative_difference_percent': rel_diff_percent,
        'is_significant': p_value <= 0.05,
        'effect_size_interpretation': effect_size_interpretation
    }


def analyze_column_significance(df_train, column_name, group_column='is_successful'):
    """
    Аналізує статистичну значущість колонки для розмежування двох груп.
    Автоматично обирає відповідний метод на основі оцінки нормальності розподілу.

    Args:
        df_train: Тренувальний DataFrame
        column_name: Назва колонки для аналізу
        group_column: Назва колонки групування

    Returns:
        Словник з результатами аналізу
    """
    logger = logging.getLogger()
    log_section(logger, f"АНАЛІЗ СТАТИСТИЧНОЇ ЗНАЧУЩОСТІ: '{column_name}'", 2)

    # Розділення даних за групами
    group_0 = df_train[df_train[group_column] == 0][column_name].dropna()
    group_1 = df_train[df_train[group_column] == 1][column_name].dropna()

    # Перевірка достатньої кількості даних
    if len(group_0) < 3 or len(group_1) < 3:
        logger.info(f"Недостатньо даних для аналізу: група 0 = {len(group_0)}, група 1 = {len(group_1)}")
        return {
            'column': column_name,
            'is_significant': False,
            'reason': 'Недостатньо даних',
            'method': None
        }

    # Оцінка нормальності розподілу
    normality_results = assess_normality_for_column(df_train, column_name, group_column)

    # Вибір методу аналізу на основі результатів оцінки нормальності
    if normality_results['both_normal']:
        # Параметричний тест для нормальних розподілів
        results = compare_groups_parametric(group_0, group_1)
    else:
        # Непараметричний тест для ненормальних розподілів
        results = compare_groups_nonparametric(group_0, group_1)

    # Визначення напрямку впливу (чи більші значення збільшують ймовірність успіху)
    if normality_results['both_normal']:
        # Для параметричного тесту використовуємо середні значення
        direction = group_1.mean() > group_0.mean()
    else:
        # Для непараметричного тесту використовуємо медіани
        direction = group_1.median() > group_0.median()

    # Визначення порогового значення для прийняття рішення
    if direction:
        # Якщо більші значення → успіх, поріг = середнє арифметичне між середніми/медіанами
        if normality_results['both_normal']:
            threshold = (group_0.mean() + group_1.mean()) / 2
        else:
            threshold = (group_0.median() + group_1.median()) / 2
    else:
        # Якщо менші значення → успіх, поріг = те ж саме
        if normality_results['both_normal']:
            threshold = (group_0.mean() + group_1.mean()) / 2
        else:
            threshold = (group_0.median() + group_1.median()) / 2

    # Логування результатів
    logger.info("\nРезультати аналізу значущості:")
    logger.info(f"Обраний метод: {'ПАРАМЕТРИЧНИЙ' if normality_results['both_normal'] else 'НЕПАРАМЕТРИЧНИЙ'}")
    logger.info(f"p-значення: {results['p_value']:.8f} (α = 0.05)")

    if results['is_significant']:
        effect_size = results.get('effect_size', 0)
        if 'effect_size_interpretation' in results:
            logger.info(f"Розмір ефекту: {effect_size:.4f} ({results['effect_size_interpretation']})")
        else:
            logger.info(f"Розмір ефекту: {effect_size:.4f}")

        # Логування напрямку впливу
        direction_text = "більші значення → вища ймовірність успіху" if direction else "менші значення → вища ймовірність успіху"
        logger.info(f"Напрямок впливу: {direction_text}")
        logger.info(f"Порогове значення: {threshold:.4f}")
    else:
        logger.info("Колонка не є статистично значущою для розмежування груп")

    # Повний результат
    return {
        'column': column_name,
        'is_significant': results['is_significant'],
        'p_value': results['p_value'],
        'effect_size': results.get('effect_size', 0),
        'effect_size_interpretation': results.get('effect_size_interpretation', ''),
        'method': 'parametric' if normality_results['both_normal'] else 'nonparametric',
        'direction': direction,
        'threshold': threshold,
        'full_results': {**results, **normality_results}
    }


def calculate_feature_weight(effect_size, method):
    """
    Розраховує вагу ознаки на основі розміру ефекту.

    Args:
        effect_size: Значення розміру ефекту (Cohen's d або AUC)
        method: Метод, використаний для розрахунку ('parametric' або 'nonparametric')

    Returns:
        Нормалізована вага ознаки від 0 до 1
    """
    logger = logging.getLogger()

    if method == 'parametric':
        # Cohen's d: нормалізація великих ефектів
        # Трансформація d в діапазон [0, 1]
        weight = 0.5 + 0.5 * np.tanh(effect_size / 2)  # tanh обмежує великі значення
    else:
        # AUC: вже в діапазоні [0, 1], масштабуємо до [0.5, 1]
        weight = 2 * abs(effect_size - 0.5) if effect_size >= 0.5 else 0

    logger.info(f"Розрахована вага фактора: {weight:.4f} (на основі розміру ефекту: {effect_size:.4f})")
    return weight


def train_statistical_model(df_train, numeric_columns, group_column='is_successful', optimize_weights=True):
    """
    Навчає статистичну модель на основі значущих колонок.
    Виконує аналіз статистичної значущості для всіх числових колонок,
    визначає порогові значення та ваги факторів.

    Args:
        df_train: Тренувальний DataFrame
        numeric_columns: Список числових колонок для аналізу
        group_column: Назва цільової колонки
        optimize_weights: Чи оптимізувати ваги ознак через крос-валідацію

    Returns:
        Словник з параметрами моделі
    """
    logger = logging.getLogger()
    log_section(logger, "НАВЧАННЯ СТАТИСТИЧНОЇ МОДЕЛІ", 1)

    # Аналіз кожної колонки
    results = {}
    significant_columns = []

    for column in numeric_columns:
        logger.info(f"\nАналіз колонки: '{column}'")
        result = analyze_column_significance(df_train, column, group_column)
        results[column] = result

        if result['is_significant']:
            significant_columns.append(column)

    # Побудова моделі на основі значущих колонок
    logger.info(
        f"\nВиявлено {len(significant_columns)} статистично значущих колонок з {len(numeric_columns)} загальних")

    model_features = {}
    for column in significant_columns:
        result = results[column]
        weight = calculate_feature_weight(result['effect_size'], result['method'])

        model_features[column] = {
            'threshold': result['threshold'],
            'direction': result['direction'],
            'weight': weight,
            'p_value': result['p_value'],
            'effect_size': result['effect_size']
        }

    # Сортування колонок за вагою (важливістю)
    sorted_features = sorted(model_features.items(), key=lambda x: x[1]['weight'], reverse=True)

    logger.info("\nЗначущі колонки (відсортовані за важливістю):")
    for i, (column, params) in enumerate(sorted_features, 1):
        direction_text = "більше → успіх" if params['direction'] else "менше → успіх"
        logger.info(
            f"{i}. {column}: поріг = {params['threshold']:.2f}, напрямок: {direction_text}, вага: {params['weight']:.4f}")

    # Підрахунок розподілу цільової змінної
    target_distribution = df_train[group_column].value_counts().to_dict()

    # Модель з усіма необхідними параметрами
    model = {
        'features': model_features,
        'metadata': {
            'total_columns': len(numeric_columns),
            'significant_columns': len(significant_columns),
            'target_distribution': target_distribution,
            # Оптимальний поріг класифікації (за замовчуванням 0.5)
            'classification_threshold': 0.5
        }
    }

    print('\n')
    print('***'*3)
    print('MODEL before optimization')
    print(model)
    print('***'*3)
    print('\n')
    # Оптимізація ваг через крос-валідацію
    if optimize_weights and significant_columns:
        logger.info("\nЗастосування оптимізації ваг через крос-валідацію...")
        model, optimized_score = optimize_weights_cv(df_train, model, group_column)
        model['metadata']['optimized'] = True
        model['metadata']['optimized_score'] = optimized_score
    else:
        model['metadata']['optimized'] = False

    print('\n')
    print('***'*3)
    print('MODEL after optimization')
    print(model)
    print('***'*3)
    print('\n')

    return model


def predict_with_statistical_model(df_test, model):
    """
    Виконує прогнозування на тестовій вибірці за допомогою статистичної моделі.

    Args:
        df_test: Тестовий DataFrame
        model: Статистична модель, навчена на тренувальних даних

    Returns:
        DataFrame з прогнозами та ймовірностями
    """
    logger = logging.getLogger()

    log_section(logger, "ПРОГНОЗУВАННЯ НА ТЕСТОВІЙ ВИБІРЦІ", 1)

    logger.info(f"Прогнозування для {len(df_test)} спостережень")

    # Копія тестових даних для додавання прогнозів
    predictions = df_test.copy()

    # Розрахунок "голосів" для кожного спостереження за кожною значущою колонкою
    for column, params in model['features'].items():
        if column not in df_test.columns:
            logger.warning(f"Колонка '{column}' відсутня у тестових даних")
            continue

        logger.info(f"Обробка колонки '{column}'...")

        # Визначення, чи перевищує значення поріг
        exceeds_threshold = df_test[column] > params['threshold']

        # Якщо більше значення = більша ймовірність успіху
        if params['direction']:
            predictions[f"{column}_vote"] = exceeds_threshold.astype(float) * params['weight']
            logger.info(f"  Напрямок: більше → успіх, поріг = {params['threshold']:.4f}, вага = {params['weight']:.4f}")
        # Якщо менше значення = більша ймовірність успіху
        else:
            predictions[f"{column}_vote"] = (~exceeds_threshold).astype(float) * params['weight']
            logger.info(f"  Напрямок: менше → успіх, поріг = {params['threshold']:.4f}, вага = {params['weight']:.4f}")

    # Сумарний "голос" усіх значущих колонок
    vote_columns = [col for col in predictions.columns if col.endswith('_vote')]
    if vote_columns:
        predictions['total_vote'] = predictions[vote_columns].sum(axis=1)

        # Нормалізація голосів для отримання "ймовірності"
        max_possible_vote = sum(model['features'][col]['weight'] for col in model['features'])
        if max_possible_vote > 0:  # Запобігаємо діленню на нуль
            predictions['probability'] = predictions['total_vote'] / max_possible_vote
            logger.info(f"Максимальний можливий голос: {max_possible_vote:.4f}")
        else:
            predictions['probability'] = 0.5  # Якщо немає значущих колонок
            logger.warning("Немає значущих колонок для прогнозування, встановлено базову ймовірність 0.5")
    else:
        # Якщо немає значущих колонок, виставляємо базову ймовірність
        logger.warning("Немає значущих колонок з голосами, використовується базова ймовірність")
        positive_ratio = model['metadata']['target_distribution'].get(1, 0) / sum(
            model['metadata']['target_distribution'].values())
        predictions['probability'] = positive_ratio
        logger.info(f"Базова ймовірність (доля позитивного класу в тренувальних даних): {positive_ratio:.4f}")

    # Класифікація на основі порогу
    classification_threshold = model['metadata']['classification_threshold']
    predictions['predicted_class'] = (predictions['probability'] > classification_threshold).astype(int)
    logger.info(f"Використаний поріг класифікації: {classification_threshold:.4f}")

    # Логування розподілу прогнозів
    pred_dist = predictions['predicted_class'].value_counts()
    logger.info("\nРозподіл прогнозованих класів:")
    logger.info(f"0 (Неуспішні): {pred_dist.get(0, 0)} ({pred_dist.get(0, 0) / len(predictions) * 100:.2f}%)")
    logger.info(f"1 (Успішні): {pred_dist.get(1, 0)} ({pred_dist.get(1, 0) / len(predictions) * 100:.2f}%)")

    return predictions


def evaluate_model_performance(y_true, y_pred, y_prob):
    """
    Оцінює якість моделі на основі різних метрик ефективності.

    Args:
        y_true: Фактичні значення цільової змінної
        y_pred: Прогнозовані класи (0 або 1)
        y_prob: Прогнозовані ймовірності для класу 1

    Returns:
        Словник з метриками ефективності
    """
    logger = logging.getLogger()
    log_section(logger, "ОЦІНКА ЯКОСТІ МОДЕЛІ", 1)

    # Базові метрики класифікації
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
    balanced_accuracy = metrics.balanced_accuracy_score(y_true, y_pred)

    # ROC AUC (для ймовірнісних прогнозів)
    try:
        roc_auc = metrics.roc_auc_score(y_true, y_prob)
    except:
        roc_auc = None
        logger.warning("Не вдалося розрахувати ROC AUC (можливо, недостатньо класів)")

    # Матриця помилок
    cm = metrics.confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Додаткові метрики
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Специфічність = TN / (TN + FP)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value

    # Логування результатів
    logger.info("\nМетрики ефективності моделі:")
    logger.info(f"Точність (Accuracy): {accuracy:.4f}")
    logger.info(f"Збалансована точність (Balanced Accuracy): {balanced_accuracy:.4f}")
    logger.info(f"Точність (Precision): {precision:.4f}")
    logger.info(f"Повнота (Recall / Sensitivity): {recall:.4f}")
    logger.info(f"Специфічність (Specificity): {specificity:.4f}")
    logger.info(f"F1-міра: {f1:.4f}")
    if roc_auc:
        logger.info(f"ROC AUC: {roc_auc:.4f}")
    logger.info(f"Negative Predictive Value: {npv:.4f}")

    # Матриця помилок
    logger.info("\nМатриця помилок:")
    logger.info("               | Прогноз: 0 | Прогноз: 1 |")
    logger.info(f"Фактично: 0 | {tn:10d} | {fp:10d} |")
    logger.info(f"Фактично: 1 | {fn:10d} | {tp:10d} |")

    # Зведена інформація по класам
    logger.info("\nЗведена інформація по класам:")
    for cls in sorted(np.unique(np.concatenate([y_true, y_pred]))):
        tpr = recall if cls == 1 else specificity
        cls_precision = precision if cls == 1 else npv
        logger.info(f"Клас {cls}: Точність = {cls_precision:.4f}, Повнота = {tpr:.4f}")

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'roc_auc': roc_auc,
        'npv': npv,
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }
    }


def plot_model_results(df_test, y_true, y_prob, feature_importance=None, output_dir='.'):
    """
    Створює візуалізації результатів моделі.

    Args:
        df_test: Тестовий набір даних
        y_true: Фактичні значення цільової змінної
        y_prob: Прогнозовані ймовірності
        feature_importance: Словник з важливістю ознак
        output_dir: Каталог для збереження графіків

    Returns:
        Список шляхів до збережених графіків
    """
    logger = logging.getLogger()
    log_section(logger, "ВІЗУАЛІЗАЦІЯ РЕЗУЛЬТАТІВ", 1)

    os.makedirs(output_dir, exist_ok=True)
    saved_plots = []

    try:
        plt.figure(figsize=(10, 8))

        # 1. ROC крива
        logger.info("Створення ROC-кривої...")
        fpr, tpr, _ = metrics.roc_curve(y_true, y_prob)
        roc_auc = metrics.auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC крива (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC крива')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        roc_path = os.path.join(output_dir, 'roc_curve.png')
        plt.savefig(roc_path)
        plt.close()
        saved_plots.append(roc_path)
        logger.info(f"ROC крива збережена у {roc_path}")

        # 2. Матриця помилок (confusion matrix)
        logger.info("Створення матриці помилок...")
        y_pred = (y_prob > 0.5).astype(int)
        cm = metrics.confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    cbar=False, annot_kws={"size": 14})
        plt.xlabel('Прогнозований клас')
        plt.ylabel('Фактичний клас')
        plt.title('Матриця помилок')
        plt.xticks([0.5, 1.5], ['0', '1'])
        plt.yticks([0.5, 1.5], ['0', '1'])

        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()
        saved_plots.append(cm_path)
        logger.info(f"Матриця помилок збережена у {cm_path}")

        # 3. Розподіл ймовірностей між класами
        logger.info("Створення гістограми розподілу ймовірностей...")
        plt.figure(figsize=(10, 6))

        sns.histplot(x=y_prob[y_true == 0], color="red", alpha=0.5,
                     bins=20, label="Фактичний клас 0", kde=True)
        sns.histplot(x=y_prob[y_true == 1], color="blue", alpha=0.5,
                     bins=20, label="Фактичний клас 1", kde=True)

        plt.axvline(x=0.5, color='black', linestyle='--')
        plt.xlabel('Прогнозована ймовірність класу 1')
        plt.ylabel('Частота')
        plt.title('Розподіл прогнозованих ймовірностей за фактичними класами')
        plt.legend()
        plt.grid(True, alpha=0.3)

        hist_path = os.path.join(output_dir, 'probability_distribution.png')
        plt.savefig(hist_path)
        plt.close()
        saved_plots.append(hist_path)
        logger.info(f"Гістограма розподілу збережена у {hist_path}")

        # 4. Важливість ознак
        if feature_importance:
            logger.info("Створення графіка важливості ознак...")
            features = list(feature_importance.keys())
            importance = [feature_importance[f] for f in features]

            feature_df = pd.DataFrame({
                'feature': features,
                'importance': importance
            }).sort_values('importance', ascending=False)

            plt.figure(figsize=(12, 8))
            sns.barplot(data=feature_df, x='importance', y='feature', palette='viridis')
            plt.title('Важливість ознак')
            plt.xlabel('Важливість')
            plt.ylabel('Ознака')
            plt.grid(True, axis='x', alpha=0.3)
            plt.tight_layout()

            feat_path = os.path.join(output_dir, 'feature_importance.png')
            plt.savefig(feat_path)
            plt.close()
            saved_plots.append(feat_path)
            logger.info(f"Графік важливості ознак збережено у {feat_path}")

    except Exception as e:
        logger.error(f"Помилка при створенні візуалізацій: {e}")
        logger.exception("Деталі помилки:")

    return saved_plots


class NumpyEncoder(json.JSONEncoder):
    """
    Спеціальний енкодер для JSON, який обробляє типи даних numpy та pandas.
    """

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (pd.Series)):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif np.isnan(obj):
            return None
        return super(NumpyEncoder, self).default(obj)


def save_model_results(model, evaluation_results, output_file='statistical_model_results.json'):
    """
    Зберігає результати моделі та її параметри у JSON форматі.

    Args:
        model: Статистична модель
        evaluation_results: Результати оцінки ефективності
        output_file: Шлях для збереження результатів

    Returns:
        Шлях до збереженого файлу
    """
    logger = logging.getLogger()
    log_section(logger, "ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ МОДЕЛІ", 1)

    try:
        # Збирання всіх результатів в один словник
        results = {
            'model': model,
            'evaluation': evaluation_results,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Збереження у JSON з використанням спеціального енкодера
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

        logger.info(f"Результати моделі збережені у файлі: {output_file}")
        return output_file

    except Exception as e:
        logger.error(f"Помилка при збереженні результатів моделі: {e}")
        logger.exception("Деталі помилки:")
        return None


def optimize_weights_cv(df, model, target_column='is_successful', n_splits=5):
    """
    Оптимізація ваг ознак за допомогою крос-валідації.

    Args:
        df: DataFrame з даними
        model: Модель зі значущими ознаками
        target_column: Назва цільової колонки
        n_splits: Кількість фолдів для крос-валідації

    Returns:
        Оновлена модель та найкращий середній F1-score
    """
    logger = logging.getLogger()
    log_section(logger, "ОПТИМІЗАЦІЯ ВАГ ОЗНАК ЧЕРЕЗ КРОС-ВАЛІДАЦІЮ", 2)

    from sklearn.model_selection import KFold
    from sklearn import metrics
    import numpy as np

    features = list(model['features'].keys())
    if not features:
        logger.warning("Немає значущих ознак для оптимізації ваг")
        return model, 0.0

    logger.info(f"Початок оптимізації ваг для {len(features)} значущих ознак")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Підготовка даних
    X = df[features]
    y = df[target_column]

    # Логування початкових ваг перед оптимізацією
    logger.info("\nПочаткові ваги ознак:")
    sorted_initial_features = sorted(model['features'].items(), key=lambda x: x[1]['weight'], reverse=True)
    for i, (column, params) in enumerate(sorted_initial_features, 1):
        direction_text = "більше → успіх" if params['direction'] else "менше → успіх"
        logger.info(
            f"{i}. {column}: вага: {params['weight']:.4f}, поріг = {params['threshold']:.2f}, напрямок: {direction_text}")

    # Обчислення початкового F1-score на всьому наборі даних
    initial_predictions = predict_with_statistical_model(df.copy(), model)
    initial_f1 = metrics.f1_score(df[target_column], initial_predictions['predicted_class'])
    logger.info(f"Початковий F1-score на всьому наборі даних: {initial_f1:.4f}")

    # Визначення можливих комбінацій ваг для пошуку
    weight_grid = []
    num_combinations = 50  # Збільшуємо кількість комбінацій

    # Початкові ваги
    initial_weights = np.array([model['features'][f]['weight'] for f in features])

    # 1. Додаємо початкові ваги
    weight_grid.append(initial_weights)

    # 2. Додаємо ваги з невеликими випадковими варіаціями від початкових
    for _ in range(15):
        # Варіюємо в межах ±30% від початкових значень
        variation_factors = np.random.uniform(0.7, 1.3, len(features))
        weights = initial_weights * variation_factors
        # Нормалізуємо
        weights /= weights.sum()
        weight_grid.append(weights)

    # 3. Додаємо ваги, що посилюють ознаки з найбільшими початковими вагами
    top_features_idx = np.argsort(initial_weights)[-3:]  # індекси топ-3 ознак
    for _ in range(10):
        weights = initial_weights.copy()
        # Збільшуємо ваги топових ознак
        boost_factors = np.random.uniform(1.2, 1.5, len(top_features_idx))
        for i, idx in enumerate(top_features_idx):
            weights[idx] *= boost_factors[i]
        # Нормалізуємо
        weights /= weights.sum()
        weight_grid.append(weights)

    # 4. Додаємо ваги, що перерозподіляють важливість між ознаками
    for _ in range(15):
        weights = np.random.uniform(0.1, 1.0, len(features))
        # Додаємо вплив p-value: важливіші ознаки отримують трохи більшу вагу
        importance_factors = np.array([1.0 / max(0.0001, model['features'][f]['p_value']) for f in features])
        importance_factors = importance_factors / importance_factors.sum()
        weights = weights * importance_factors
        # Нормалізуємо
        weights /= weights.sum()
        weight_grid.append(weights)

    # 5. Додаємо повністю рандомні ваги для різноманітності
    for _ in range(10):
        weights = np.random.uniform(0.1, 1.0, len(features))
        weights /= weights.sum()
        weight_grid.append(weights)

    # Ініціалізуємо best_score початковим значенням F1-score
    best_score = initial_f1
    best_weights = initial_weights.copy()

    logger.info(f"Перевірка {len(weight_grid)} комбінацій ваг з використанням {n_splits}-фолдової крос-валідації")

    # Зберігаємо всі результати для аналізу
    all_combinations = []

    for weights_idx, weights in enumerate(weight_grid):
        fold_scores = []

        for train_idx, val_idx in kf.split(X):
            # Робимо копію моделі і встановлюємо поточні ваги
            model_copy = {
                'features': {f: model['features'][f].copy() for f in features},
                'metadata': model['metadata'].copy()
            }

            for i, feature in enumerate(features):
                model_copy['features'][feature]['weight'] = float(weights[i])

            # Прогнозуємо на валідаційній вибірці
            val_df = df.iloc[val_idx].copy()
            predictions = predict_with_statistical_model(val_df, model_copy)
            y_true = val_df[target_column]
            y_pred = predictions['predicted_class']

            # Обчислюємо метрику (F1-score)
            fold_score = metrics.f1_score(y_true, y_pred)
            fold_scores.append(fold_score)

        # Середній результат по всіх фолдах
        avg_score = np.mean(fold_scores)

        # Зберігаємо результат для аналізу
        all_combinations.append((weights, avg_score))

        # Перевіряємо, чи це найкращий результат
        if avg_score > best_score:
            best_score = avg_score
            best_weights = weights.copy()
            logger.info(f"Нова найкраща комбінація знайдена! F1: {best_score:.4f}")

        if (weights_idx + 1) % 10 == 0 or weights_idx == 0:
            logger.info(
                f"Перевірено {weights_idx + 1}/{len(weight_grid)} комбінацій. Поточний найкращий F1: {best_score:.4f}")

    # Виводимо топ-5 комбінацій для аналізу
    logger.info("\nТоп-5 найкращих комбінацій ваг:")
    sorted_combinations = sorted(all_combinations, key=lambda x: x[1], reverse=True)
    for i, (weights, score) in enumerate(sorted_combinations[:5], 1):
        logger.info(f"Комбінація #{i}: F1-score = {score:.4f}")
        # Виводимо найбільші ваги з цієї комбінації
        top_weight_indices = np.argsort(weights)[-3:]  # індекси топ-3 ваг
        for idx in reversed(top_weight_indices):
            feature_name = features[idx]
            logger.info(f"  - {feature_name}: {weights[idx]:.4f}")

    # Оновлюємо ваги в моделі
    for i, feature in enumerate(features):
        model['features'][feature]['weight'] = float(best_weights[i])

    logger.info(f"\nРезультати оптимізації:")
    logger.info(f"Початковий F1-score: {initial_f1:.4f}")
    logger.info(f"Оптимізований F1-score: {best_score:.4f}")
    logger.info(f"Покращення: {(best_score - initial_f1) * 100:.2f}%")

    # Виводимо оптимізовані ваги
    logger.info("\nОптимізовані ваги ознак:")
    sorted_features = sorted(model['features'].items(), key=lambda x: x[1]['weight'], reverse=True)
    for i, (column, params) in enumerate(sorted_features, 1):
        direction_text = "більше → успіх" if params['direction'] else "менше → успіх"
        logger.info(
            f"{i}. {column}: вага: {params['weight']:.4f}, поріг = {params['threshold']:.2f}, напрямок: {direction_text}")

    # Переконуємось, що модель містить правильні ваги
    model_check_weights = [model['features'][f]['weight'] for f in features]
    logger.info(f"\nПеревірка ваг після оптимізації: {np.allclose(model_check_weights, best_weights)}")

    return model, best_score


def main():
    """
    Основна функція для запуску повного статистичного аналізу та моделювання.
    """
    # Налаштування логування
    setup_logging()
    logger = logging.getLogger()
    log_section(logger, "СТАТИСТИЧНИЙ АНАЛІЗ ТА ПРОГНОЗУВАННЯ УСПІШНОСТІ ЗАМОВЛЕНЬ", 0)



    try:
        # Завантаження та попередня обробка даних
        logger.info("Завантаження даних...")
        data_file = 'cleanest_data.csv'
        df, numeric_columns = load_data(data_file,
                                        handle_outliers_flag=False)  # Спершу завантаження без обробки викидів
        print("\nBEFORE")
        # Тимчасово змінюємо налаштування
        with pd.option_context('display.max_columns', None):
            print(df.describe())
        # Обробка викидів після завантаження даних
        if HANDLE_OUTLIERS:
            log_section(logger, "ОБРОБКА ВИКИДІВ У ДАНИХ", 2)
            logger.info(f"Метод виявлення викидів: {OUTLIER_METHOD}")
            logger.info(f"Метод обробки викидів: {OUTLIER_TREATMENT}")
            logger.info(f"Поріг для виявлення: {OUTLIER_THRESHOLD}")

            df, outlier_report = handle_all_outliers(
                df, numeric_columns,
                method=OUTLIER_METHOD,
                treatment=OUTLIER_TREATMENT,
                threshold=OUTLIER_THRESHOLD
            )

            # Створення візуалізацій для колонок з найбільшою кількістю викидів
            if outlier_report:
                top_outliers = sorted(
                    outlier_report.items(),
                    key=lambda x: x[1]['outliers_found'] if 'outliers_found' in x[1] else 0,
                    reverse=True
                )[:5]  # Топ-5 колонок з найбільшою кількістю викидів

                output_dir = 'outlier_plots'
                os.makedirs(output_dir, exist_ok=True)

                for col, _ in top_outliers:
                    plot_outliers(df, col, output_dir=output_dir)
        print("\nAFTER")
        # Тимчасово змінюємо налаштування
        with pd.option_context('display.max_columns', None):
            print(df.describe())
        # Розділення на тренувальну та тестову вибірки
        df_train, df_test = split_train_test(df, test_size=0.3)

        # Навчання статистичної моделі
        logger.info("Навчання статистичної моделі...")
        model = train_statistical_model(df_train, numeric_columns, group_column='is_successful', optimize_weights=True)

        # Прогнозування
        logger.info("Застосування моделі до тестової вибірки...")

        # Логування ваг перед застосуванням моделі для перевірки, що оптимізація спрацювала
        log_section(logger, "ВАГИ ОЗНАК ПЕРЕД ПРОГНОЗУВАННЯМ", 3)
        sorted_features = sorted(model['features'].items(), key=lambda x: x[1]['weight'], reverse=True)
        for i, (column, params) in enumerate(sorted_features, 1):
            direction_text = "більше → успіх" if params['direction'] else "менше → успіх"
            logger.info(
                f"{i}. {column}: вага: {params['weight']:.4f}, поріг = {params['threshold']:.2f}, напрямок: {direction_text}")

        predictions = predict_with_statistical_model(df_test, model)

        # Оцінка якості моделі
        logger.info("Оцінка якості моделі...")
        y_true = df_test['is_successful'].values
        y_pred = predictions['predicted_class'].values
        y_prob = predictions['probability'].values

        evaluation_results = evaluate_model_performance(y_true, y_pred, y_prob)

        # Виводимо інформацію про оптимізацію
        if model.get('metadata', {}).get('optimized', False):
            log_section(logger, "ІНФОРМАЦІЯ ПРО ОПТИМІЗАЦІЮ МОДЕЛІ", 3)
            logger.info(f"Модель була оптимізована: {model['metadata']['optimized']}")
            logger.info(f"Оптимізований F1-score: {model['metadata'].get('optimized_score', 0):.4f}")

        # Створення візуалізацій
        logger.info("Створення візуалізацій...")
        output_dir = 'results_visualizations'

        # Підготовка даних про важливість ознак для візуалізації
        feature_importance = {col: params['weight'] for col, params in model['features'].items()}

        saved_plots = plot_model_results(df_test, y_true, y_prob,
                                         feature_importance=feature_importance,
                                         output_dir=output_dir)

        # Збереження результатів моделі
        logger.info("Збереження результатів моделі...")
        output_file = 'statistical_model_results.json'
        save_model_results(model, evaluation_results, output_file)

        logger.info("\nАналіз завершено успішно!")
        logger.info(f"Візуалізації збережені в директорії: {output_dir}")
        logger.info(f"Результати моделі збережені у файлі: {output_file}")

    except Exception as e:
        logger.error(f"Помилка під час виконання аналізу: {str(e)}")
        logger.exception("Детальна інформація про помилку:")
        raise


if __name__ == "__main__":
    main()
