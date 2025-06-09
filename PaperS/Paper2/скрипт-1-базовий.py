# statistical_prediction_model.py

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def load_data(file_path, group_column='is_successful'):
    """
    Завантажує дані, замінює негативні значення на нуль та виділяє числові колонки.

    Args:
        file_path: Шлях до CSV-файлу з даними
        group_column: Назва цільової колонки

    Returns:
        DataFrame з даними та список числових колонок
    """
    df = pd.read_csv(file_path)
    df[group_column] = df[group_column].astype(int)

    # Виділення числових колонок
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if group_column in numeric_columns:
        numeric_columns.remove(group_column)

    # Заміна негативних значень нулями
    for col in numeric_columns:
        df[col] = df[col].apply(lambda x: max(0, x) if not pd.isna(x) else x)

    return df, numeric_columns


def split_train_test(df, test_size=0.3, random_state=42):
    """
    Розбиває дані на тренувальну та тестову вибірки.

    Args:
        df: DataFrame з даними
        test_size: Частка тестової вибірки (0-1)
        random_state: Зерно для відтворюваності результатів

    Returns:
        Тренувальна та тестова вибірки
    """
    return train_test_split(df, test_size=test_size, random_state=random_state)


def test_normality(data):
    """
    Перевіряє нормальність розподілу даних.

    Args:
        data: Серія даних для аналізу

    Returns:
        Словник з результатами тестів на нормальність
    """
    # Тест Шапіро-Вілка
    if len(data) <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro(data)
        shapiro_normal = shapiro_p > 0.05
    else:
        shapiro_stat, shapiro_p = None, None
        shapiro_normal = None

    # Тест Д'Агостіно-Пірсона
    dagostino_stat, dagostino_p = stats.normaltest(data)
    dagostino_normal = dagostino_p > 0.05

    # Тест Андерсона-Дарлінга
    anderson_result = stats.anderson(data)
    anderson_stat = anderson_result.statistic
    anderson_critical = anderson_result.critical_values[2]  # для 5% рівня значущості
    anderson_normal = anderson_stat < anderson_critical

    # Загальний висновок: нормальний розподіл, якщо всі тести підтверджують
    if shapiro_normal is None:
        is_normal = dagostino_normal and anderson_normal
    else:
        is_normal = shapiro_normal and dagostino_normal and anderson_normal

    return {
        'shapiro': {'statistic': shapiro_stat, 'p_value': shapiro_p, 'normal': shapiro_normal},
        'dagostino': {'statistic': dagostino_stat, 'p_value': dagostino_p, 'normal': dagostino_normal},
        'anderson': {'statistic': anderson_stat, 'critical_value': anderson_critical, 'normal': anderson_normal},
        'is_normal': is_normal
    }


def analyze_column_significance(df_train, column, group_column='is_successful', alpha=0.05):
    """
    Аналізує статистичну значущість різниці між групами для заданої колонки.

    Args:
        df_train: Тренувальний набір даних
        column: Назва колонки для аналізу
        group_column: Назва цільової колонки
        alpha: Рівень значущості

    Returns:
        Словник з результатами статистичного аналізу колонки
    """
    # Розділення даних за групами
    group_0 = df_train[df_train[group_column] == 0][column].dropna()
    group_1 = df_train[df_train[group_column] == 1][column].dropna()

    if len(group_0) < 2 or len(group_1) < 2:
        return {'is_significant': False, 'reason': 'Недостатньо даних', 'column': column}

    # Перевірка нормальності розподілу
    normality_0 = test_normality(group_0)
    normality_1 = test_normality(group_1)
    both_normal = normality_0['is_normal'] and normality_1['is_normal']

    # Розрахунок базової статистики
    stats_0 = {
        'mean': group_0.mean(),
        'median': group_0.median(),
        'std': group_0.std(),
        'min': group_0.min(),
        'max': group_0.max(),
        'q1': group_0.quantile(0.25),
        'q3': group_0.quantile(0.75)
    }

    stats_1 = {
        'mean': group_1.mean(),
        'median': group_1.median(),
        'std': group_1.std(),
        'min': group_1.min(),
        'max': group_1.max(),
        'q1': group_1.quantile(0.25),
        'q3': group_1.quantile(0.75)
    }

    # Проведення статистичних тестів
    if both_normal:
        # Параметричний тест (t-тест Велча)
        t_stat, p_value = stats.ttest_ind(group_0, group_1, equal_var=False)
        test_name = "Welch's t-test"
        significant = p_value < alpha

        # Розрахунок розміру ефекту (Cohen's d)
        pooled_std = np.sqrt(((len(group_0) - 1) * group_0.var() + (len(group_1) - 1) * group_1.var()) /
                             (len(group_0) + len(group_1) - 2))
        effect_size = (group_1.mean() - group_0.mean()) / pooled_std
        effect_type = 'cohen_d'
    else:
        # Непараметричний тест (Манна-Уітні)
        u_stat, p_value = stats.mannwhitneyu(group_0, group_1)
        test_name = "Mann-Whitney U test"
        significant = p_value < alpha

        # Розрахунок AUC як міри ефекту
        all_values = pd.concat([group_0, group_1]).values
        all_labels = np.concatenate([np.zeros(len(group_0)), np.ones(len(group_1))])
        effect_size = roc_auc_score(all_labels, all_values)
        if effect_size < 0.5:
            effect_size = 1 - effect_size
        effect_type = 'auc'

    # Визначення "порогу рішення" на основі розподілу
    # Для нормального розподілу: середнє між середніми значеннями груп
    # Для ненормального: медіана між медіанами груп
    if both_normal:
        threshold = (stats_0['mean'] + stats_1['mean']) / 2
        threshold_type = 'mean'
    else:
        threshold = (stats_0['median'] + stats_1['median']) / 2
        threshold_type = 'median'

    # Визначення напрямку впливу: true, якщо більше значення → більша ймовірність успіху
    if both_normal:
        direction = stats_1['mean'] > stats_0['mean']
    else:
        direction = stats_1['median'] > stats_0['median']

    # Відносна різниця (для оцінки важливості)
    if both_normal:
        base_value = abs(stats_0['mean'])
        diff = stats_1['mean'] - stats_0['mean']
    else:
        base_value = abs(stats_0['median'])
        diff = stats_1['median'] - stats_0['median']

    relative_diff = (diff / base_value) if base_value != 0 else float('inf')

    # Обчислення "ваги" колонки на основі статистичних показників
    # Нормалізація p-value: менше p → більша вага
    p_weight = 1 - min(p_value, alpha) / alpha

    # Нормалізація ефекту: більший ефект → більша вага
    if effect_type == 'cohen_d':
        # Для Cohen's d: 0.2 малий, 0.5 середній, 0.8 великий
        effect_weight = min(abs(effect_size) / 0.8, 1.0)
    else:  # AUC
        # Для AUC: 0.5 - випадковий, 1.0 - ідеальний
        effect_weight = 2 * (abs(effect_size) - 0.5)

    # Вага колонки: комбінація p-value та розміру ефекту
    column_weight = (p_weight + effect_weight) / 2

    result = {
        'column': column,
        'is_significant': significant,
        'p_value': p_value,
        'test': test_name,
        'both_normal': both_normal,
        'group_0_stats': stats_0,
        'group_1_stats': stats_1,
        'effect': {
            'type': effect_type,
            'value': effect_size
        },
        'threshold': {
            'value': threshold,
            'type': threshold_type
        },
        'direction': direction,
        'relative_diff': relative_diff,
        'weight': column_weight
    }

    return result


def train_statistical_model(df_train, numeric_columns, group_column='is_successful', alpha=0.05):
    """
    Навчає статистичну модель на тренувальних даних.

    Args:
        df_train: Тренувальний набір даних
        numeric_columns: Список числових колонок для аналізу
        group_column: Назва цільової колонки
        alpha: Рівень значущості

    Returns:
        Модель у вигляді словника з параметрами для кожної значущої колонки
    """
    model = {'features': {}, 'metadata': {}}

    # Аналіз кожної колонки
    for column in numeric_columns:
        result = analyze_column_significance(df_train, column, group_column, alpha)
        if result['is_significant']:
            model['features'][column] = {
                'threshold': result['threshold']['value'],
                'threshold_type': result['threshold']['type'],
                'direction': result['direction'],
                'weight': result['weight']
            }

    # Метадані моделі
    model['metadata'] = {
        'total_columns': len(numeric_columns),
        'significant_columns': len(model['features']),
        'alpha': alpha,
        'target_column': group_column,
        'model_type': 'statistical_weighted_voting',
        'target_distribution': df_train[group_column].value_counts().to_dict()
    }

    # Розрахунок порогу для класифікації (пропорційно розподілу класів)
    positive_ratio = df_train[group_column].mean()
    model['metadata']['classification_threshold'] = positive_ratio

    return model


def predict_with_statistical_model(df_test, model):
    """
    Виконує прогнозування на тестовій вибірці за допомогою статистичної моделі.

    Args:
        df_test: Тестовий набір даних
        model: Статистична модель, навчена на тренувальних даних

    Returns:
        DataFrame з прогнозами та ймовірностями
    """
    # Копія тестових даних для додавання прогнозів
    predictions = df_test.copy()

    # Розрахунок "ваги" кожного спостереження для кожної значущої колонки
    for column, params in model['features'].items():
        if column not in df_test.columns:
            continue

        # Для кожного спостереження визначаємо, чи перевищує воно поріг і чи це "правильний" напрямок
        exceeds_threshold = df_test[column] > params['threshold']

        # Якщо більше значення = більша ймовірність успіху
        if params['direction']:
            predictions[f"{column}_vote"] = exceeds_threshold.astype(float) * params['weight']
        # Якщо більше значення = менша ймовірність успіху
        else:
            predictions[f"{column}_vote"] = (~exceeds_threshold).astype(float) * params['weight']

    # Сумарний "голос" усіх значущих колонок
    vote_columns = [col for col in predictions.columns if col.endswith('_vote')]
    if vote_columns:
        predictions['total_vote'] = predictions[vote_columns].sum(axis=1)

        # Нормалізація голосів для отримання "ймовірності"
        max_possible_vote = sum(model['features'][col]['weight'] for col in model['features'])
        if max_possible_vote > 0:  # Запобігаємо діленню на нуль
            predictions['probability'] = predictions['total_vote'] / max_possible_vote
        else:
            predictions['probability'] = 0.5  # Якщо немає значущих колонок
    else:
        # Якщо немає значущих колонок, виставляємо базову ймовірність
        positive_ratio = model['metadata']['target_distribution'].get(1, 0) / sum(
            model['metadata']['target_distribution'].values())
        predictions['probability'] = positive_ratio

    # Класифікація на основі порогу
    classification_threshold = model['metadata']['classification_threshold']
    predictions['predicted_class'] = (predictions['probability'] > classification_threshold).astype(int)

    return predictions


def evaluate_model(df_test, predictions, target_column='is_successful'):
    """
    Оцінює якість прогнозування моделі.

    Args:
        df_test: Тестовий набір даних з фактичними значеннями
        predictions: DataFrame з прогнозованими значеннями
        target_column: Назва цільової колонки

    Returns:
        Словник з метриками якості моделі
    """
    actual = df_test[target_column]
    predicted = predictions['predicted_class']
    probabilities = predictions['probability']

    metrics = {
        'accuracy': accuracy_score(actual, predicted),
        'precision': precision_score(actual, predicted, zero_division=0),
        'recall': recall_score(actual, predicted, zero_division=0),
        'f1': f1_score(actual, predicted, zero_division=0),
        'roc_auc': roc_auc_score(actual, probabilities),
        'confusion_matrix': confusion_matrix(actual, predicted).tolist()
    }

    # Розрахунок специфічності (true negative rate)
    tn, fp, fn, tp = metrics['confusion_matrix'][0][0], metrics['confusion_matrix'][0][1], \
    metrics['confusion_matrix'][1][0], metrics['confusion_matrix'][1][1]
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Додаткові метрики
    metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2

    return metrics


class NumpyEncoder(json.JSONEncoder):
    """Спеціальний клас для кодування объектів NumPy в JSON."""

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


def save_model_and_results(model, evaluation, file_path='statistical_model'):
    """
    Зберігає модель та результати в JSON-файли з підтримкою різних типів даних.

    Args:
        model: Статистична модель
        evaluation: Результати оцінки моделі
        file_path: Базовий шлях для збереження файлів
    """
    # Перевірка та створення каталогу для збереження результатів
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Збереження моделі
    with open(f"{file_path}_model.json", 'w', encoding='utf-8') as f:
        json.dump(model, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)

    # Збереження результатів оцінки
    with open(f"{file_path}_evaluation.json", 'w', encoding='utf-8') as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)


def visualize_results(model, evaluation, predictions, target_column='is_successful', output_dir='.'):
    """
    Візуалізує результати роботи моделі.

    Args:
        model: Статистична модель
        evaluation: Результати оцінки моделі
        predictions: DataFrame з прогнозованими значеннями
        target_column: Назва цільової колонки
        output_dir: Директорія для збереження візуалізацій
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Важливість колонок
    plt.figure(figsize=(12, 8))
    column_weights = {col: params['weight'] for col, params in model['features'].items()}
    sorted_columns = sorted(column_weights.items(), key=lambda x: x[1], reverse=True)

    plt.barh([col for col, _ in sorted_columns], [weight for _, weight in sorted_columns])
    plt.title('Важливість колонок у прогнозуванні')
    plt.xlabel('Вага')
    plt.ylabel('Колонка')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
    plt.close()

    # 2. ROC-крива
    plt.figure(figsize=(8, 8))
    actual = predictions[target_column]
    probabilities = predictions['probability']

    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(actual, probabilities)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC крива (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-крива')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300)
    plt.close()

    # 3. Матриця помилок
    plt.figure(figsize=(8, 6))
    cm = np.array(evaluation['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Неуспішні', 'Успішні'],
                yticklabels=['Неуспішні', 'Успішні'])
    plt.title('Матриця помилок')
    plt.ylabel('Фактичні значення')
    plt.xlabel('Прогнозовані значення')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()

    # 4. Розподіл ймовірностей
    plt.figure(figsize=(10, 6))
    sns.histplot(data=predictions, x='probability', hue=target_column, bins=50, kde=True)
    plt.axvline(x=model['metadata']['classification_threshold'], color='red', linestyle='--',
                label=f'Поріг класифікації ({model["metadata"]["classification_threshold"]:.2f})')
    plt.title('Розподіл спрогнозованих ймовірностей')
    plt.xlabel('Ймовірність успішності')
    plt.ylabel('Кількість спостережень')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'probability_distribution.png'), dpi=300)
    plt.close()


def run_statistical_prediction_model(file_path, group_column='is_successful', test_size=0.3, random_state=42,
                                     output_dir='.'):
    """
    Запускає повний цикл аналізу: від завантаження даних до оцінки моделі.

    Args:
        file_path: Шлях до файлу з даними
        group_column: Назва цільової колонки
        test_size: Частка тестової вибірки
        random_state: Зерно для відтворюваності результатів
        output_dir: Директорія для збереження результатів

    Returns:
        Словник з моделлю та результатами оцінки
    """
    # Завантаження даних
    print("Завантаження даних...")
    df, numeric_columns = load_data(file_path, group_column)
    print(f"Завантажено {len(df)} записів з {len(numeric_columns)} числовими колонками.")

    # Розбиття на тренувальну та тестову вибірки
    print("Розбиття на тренувальну та тестову вибірки...")
    df_train, df_test = split_train_test(df, test_size, random_state)
    print(f"Тренувальна вибірка: {len(df_train)} записів, Тестова вибірка: {len(df_test)} записів.")

    # Навчання моделі
    print("Аналіз статистичної значущості колонок...")
    model = train_statistical_model(df_train, numeric_columns, group_column)
    print(
        f"Виявлено {model['metadata']['significant_columns']} статистично значущих колонок з {model['metadata']['total_columns']} загальних.")

    # Список значущих колонок
    print("\nСтатистично значущі колонки:")
    for i, (column, params) in enumerate(model['features'].items(), 1):
        direction = "більше → успіх" if params['direction'] else "менше → успіх"
        print(f"{i}. {column}: поріг = {params['threshold']:.2f}, напрямок: {direction}, вага: {params['weight']:.4f}")

    # Прогнозування
    print("\nПрогнозування на тестовій вибірці...")
    predictions = predict_with_statistical_model(df_test, model)

    # Оцінка моделі
    print("Оцінка якості моделі...")
    evaluation = evaluate_model(df_test, predictions, group_column)

    # Вивід результатів
    print("\nРезультати оцінки моделі:")
    print(f"Точність (Accuracy): {evaluation['accuracy']:.4f}")
    print(f"Точність (Precision): {evaluation['precision']:.4f}")
    print(f"Повнота (Recall): {evaluation['recall']:.4f}")
    print(f"F1-міра: {evaluation['f1']:.4f}")
    print(f"ROC AUC: {evaluation['roc_auc']:.4f}")
    print(f"Збалансована точність: {evaluation['balanced_accuracy']:.4f}")

    # Збереження результатів
    # Збереження результатів
    print("\nЗбереження моделі та результатів...")
    os.makedirs(output_dir, exist_ok=True)  # Переконуємось, що директорія існує
    save_model_and_results(model, evaluation, os.path.join(output_dir, 'statistical_model'))

    # Візуалізація результатів
    print("Візуалізація результатів...")
    visualize_results(model, evaluation, predictions, group_column, output_dir)

    return {
        'model': model,
        'evaluation': evaluation,
        'predictions': predictions,
        'df_train': df_train,
        'df_test': df_test
    }


# Приклад використання:
if __name__ == "__main__":
    # Шлях до файлу з даними
    DATA_FILE = 'cleanest_data.csv'

    # Створення директорії для результатів
    output_dir = r'D:\PROJECTs\MY\ScientificCalculations\SC\ScientificCalculations\PaperS\Paper2\results'
    os.makedirs(output_dir, exist_ok=True)

    # Запуск моделі
    results = run_statistical_prediction_model(
        file_path=DATA_FILE,
        group_column='is_successful',
        test_size=0.3,
        random_state=42,
        output_dir=r'D:\PROJECTs\MY\ScientificCalculations\SC\ScientificCalculations\PaperS\Paper2\results'
    )

    print("\nГотово! Результати збережено в директорії 'results'.")