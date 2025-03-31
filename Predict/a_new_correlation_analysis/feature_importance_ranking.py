import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')  # Встановлюємо Agg бекенд
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate
import os

# Створюємо директорію для збереження результатів
results_dir = f"feature_importance_results"
os.makedirs(results_dir, exist_ok=True)

# Налаштування логування
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Створюємо словник для перекладу назв ознак
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

# Функція для отримання українських назв ознак
def get_ua_feature_name(feature):
    """
    Повертає українську назву ознаки, якщо вона є в словнику

    Args:
        feature (str): Назва ознаки англійською

    Returns:
        str: Назва ознаки українською або оригінальна назва
    """
    return feature_names_ua.get(feature, feature)

# Функція для обчислення вартості ознак різними методами
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

    # 4. Logistic Regression Coefficients
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

    # 5. Decision Tree Feature Importance з крос-валідацією
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

    # 6. Random Forest Importance з крос-валідацією
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

    # 7. XGBoost Feature Importance з крос-валідацією
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
        results_df[f'{col} Rank'] = results_df[col].rank(method='average')

    # Обчислюємо загальний рейтинг (сума всіх рангів)
    results_df['Total Importance Score'] = results_df[[
        'MI Score Rank', 'F Score Rank', 'Spearman Score Rank',
        'Absolute Coefficient Rank', 'DT Score Rank', 'RF Score Rank', 'XGBoost Score Rank'
    ]].sum(axis=1)

    # Нормалізуємо до 100%
    max_score = results_df['Total Importance Score'].max()
    results_df['Total Importance Rank'] = round((results_df['Total Importance Score'] / max_score) * 100)

    # Сортуємо за спаданням (найважливіші ознаки зверху)
    results_df = results_df.sort_values(by='Total Importance Rank', ascending=False)

    return results_df, models_dict

# Функція для побудови графіків важливості ознак
def plot_feature_importance(results_df, top_n=15, title='Важливість ознак', save_path=None):
    """
    Будує графік важливості ознак

    Args:
        results_df (pd.DataFrame): DataFrame з результатами оцінки ознак
        top_n (int): Кількість найважливіших ознак для відображення
        title (str): Заголовок графіка
        save_path (str): Шлях для збереження графіка, якщо None - не зберігати
    """
    plt.figure(figsize=(12, 8))

    # Вибираємо топ-N ознак
    plot_df = results_df.head(top_n).copy()

    # Додаємо українські назви для ознак
    plot_df['Feature_UA'] = plot_df['Feature'].apply(get_ua_feature_name)

    # Будуємо графік
    plt.barh(plot_df['Feature_UA'], plot_df['Total Importance Rank'], color='skyblue')
    plt.xlabel('Відносна важливість (%)')
    plt.ylabel('Ознака')
    plt.title(title)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.gca().invert_yaxis()  # Найважливіша ознака зверху

    # Додаємо значення на графіку
    for i, v in enumerate(plot_df['Total Importance Rank']):
        plt.text(v + 1, i, f"{v:.0f}%", va='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.info(f"Графік збережено: {save_path}")

    plt.close()

# Функція для виведення рейтингів
def print_importance_rankings(results_df, top_n=20):
    """
    Виводить рейтинги важливості ознак

    Args:
        results_df (pd.DataFrame): DataFrame з результатами оцінки ознак
        top_n (int): Кількість найважливіших ознак для відображення
    """
    # Створюємо копію та додаємо українські назви
    display_df = results_df.head(top_n).copy()
    display_df['Feature_UA'] = display_df['Feature'].apply(get_ua_feature_name)

    # Виводимо додаткові рейтинги для різних методів
    methods = [
        ('MI Score', 'Mutual Information'),
        ('F Score', 'ANOVA F-test'),
        ('Spearman Score', 'Spearman Correlation'),
        ('Absolute Coefficient', 'Logistic Regression'),
        ('DT Score', 'Decision Tree'),
        ('RF Score', 'Random Forest'),
        ('XGBoost Score', 'XGBoost')
    ]

    # Створюємо окремі таблиці для кожного методу
    for score_col, method_name in methods:
        # Сортуємо за поточним показником
        method_df = results_df.sort_values(by=score_col, ascending=False).head(top_n).copy()
        method_df['Feature_UA'] = method_df['Feature'].apply(get_ua_feature_name)
        method_df['Rank'] = range(1, len(method_df) + 1)

        print(f"\n=== Рейтинг за методом: {method_name} ===")
        method_table = method_df[['Rank', 'Feature', 'Feature_UA', score_col]].values.tolist()
        method_headers = ['Ранг', 'Ознака', 'Назва українською', 'Оцінка']
        print("\n" + tabulate(method_table, headers=method_headers, tablefmt='fancy_grid', floatfmt=".4f"))

    # Виводимо загальний рейтинг в кінці
    print("\n=== Загальний рейтинг важливості ознак ===")
    table_data = display_df[['Feature', 'Feature_UA', 'Total Importance Rank']].values.tolist()
    headers = ['Ознака', 'Назва українською', 'Загальна важливість (%)']
    print("\n" + tabulate(table_data, headers=headers, tablefmt='fancy_grid'))

# Головна функція
def main():
    """
    Головна функція програми
    """
    logger.info("Початок роботи програми розрахунку важливості ознак")

    # Запитуємо шлях до файлу даних
    data_file = 'cleaned_result.csv'

    try:
        # Завантаження даних
        logger.info(f"Завантаження даних з {data_file}...")
        data = pd.read_csv(data_file)
        logger.info(f"Завантажено {data.shape[0]} рядків та {data.shape[1]} колонок")

        # Перевірка наявності цільової змінної
        target_column = 'is_successful'
        if target_column not in data.columns:
            target_column = input(f"Колонку '{target_column}' не знайдено. Введіть назву цільової колонки: ")

        # Підготовка даних
        logger.info("Підготовка даних...")

        # Обробка цільової змінної
        y = data[target_column].copy()
        logger.info(f"Розподіл цільової змінної: {y.value_counts().to_dict()}")

        # Видалення цільової змінної та інших неінформативних колонок з набору ознак
        exclude_columns = [target_column, 'id', 'order_id', 'partner_id', 'date', 'timestamp']
        feature_columns = [col for col in data.columns if col not in exclude_columns]

        # Видалення колонок-ідентифікаторів та дат
        X = data[feature_columns].copy()
        logger.info(f"Використовуємо {X.shape[1]} ознак для аналізу")

        # Обробка категоріальних ознак
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_features:
            logger.info(f"Знайдено {len(categorical_features)} категоріальних ознак")

            # Створюємо та застосовуємо LabelEncoder для кожної категоріальної ознаки
            for feature in categorical_features:
                logger.info(f"Кодування категоріальної ознаки: {feature}")
                X[feature] = LabelEncoder().fit_transform(X[feature].astype(str))

        # Обробка пропущених значень
        if X.isnull().sum().sum() > 0:
            logger.info("Заповнення пропущених значень...")
            numeric_columns = X.select_dtypes(include=['number']).columns
            X[numeric_columns] = SimpleImputer(strategy='median').fit_transform(X[numeric_columns])

            # Якщо залишились пропущені значення в нечислових колонках
            if X.isnull().sum().sum() > 0:
                # Заповнюємо їх найчастішими значеннями
                X = SimpleImputer(strategy='most_frequent').fit_transform(X)
                X = pd.DataFrame(X, columns=feature_columns)

        # Оцінка важливості ознак
        results_df, models_dict = evaluate_features(X, y)

        # Зберігаємо результати оцінки важливості ознак
        results_path = f"{results_dir}/feature_importance_rankings.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Збережено результати рейтингу важливості ознак: {results_path}")

        # Виводимо загальний рейтинг
        print_importance_rankings(results_df)

        # Будуємо та зберігаємо графіки
        # Загальний рейтинг
        plot_feature_importance(
            results_df,
            title='Загальний рейтинг важливості ознак',
            save_path=f"{results_dir}/overall_importance.png"
        )

        # Будуємо графіки для кожного методу
        methods = [
            ('MI Score', 'Mutual Information'),
            ('F Score', 'ANOVA F-test'),
            ('Spearman Score', 'Spearman Correlation'),
            ('Absolute Coefficient', 'Logistic Regression'),
            ('DT Score', 'Decision Tree'),
            ('RF Score', 'Random Forest'),
            ('XGBoost Score', 'XGBoost')
        ]

        for score_col, method_name in methods:
            # Сортуємо за поточним показником
            method_df = results_df.sort_values(by=score_col, ascending=False).copy()
            plot_feature_importance(
                method_df,
                title=f'Важливість ознак за методом: {method_name}',
                save_path=f"{results_dir}/{score_col.replace(' ', '_').lower()}_importance.png"
            )

        logger.info(f"Завершено аналіз важливості ознак. Всі результати збережено в директорії: {results_dir}")

    except Exception as e:
        logger.error(f"Помилка: {str(e)}", exc_info=True)
        return

if __name__ == "__main__":
    main()
