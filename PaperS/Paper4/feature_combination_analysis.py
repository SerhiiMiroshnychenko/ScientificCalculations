"""
🔬 АНАЛІЗ КОМБІНАЦІЙ ОЗНАК З ОПТИМАЛЬНИМИ ПАРАМЕТРАМИ XGBOOST

Цей скрипт тестує модель XGBoost з найкращими знайденими параметрами
на всіх можливих комбінаціях ознак для визначення їх важливості.

Етапи роботи:
1. Завантаження даних та підготовка ознак
2. Використання оптимальних параметрів з hyperparameter research
3. Тестування всіх комбінацій ознак (1, 2, 3, ..., n)
4. Розрахунок метрик для кожної комбінації
5. Створення візуалізацій та рейтингів
6. Генерація висновків про важливість ознак

Автор: AI Assistant
Дата: 2024
"""

import pandas as pd
import numpy as np

# Встановлення non-interactive backend для matplotlib (вирішує проблеми з Tcl/Tk)
import matplotlib
matplotlib.use('Agg')  # Backend без GUI - вирішує проблеми з Tcl/Tk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import itertools
import warnings
import os
import time

import pickle

warnings.filterwarnings('ignore')

class FeatureCombinationAnalyzer:
    """
    Клас для аналізу комбінацій топ ознак з оптимальними параметрами XGBoost
    """

    def __init__(self, data_path, random_state=42, top_features_count=5):
        self.data_path = data_path
        self.random_state = random_state
        self.top_features_count = top_features_count  # Кількість топ ознак для аналізу

        # ОПТИМАЛЬНІ ПАРАМЕТРИ з hyperparameter research
        self.optimal_params = {
            'subsample': 1.0,
            'reg_lambda': 2.0,
            'reg_alpha': 0.5,
            'n_estimators': 800,
            'min_child_weight': 3,
            'max_depth': 7,
            'learning_rate': 0.1,
            'gamma': 0.2,
            'colsample_bytree': 0.7,
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',  # AUC-PR метрика в XGBoost
            'random_state': random_state,
            'n_jobs': -1,
            'verbosity': 0
        }

        # Змінні для збереження результатів
        self.data = None
        self.feature_names = None
        self.top_feature_names = None  # Топ ознаки за важливістю
        self.feature_importance_scores = None  # Оцінки важливості ознак
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.results = []
        self.best_combinations = {}

        # Створення папки для результатів
        self.results_dir = f"feature_analysis_results"
        os.makedirs(self.results_dir, exist_ok=True)

        print(f"🔬 === АНАЛІЗ КОМБІНАЦІЙ ТОП-{self.top_features_count} ОЗНАК З ОПТИМАЛЬНИМИ ПАРАМЕТРАМИ ===")
        print(f"📁 Результати будуть збережені в: {self.results_dir}/")
        print(f"🎯 Використовуємо найкращі параметри з hyperparameter research")
        print(f"⚡ Оптимізація: аналіз тільки топ-{self.top_features_count} найважливіших ознак")
        print(f"🖼️ Візуалізації будуть збережені як PNG файли (non-interactive режим)")

    def load_and_prepare_data(self):
        """
        ЕТАП 0: Завантаження та підготовка даних
        """
        print(f"\n🔄 === ЕТАП 0: ЗАВАНТАЖЕННЯ ТА ПІДГОТОВКА ДАНИХ ===")

        # Завантаження даних
        print(f"📂 Завантаження даних з: {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        print(f"📊 Розмір даних: {self.data.shape}")

        # Перевірка наявності цільової змінної
        if 'is_successful' not in self.data.columns:
            raise ValueError("❌ Цільова змінна 'is_successful' не знайдена!")

        # Підготовка ознак (тільки числові ознаки, як в оригінальному скрипті)
        exclude_columns = ['order_id', 'is_successful', 'create_date', 'partner_id']

        # Включаємо тільки числові ознаки, що використовуються в оригінальному скрипті
        numeric_features = [
            'order_amount', 'order_messages', 'order_changes',
            'partner_success_rate', 'partner_total_orders', 'partner_order_age_days',
            'partner_avg_amount', 'partner_success_avg_amount', 'partner_fail_avg_amount',
            'partner_total_messages', 'partner_success_avg_messages', 'partner_fail_avg_messages',
            'partner_avg_changes', 'partner_success_avg_changes', 'partner_fail_avg_changes'
        ]

        # Відбираємо тільки ті ознаки, що реально є в даних
        self.feature_names = [col for col in numeric_features if col in self.data.columns]

        print(f"🔍 Знайдено ознак: {len(self.feature_names)}")
        print(f"📋 Список ознак: {self.feature_names}")

        # Створення додаткових ознак (як в оригінальному скрипті)
        print(f"🔧 Створення додаткових ознак...")
        data_with_features = self._create_additional_features(self.data)

        # Оновлюємо список ознак включаючи додаткові
        additional_features = [col for col in data_with_features.columns
                             if col.endswith('_diff') or col.endswith('_ratio')]
        all_features = self.feature_names + additional_features

        # Підготовка X та y
        X = data_with_features[all_features]
        y = data_with_features['is_successful']

        # Оновлюємо список ознак
        self.feature_names = all_features

        print(f"🎯 Розподіл цільової змінної:")
        print(f"  • Успішні замовлення: {y.sum()} ({y.mean():.1%})")
        print(f"  • Неуспішні замовлення: {(~y.astype(bool)).sum()} ({(1-y.mean()):.1%})")

        # Розділення на train/test (70%/30%)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )

        print(f"📊 Розділення даних:")
        print(f"  • Тренувальна вибірка: {self.X_train.shape[0]} зразків")
        print(f"  • Тестова вибірка: {self.X_test.shape[0]} зразків")

        # Масштабування ознак
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print(f"✅ Дані підготовлені для аналізу")

    def select_top_features(self):
        """
        ЕТАП 0.5: Визначення топ ознак за важливістю XGBoost
        """
        print(f"\n🔄 === ЕТАП 0.5: ВИЗНАЧЕННЯ ТОП-{self.top_features_count} ОЗНАК ===")

        # Тренування моделі на всіх ознаках для отримання feature importance
        print(f"🤖 Тренування XGBoost на всіх ознаках для визначення важливості...")
        importance_model = XGBClassifier(**self.optimal_params)
        importance_model.fit(self.X_train_scaled, self.y_train)

        # Отримання важливості ознак
        feature_importance = importance_model.feature_importances_

        # Створення DataFrame для сортування
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        print(f"📊 Важливість усіх ознак:")
        for i, (_, row) in enumerate(importance_df.iterrows(), 1):
            print(f"  {i:2d}. {row['feature']:<35} {row['importance']:.6f}")

        # Вибір топ ознак
        self.top_feature_names = importance_df.head(self.top_features_count)['feature'].tolist()
        self.feature_importance_scores = importance_df.head(self.top_features_count)

        print(f"\n🏆 Вибрані топ-{self.top_features_count} ознак для аналізу:")
        for i, feature in enumerate(self.top_feature_names, 1):
            importance_score = importance_df[importance_df['feature'] == feature]['importance'].iloc[0]
            print(f"  {i}. {feature} (важливість: {importance_score:.6f})")

        # Отримання індексів топ ознак
        self.top_feature_indices = [self.feature_names.index(feature) for feature in self.top_feature_names]

        # Оновлення даних для роботи тільки з топ ознаками
        self.X_train_top = self.X_train_scaled[:, self.top_feature_indices]
        self.X_test_top = self.X_test_scaled[:, self.top_feature_indices]

        print(f"✅ Дані оновлені для роботи з топ-{self.top_features_count} ознаками")

        # Збереження важливості ознак
        importance_df.to_csv(f'{self.results_dir}/feature_importance_all.csv', index=False)
        self.feature_importance_scores.to_csv(f'{self.results_dir}/top_features_selected.csv', index=False)

        return importance_df

    def _create_additional_features(self, df):
        """Створення додаткових ознак (точно як в оригінальному скрипті)"""
        df = df.copy()

        # Різниця між середніми сумами успішних та неуспішних замовлень
        if ('partner_success_avg_amount' in df.columns and
            'partner_fail_avg_amount' in df.columns):
            df['amount_success_fail_diff'] = (df['partner_success_avg_amount'] -
                                            df['partner_fail_avg_amount'])

        # Різниця між середньою кількістю повідомлень успішних та неуспішних замовлень
        if ('partner_success_avg_messages' in df.columns and
            'partner_fail_avg_messages' in df.columns):
            df['messages_success_fail_diff'] = (df['partner_success_avg_messages'] -
                                              df['partner_fail_avg_messages'])

        # Різниця між середньою кількістю змін успішних та неуспішних замовлень
        if ('partner_success_avg_changes' in df.columns and
            'partner_fail_avg_changes' in df.columns):
            df['changes_success_fail_diff'] = (df['partner_success_avg_changes'] -
                                             df['partner_fail_avg_changes'])

        # Відношення суми замовлення до середньої суми замовлень партнера
        if ('order_amount' in df.columns and
            'partner_avg_amount' in df.columns):
            df['order_amount_to_avg_ratio'] = df['order_amount'] / df['partner_avg_amount']

        # Заміна нескінченних значень та NaN на 0
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        additional_features = [col for col in df.columns if col.endswith('_diff') or col.endswith('_ratio')]
        if additional_features:
            print(f"  ✅ Створені додаткові ознаки: {additional_features}")

        return df

    def test_feature_combination(self, feature_indices, combination_size):
        """
        Тестування конкретної комбінації топ ознак
        """
        # Вибір ознак з топ ознак (індекси відносно топ ознак)
        X_train_combo = self.X_train_top[:, feature_indices]
        X_test_combo = self.X_test_top[:, feature_indices]

        feature_combo_names = [self.top_feature_names[i] for i in feature_indices]

        # Створення моделі з оптимальними параметрами
        model = XGBClassifier(**self.optimal_params)

        # Cross-validation на тренувальній вибірці (5-fold)
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        cv_f1_scores = cross_val_score(model, X_train_combo, self.y_train, cv=cv_strategy, scoring='f1')
        cv_auc_roc_scores = cross_val_score(model, X_train_combo, self.y_train, cv=cv_strategy, scoring='roc_auc')
        cv_auc_pr_scores = cross_val_score(model, X_train_combo, self.y_train, cv=cv_strategy, scoring='average_precision')

        # Тренування для тестування
        model.fit(X_train_combo, self.y_train)

        # Прогнози на тестовій вибірці
        y_pred = model.predict(X_test_combo)
        y_pred_proba = model.predict_proba(X_test_combo)[:, 1]

        # Розрахунок метрик
        metrics = {
            'combination_size': combination_size,
            'feature_indices': feature_indices,
            'feature_names': feature_combo_names,
            'feature_names_str': ' + '.join(feature_combo_names),

            # CV метрики
            'cv_f1_mean': cv_f1_scores.mean(),
            'cv_f1_std': cv_f1_scores.std(),
            'cv_auc_roc_mean': cv_auc_roc_scores.mean(),
            'cv_auc_roc_std': cv_auc_roc_scores.std(),
            'cv_auc_pr_mean': cv_auc_pr_scores.mean(),
            'cv_auc_pr_std': cv_auc_pr_scores.std(),

            # Test метрики
            'test_f1': f1_score(self.y_test, y_pred),
            'test_accuracy': accuracy_score(self.y_test, y_pred),
            'test_precision': precision_score(self.y_test, y_pred),
            'test_recall': recall_score(self.y_test, y_pred),
            'test_auc_roc': roc_auc_score(self.y_test, y_pred_proba),
            'test_auc_pr': average_precision_score(self.y_test, y_pred_proba),

            # Різниця CV vs Test (ознака перенавчання)
            'overfitting_gap_f1': cv_f1_scores.mean() - f1_score(self.y_test, y_pred),
            'overfitting_gap_auc_pr': cv_auc_pr_scores.mean() - average_precision_score(self.y_test, y_pred_proba)
        }

        return metrics

    def analyze_all_combinations(self):
        """
        ЕТАП 1: Аналіз всіх можливих комбінацій топ ознак
        """
        print(f"\n🔄 === ЕТАП 1: АНАЛІЗ ВСІХ КОМБІНАЦІЙ ТОП-{self.top_features_count} ОЗНАК ===")

        n_features = self.top_features_count  # Працюємо тільки з топ ознаками
        total_combinations = 2**n_features - 1  # Всі комбінації крім пустої

        print(f"🔢 Загальна кількість комбінацій: {total_combinations}")
        print(f"⚡ Топ ознаки: {', '.join(self.top_feature_names)}")

        max_combination_size = n_features  # Аналізуємо всі можливі комбінації топ ознак

        start_time = time.time()
        processed = 0

        # Тестування комбінацій від 1 до max_combination_size ознак
        for combination_size in range(1, max_combination_size + 1):
            print(f"\n📊 Тестування комбінацій з {combination_size} ознак...")

            combinations = list(itertools.combinations(range(n_features), combination_size))
            n_combinations = len(combinations)

            print(f"🔢 Кількість комбінацій: {n_combinations}")

            for i, feature_indices in enumerate(combinations):
                if i % max(1, n_combinations // 10) == 0:
                    progress = (i / n_combinations) * 100
                    print(f"  📈 Прогрес: {progress:.1f}% ({i}/{n_combinations})")

                try:
                    metrics = self.test_feature_combination(feature_indices, combination_size)
                    self.results.append(metrics)
                    processed += 1

                except Exception as e:
                    print(f"  ⚠️ Помилка для комбінації {feature_indices}: {e}")
                    continue

        total_time = time.time() - start_time
        print(f"\n✅ Аналіз завершено!")
        print(f"⏱️ Загальний час: {total_time:.2f} секунд")
        print(f"📊 Оброблено комбінацій: {processed}")

        # Створення DataFrame з результатами
        self.results_df = pd.DataFrame(self.results)

        # Знаходження найкращих комбінацій для кожного розміру (за AUC-PR)
        for size in range(1, max_combination_size + 1):
            size_results = self.results_df[self.results_df['combination_size'] == size]
            if not size_results.empty:
                best_idx = size_results['test_auc_pr'].idxmax()  # Оптимізуємо за AUC-PR
                self.best_combinations[size] = self.results_df.loc[best_idx]

        print(f"🏆 Найкращі комбінації знайдено для кожного розміру")

    def create_visualizations(self):
        """
        ЕТАП 2: Створення візуалізацій
        """
        print(f"\n🔄 === ЕТАП 2: СТВОРЕННЯ ВІЗУАЛІЗАЦІЙ ===")
        print(f"🖼️ Використовуємо matplotlib backend 'Agg' (non-interactive)")

        try:
            # 1. Розподіл метрик по розмірах комбінацій
            print(f"  📊 Створення графіків метрик по розмірах...")
            self._plot_metrics_by_size()

            # 2. Топ-10 найкращих комбінацій
            print(f"  🏆 Створення графіків топ комбінацій...")
            self._plot_top_combinations()

            # 3. Аналіз важливості індивідуальних ознак
            print(f"  🎯 Створення аналізу індивідуальних ознак...")
            self._plot_individual_features()

            # 4. Heatmap кореляції між метриками
            print(f"  🔥 Створення heatmap кореляції...")
            self._plot_metrics_correlation()

            # 5. Аналіз перенавчання
            print(f"  📈 Створення аналізу перенавчання...")
            self._plot_overfitting_analysis()

            print(f"✅ Всі візуалізації створено та збережено!")

        except Exception as e:
            print(f"❌ Помилка при створенні візуалізацій: {e}")
            print(f"⚠️ Продовжуємо без візуалізацій...")
            return False

        return True

    def _plot_metrics_by_size(self):
        """Графік метрик по розмірах комбінацій"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Аналіз метрик по розмірах комбінацій ознак', fontsize=16, fontweight='bold')

        metrics = ['test_f1', 'test_auc_pr', 'test_auc_roc', 'overfitting_gap_auc_pr']
        titles = ['F1 Score', 'AUC-PR (Primary)', 'AUC-ROC', 'Overfitting Gap AUC-PR (CV - Test)']

        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i//2, i%2]

            # Box plot для кожного розміру
            sizes = sorted(self.results_df['combination_size'].unique())
            data_by_size = [self.results_df[self.results_df['combination_size'] == size][metric]
                           for size in sizes]

            bp = ax.boxplot(data_by_size, labels=sizes, patch_artist=True)

            # Кольорове кодування
            colors = plt.cm.viridis(np.linspace(0, 1, len(sizes)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Кількість ознак в комбінації')
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/metrics_by_combination_size.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_top_combinations(self):
        """Топ-10 найкращих комбінацій"""
        # Сортування по test_auc_pr (основна метрика)
        top_combinations = self.results_df.nlargest(10, 'test_auc_pr')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Графік 1: AUC-PR Score топ-10
        bars1 = ax1.barh(range(len(top_combinations)), top_combinations['test_auc_pr'],
                        color=plt.cm.viridis(np.linspace(0, 1, len(top_combinations))))

        ax1.set_yticks(range(len(top_combinations)))
        ax1.set_yticklabels([f"{row['combination_size']} ознак:\n{row['feature_names_str'][:50]}..."
                            if len(row['feature_names_str']) > 50
                            else f"{row['combination_size']} ознак:\n{row['feature_names_str']}"
                            for _, row in top_combinations.iterrows()], fontsize=10)
        ax1.set_xlabel('Test AUC-PR Score')
        ax1.set_title('Топ-10 найкращих комбінацій ознак (AUC-PR)', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Додавання значень на bars
        for i, bar in enumerate(bars1):
            width = bar.get_width()
            ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{width:.4f}', ha='left', va='center', fontweight='bold')

        # Графік 2: Порівняння CV vs Test AUC-PR для топ-10
        x_pos = np.arange(len(top_combinations))
        width = 0.35

        bars1 = ax2.bar(x_pos - width/2, top_combinations['cv_auc_pr_mean'], width,
                       label='CV AUC-PR Score', alpha=0.8, color='lightblue')
        bars2 = ax2.bar(x_pos + width/2, top_combinations['test_auc_pr'], width,
                       label='Test AUC-PR Score', alpha=0.8, color='lightcoral')

        ax2.set_xlabel('Комбінація ознак')
        ax2.set_ylabel('AUC-PR Score')
        ax2.set_title('CV vs Test AUC-PR Score для топ-10 комбінацій', fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f"{i+1}" for i in range(len(top_combinations))])
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/top_combinations.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_individual_features(self):
        """Аналіз важливості індивідуальних ознак"""
        # Результати для одиночних ознак
        single_features = self.results_df[self.results_df['combination_size'] == 1].copy()
        single_features = single_features.sort_values('test_auc_pr', ascending=True)  # Сортуємо за AUC-PR

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Графік 1: Рейтинг індивідуальних ознак за AUC-PR
        bars = ax1.barh(range(len(single_features)), single_features['test_auc_pr'],
                       color=plt.cm.RdYlGn(np.linspace(0.3, 1, len(single_features))))

        ax1.set_yticks(range(len(single_features)))
        ax1.set_yticklabels(single_features['feature_names_str'], fontsize=12)
        ax1.set_xlabel('Test AUC-PR Score')
        ax1.set_title('Рейтинг індивідуальних ознак (AUC-PR)', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Додавання значень
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{width:.4f}', ha='left', va='center', fontweight='bold')

        # Графік 2: Частота появи ознак в топ-20 комбінаціях
        top_20 = self.results_df.nlargest(20, 'test_auc_pr')  # За AUC-PR
        feature_counts = {}

        for _, row in top_20.iterrows():
            for feature in row['feature_names']:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1

        features_sorted = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        features, counts = zip(*features_sorted)

        bars2 = ax2.bar(range(len(features)), counts,
                       color=plt.cm.plasma(np.linspace(0, 1, len(features))))

        ax2.set_xticks(range(len(features)))
        ax2.set_xticklabels(features, rotation=45, ha='right')
        ax2.set_ylabel('Частота появи в топ-20')
        ax2.set_title('Частота появи ознак в найкращих комбінаціях', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3)

        # Додавання значень
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/individual_features_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_metrics_correlation(self):
        """Heatmap кореляції між метриками"""
        metrics_cols = ['test_f1', 'test_auc_pr', 'test_auc_roc', 'test_accuracy',
                       'test_precision', 'test_recall', 'cv_auc_pr_mean', 'overfitting_gap_auc_pr']

        correlation_matrix = self.results_df[metrics_cols].corr()

        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.3f')

        plt.title('Кореляція між метриками якості', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/metrics_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_overfitting_analysis(self):
        """Аналіз перенавчання"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Графік 1: Scatter plot CV vs Test AUC-PR
        scatter = ax1.scatter(self.results_df['cv_auc_pr_mean'], self.results_df['test_auc_pr'],
                            c=self.results_df['combination_size'], cmap='viridis', alpha=0.6)

        # Лінія ідеального співпадіння
        min_val = min(self.results_df['cv_auc_pr_mean'].min(), self.results_df['test_auc_pr'].min())
        max_val = max(self.results_df['cv_auc_pr_mean'].max(), self.results_df['test_auc_pr'].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)

        ax1.set_xlabel('CV AUC-PR Score')
        ax1.set_ylabel('Test AUC-PR Score')
        ax1.set_title('CV vs Test AUC-PR Score', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Кількість ознак')

        # Графік 2: Розподіл overfitting gap для AUC-PR
        ax2.hist(self.results_df['overfitting_gap_auc_pr'], bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Немає перенавчання')
        ax2.axvline(self.results_df['overfitting_gap_auc_pr'].mean(), color='blue', linestyle='-',
                   linewidth=2, label=f'Середнє: {self.results_df["overfitting_gap_auc_pr"].mean():.4f}')

        ax2.set_xlabel('Overfitting Gap AUC-PR (CV - Test)')
        ax2.set_ylabel('Частота')
        ax2.set_title('Розподіл перенавчання (AUC-PR)', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/overfitting_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_report(self):
        """
        ЕТАП 3: Генерація детального звіту
        """
        print(f"\n🔄 === ЕТАП 3: ГЕНЕРАЦІЯ ЗВІТУ ===")

        # Збереження результатів у CSV
        self.results_df.to_csv(f'{self.results_dir}/all_combinations_results.csv', index=False)

        # Збереження топ-комбінацій (за AUC-PR)
        top_10 = self.results_df.nlargest(10, 'test_auc_pr')
        top_10.to_csv(f'{self.results_dir}/top_10_combinations.csv', index=False)

        # Створення markdown звіту
        report_path = f'{self.results_dir}/analysis_report.md'

        with open(report_path, 'w', encoding='utf-8') as f:
            # Заголовок звіту
            f.write("# 🔬 ЗВІТ АНАЛІЗУ КОМБІНАЦІЙ ОЗНАК\n\n")

            # Загальна інформація
            f.write("## 📊 Загальна інформація\n")
            f.write(f"- Загальна кількість ознак у датасеті: {len(self.feature_names)}\n")
            f.write(f"- Топ ознак для аналізу: {self.top_features_count}\n")
            f.write(f"- Загальна кількість протестованих комбінацій: {len(self.results_df)}\n")
            f.write(f"- Використані оптимальні параметри XGBoost\n")
            f.write(f"- Основна метрика оптимізації: AUC-PR (Area Under Precision-Recall Curve)\n\n")

            # Топ ознаки
            f.write("## 🏆 Вибрані топ ознаки (за важливістю XGBoost)\n")
            for i, (_, row) in enumerate(self.feature_importance_scores.iterrows(), 1):
                f.write(f"{i}. **{row['feature']}** - важливість: {row['importance']:.6f}\n")
            f.write("\n")

            # Оптимальні параметри
            f.write("## 🎯 Оптимальні параметри XGBoost\n")
            for param, value in self.optimal_params.items():
                f.write(f"- {param}: {value}\n")
            f.write("\n")

            # Топ-5 комбінацій
            f.write("## 🏆 ТОП-5 НАЙКРАЩИХ КОМБІНАЦІЙ (за AUC-PR)\n\n")
            top_5 = self.results_df.nlargest(5, 'test_auc_pr')
            for i, (_, row) in enumerate(top_5.iterrows(), 1):
                f.write(f"### {i}. Комбінація з {row['combination_size']} ознак\n")
                f.write(f"- **Ознаки**: {row['feature_names_str']}\n")
                f.write(f"- **Test AUC-PR**: {row['test_auc_pr']:.6f} (Primary)\n")
                f.write(f"- **Test F1 Score**: {row['test_f1']:.6f}\n")
                f.write(f"- **Test AUC-ROC**: {row['test_auc_roc']:.6f}\n")
                f.write(f"- **Test Accuracy**: {row['test_accuracy']:.6f}\n")
                f.write(f"- **CV AUC-PR**: {row['cv_auc_pr_mean']:.6f} ± {row['cv_auc_pr_std']:.6f}\n")
                f.write(f"- **Overfitting Gap (AUC-PR)**: {row['overfitting_gap_auc_pr']:.6f}\n\n")

            # Найкращі індивідуальні ознаки
            single_features = self.results_df[self.results_df['combination_size'] == 1].nlargest(5, 'test_auc_pr')
            f.write("## 🥇 ТОП-5 ІНДИВІДУАЛЬНИХ ОЗНАК (за AUC-PR)\n\n")

            for i, (_, row) in enumerate(single_features.iterrows(), 1):
                f.write(f"### {i}. {row['feature_names_str']}\n")
                f.write(f"- **Test AUC-PR**: {row['test_auc_pr']:.6f} (Primary)\n")
                f.write(f"- **Test F1 Score**: {row['test_f1']:.6f}\n")
                f.write(f"- **Test AUC-ROC**: {row['test_auc_roc']:.6f}\n")
                f.write(f"- **Test Accuracy**: {row['test_accuracy']:.6f}\n\n")

            # Статистика по розмірах комбінацій
            f.write("## 📈 СТАТИСТИКА ПО РОЗМІРАХ КОМБІНАЦІЙ\n\n")
            f.write("| Розмір | Кількість | Середній AUC-PR | Найкращий AUC-PR | Стандартне відхилення |\n")
            f.write("|--------|-----------|-----------------|------------------|-----------------------|\n")

            for size in sorted(self.results_df['combination_size'].unique()):
                size_data = self.results_df[self.results_df['combination_size'] == size]
                f.write(f"| {size} | {len(size_data)} | {size_data['test_auc_pr'].mean():.6f} | {size_data['test_auc_pr'].max():.6f} | {size_data['test_auc_pr'].std():.6f} |\n")

            # Висновки
            best_overall = self.results_df.loc[self.results_df['test_auc_pr'].idxmax()]
            best_single = single_features.iloc[0]

            f.write("\n## 🎯 КЛЮЧОВІ ВИСНОВКИ\n\n")

            f.write("### 1. Найкраща загальна комбінація\n")
            f.write(f"- **{best_overall['combination_size']} ознак**: {best_overall['feature_names_str']}\n")
            f.write(f"- **AUC-PR**: {best_overall['test_auc_pr']:.6f} (Primary)\n")
            f.write(f"- **F1 Score**: {best_overall['test_f1']:.6f}\n")
            improvement = ((best_overall['test_auc_pr'] / best_single['test_auc_pr'] - 1) * 100)
            f.write(f"- Покращення відносно найкращої індивідуальної ознаки: {improvement:.2f}%\n\n")

            f.write("### 2. Найкраща індивідуальна ознака\n")
            f.write(f"- **{best_single['feature_names_str']}**\n")
            f.write(f"- **AUC-PR**: {best_single['test_auc_pr']:.6f} (Primary)\n")
            f.write(f"- **F1 Score**: {best_single['test_f1']:.6f}\n\n")

            f.write("### 3. Аналіз перенавчання\n")
            f.write(f"- Середній overfitting gap (AUC-PR): {self.results_df['overfitting_gap_auc_pr'].mean():.6f}\n")
            no_overfitting = (self.results_df['overfitting_gap_auc_pr'] <= 0).sum()
            f.write(f"- Комбінацій без перенавчання: {no_overfitting} з {len(self.results_df)}\n\n")

            f.write("### 4. Рекомендації\n")
            if best_overall['combination_size'] == 1:
                f.write("- Найкраща модель використовує лише одну ознаку - це свідчить про високу якість даних\n")
            else:
                f.write(f"- Оптимальна кількість ознак: {best_overall['combination_size']}\n")
                f.write("- Комбінування ознак дає значне покращення результатів\n")

            if self.results_df['overfitting_gap_auc_pr'].mean() > 0.02:
                f.write("- Модель схильна до перенавчання - варто збільшити регуляризацію\n")
            else:
                f.write("- Модель добре генералізує - параметри підібрані оптимально\n")

            f.write("\n## 📁 Створені файли\n")
            f.write("- `all_combinations_results.csv` - Повні результати всіх комбінацій\n")
            f.write("- `top_10_combinations.csv` - Топ-10 найкращих комбінацій (за AUC-PR)\n")
            f.write("- `feature_importance_all.csv` - Важливість всіх ознак\n")
            f.write("- `top_features_selected.csv` - Вибрані топ ознаки\n")
            f.write("- Візуалізації у форматі PNG\n")

        print(f"📄 Звіт збережено: {report_path}")
        print(f"💾 CSV файли збережено в папці: {self.results_dir}/")

    def run_analysis(self):
        """
        Запуск повного аналізу
        """
        print(f"🚀 === ПОЧАТОК АНАЛІЗУ КОМБІНАЦІЙ ОЗНАК ===")
        start_time = time.time()

        try:
            # Виконання всіх етапів
            self.load_and_prepare_data()
            self.select_top_features()  # Новий етап: вибір топ ознак
            self.analyze_all_combinations()

            # Створення візуалізацій (з обробкою помилок)
            visualization_success = self.create_visualizations()
            if not visualization_success:
                print(f"⚠️ Візуалізації не створені через технічні проблеми")
                print(f"📊 Але всі дані та звіти будуть збережені!")

            self.generate_report()

            total_time = time.time() - start_time

            print(f"\n✅ === АНАЛІЗ ЗАВЕРШЕНО УСПІШНО ===")
            print(f"⏱️ Загальний час виконання: {total_time:.2f} секунд")
            print(f"📊 Протестовано комбінацій: {len(self.results_df)}")
            print(f"🏆 Найкращий AUC-PR Score: {self.results_df['test_auc_pr'].max():.6f}")
            print(f"📁 Результати збережено в: {self.results_dir}/")

            # Виведення топ-3 результатів
            print(f"\n🥇 ТОП-3 НАЙКРАЩИХ КОМБІНАЦІЙ (за AUC-PR):")
            top_3 = self.results_df.nlargest(3, 'test_auc_pr')
            for i, (_, row) in enumerate(top_3.iterrows(), 1):
                print(f"{i}. {row['feature_names_str']} (AUC-PR: {row['test_auc_pr']:.6f})")

        except Exception as e:
            print(f"❌ ПОМИЛКА: {e}")
            raise


# Приклад використання
if __name__ == "__main__":
    # Шлях до файлу з даними
    data_path = "b2b.csv"

    # Кількість топ ознак для аналізу (можна змінювати)
    TOP_FEATURES_COUNT = 10
    
    # Створення аналізатора
    analyzer = FeatureCombinationAnalyzer(
        data_path=data_path, 
        random_state=42,
        top_features_count=TOP_FEATURES_COUNT
    )
    
    # Запуск аналізу
    analyzer.run_analysis() 