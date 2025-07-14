# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
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

# === ДОДАНО: Функції для обробки циклічних ознак (як у analyse.py) ===
def identify_cyclical_features(feature_names):
    cyclical_pairs = {}
    sin_features = [f for f in feature_names if f.endswith('_sin')]
    for sin_feature in sin_features:
        base_name = sin_feature[:-4]
        cos_feature = f"{base_name}_cos"
        if cos_feature in feature_names:
            cyclical_pairs[base_name] = [sin_feature, cos_feature]
    return cyclical_pairs

def preprocess_features_for_analysis(X):
    feature_names = X.columns
    cyclical_pairs = identify_cyclical_features(feature_names)
    X_processed = pd.DataFrame()
    for feature in feature_names:
        is_part_of_pair = False
        for pair in cyclical_pairs.values():
            if feature in pair:
                is_part_of_pair = True
                break
        if not is_part_of_pair:
            X_processed[feature] = X[feature]
    for base_name, pair in cyclical_pairs.items():
        sin_feature, cos_feature = pair
        X_processed[base_name] = np.sqrt(X[sin_feature]**2 + X[cos_feature]**2)
    return X_processed

class FeatureCombinationAnalyzer:
    def __init__(self, data_path, random_state=42, top_features_count=5):
        self.data_path = data_path
        self.random_state = random_state
        self.top_features_count = top_features_count
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
            'eval_metric': 'aucpr',
            'random_state': random_state,
            'n_jobs': -1,
            'verbosity': 0,
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
        }
        self.data = None
        self.feature_names = None
        self.top_feature_names = None
        self.feature_importance_scores = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.results = []
        self.best_combinations = {}
        self.extended_results = []  # Для розширених комбінацій
        self.results_dir = f"feature_analysis_results_v2"
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"🔬 === АНАЛІЗ КОМБІНАЦІЙ ТОП-{self.top_features_count} ОЗНАК З ОПТИМАЛЬНИМИ ПАРАМЕТРАМИ ===")
        print(f"📁 Результати будуть збережені в: {self.results_dir}/")
        print(f"🎯 Використовуємо найкращі параметри з hyperparameter research")
        print(f"⚡ Оптимізація: аналіз тільки топ-{self.top_features_count} найважливіших ознак")
        print(f"🖼️ Візуалізації будуть збережені як PNG файли (non-interactive режим)")

    def load_and_prepare_data(self):
        print(f"\n🔄 === ЕТАП 0: ЗАВАНТАЖЕННЯ ТА ПІДГОТОВКА ДАНИХ ===")
        print(f"📂 Завантаження даних з: {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        print(f"📊 Розмір даних: {self.data.shape}")
        if 'is_successful' not in self.data.columns:
            raise ValueError("❌ Цільова змінна 'is_successful' не знайдена!")
        # Відокремлюємо цільову змінну
        X = self.data.drop('is_successful', axis=1)
        y = self.data['is_successful']
        # Використовуємо лише числові ознаки
        X = X.select_dtypes(exclude=['object'])
        # === ДОДАНО: Об'єднання циклічних ознак ===
        X_processed = preprocess_features_for_analysis(X)
        self.feature_names = list(X_processed.columns)
        print(f"🔍 Залишилось числових ознак після обробки циклічних: {len(self.feature_names)}")
        # Ділимо на train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_processed, y, test_size=0.3, random_state=self.random_state, stratify=y
        )
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print(f"✅ Дані підготовлені для аналізу")

    def select_top_features(self):
        print(f"\n🔄 === ЕТАП 0.5: ВИЗНАЧЕННЯ ТОП-{self.top_features_count} ОЗНАК ===")
        importance_model = XGBClassifier(**self.optimal_params)
        importance_model.fit(self.X_train_scaled, self.y_train)
        feature_importance = importance_model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        print(f"📊 Важливість усіх ознак:")
        for i, (_, row) in enumerate(importance_df.iterrows(), 1):
            print(f"  {i:2d}. {row['feature']:<35} {row['importance']:.6f}")
        self.top_feature_names = importance_df.head(self.top_features_count)['feature'].tolist()
        self.feature_importance_scores = importance_df.head(self.top_features_count)
        self.top_feature_indices = [self.feature_names.index(feature) for feature in self.top_feature_names]
        self.X_train_top = self.X_train_scaled[:, self.top_feature_indices]
        self.X_test_top = self.X_test_scaled[:, self.top_feature_indices]
        print(f"✅ Дані оновлені для роботи з топ-{self.top_features_count} ознаками")

    def test_feature_combination(self, feature_indices, combination_size):
        X_train_combo = self.X_train_top[:, feature_indices]
        X_test_combo = self.X_test_top[:, feature_indices]
        feature_combo_names = [self.top_feature_names[i] for i in feature_indices]
        model = XGBClassifier(**self.optimal_params)
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_f1_scores = cross_val_score(model, X_train_combo, self.y_train, cv=cv_strategy, scoring='f1')
        cv_auc_roc_scores = cross_val_score(model, X_train_combo, self.y_train, cv=cv_strategy, scoring='roc_auc')
        cv_auc_pr_scores = cross_val_score(model, X_train_combo, self.y_train, cv=cv_strategy, scoring='average_precision')
        model.fit(X_train_combo, self.y_train)
        y_pred = model.predict(X_test_combo)
        y_pred_proba = model.predict_proba(X_test_combo)[:, 1]
        metrics = {
            'combination_size': combination_size,
            'feature_indices': feature_indices,
            'feature_names': feature_combo_names,
            'feature_names_str': ' + '.join(feature_combo_names),
            'cv_f1_mean': cv_f1_scores.mean(),
            'cv_f1_std': cv_f1_scores.std(),
            'cv_auc_roc_mean': cv_auc_roc_scores.mean(),
            'cv_auc_roc_std': cv_auc_roc_scores.std(),
            'cv_auc_pr_mean': cv_auc_pr_scores.mean(),
            'cv_auc_pr_std': cv_auc_pr_scores.std(),
            'test_f1': f1_score(self.y_test, y_pred),
            'test_accuracy': accuracy_score(self.y_test, y_pred),
            'test_precision': precision_score(self.y_test, y_pred),
            'test_recall': recall_score(self.y_test, y_pred),
            'test_auc_roc': roc_auc_score(self.y_test, y_pred_proba),
            'test_auc_pr': average_precision_score(self.y_test, y_pred_proba),
            'overfitting_gap_f1': cv_f1_scores.mean() - f1_score(self.y_test, y_pred),
            'overfitting_gap_auc_pr': cv_auc_pr_scores.mean() - average_precision_score(self.y_test, y_pred_proba)
        }
        return metrics

    def analyze_all_combinations(self):
        print(f"\n🔄 === ЕТАП 1: АНАЛІЗ ВСІХ КОМБІНАЦІЙ ТОП-{self.top_features_count} ОЗНАК ===")
        n_features = self.top_features_count
        total_combinations = 2**n_features - 1
        print(f"🔢 Загальна кількість комбінацій: {total_combinations}")
        print(f"⚡ Топ ознаки: {', '.join(self.top_feature_names)}")
        max_combination_size = n_features
        start_time = time.time()
        processed = 0
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
        self.results_df = pd.DataFrame(self.results)
        for size in range(1, max_combination_size + 1):
            size_results = self.results_df[self.results_df['combination_size'] == size]
            if not size_results.empty:
                best_idx = size_results['test_auc_pr'].idxmax()
                self.best_combinations[size] = self.results_df.loc[best_idx]

        # === ДОДАНО: Повний перебір для залишених ознак ===
        print(f"\n🔄 === ЕТАП 1.5: ПОВНИЙ ПЕРЕБІР ДЛЯ ЗАЛИШЕНИХ ОЗНАК ===")
        all_features = list(self.feature_names)
        top_set = set(self.top_feature_names)
        rest_features = [f for f in all_features if f not in top_set]
        print(f"Залишені ознаки: {rest_features}")
        self.rest_results = []
        if rest_features:
            rest_indices = [self.feature_names.index(f) for f in rest_features]
            n_rest = len(rest_features)
            for combination_size in range(1, n_rest + 1):
                print(f"\n📊 Тестування комбінацій з {combination_size} залишених ознак...")
                combinations = list(itertools.combinations(range(n_rest), combination_size))
                n_combinations = len(combinations)
                print(f"🔢 Кількість комбінацій: {n_combinations}")
                for i, feature_indices in enumerate(combinations):
                    if i % max(1, n_combinations // 10) == 0:
                        progress = (i / n_combinations) * 100
                        print(f"  📈 Прогрес: {progress:.1f}% ({i}/{n_combinations})")
                    try:
                        indices = [rest_indices[idx] for idx in feature_indices]
                        feature_names = [rest_features[idx] for idx in feature_indices]
                        X_train_rest = self.X_train_scaled[:, indices]
                        X_test_rest = self.X_test_scaled[:, indices]
                        model = XGBClassifier(**self.optimal_params)
                        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
                        cv_f1_scores = cross_val_score(model, X_train_rest, self.y_train, cv=cv_strategy, scoring='f1')
                        cv_auc_roc_scores = cross_val_score(model, X_train_rest, self.y_train, cv=cv_strategy, scoring='roc_auc')
                        cv_auc_pr_scores = cross_val_score(model, X_train_rest, self.y_train, cv=cv_strategy, scoring='average_precision')
                        model.fit(X_train_rest, self.y_train)
                        y_pred = model.predict(X_test_rest)
                        y_pred_proba = model.predict_proba(X_test_rest)[:, 1]
                        metrics = {
                            'combination_size': len(feature_names),
                            'feature_indices': indices,
                            'feature_names': feature_names,
                            'feature_names_str': ' + '.join(feature_names),
                            'cv_f1_mean': cv_f1_scores.mean(),
                            'cv_f1_std': cv_f1_scores.std(),
                            'cv_auc_roc_mean': cv_auc_roc_scores.mean(),
                            'cv_auc_roc_std': cv_auc_roc_scores.std(),
                            'cv_auc_pr_mean': cv_auc_pr_scores.mean(),
                            'cv_auc_pr_std': cv_auc_pr_scores.std(),
                            'test_f1': f1_score(self.y_test, y_pred),
                            'test_accuracy': accuracy_score(self.y_test, y_pred),
                            'test_precision': precision_score(self.y_test, y_pred),
                            'test_recall': recall_score(self.y_test, y_pred),
                            'test_auc_roc': roc_auc_score(self.y_test, y_pred_proba),
                            'test_auc_pr': average_precision_score(self.y_test, y_pred_proba),
                            'overfitting_gap_f1': cv_f1_scores.mean() - f1_score(self.y_test, y_pred),
                            'overfitting_gap_auc_pr': cv_auc_pr_scores.mean() - average_precision_score(self.y_test, y_pred_proba)
                        }
                        self.rest_results.append(metrics)
                    except Exception as e:
                        print(f"  ⚠️ Помилка для комбінації {feature_names}: {e}")
                        continue
            self.rest_results_df = pd.DataFrame(self.rest_results)
            print(f"✅ Повний перебір для залишених ознак завершено!")
        else:
            self.rest_results_df = pd.DataFrame()

    def _plot_aucpr_vs_num_features(self):
        """Scatter plot: Кількість ознак vs AUC-PR"""
        plt.figure(figsize=(12, 7))
        x = self.results_df['combination_size']
        y = self.results_df['test_auc_pr']
        plt.scatter(x, y, alpha=0.5, c=x, cmap='viridis', edgecolor='k')
        plt.xlabel('Кількість ознак у комбінації', fontsize=14)
        plt.ylabel('AUC-PR (test_auc_pr)', fontsize=14)
        plt.title('Залежність AUC-PR від кількості ознак у комбінації', fontsize=16, fontweight='bold')
        # Додаємо padding по осі Y
        y_min, y_max = y.min(), y.max()
        y_range = y_max - y_min
        pad = y_range * 0.05 if y_range > 0 else 0.01
        plt.xlim(0.5, self.top_features_count + 0.5)
        plt.ylim(y_min - pad, y_max + pad)
        plt.xticks(range(1, self.top_features_count + 1))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/aucpr_vs_num_features.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_aucpr_vs_num_features_extended(self):
        """Scatter plot: Кількість ознак vs AUC-PR (розширений)"""
        plt.figure(figsize=(12, 7))
        # Основні комбінації
        x1 = self.results_df['combination_size']
        y1 = self.results_df['test_auc_pr']
        # Розширені комбінації
        x2 = self.extended_results_df['combination_size'] if hasattr(self, 'extended_results_df') else []
        y2 = self.extended_results_df['test_auc_pr'] if hasattr(self, 'extended_results_df') else []
        plt.scatter(x1, y1, alpha=0.5, c=x1, cmap='viridis', edgecolor='k', label='Комбінації топових ознак')
        if len(x2) > 0:
            plt.scatter(x2, y2, alpha=0.9, c='red', marker='*', s=120, label='Розширені набори (top + 1)')
        plt.xlabel('Кількість ознак у комбінації', fontsize=14)
        plt.ylabel('AUC-PR (test_auc_pr)', fontsize=14)
        plt.title('AUC-PR для комбінацій топових та розширених наборів ознак', fontsize=16, fontweight='bold')
        # Додаємо padding по осі Y
        all_y = pd.concat([y1, y2]) if len(y2) > 0 else y1
        y_min, y_max = all_y.min(), all_y.max()
        y_range = y_max - y_min
        pad = y_range * 0.05 if y_range > 0 else 0.01
        plt.xlim(0.5, self.top_features_count + len(x2) + 0.5)
        plt.ylim(y_min - pad, y_max + pad)
        plt.xticks(range(1, self.top_features_count + len(x2) + 1))
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/aucpr_vs_num_features_extended.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_visualizations(self):
        print(f"\n🔄 === ЕТАП 2: СТВОРЕННЯ ВІЗУАЛІЗАЦІЙ ===")
        print(f"🖼️ Використовуємо matplotlib backend 'Agg' (non-interactive)")
        try:
            print(f"  📊 Створення графіків метрик по розмірах...")
            self._plot_metrics_by_size()
            print(f"  🏆 Створення графіків топ комбінацій...")
            self._plot_top_combinations()
            print(f"  🎯 Створення аналізу індивідуальних ознак...")
            self._plot_individual_features()
            print(f"  🔥 Створення heatmap кореляції...")
            self._plot_metrics_correlation()
            print(f"  📈 Створення аналізу перенавчання...")
            self._plot_overfitting_analysis()
            print(f"  🟢 Створення scatter-графіка AUC-PR vs Кількість ознак...")
            self._plot_aucpr_vs_num_features()
            # Візуалізації для залишених ознак
            if not self.rest_results_df.empty:
                print(f"  🟠 Створення графіків для залишених ознак...")
                self._plot_metrics_by_size_rest()
                self._plot_top_combinations_rest()
                self._plot_individual_features_rest()
                self._plot_metrics_correlation_rest()
                self._plot_overfitting_analysis_rest()
                self._plot_aucpr_vs_num_features_rest()
            print(f"✅ Всі візуалізації створено та збережено!")
        except Exception as e:
            print(f"❌ Помилка при створенні візуалізацій: {e}")
            print(f"⚠️ Продовжуємо без візуалізацій...")
            return False
        return True

    def _plot_metrics_by_size(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Аналіз метрик по розмірах комбінацій ознак', fontsize=16, fontweight='bold')
        metrics = ['test_f1', 'test_auc_pr', 'test_auc_roc', 'overfitting_gap_auc_pr']
        titles = ['F1 Score', 'AUC-PR (Primary)', 'AUC-ROC', 'Overfitting Gap AUC-PR (CV - Test)']
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i//2, i%2]
            sizes = sorted(self.results_df['combination_size'].unique())
            data_by_size = [self.results_df[self.results_df['combination_size'] == size][metric]
                           for size in sizes]
            bp = ax.boxplot(data_by_size, labels=sizes, patch_artist=True)
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
        top_combinations = self.results_df.nlargest(10, 'test_auc_pr')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
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
        for i, bar in enumerate(bars1):
            width = bar.get_width()
            ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{width:.4f}', ha='left', va='center', fontweight='bold')
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
        single_features = self.results_df[self.results_df['combination_size'] == 1].copy()
        single_features = single_features.sort_values('test_auc_pr', ascending=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        bars = ax1.barh(range(len(single_features)), single_features['test_auc_pr'],
                       color=plt.cm.RdYlGn(np.linspace(0.3, 1, len(single_features))))
        ax1.set_yticks(range(len(single_features)))
        ax1.set_yticklabels(single_features['feature_names_str'], fontsize=12)
        ax1.set_xlabel('Test AUC-PR Score')
        ax1.set_title('Рейтинг індивідуальних ознак (AUC-PR)', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3)
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{width:.4f}', ha='left', va='center', fontweight='bold')
        top_20 = self.results_df.nlargest(20, 'test_auc_pr')
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
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/individual_features_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_metrics_correlation(self):
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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        scatter = ax1.scatter(self.results_df['cv_auc_pr_mean'], self.results_df['test_auc_pr'],
                            c=self.results_df['combination_size'], cmap='viridis', alpha=0.6)
        min_val = min(self.results_df['cv_auc_pr_mean'].min(), self.results_df['test_auc_pr'].min())
        max_val = max(self.results_df['cv_auc_pr_mean'].max(), self.results_df['test_auc_pr'].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        ax1.set_xlabel('CV AUC-PR Score')
        ax1.set_ylabel('Test AUC-PR Score')
        ax1.set_title('CV vs Test AUC-PR Score', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Кількість ознак')
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
        print(f"\n🔄 === ЕТАП 3: ГЕНЕРАЦІЯ ЗВІТУ ===")
        self.results_df.to_csv(f'{self.results_dir}/all_combinations_results.csv', index=False)
        top_10 = self.results_df.nlargest(10, 'test_auc_pr')
        top_10.to_csv(f'{self.results_dir}/top_10_combinations.csv', index=False)
        report_path = f'{self.results_dir}/analysis_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 🔬 ЗВІТ АНАЛІЗУ КОМБІНАЦІЙ ОЗНАК\n\n")
            f.write("## 📊 Загальна інформація\n")
            f.write(f"- Загальна кількість ознак у датасеті: {len(self.feature_names)}\n")
            f.write(f"- Топ ознак для аналізу: {self.top_features_count}\n")
            f.write(f"- Загальна кількість протестованих комбінацій: {len(self.results_df)}\n")
            f.write(f"- Використані оптимальні параметри XGBoost\n")
            f.write(f"- Основна метрика оптимізації: AUC-PR (Area Under Precision-Recall Curve)\n\n")
            f.write("## 🏆 Вибрані топ ознаки (за важливістю XGBoost)\n")
            for i, (_, row) in enumerate(self.feature_importance_scores.iterrows(), 1):
                f.write(f"{i}. **{row['feature']}** - важливість: {row['importance']:.6f}\n")
            f.write("\n")
            f.write("## 🎯 Оптимальні параметри XGBoost\n")
            for param, value in self.optimal_params.items():
                f.write(f"- {param}: {value}\n")
            f.write("\n")
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
            single_features = self.results_df[self.results_df['combination_size'] == 1].nlargest(5, 'test_auc_pr')
            f.write("## 🥇 ТОП-5 ІНДИВІДУАЛЬНИХ ОЗНАК (за AUC-PR)\n\n")
            for i, (_, row) in enumerate(single_features.iterrows(), 1):
                f.write(f"### {i}. {row['feature_names_str']}\n")
                f.write(f"- **Test AUC-PR**: {row['test_auc_pr']:.6f} (Primary)\n")
                f.write(f"- **Test F1 Score**: {row['test_f1']:.6f}\n")
                f.write(f"- **Test AUC-ROC**: {row['test_auc_roc']:.6f}\n")
                f.write(f"- **Test Accuracy**: {row['test_accuracy']:.6f}\n\n")
            f.write("## 📈 СТАТИСТИКА ПО РОЗМІРАХ КОМБІНАЦІЙ\n\n")
            f.write("| Розмір | Кількість | Середній AUC-PR | Найкращий AUC-PR | Стандартне відхилення |\n")
            f.write("|--------|-----------|-----------------|------------------|-----------------------|\n")
            for size in sorted(self.results_df['combination_size'].unique()):
                size_data = self.results_df[self.results_df['combination_size'] == size]
                f.write(f"| {size} | {len(size_data)} | {size_data['test_auc_pr'].mean():.6f} | {size_data['test_auc_pr'].max():.6f} | {size_data['test_auc_pr'].std():.6f} |\n")
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
        print(f"🚀 === ПОЧАТОК АНАЛІЗУ КОМБІНАЦІЙ ОЗНАК ===")
        start_time = time.time()
        try:
            self.load_and_prepare_data()
            self.select_top_features()
            self.analyze_all_combinations()
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
            print(f"\n🥇 ТОП-3 НАЙКРАЩИХ КОМБІНАЦІЙ (за AUC-PR):")
            top_3 = self.results_df.nlargest(3, 'test_auc_pr')
            for i, (_, row) in enumerate(top_3.iterrows(), 1):
                print(f"{i}. {row['feature_names_str']} (AUC-PR: {row['test_auc_pr']:.6f})")
        except Exception as e:
            print(f"❌ ПОМИЛКА: {e}")
            raise

if __name__ == '__main__':
    analyzer = FeatureCombinationAnalyzer(
        data_path=r'D:\PROJECTs\MY\ScientificCalculations\SC\ScientificCalculations\PaperS\Paper4\preprocessed_data2.csv',
        random_state=42,
        top_features_count=12
    )
    analyzer.load_and_prepare_data()
    analyzer.select_top_features()
    analyzer.analyze_all_combinations()
    analyzer.create_visualizations()  # За потреби
    analyzer.generate_report()        # За потреби