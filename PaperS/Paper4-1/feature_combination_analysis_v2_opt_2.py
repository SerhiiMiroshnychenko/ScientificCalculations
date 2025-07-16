# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import itertools
import warnings
import os
import time
import pickle
# from joblib import Parallel, delayed  # Видалено joblib

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
        self.results_dir = f"feature_analysis_results_v2_opt_2"
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
        # Без крос-валідації: тільки train/test
        model.fit(X_train_combo, self.y_train)
        y_pred = model.predict(X_test_combo)
        y_pred_proba = model.predict_proba(X_test_combo)[:, 1]
        metrics = {
            'combination_size': combination_size,
            'feature_indices': feature_indices,
            'feature_names': feature_combo_names,
            'feature_names_str': ' + '.join(feature_combo_names),
            # CV метрики видалено
            # Test метрики
            'test_f1': f1_score(self.y_test, y_pred),
            'test_accuracy': accuracy_score(self.y_test, y_pred),
            'test_precision': precision_score(self.y_test, y_pred),
            'test_recall': recall_score(self.y_test, y_pred),
            'test_auc_roc': roc_auc_score(self.y_test, y_pred_proba),
            'test_auc_pr': average_precision_score(self.y_test, y_pred_proba),
            # Overfitting gap не рахуємо
        }
        return metrics

    def analyze_all_combinations(self):
        print(f"\n🔄 === ЕТАП 1: АНАЛІЗ ВСІХ КОМБІНАЦІЙ ТОП-{self.top_features_count} ОЗНАК ===")
        n_features = self.top_features_count
        # Збираємо всі комбінації (індекси та розмір)
        all_combinations = [(comb, size) for size in range(1, n_features + 1)
                            for comb in itertools.combinations(range(n_features), size)]
        total_combinations = len(all_combinations)
        print(f"🔢 Загальна кількість комбінацій: {total_combinations}")
        print(f"⚡ Топ ознаки: {', '.join(self.top_feature_names)}")
        start_time = time.time()
        # === Замість Parallel — звичайний цикл ===
        results = []
        counter = total_combinations
        for comb, size in all_combinations:
            print(f"{counter}: {comb}")
            metrics = self.test_feature_combination(comb, size)
            results.append(metrics)
            counter -= 1
        self.results = results
        self.results_df = pd.DataFrame(self.results)
        for size in range(1, n_features + 1):
            size_results = self.results_df[self.results_df['combination_size'] == size]
            if not size_results.empty:
                best_idx = size_results['test_auc_pr'].idxmax()
                self.best_combinations[size] = self.results_df.loc[best_idx]
        total_time = time.time() - start_time
        print(f"\n✅ Аналіз завершено!")
        print(f"⏱️ Загальний час: {total_time:.2f} секунд")
        print(f"📊 Оброблено комбінацій: {total_combinations}")

        # === ДОДАНО: Розширений аналіз з ПОСЛІДОВНИМ додаванням залишених ознак ===
        print(f"\n🔄 === ЕТАП 1.5: ПОСЛІДОВНЕ ДОДАВАННЯ ЗАЛИШЕНИХ ОЗНАК ДО ТОП-{self.top_features_count} ===")
        all_features = list(self.feature_names)
        top_set = set(self.top_feature_names)
        rest_features = [f for f in all_features if f not in top_set]
        print(f"Залишені ознаки: {rest_features}")
        self.extended_results = []
        extended_features = list(self.top_feature_names)
        for i, add_feature in enumerate(rest_features):
            extended_features.append(add_feature)
            indices = [self.feature_names.index(f) for f in extended_features]
            X_train_ext = self.X_train_scaled[:, indices]
            X_test_ext = self.X_test_scaled[:, indices]
            model = XGBClassifier(**self.optimal_params)
            model.fit(X_train_ext, self.y_train)
            y_pred = model.predict(X_test_ext)
            y_pred_proba = model.predict_proba(X_test_ext)[:, 1]
            metrics = {
                'combination_size': len(extended_features),
                'feature_indices': indices,
                'feature_names': list(extended_features),
                'feature_names_str': ' + '.join(extended_features),
                'test_f1': f1_score(self.y_test, y_pred),
                'test_accuracy': accuracy_score(self.y_test, y_pred),
                'test_precision': precision_score(self.y_test, y_pred),
                'test_recall': recall_score(self.y_test, y_pred),
                'test_auc_roc': roc_auc_score(self.y_test, y_pred_proba),
                'test_auc_pr': average_precision_score(self.y_test, y_pred_proba),
            }
            self.extended_results.append(metrics)
        self.extended_results_df = pd.DataFrame(self.extended_results)
        print(f"✅ Додатковий аналіз з послідовним нарощуванням завершено!")

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
        # === Об'єднуємо всі результати в один датафрейм ===
        if len(x2) > 0:
            df_all = pd.concat([self.results_df, self.extended_results_df], ignore_index=True)
        else:
            df_all = self.results_df.copy()
        x = df_all['combination_size']
        y = df_all['test_auc_pr']
        y_min, y_max = y.min(), y.max()  # Визначаємо до використання у підписах
        plt.scatter(x, y, alpha=0.7, c=x, cmap='viridis', edgecolor='k', marker='o', s=80)
        # === Додаємо підписи максимальних значень AUC-PR для кожної кількості ознак ===
        for size in sorted(df_all['combination_size'].unique()):
            size_df = df_all[df_all['combination_size'] == size]
            if not size_df.empty:
                idx_max = size_df['test_auc_pr'].idxmax()
                row = size_df.loc[idx_max]
                x_ = row['combination_size']
                y_ = row['test_auc_pr']
                y_offset = (y_max - y_min) * 0.03 if y_max > y_min else 0.01
                plt.text(x_, y_ + y_offset, f'{y_:.3f}', fontsize=8, color='black',
                         ha='center', va='bottom',
                         bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.08'))
        plt.xlabel('Кількість ознак у комбінації', fontsize=14)
        plt.ylabel('AUC-PR (test_auc_pr)', fontsize=14)
        plt.title('AUC-PR для всіх комбінацій ознак', fontsize=16, fontweight='bold')
        # Додаємо padding по осі Y
        y_min, y_max = y.min(), y.max()
        y_range = y_max - y_min
        pad = y_range * 0.05 if y_range > 0 else 0.01
        plt.xlim(0.5, df_all['combination_size'].max() + 0.5)
        plt.ylim(y_min - pad, y_max + pad * 2)  # Додаємо більше місця зверху
        plt.xticks(range(1, df_all['combination_size'].max() + 1))
        plt.grid(True, alpha=0.3)
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
            print(f"  🟣 Створення розширеного scatter-графіка AUC-PR vs Кількість ознак...")
            self._plot_aucpr_vs_num_features_extended()
            print(f"✅ Всі візуалізації створено та збережено!")
        except Exception as e:
            print(f"❌ Помилка при створенні візуалізацій: {e}")
            print(f"⚠️ Продовжуємо без візуалізацій...")
            return False
        return True

    def _plot_metrics_by_size(self):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Аналіз метрик по розмірах комбінацій ознак', fontsize=16, fontweight='bold')
        metrics = ['test_f1', 'test_auc_pr', 'test_auc_roc']
        titles = ['F1 Score', 'AUC-PR (Primary)', 'AUC-ROC']
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i]
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
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
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
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/top_combinations.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_individual_features(self):
        single_features = self.results_df[self.results_df['combination_size'] == 1].copy()
        single_features = single_features.sort_values('test_auc_pr', ascending=True)
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
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
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/individual_features_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_metrics_correlation(self):
        metrics_cols = ['test_f1', 'test_auc_pr', 'test_auc_roc', 'test_accuracy',
                       'test_precision', 'test_recall']
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
        # Видаляємо цей графік, бо overfitting gap більше не рахується
        pass

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
                # CV та overfitting gap не виводимо
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
            # Overfitting gap не виводимо
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