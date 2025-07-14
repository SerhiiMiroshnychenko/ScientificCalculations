"""
🔬 АНАЛІЗ ТА ВИРІШЕННЯ МУЛЬТИКОЛІНЕАРНОСТІ В ОЗНАКАХ

Цей скрипт проводить комплексний аналіз кореляцій між ознаками та застосовує
сучасні методи для вирішення проблеми мультиколінеарності.

Етапи роботи:
1. Попередній аналіз кореляції між ознаками (кореляційна матриця, тепловова карта)
2. VIF (Variance Inflation Factor) аналіз для виявлення мультиколінеарності
3. Застосування різних методів вирішення:
   - PCA (Principal Component Analysis)
   - ICA (Independent Component Analysis)
   - Ridge/Lasso/Elastic Net регуляризація
   - Feature selection методи
4. Порівняння результатів різних підходів
5. Рекомендації щодо оптимального рішення

Автор: AI Assistant
Дата: 2024
"""

import pandas as pd
import numpy as np

# Встановлення non-interactive backend для matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
import os
import time
from itertools import combinations

warnings.filterwarnings('ignore')

class MulticollinearityAnalyzer:
    """
    Клас для комплексного аналізу та вирішення мультиколінеарності
    """
    
    def __init__(self, data_path, random_state=42, correlation_threshold=0.8, vif_threshold=5.0):
        self.data_path = data_path
        self.random_state = random_state
        self.correlation_threshold = correlation_threshold  # Поріг для сильної кореляції
        self.vif_threshold = vif_threshold  # Поріг VIF для мультиколінеарності
        
        # ОПТИМАЛЬНІ ПАРАМЕТРИ XGBOOST
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
            'verbosity': 0
        }
        
        # Змінні для збереження результатів
        self.data = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        
        # Результати аналізу
        self.correlation_matrix = None
        self.vif_scores = None
        self.highly_correlated_pairs = []
        self.high_vif_features = []
        
        # Результати різних методів
        self.method_results = {}
        
        # Створення папки для результатів
        self.results_dir = "multicollinearity_analysis_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"🔬 === АНАЛІЗ ТА ВИРІШЕННЯ МУЛЬТИКОЛІНЕАРНОСТІ ===")
        print(f"📁 Результати будуть збережені в: {self.results_dir}/")
        print(f"🎯 Поріг кореляції: {self.correlation_threshold}")
        print(f"🎯 Поріг VIF: {self.vif_threshold}")
        print(f"🖼️ Візуалізації будуть збережені як PNG файли")

    def load_and_prepare_data(self):
        """
        ЕТАП 1: Завантаження та підготовка даних
        """
        print(f"\n🔄 === ЕТАП 1: ЗАВАНТАЖЕННЯ ТА ПІДГОТОВКА ДАНИХ ===")
        
        # Завантаження даних
        print(f"📂 Завантаження даних з: {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        print(f"📊 Розмір даних: {self.data.shape}")
        
        # Перевірка наявності цільової змінної
        if 'is_successful' not in self.data.columns:
            raise ValueError("❌ Цільова змінна 'is_successful' не знайдена!")
        
        # Підготовка ознак
        exclude_columns = ['order_id', 'is_successful', 'create_date', 'partner_id']
        
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
        
        # Створення додаткових ознак
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

    def _create_additional_features(self, df):
        """
        Створення додаткових ознак (як в оригінальному скрипті)
        """
        data_copy = df.copy()
        
        # Створення додаткових ознак
        if 'partner_success_avg_amount' in data_copy.columns and 'partner_fail_avg_amount' in data_copy.columns:
            data_copy['success_fail_amount_diff'] = data_copy['partner_success_avg_amount'] - data_copy['partner_fail_avg_amount']
            data_copy['success_fail_amount_ratio'] = data_copy['partner_success_avg_amount'] / (data_copy['partner_fail_avg_amount'] + 1e-8)
        
        if 'partner_success_avg_messages' in data_copy.columns and 'partner_fail_avg_messages' in data_copy.columns:
            data_copy['success_fail_messages_diff'] = data_copy['partner_success_avg_messages'] - data_copy['partner_fail_avg_messages']
            data_copy['success_fail_messages_ratio'] = data_copy['partner_success_avg_messages'] / (data_copy['partner_fail_avg_messages'] + 1e-8)
        
        if 'partner_success_avg_changes' in data_copy.columns and 'partner_fail_avg_changes' in data_copy.columns:
            data_copy['success_fail_changes_diff'] = data_copy['partner_success_avg_changes'] - data_copy['partner_fail_avg_changes']
            data_copy['success_fail_changes_ratio'] = data_copy['partner_success_avg_changes'] / (data_copy['partner_fail_avg_changes'] + 1e-8)
        
        # Замінюємо inf та -inf значення на NaN, потім на 0
        data_copy = data_copy.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return data_copy 

    def analyze_correlation(self):
        """
        ЕТАП 2: Аналіз кореляції між ознаками
        """
        print(f"\n🔄 === ЕТАП 2: АНАЛІЗ КОРЕЛЯЦІЇ МІЖ ОЗНАКАМИ ===")
        
        # Розрахунок кореляційної матриці
        print(f"📊 Розрахунок кореляційної матриці...")
        self.correlation_matrix = pd.DataFrame(
            self.X_train_scaled, 
            columns=self.feature_names
        ).corr()
        
        # Пошук сильно корелюючих пар
        print(f"🔍 Пошук пар з кореляцією > {self.correlation_threshold}...")
        
        # Отримуємо верхній трикутник матриці (без діагоналі)
        mask = np.triu(np.ones_like(self.correlation_matrix, dtype=bool), k=1)
        high_corr_pairs = []
        
        for i in range(len(self.feature_names)):
            for j in range(i+1, len(self.feature_names)):
                corr_value = abs(self.correlation_matrix.iloc[i, j])
                if corr_value > self.correlation_threshold:
                    high_corr_pairs.append({
                        'feature1': self.feature_names[i],
                        'feature2': self.feature_names[j],
                        'correlation': self.correlation_matrix.iloc[i, j],
                        'abs_correlation': corr_value
                    })
        
        # Сортуємо за абсолютним значенням кореляції
        self.highly_correlated_pairs = sorted(high_corr_pairs, 
                                            key=lambda x: x['abs_correlation'], 
                                            reverse=True)
        
        print(f"🎯 Знайдено {len(self.highly_correlated_pairs)} пар з високою кореляцією:")
        for pair in self.highly_correlated_pairs:
            print(f"  • {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.4f}")
        
        # Створення візуалізації кореляційної матриці
        self._plot_correlation_matrix()
        
        print(f"✅ Аналіз кореляції завершено")

    def analyze_vif(self):
        """
        ЕТАП 3: VIF (Variance Inflation Factor) аналіз
        """
        print(f"\n🔄 === ЕТАП 3: VIF АНАЛІЗ МУЛЬТИКОЛІНЕАРНОСТІ ===")
        
        print(f"📊 Розрахунок VIF для всіх ознак...")
        
        # Створюємо DataFrame для VIF розрахунку
        vif_data = pd.DataFrame(self.X_train_scaled, columns=self.feature_names)
        
        # Розрахунок VIF для кожної ознаки
        vif_scores = []
        for i, feature in enumerate(self.feature_names):
            try:
                vif_value = variance_inflation_factor(vif_data.values, i)
                vif_scores.append({
                    'feature': feature,
                    'vif': vif_value
                })
            except Exception as e:
                print(f"⚠️ Помилка при розрахунку VIF для {feature}: {e}")
                vif_scores.append({
                    'feature': feature,
                    'vif': np.inf
                })
        
        # Створюємо DataFrame з результатами
        self.vif_scores = pd.DataFrame(vif_scores).sort_values('vif', ascending=False)
        
        # Знаходимо ознаки з високим VIF
        self.high_vif_features = self.vif_scores[
            self.vif_scores['vif'] > self.vif_threshold
        ]['feature'].tolist()
        
        print(f"📋 VIF оцінки для всіх ознак:")
        for _, row in self.vif_scores.iterrows():
            status = "🔴 ВИСОКИЙ" if row['vif'] > self.vif_threshold else "🟢 НОРМАЛЬНИЙ"
            print(f"  • {row['feature']:<35} VIF: {row['vif']:.2f} {status}")
        
        print(f"\n🎯 Ознаки з VIF > {self.vif_threshold}: {len(self.high_vif_features)}")
        for feature in self.high_vif_features:
            vif_value = self.vif_scores[self.vif_scores['feature'] == feature]['vif'].iloc[0]
            print(f"  • {feature}: {vif_value:.2f}")
        
        # Створення візуалізації VIF
        self._plot_vif_scores()
        
        print(f"✅ VIF аналіз завершено")

    def _plot_correlation_matrix(self):
        """
        Створення візуалізації кореляційної матриці
        """
        print(f"🎨 Створення візуалізації кореляційної матриці...")
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Повна кореляційна матриця
            sns.heatmap(self.correlation_matrix, 
                       annot=True, 
                       cmap='coolwarm', 
                       center=0, 
                       square=True,
                       fmt='.2f',
                       cbar_kws={'shrink': 0.8},
                       ax=ax1)
            ax1.set_title('Повна кореляційна матриця ознак', fontsize=14, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            ax1.tick_params(axis='y', rotation=0)
            
            # Матриця тільки сильних кореляцій
            high_corr_matrix = self.correlation_matrix.copy()
            mask = np.abs(high_corr_matrix) < self.correlation_threshold
            high_corr_matrix[mask] = 0
            
            sns.heatmap(high_corr_matrix, 
                       annot=True, 
                       cmap='coolwarm', 
                       center=0, 
                       square=True,
                       fmt='.2f',
                       cbar_kws={'shrink': 0.8},
                       ax=ax2)
            ax2.set_title(f'Сильні кореляції (|r| > {self.correlation_threshold})', fontsize=14, fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
            ax2.tick_params(axis='y', rotation=0)
            
            plt.tight_layout()
            plt.savefig(f'{self.results_dir}/correlation_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"💾 Збережено: {self.results_dir}/correlation_analysis.png")
            
        except Exception as e:
            print(f"⚠️ Помилка при створенні візуалізації кореляції: {e}")

    def _plot_vif_scores(self):
        """
        Створення візуалізації VIF оцінок
        """
        print(f"🎨 Створення візуалізації VIF оцінок...")
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
            
            # График VIF для всіх ознак
            colors = ['red' if vif > self.vif_threshold else 'green' for vif in self.vif_scores['vif']]
            
            ax1.barh(range(len(self.vif_scores)), self.vif_scores['vif'], color=colors, alpha=0.7)
            ax1.set_yticks(range(len(self.vif_scores)))
            ax1.set_yticklabels(self.vif_scores['feature'], fontsize=10)
            ax1.axvline(x=self.vif_threshold, color='red', linestyle='--', linewidth=2, 
                       label=f'Поріг VIF = {self.vif_threshold}')
            ax1.set_xlabel('VIF Score')
            ax1.set_title('VIF оцінки для всіх ознак', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # График тільки високих VIF (якщо є)
            if len(self.high_vif_features) > 0:
                high_vif_data = self.vif_scores[self.vif_scores['vif'] > self.vif_threshold]
                
                ax2.barh(range(len(high_vif_data)), high_vif_data['vif'], color='red', alpha=0.7)
                ax2.set_yticks(range(len(high_vif_data)))
                ax2.set_yticklabels(high_vif_data['feature'], fontsize=10)
                ax2.axvline(x=self.vif_threshold, color='red', linestyle='--', linewidth=2,
                           label=f'Поріг VIF = {self.vif_threshold}')
                ax2.set_xlabel('VIF Score')
                ax2.set_title(f'Ознаки з високим VIF (> {self.vif_threshold})', fontsize=14, fontweight='bold')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'Немає ознак з високим VIF', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                ax2.set_title('Ознаки з високим VIF', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'{self.results_dir}/vif_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"💾 Збережено: {self.results_dir}/vif_analysis.png")
            
        except Exception as e:
            print(f"⚠️ Помилка при створенні візуалізації VIF: {e}")

    def evaluate_model(self, X_train, X_test, y_train, y_test, method_name="baseline"):
        """
        Оцінка моделі з розрахунком усіх метрик
        """
        # Cross-validation на тренувальних даних
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        model = XGBClassifier(**self.optimal_params)
        
        # CV метрики
        cv_auc_roc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        cv_auc_pr_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='average_precision')
        cv_f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
        
        # Тренування та тест на відкладеній вибірці
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Тестові метрики
        test_auc_roc = roc_auc_score(y_test, y_pred_proba)
        test_auc_pr = average_precision_score(y_test, y_pred_proba)
        test_f1 = f1_score(y_test, y_pred)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred)
        test_recall = recall_score(y_test, y_pred)
        
        return {
            'method': method_name,
            'n_features': X_train.shape[1],
            'cv_auc_roc_mean': cv_auc_roc_scores.mean(),
            'cv_auc_roc_std': cv_auc_roc_scores.std(),
            'cv_auc_pr_mean': cv_auc_pr_scores.mean(),
            'cv_auc_pr_std': cv_auc_pr_scores.std(),
            'cv_f1_mean': cv_f1_scores.mean(),
            'cv_f1_std': cv_f1_scores.std(),
            'test_auc_roc': test_auc_roc,
            'test_auc_pr': test_auc_pr,
            'test_f1': test_f1,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'overfitting_gap_auc_roc': cv_auc_roc_scores.mean() - test_auc_roc,
            'overfitting_gap_auc_pr': cv_auc_pr_scores.mean() - test_auc_pr,
            'overfitting_gap_f1': cv_f1_scores.mean() - test_f1
        } 

    def apply_pca_method(self, n_components=None):
        """
        МЕТОД 1: PCA (Principal Component Analysis)
        """
        print(f"\n🔄 === МЕТОД 1: PCA (PRINCIPAL COMPONENT ANALYSIS) ===")
        
        if n_components is None:
            # Автоматичний вибір компонентів для збереження 95% дисперсії
            n_components = 0.95
        
        print(f"🎯 Цільова дисперсія: {n_components}")
        
        # Застосування PCA
        pca = PCA(n_components=n_components, random_state=self.random_state)
        X_train_pca = pca.fit_transform(self.X_train_scaled)
        X_test_pca = pca.transform(self.X_test_scaled)
        
        print(f"📊 Кількість компонентів: {pca.n_components_}")
        print(f"📊 Пояснена дисперсія: {pca.explained_variance_ratio_.sum():.4f}")
        
        # Оцінка моделі
        results = self.evaluate_model(X_train_pca, X_test_pca, self.y_train, self.y_test, "PCA")
        self.method_results['PCA'] = results
        
        # Додаткова інформація про PCA
        results['explained_variance_ratio'] = pca.explained_variance_ratio_.sum()
        results['n_components'] = pca.n_components_
        results['original_features'] = len(self.feature_names)
        
        print(f"✅ PCA метод завершено: {len(self.feature_names)} → {pca.n_components_} компонентів")
        
        return X_train_pca, X_test_pca, pca

    def apply_ica_method(self, n_components=None):
        """
        МЕТОД 2: ICA (Independent Component Analysis)
        """
        print(f"\n🔄 === МЕТОД 2: ICA (INDEPENDENT COMPONENT ANALYSIS) ===")
        
        if n_components is None:
            # Використовуємо таку ж кількість компонентів як у PCA
            if 'PCA' in self.method_results:
                n_components = self.method_results['PCA']['n_components']
            else:
                n_components = min(10, len(self.feature_names))
        
        print(f"🎯 Кількість компонентів: {n_components}")
        
        # Застосування ICA
        ica = FastICA(n_components=n_components, random_state=self.random_state, max_iter=1000)
        X_train_ica = ica.fit_transform(self.X_train_scaled)
        X_test_ica = ica.transform(self.X_test_scaled)
        
        # Оцінка моделі
        results = self.evaluate_model(X_train_ica, X_test_ica, self.y_train, self.y_test, "ICA")
        self.method_results['ICA'] = results
        
        # Додаткова інформація про ICA
        results['n_components'] = n_components
        results['original_features'] = len(self.feature_names)
        
        print(f"✅ ICA метод завершено: {len(self.feature_names)} → {n_components} компонентів")
        
        return X_train_ica, X_test_ica, ica

    def apply_ridge_method(self, alpha=1.0):
        """
        МЕТОД 3: Ridge регресія (L2 регуляризація)
        """
        print(f"\n🔄 === МЕТОД 3: RIDGE РЕГРЕСІЯ (L2 РЕГУЛЯРИЗАЦІЯ) ===")
        
        print(f"🎯 Параметр регуляризації α: {alpha}")
        
        # Модифікуємо параметри XGBoost для L2 регуляризації
        ridge_params = self.optimal_params.copy()
        ridge_params['reg_lambda'] = alpha
        
        # Оцінка моделі з L2 регуляризацією
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        model = XGBClassifier(**ridge_params)
        
        # CV метрики
        cv_auc_roc_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=cv, scoring='roc_auc')
        cv_auc_pr_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=cv, scoring='average_precision')
        cv_f1_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=cv, scoring='f1')
        
        # Тренування та тест
        model.fit(self.X_train_scaled, self.y_train)
        y_pred = model.predict(self.X_test_scaled)
        y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Тестові метрики
        test_auc_roc = roc_auc_score(self.y_test, y_pred_proba)
        test_auc_pr = average_precision_score(self.y_test, y_pred_proba)
        test_f1 = f1_score(self.y_test, y_pred)
        test_accuracy = accuracy_score(self.y_test, y_pred)
        test_precision = precision_score(self.y_test, y_pred)
        test_recall = recall_score(self.y_test, y_pred)
        
        results = {
            'method': 'Ridge',
            'n_features': len(self.feature_names),
            'cv_auc_roc_mean': cv_auc_roc_scores.mean(),
            'cv_auc_roc_std': cv_auc_roc_scores.std(),
            'cv_auc_pr_mean': cv_auc_pr_scores.mean(),
            'cv_auc_pr_std': cv_auc_pr_scores.std(),
            'cv_f1_mean': cv_f1_scores.mean(),
            'cv_f1_std': cv_f1_scores.std(),
            'test_auc_roc': test_auc_roc,
            'test_auc_pr': test_auc_pr,
            'test_f1': test_f1,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'overfitting_gap_auc_roc': cv_auc_roc_scores.mean() - test_auc_roc,
            'overfitting_gap_auc_pr': cv_auc_pr_scores.mean() - test_auc_pr,
            'overfitting_gap_f1': cv_f1_scores.mean() - test_f1,
            'regularization_alpha': alpha
        }
        
        self.method_results['Ridge'] = results
        
        print(f"✅ Ridge метод завершено з α = {alpha}")
        
        return model

    def apply_lasso_method(self, alpha=1.0):
        """
        МЕТОД 4: Lasso регресія (L1 регуляризація)
        """
        print(f"\n🔄 === МЕТОД 4: LASSO РЕГРЕСІЯ (L1 РЕГУЛЯРИЗАЦІЯ) ===")
        
        print(f"🎯 Параметр регуляризації α: {alpha}")
        
        # Модифікуємо параметри XGBoost для L1 регуляризації
        lasso_params = self.optimal_params.copy()
        lasso_params['reg_alpha'] = alpha
        
        # Оцінка моделі з L1 регуляризацією
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        model = XGBClassifier(**lasso_params)
        
        # CV метрики
        cv_auc_roc_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=cv, scoring='roc_auc')
        cv_auc_pr_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=cv, scoring='average_precision')
        cv_f1_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=cv, scoring='f1')
        
        # Тренування та тест
        model.fit(self.X_train_scaled, self.y_train)
        y_pred = model.predict(self.X_test_scaled)
        y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Тестові метрики
        test_auc_roc = roc_auc_score(self.y_test, y_pred_proba)
        test_auc_pr = average_precision_score(self.y_test, y_pred_proba)
        test_f1 = f1_score(self.y_test, y_pred)
        test_accuracy = accuracy_score(self.y_test, y_pred)
        test_precision = precision_score(self.y_test, y_pred)
        test_recall = recall_score(self.y_test, y_pred)
        
        results = {
            'method': 'Lasso',
            'n_features': len(self.feature_names),
            'cv_auc_roc_mean': cv_auc_roc_scores.mean(),
            'cv_auc_roc_std': cv_auc_roc_scores.std(),
            'cv_auc_pr_mean': cv_auc_pr_scores.mean(),
            'cv_auc_pr_std': cv_auc_pr_scores.std(),
            'cv_f1_mean': cv_f1_scores.mean(),
            'cv_f1_std': cv_f1_scores.std(),
            'test_auc_roc': test_auc_roc,
            'test_auc_pr': test_auc_pr,
            'test_f1': test_f1,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'overfitting_gap_auc_roc': cv_auc_roc_scores.mean() - test_auc_roc,
            'overfitting_gap_auc_pr': cv_auc_pr_scores.mean() - test_auc_pr,
            'overfitting_gap_f1': cv_f1_scores.mean() - test_f1,
            'regularization_alpha': alpha
        }
        
        self.method_results['Lasso'] = results
        
        print(f"✅ Lasso метод завершено з α = {alpha}")
        
        return model

    def apply_elastic_net_method(self, alpha=1.0, l1_ratio=0.5):
        """
        МЕТОД 5: Elastic Net (комбінація L1 + L2 регуляризації)
        """
        print(f"\n🔄 === МЕТОД 5: ELASTIC NET (L1 + L2 РЕГУЛЯРИЗАЦІЯ) ===")
        
        print(f"🎯 Параметр регуляризації α: {alpha}")
        print(f"🎯 L1 ratio: {l1_ratio}")
        
        # Модифікуємо параметри XGBoost для Elastic Net
        elastic_params = self.optimal_params.copy()
        elastic_params['reg_alpha'] = alpha * l1_ratio  # L1 компонент
        elastic_params['reg_lambda'] = alpha * (1 - l1_ratio)  # L2 компонент
        
        # Оцінка моделі з Elastic Net регуляризацією
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        model = XGBClassifier(**elastic_params)
        
        # CV метрики
        cv_auc_roc_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=cv, scoring='roc_auc')
        cv_auc_pr_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=cv, scoring='average_precision')
        cv_f1_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=cv, scoring='f1')
        
        # Тренування та тест
        model.fit(self.X_train_scaled, self.y_train)
        y_pred = model.predict(self.X_test_scaled)
        y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Тестові метрики
        test_auc_roc = roc_auc_score(self.y_test, y_pred_proba)
        test_auc_pr = average_precision_score(self.y_test, y_pred_proba)
        test_f1 = f1_score(self.y_test, y_pred)
        test_accuracy = accuracy_score(self.y_test, y_pred)
        test_precision = precision_score(self.y_test, y_pred)
        test_recall = recall_score(self.y_test, y_pred)
        
        results = {
            'method': 'Elastic Net',
            'n_features': len(self.feature_names),
            'cv_auc_roc_mean': cv_auc_roc_scores.mean(),
            'cv_auc_roc_std': cv_auc_roc_scores.std(),
            'cv_auc_pr_mean': cv_auc_pr_scores.mean(),
            'cv_auc_pr_std': cv_auc_pr_scores.std(),
            'cv_f1_mean': cv_f1_scores.mean(),
            'cv_f1_std': cv_f1_scores.std(),
            'test_auc_roc': test_auc_roc,
            'test_auc_pr': test_auc_pr,
            'test_f1': test_f1,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'overfitting_gap_auc_roc': cv_auc_roc_scores.mean() - test_auc_roc,
            'overfitting_gap_auc_pr': cv_auc_pr_scores.mean() - test_auc_pr,
            'overfitting_gap_f1': cv_f1_scores.mean() - test_f1,
            'regularization_alpha': alpha,
            'l1_ratio': l1_ratio
        }
        
        self.method_results['Elastic Net'] = results
        
        print(f"✅ Elastic Net метод завершено з α = {alpha}, l1_ratio = {l1_ratio}")
        
        return model

    def apply_feature_selection_vif(self):
        """
        МЕТОД 6: Видалення ознак з високим VIF
        """
        print(f"\n🔄 === МЕТОД 6: ВИДАЛЕННЯ ОЗНАК З ВИСОКИМ VIF ===")
        
        if self.vif_scores is None:
            print("⚠️ Спочатку потрібно виконати VIF аналіз!")
            return None, None
        
        # Ознаки з нормальним VIF
        low_vif_features = self.vif_scores[
            self.vif_scores['vif'] <= self.vif_threshold
        ]['feature'].tolist()
        
        print(f"🎯 Видаляємо {len(self.high_vif_features)} ознак з високим VIF")
        print(f"🎯 Залишаємо {len(low_vif_features)} ознак з нормальним VIF")
        
        # Індекси ознак з низьким VIF
        low_vif_indices = [self.feature_names.index(feature) for feature in low_vif_features]
        
        # Створення нових даних без ознак з високим VIF
        X_train_vif = self.X_train_scaled[:, low_vif_indices]
        X_test_vif = self.X_test_scaled[:, low_vif_indices]
        
        # Оцінка моделі
        results = self.evaluate_model(X_train_vif, X_test_vif, self.y_train, self.y_test, "VIF Selection")
        self.method_results['VIF Selection'] = results
        
        # Додаткова інформація
        results['removed_features'] = len(self.high_vif_features)
        results['remaining_features'] = len(low_vif_features)
        results['original_features'] = len(self.feature_names)
        
        print(f"✅ VIF Selection метод завершено: {len(self.feature_names)} → {len(low_vif_features)} ознак")
        
        return X_train_vif, X_test_vif

    def apply_correlation_removal(self):
        """
        МЕТОД 7: Видалення одного з пари сильно корелюючих ознак
        """
        print(f"\n🔄 === МЕТОД 7: ВИДАЛЕННЯ КОРЕЛЮЮЧИХ ОЗНАК ===")
        
        if not self.highly_correlated_pairs:
            print("✅ Немає сильно корелюючих пар ознак для видалення")
            # Повертаємо оригінальні дані
            results = self.evaluate_model(self.X_train_scaled, self.X_test_scaled, 
                                        self.y_train, self.y_test, "Correlation Removal")
            self.method_results['Correlation Removal'] = results
            return self.X_train_scaled, self.X_test_scaled
        
        # Визначаємо ознаки для видалення
        features_to_remove = set()
        
        # Для кожної пари видаляємо ознаку з меншою важливістю
        # Спочатку отримуємо важливість ознак з XGBoost
        importance_model = XGBClassifier(**self.optimal_params)
        importance_model.fit(self.X_train_scaled, self.y_train)
        feature_importance = importance_model.feature_importances_
        
        importance_dict = {feature: importance for feature, importance 
                          in zip(self.feature_names, feature_importance)}
        
        for pair in self.highly_correlated_pairs:
            feature1 = pair['feature1']
            feature2 = pair['feature2']
            
            # Видаляємо ознаку з меншою важливістю
            if importance_dict[feature1] > importance_dict[feature2]:
                features_to_remove.add(feature2)
                print(f"  • Видаляємо {feature2} (важливість: {importance_dict[feature2]:.4f})")
                print(f"    Залишаємо {feature1} (важливість: {importance_dict[feature1]:.4f})")
            else:
                features_to_remove.add(feature1)
                print(f"  • Видаляємо {feature1} (важливість: {importance_dict[feature1]:.4f})")
                print(f"    Залишаємо {feature2} (важливість: {importance_dict[feature2]:.4f})")
        
        # Ознаки, що залишаються
        remaining_features = [f for f in self.feature_names if f not in features_to_remove]
        remaining_indices = [self.feature_names.index(f) for f in remaining_features]
        
        print(f"🎯 Видаляємо {len(features_to_remove)} ознак")
        print(f"🎯 Залишаємо {len(remaining_features)} ознак")
        
        # Створення нових даних
        X_train_corr = self.X_train_scaled[:, remaining_indices]
        X_test_corr = self.X_test_scaled[:, remaining_indices]
        
        # Оцінка моделі
        results = self.evaluate_model(X_train_corr, X_test_corr, self.y_train, self.y_test, "Correlation Removal")
        self.method_results['Correlation Removal'] = results
        
        # Додаткова інформація
        results['removed_features'] = len(features_to_remove)
        results['remaining_features'] = len(remaining_features)
        results['original_features'] = len(self.feature_names)
        
        print(f"✅ Correlation Removal метод завершено: {len(self.feature_names)} → {len(remaining_features)} ознак")
        
        return X_train_corr, X_test_corr

    def create_comparison_visualizations(self):
        """
        Створення візуалізацій для порівняння всіх методів
        """
        print(f"\n🎨 === СТВОРЕННЯ ВІЗУАЛІЗАЦІЙ ПОРІВНЯННЯ ===")
        
        if not self.method_results:
            print("⚠️ Немає результатів для візуалізації")
            return
        
        try:
            # Перевіряємо, чи є Baseline в результатах
            if 'Baseline' not in self.method_results:
                # Додавання baseline (оригінальні дані)
                baseline_results = self.evaluate_model(self.X_train_scaled, self.X_test_scaled,
                                                     self.y_train, self.y_test, "Baseline")
                self.method_results['Baseline'] = baseline_results

            # Створення DataFrame з результатами
            results_df = pd.DataFrame(self.method_results).T

            # Отримуємо методи з фактичних результатів
            methods = list(results_df.index)

            print(f"📊 Знайдено методів для візуалізації: {len(methods)}")
            print(f"📋 Методи: {methods}")

            # Перевіряємо, чи всі необхідні колонки є в результатах
            required_columns = ['cv_auc_roc_mean', 'cv_auc_roc_std', 'cv_auc_pr_mean', 'cv_auc_pr_std',
                              'cv_f1_mean', 'cv_f1_std', 'test_auc_roc', 'test_auc_pr', 'test_f1',
                              'n_features', 'overfitting_gap_auc_pr']

            missing_columns = [col for col in required_columns if col not in results_df.columns]
            if missing_columns:
                print(f"⚠️ Відсутні колонки: {missing_columns}")
                return

            # 1. Порівняння основних метрик
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Порівняння методів вирішення мультиколінеарності', fontsize=16, fontweight='bold')

            # AUC-ROC CV
            ax1 = axes[0, 0]
            print(f"🔧 Створення барчарта AUC-ROC для {len(methods)} методів")
            bars1 = ax1.bar(range(len(methods)), results_df['cv_auc_roc_mean'],
                           yerr=results_df['cv_auc_roc_std'], capsize=5, alpha=0.7)
            ax1.set_title('AUC-ROC (Cross-Validation)', fontweight='bold')
            ax1.set_ylabel('AUC-ROC Score')
            ax1.set_xticks(range(len(methods)))
            ax1.set_xticklabels(methods, rotation=45)
            ax1.grid(True, alpha=0.3)

            # AUC-PR CV
            ax2 = axes[0, 1]
            bars2 = ax2.bar(range(len(methods)), results_df['cv_auc_pr_mean'],
                           yerr=results_df['cv_auc_pr_std'], capsize=5, alpha=0.7, color='orange')
            ax2.set_title('AUC-PR (Cross-Validation)', fontweight='bold')
            ax2.set_ylabel('AUC-PR Score')
            ax2.set_xticks(range(len(methods)))
            ax2.set_xticklabels(methods, rotation=45)
            ax2.grid(True, alpha=0.3)

            # F1 CV
            ax3 = axes[0, 2]
            bars3 = ax3.bar(range(len(methods)), results_df['cv_f1_mean'],
                           yerr=results_df['cv_f1_std'], capsize=5, alpha=0.7, color='green')
            ax3.set_title('F1-Score (Cross-Validation)', fontweight='bold')
            ax3.set_ylabel('F1 Score')
            ax3.set_xticks(range(len(methods)))
            ax3.set_xticklabels(methods, rotation=45)
            ax3.grid(True, alpha=0.3)

            # Test метрики
            ax4 = axes[1, 0]
            test_metrics = ['test_auc_roc', 'test_auc_pr', 'test_f1']

            x = np.arange(len(methods))
            width = 0.25

            ax4.bar(x - width, results_df['test_auc_roc'].values, width, label='AUC-ROC', alpha=0.7)
            ax4.bar(x, results_df['test_auc_pr'].values, width, label='AUC-PR', alpha=0.7)
            ax4.bar(x + width, results_df['test_f1'].values, width, label='F1-Score', alpha=0.7)

            ax4.set_title('Тестові метрики', fontweight='bold')
            ax4.set_ylabel('Score')
            ax4.set_xticks(x)
            ax4.set_xticklabels(methods, rotation=45)
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            # Кількість ознак
            ax5 = axes[1, 1]
            bars5 = ax5.bar(range(len(methods)), results_df['n_features'], alpha=0.7, color='purple')
            ax5.set_title('Кількість ознак', fontweight='bold')
            ax5.set_ylabel('Кількість ознак')
            ax5.set_xticks(range(len(methods)))
            ax5.set_xticklabels(methods, rotation=45)
            ax5.grid(True, alpha=0.3)

            # Додавання значень на стовпці
            for i, bar in enumerate(bars5):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom')

            # Overfitting gap (AUC-PR)
            ax6 = axes[1, 2]
            gap_colors = ['red' if gap > 0.05 else 'green' for gap in results_df['overfitting_gap_auc_pr']]
            bars6 = ax6.bar(range(len(methods)), results_df['overfitting_gap_auc_pr'],
                           alpha=0.7, color=gap_colors)
            ax6.set_title('Overfitting Gap (AUC-PR)', fontweight='bold')
            ax6.set_ylabel('CV - Test AUC-PR')
            ax6.set_xticks(range(len(methods)))
            ax6.set_xticklabels(methods, rotation=45)
            ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax6.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='Поріг (0.05)')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'{self.results_dir}/method_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

            print(f"💾 Збережено: {self.results_dir}/method_comparison.png")

            # 2. ROC та PR криві для найкращих методів
            self._plot_best_methods_curves()

            # 3. Таблиця результатів
            self._save_results_table()

        except Exception as e:
            print(f"⚠️ Помилка при створенні візуалізацій: {e}")

    def _plot_best_methods_curves(self):
        """
        Створення ROC та PR кривих для найкращих методів
        """
        print(f"🎨 Створення ROC та PR кривих...")

        try:
            # Створення DataFrame з результатами
            results_df = pd.DataFrame(self.method_results).T

            # Перевіряємо, чи є достатньо методів
            if len(results_df) == 0:
                print("⚠️ Немає результатів для створення кривих")
                return

            # Вибираємо до 3 найкращих методів за AUC-PR (або менше, якщо методів менше)
            n_methods = min(3, len(results_df))
            best_methods = results_df.nlargest(n_methods, 'test_auc_pr').index.tolist()

            print(f"📊 Створення кривих для {len(best_methods)} найкращих методів: {best_methods}")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

            for i, method in enumerate(best_methods):
                # Переконуємося, що не виходимо за межі кольорів
                color = colors[i % len(colors)]
                # Отримуємо дані для побудови кривих
                if method == 'Baseline':
                    X_test_method = self.X_test_scaled
                elif method == 'PCA':
                    pca = PCA(n_components=0.95, random_state=self.random_state)
                    pca.fit(self.X_train_scaled)
                    X_test_method = pca.transform(self.X_test_scaled)
                elif method == 'VIF Selection':
                    low_vif_features = self.vif_scores[
                        self.vif_scores['vif'] <= self.vif_threshold
                    ]['feature'].tolist()
                    low_vif_indices = [self.feature_names.index(f) for f in low_vif_features]
                    X_test_method = self.X_test_scaled[:, low_vif_indices]
                else:
                    X_test_method = self.X_test_scaled  # Для регуляризаційних методів

                # Тренування моделі для цього методу
                if method in ['Ridge', 'Lasso', 'Elastic Net']:
                    # Використовуємо відповідні параметри регуляризації
                    model_params = self.optimal_params.copy()
                    if method == 'Ridge':
                        model_params['reg_lambda'] = 1.0
                    elif method == 'Lasso':
                        model_params['reg_alpha'] = 1.0
                    elif method == 'Elastic Net':
                        model_params['reg_alpha'] = 0.5
                        model_params['reg_lambda'] = 0.5
                    model = XGBClassifier(**model_params)
                    model.fit(self.X_train_scaled, self.y_train)
                    y_pred_proba = model.predict_proba(X_test_method)[:, 1]
                else:
                    model = XGBClassifier(**self.optimal_params)
                    if method == 'PCA':
                        model.fit(pca.transform(self.X_train_scaled), self.y_train)
                    elif method == 'VIF Selection':
                        model.fit(self.X_train_scaled[:, low_vif_indices], self.y_train)
                    else:
                        model.fit(self.X_train_scaled, self.y_train)
                    y_pred_proba = model.predict_proba(X_test_method)[:, 1]

                # ROC крива
                fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
                roc_auc = roc_auc_score(self.y_test, y_pred_proba)
                ax1.plot(fpr, tpr, color=color, lw=2,
                        label=f'{method} (AUC = {roc_auc:.3f})')

                # PR крива
                precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
                pr_auc = average_precision_score(self.y_test, y_pred_proba)
                ax2.plot(recall, precision, color=color, lw=2,
                        label=f'{method} (AUC = {pr_auc:.3f})')
            
            # Оформлення ROC кривої
            ax1.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
            ax1.set_xlim([0.0, 1.0])
            ax1.set_ylim([0.0, 1.05])
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            ax1.set_title('ROC криві для найкращих методів', fontweight='bold')
            ax1.legend(loc="lower right")
            ax1.grid(True, alpha=0.3)
            
            # Оформлення PR кривої
            baseline_precision = self.y_test.sum() / len(self.y_test)
            ax2.axhline(y=baseline_precision, color='k', linestyle='--', lw=1, alpha=0.5,
                       label=f'Baseline ({baseline_precision:.3f})')
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.set_title('Precision-Recall криві для найкращих методів', fontweight='bold')
            ax2.legend(loc="lower left")
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{self.results_dir}/best_methods_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"💾 Збережено: {self.results_dir}/best_methods_curves.png")
            
        except Exception as e:
            print(f"⚠️ Помилка при створенні кривих: {e}")

    def _save_results_table(self):
        """
        Збереження детальної таблиці результатів
        """
        print(f"📊 Збереження детальної таблиці результатів...")
        
        try:
            results_df = pd.DataFrame(self.method_results).T
            
            # Сортуємо за AUC-PR
            results_df = results_df.sort_values('test_auc_pr', ascending=False)
            
            # Округлюємо числові значення
            numeric_columns = results_df.select_dtypes(include=[np.number]).columns
            results_df[numeric_columns] = results_df[numeric_columns].round(4)
            
            # Збереження в CSV
            results_df.to_csv(f'{self.results_dir}/detailed_results.csv')
            print(f"💾 Збережено: {self.results_dir}/detailed_results.csv")
            
            # Збереження в Excel з форматуванням
            with pd.ExcelWriter(f'{self.results_dir}/detailed_results.xlsx', engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name='Results', index=True)
            print(f"💾 Збережено: {self.results_dir}/detailed_results.xlsx")
            
        except Exception as e:
            print(f"⚠️ Помилка при збереженні таблиці: {e}")

    def generate_recommendations(self):
        """
        Генерація рекомендацій на основі результатів аналізу
        """
        print(f"\n📝 === ГЕНЕРАЦІЯ РЕКОМЕНДАЦІЙ ===")
        
        if not self.method_results:
            print("⚠️ Немає результатів для аналізу")
            return
        
        results_df = pd.DataFrame(self.method_results).T
        
        # Найкращий метод за AUC-PR
        best_method = results_df.loc[results_df['test_auc_pr'].idxmax()]
        
        # Аналіз результатів
        print(f"🏆 НАЙКРАЩИЙ МЕТОД: {best_method.name}")
        print(f"   • Test AUC-PR: {best_method['test_auc_pr']:.4f}")
        print(f"   • Test AUC-ROC: {best_method['test_auc_roc']:.4f}")
        print(f"   • Test F1: {best_method['test_f1']:.4f}")
        print(f"   • Кількість ознак: {best_method['n_features']}")
        print(f"   • Overfitting gap: {best_method['overfitting_gap_auc_pr']:.4f}")
        
        recommendations = []
        
        # Рекомендації на основі проблем мультиколінеарності
        if len(self.highly_correlated_pairs) > 0:
            recommendations.append(f"🔍 Виявлено {len(self.highly_correlated_pairs)} пар сильно корелюючих ознак")
        
        if len(self.high_vif_features) > 0:
            recommendations.append(f"⚠️ Знайдено {len(self.high_vif_features)} ознак з високим VIF > {self.vif_threshold}")
        
        # Рекомендації за результатами
        baseline_auc_pr = results_df.loc['Baseline', 'test_auc_pr']
        improvement = best_method['test_auc_pr'] - baseline_auc_pr
        
        if improvement > 0.01:
            recommendations.append(f"✅ Метод {best_method.name} покращує AUC-PR на {improvement:.4f}")
        elif improvement > -0.01:
            recommendations.append(f"⚖️ Метод {best_method.name} показує схожі результати з baseline")
        else:
            recommendations.append(f"❌ Методи вирішення мультиколінеарності погіршують результати")
        
        # Специфічні рекомендації
        if best_method.name == 'PCA':
            recommendations.append("📊 PCA рекомендується коли важлива максимальна точність, але втрачається інтерпретованість")
        elif best_method.name in ['Ridge', 'Lasso', 'Elastic Net']:
            recommendations.append("🎛️ Регуляризація рекомендується для збереження всіх ознак з контролем впливу")
        elif best_method.name == 'VIF Selection':
            recommendations.append("🎯 Видалення ознак з високим VIF рекомендується для збереження інтерпретованості")
        
        print(f"\n📋 РЕКОМЕНДАЦІЇ:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        # Збереження рекомендацій у файл
        with open(f'{self.results_dir}/recommendations.txt', 'w', encoding='utf-8') as f:
            f.write("АНАЛІЗ МУЛЬТИКОЛІНЕАРНОСТІ - РЕКОМЕНДАЦІЇ\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Найкращий метод: {best_method.name}\n")
            f.write(f"Test AUC-PR: {best_method['test_auc_pr']:.4f}\n")
            f.write(f"Покращення відносно baseline: {improvement:.4f}\n\n")
            f.write("Рекомендації:\n")
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")
        
        print(f"💾 Збережено: {self.results_dir}/recommendations.txt")

    def run_analysis(self, data_path_param=None):
        """
        Основний метод для запуску повного аналізу мультиколінеарності
        """
        start_time = time.time()
        
        if data_path_param:
            self.data_path = data_path_param
        
        print(f"🚀 === ПОЧАТОК АНАЛІЗУ МУЛЬТИКОЛІНЕАРНОСТІ ===")
        print(f"⏰ Час початку: {time.strftime('%H:%M:%S')}")
        
        try:
            # ЕТАП 1: Підготовка даних
            self.load_and_prepare_data()
            
            # ЕТАП 2: Аналіз кореляції
            self.analyze_correlation()
            
            # ЕТАП 3: VIF аналіз
            self.analyze_vif()
            
            # ЕТАП 4: Застосування різних методів
            print(f"\n🔄 === ЕТАП 4: ЗАСТОСУВАННЯ МЕТОДІВ ВИРІШЕННЯ ===")
            
            # Baseline (для порівняння)
            print(f"📊 Оцінка baseline моделі...")
            baseline_results = self.evaluate_model(self.X_train_scaled, self.X_test_scaled, 
                                                  self.y_train, self.y_test, "Baseline")
            self.method_results['Baseline'] = baseline_results
            
            # Методи зменшення розмірності
            self.apply_pca_method()
            self.apply_ica_method()
            
            # Регуляризаційні методи
            self.apply_ridge_method(alpha=1.0)
            self.apply_lasso_method(alpha=1.0)
            self.apply_elastic_net_method(alpha=1.0, l1_ratio=0.5)
            
            # Feature selection методи
            self.apply_feature_selection_vif()
            self.apply_correlation_removal()
            
            # ЕТАП 5: Візуалізація та порівняння
            print(f"\n🔄 === ЕТАП 5: ВІЗУАЛІЗАЦІЯ ТА ПОРІВНЯННЯ ===")
            self.create_comparison_visualizations()
            
            # ЕТАП 6: Рекомендації
            print(f"\n🔄 === ЕТАП 6: ГЕНЕРАЦІЯ РЕКОМЕНДАЦІЙ ===")
            self.generate_recommendations()
            
            execution_time = time.time() - start_time
            print(f"\n✅ === АНАЛІЗ ЗАВЕРШЕНО ===")
            print(f"⏱️ Час виконання: {execution_time:.1f} секунд")
            print(f"📁 Усі результати збережені в: {self.results_dir}/")
            
        except Exception as e:
            print(f"❌ Помилка під час аналізу: {e}")
            raise

# Основна функція для запуску аналізу
if __name__ == "__main__":
    # Шлях до файлу з даними - змініть на ваш
    DATA_PATH = "b2b.csv"  # Замініть на правильний шлях
    
    # Створення та запуск аналізатора
    analyzer = MulticollinearityAnalyzer(
        data_path=DATA_PATH,
        random_state=42,
        correlation_threshold=0.8,  # Поріг для сильної кореляції
        vif_threshold=5.0  # Поріг VIF для мультиколінеарності
    )
    
    analyzer.run_analysis() 