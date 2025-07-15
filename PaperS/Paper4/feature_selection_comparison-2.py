# -*- coding: utf-8 -*-
"""
Скрипт для порівняння методів відбору ознак:
- XGBoost importance (gain/cover/weight)
- RFE (з XGBoost)
- BoostARoota
- SHAP

Перед запуском встановіть додаткові бібліотеки:
pip install BoostARoota shap
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE
import warnings
import os
import time

warnings.filterwarnings('ignore')

# BoostARoota та SHAP імпортуємо з перевіркою
try:
    from boostaroota import BoostARoota  # виправлено регістр
except ImportError:
    BoostARoota = None
try:
    import shap
except ImportError:
    shap = None

# === Функції для обробки циклічних ознак ===
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

class FeatureSelectionComparison:
    def __init__(self, data_path, random_state=42):
        self.data_path = data_path
        self.random_state = random_state
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
        self.results_dir = f"feature_selection_comparison_results_2"
        os.makedirs(self.results_dir, exist_ok=True)

    def load_and_prepare_data(self):
        print(f"\n🔄 === ЗАВАНТАЖЕННЯ ТА ПІДГОТОВКА ДАНИХ ===")
        data = pd.read_csv(self.data_path)
        if 'is_successful' not in data.columns:
            raise ValueError("❌ Цільова змінна 'is_successful' не знайдена!")
        X = data.drop('is_successful', axis=1)
        y = data['is_successful']
        X = X.select_dtypes(exclude=['object'])
        X_processed = preprocess_features_for_analysis(X)
        self.feature_names = list(X_processed.columns)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_processed, y, test_size=0.3, random_state=self.random_state, stratify=y
        )
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print(f"✅ Дані підготовлені для аналізу. Кількість ознак: {len(self.feature_names)}")

    def xgboost_importance(self):
        print(f"\n🔄 === XGBoost Feature Importance ===")
        model = XGBClassifier(**self.optimal_params)
        model.fit(self.X_train_scaled, self.y_train)
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        importance_df.to_csv(f'{self.results_dir}/xgboost_importance.csv', index=False)
        self.xgb_importance_df = importance_df
        print(importance_df)

    def rfe_selection(self, n_features_to_select=10):
        pass  # RFE видалено

    def shap_importance(self):
        if shap is None:
            print("❌ SHAP не встановлено! Встановіть через pip install shap")
            return
        print(f"\n🔄 === SHAP Importance ===")
        model = XGBClassifier(**self.optimal_params)
        model.fit(self.X_train_scaled, self.y_train)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(self.X_train_scaled)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
        shap_df.to_csv(f'{self.results_dir}/shap_importance.csv', index=False)
        self.shap_df = shap_df
        print(shap_df)

    def greedy_feature_ranking(self):
        print(f"\n🔄 === ЖАДІБНИЙ ВІДБІР ОЗНАК ===")
        remaining = set(range(len(self.feature_names)))
        selected = []
        greedy_scores = []
        for step in range(len(self.feature_names)):
            best_score = -np.inf
            best_idx = None
            for idx in remaining:
                current_features = selected + [idx]
                X_train_sel = self.X_train_scaled[:, current_features]
                X_test_sel = self.X_test_scaled[:, current_features]
                model = XGBClassifier(**self.optimal_params)
                model.fit(X_train_sel, self.y_train)
                y_pred_proba = model.predict_proba(X_test_sel)[:, 1]
                score = average_precision_score(self.y_test, y_pred_proba)
                if score > best_score:
                    best_score = score
                    best_idx = idx
            selected.append(best_idx)
            remaining.remove(best_idx)
            greedy_scores.append({'feature': self.feature_names[best_idx], 'score': best_score, 'step': step+1})
            print(f"Крок {step+1}: {self.feature_names[best_idx]} (AUC-PR: {best_score:.4f})")
        self.greedy_ranking_df = pd.DataFrame(greedy_scores)
        self.greedy_ranking_df.to_csv(f'{self.results_dir}/greedy_ranking.csv', index=False)

    def compare_and_visualize(self):
        print(f"\n🔄 === Порівняння та візуалізація ===")
        df = pd.DataFrame({'feature': self.feature_names})
        if hasattr(self, 'xgb_importance_df'):
            df = df.merge(self.xgb_importance_df[['feature', 'importance']], on='feature', how='left')
            df['xgb_rank'] = df['importance'].rank(ascending=False, method='min')
        if hasattr(self, 'shap_df'):
            df = df.merge(self.shap_df[['feature', 'mean_abs_shap']], on='feature', how='left')
            df['shap_rank'] = df['mean_abs_shap'].rank(ascending=False, method='min')
        if hasattr(self, 'greedy_ranking_df'):
            greedy_map = {row['feature']: i+1 for i, row in self.greedy_ranking_df.iterrows()}
            df['greedy_rank'] = df['feature'].map(greedy_map)
        # Додаємо середній ранг
        rank_cols = ['xgb_rank', 'shap_rank', 'greedy_rank']
        df['mean_rank'] = df[rank_cols].mean(axis=1)
        df = df.sort_values('mean_rank', ascending=True)
        df.to_csv(f'{self.results_dir}/all_methods_comparison.csv', index=False)
        # HEATMAP: ознаки по вертикалі, методи по горизонталі + середній ранг
        heatmap_df = df.set_index('feature')[rank_cols + ['mean_rank']]
        plt.figure(figsize=(8, max(8, len(self.feature_names)*0.4)))
        sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='YlGnBu', cbar_kws={'label': 'Ранг ознаки'})
        plt.ylabel('Ознаки (відсортовано за середнім рангом)')
        plt.xlabel('Метод')
        plt.title('Теплова карта рангів ознак за різними методами та середній ранг')
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/feature_ranking_heatmap.png', dpi=300)
        plt.close()
        # Barplot для кожного методу (залишаю для зручності)
        plt.figure(figsize=(14, 7))
        if hasattr(self, 'xgb_importance_df'):
            sns.barplot(x='importance', y='feature', data=self.xgb_importance_df.head(15), color='skyblue', label='XGBoost')
            plt.title('XGBoost Feature Importance (Top 15)')
            plt.tight_layout()
            plt.savefig(f'{self.results_dir}/xgboost_importance_barplot.png', dpi=300)
            plt.close()
        if hasattr(self, 'shap_df'):
            plt.figure(figsize=(14, 7))
            sns.barplot(x='mean_abs_shap', y='feature', data=self.shap_df.head(15), color='green', label='SHAP')
            plt.title('SHAP Feature Importance (Top 15)')
            plt.tight_layout()
            plt.savefig(f'{self.results_dir}/shap_importance_barplot.png', dpi=300)
            plt.close()
        if hasattr(self, 'xgb_importance_df') and hasattr(self, 'shap_df'):
            merged = pd.merge(self.xgb_importance_df.reset_index().rename(columns={'index': 'xgb_rank'}),
                              self.shap_df.reset_index().rename(columns={'index': 'shap_rank'}),
                              on='feature', how='inner')
            plt.figure(figsize=(8, 8))
            plt.scatter(merged['xgb_rank']+1, merged['shap_rank']+1, alpha=0.7)
            for _, row in merged.iterrows():
                dx, dy = 0.2, 0.2
                plt.text(row['xgb_rank']+1+dx, row['shap_rank']+1+dy, row['feature'], fontsize=7, alpha=0.7)
            plt.xlabel('XGBoost Rank')
            plt.ylabel('SHAP Rank')
            plt.title('Порівняння позицій ознак: XGBoost vs SHAP', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{self.results_dir}/xgb_vs_shap_scatter.png', dpi=300)
            plt.close()
        print(f'✅ Всі результати та графіки збережено у {self.results_dir}/')

    def run(self):
        self.load_and_prepare_data()
        self.xgboost_importance()
        self.shap_importance()
        self.greedy_feature_ranking()
        self.compare_and_visualize()

if __name__ == '__main__':
    analyzer = FeatureSelectionComparison(
        data_path=r'D:\PROJECTs\MY\ScientificCalculations\SC\ScientificCalculations\PaperS\Paper4\preprocessed_data2.csv',
        random_state=42
    )
    analyzer.run() 