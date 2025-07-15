# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import warnings
import os
import time

warnings.filterwarnings('ignore')

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

class GreedyVsImportanceAnalyzer:
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
        self.data = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.results_dir = f"greedy_vs_importance_results"
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"\n🔬 === ПОРІВНЯННЯ ЖАДІБНОГО ВІДБОРУ ТА XGBOOST IMPORTANCE ===")
        print(f"📁 Результати будуть збережені в: {self.results_dir}/")

    def load_and_prepare_data(self):
        print(f"\n🔄 === ЗАВАНТАЖЕННЯ ТА ПІДГОТОВКА ДАНИХ ===")
        self.data = pd.read_csv(self.data_path)
        if 'is_successful' not in self.data.columns:
            raise ValueError("❌ Цільова змінна 'is_successful' не знайдена!")
        X = self.data.drop('is_successful', axis=1)
        y = self.data['is_successful']
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

    def get_xgboost_importance(self):
        print(f"\n🔄 === ВИЗНАЧЕННЯ ВАЖЛИВОСТІ ОЗНАК XGBOOST ===")
        model = XGBClassifier(**self.optimal_params)
        model.fit(self.X_train_scaled, self.y_train)
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        self.xgb_importance_df = importance_df
        print(importance_df.head(24))
        importance_df.to_csv(f'{self.results_dir}/xgboost_importance.csv', index=False)

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
        print(f"\n🔄 === ВІЗУАЛІЗАЦІЯ ТА ПОРІВНЯННЯ РЕЙТИНГІВ ===")
        # Barplot: XGBoost importance
        plt.figure(figsize=(12, 6))
        sns.barplot(x='importance', y='feature', data=self.xgb_importance_df.head(20), color='skyblue')
        plt.title('XGBoost Feature Importance (Top 20)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/xgboost_importance_barplot.png', dpi=300)
        plt.close()
        # Barplot: Greedy ranking
        plt.figure(figsize=(12, 6))
        sns.barplot(x='score', y='feature', data=self.greedy_ranking_df, color='lightgreen')
        plt.title('Greedy Feature Ranking (AUC-PR by step)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/greedy_ranking_barplot.png', dpi=300)
        plt.close()
        # Scatter plot: Порівняння позицій
        merged = pd.merge(
            self.xgb_importance_df.reset_index().rename(columns={'index': 'xgb_rank'}),
            self.greedy_ranking_df.reset_index().rename(columns={'index': 'greedy_rank'}),
            on='feature', how='inner'
        )
        plt.figure(figsize=(8, 8))
        plt.scatter(merged['xgb_rank']+1, merged['greedy_rank']+1, alpha=0.7)
        for _, row in merged.iterrows():
            plt.text(row['xgb_rank']+1, row['greedy_rank']+1, row['feature'], fontsize=7, alpha=0.7)
        plt.xlabel('XGBoost Rank')
        plt.ylabel('Greedy Rank')
        plt.title('Порівняння позицій ознак у рейтингах', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/ranking_scatter.png', dpi=300)
        plt.close()
        # Кореляція
        corr = merged[['xgb_rank', 'greedy_rank']].corr().iloc[0,1]
        with open(f'{self.results_dir}/ranking_correlation.txt', 'w', encoding='utf-8') as f:
            f.write(f'Кореляція між рейтингами (Spearman): {corr:.4f}\n')
        print(f'Кореляція між рейтингами (Spearman): {corr:.4f}')

    def run(self):
        self.load_and_prepare_data()
        self.get_xgboost_importance()
        self.greedy_feature_ranking()
        self.compare_and_visualize()
        print(f"\n✅ Аналіз завершено! Всі результати збережено у {self.results_dir}/")

if __name__ == '__main__':
    analyzer = GreedyVsImportanceAnalyzer(
        data_path=r'D:\PROJECTs\MY\ScientificCalculations\SC\ScientificCalculations\PaperS\Paper4\preprocessed_data2.csv',
        random_state=42
    )
    analyzer.run() 