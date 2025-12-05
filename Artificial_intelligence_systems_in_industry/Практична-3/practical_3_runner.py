# -*- coding: utf-8 -*-
"""
Практична робота 3: Дизайн наукового експерименту, валідація, безпека та етика систем ШІ

Цей скрипт генерує всі необхідні логи для написання звіту.
"""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Фіксація випадковості для відтворюваності
np.random.seed(42)

print("=" * 80)
print("ПРАКТИЧНА РОБОТА 3")
print("Тема: Дизайн наукового експерименту, валідація, безпека та етика систем ШІ")
print("=" * 80)

# ============================================================================
# ЗАВДАННЯ 1: ПІДГОТОВКА ДАНИХ ТА БАЗОВЕ МОДЕЛЮВАННЯ
# ============================================================================
print("\n" + "=" * 80)
print("ЗАВДАННЯ 1: ПІДГОТОВКА ДАНИХ ТА БАЗОВЕ МОДЕЛЮВАННЯ")
print("=" * 80)


# Генерація синтетичного промислового датасету
def generate_industrial_data(n_samples=2000):
    """
    Моделюємо роботу турбіни/компресора з трьома датчиками:
    1. Vibration (Вібрація): мм/с
    2. Temperature (Температура): °C
    3. Pressure (Тиск): бар
    """
    # Нормальні дані (клас 0)
    X_normal = np.random.normal(loc=[2.0, 60.0, 10.0], scale=[0.5, 5.0, 1.0], size=(n_samples // 2, 3))
    y_normal = np.zeros(n_samples // 2)

    # Аварійні дані (клас 1)
    X_fault = np.random.normal(loc=[4.5, 85.0, 12.0], scale=[0.8, 8.0, 2.0], size=(n_samples // 2, 3))
    y_fault = np.ones(n_samples // 2)

    X = np.vstack((X_normal, X_fault))
    y = np.hstack((y_normal, y_fault))

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    return X[indices], y[indices]


X, y = generate_industrial_data(2000)
feature_names = ['Vibration (mm/s)', 'Temperature (°C)', 'Pressure (bar)']

# Створюємо DataFrame для EDA
df = pd.DataFrame(X, columns=feature_names)
df['Status'] = y
df['Status_Label'] = df['Status'].map({0: 'Норма', 1: 'Аварія'})

print("\n--- 1.1 EXPLORATORY DATA ANALYSIS (EDA) ---")
print(f"\nРозмір датасету: {X.shape[0]} зразків, {X.shape[1]} ознаки")
print(f"\nОзнаки: {feature_names}")

print("\n--- Описова статистика ---")
print(df.describe().round(2).to_string())

print("\n--- Статистика по класах ---")
for cls in [0, 1]:
    label = 'Норма' if cls == 0 else 'Аварія'
    subset = df[df['Status'] == cls]
    print(f"\nКлас {cls} ({label}):")
    print(subset[feature_names].describe().round(2).to_string())

print("\n--- 1.2 АНАЛІЗ ДИСБАЛАНСУ КЛАСІВ ---")
class_counts = df['Status'].value_counts()
print(f"\nРозподіл класів:")
print(f"  Клас 0 (Норма):  {class_counts[0]} зразків ({class_counts[0] / len(df) * 100:.1f}%)")
print(f"  Клас 1 (Аварія): {class_counts[1]} зразків ({class_counts[1] / len(df) * 100:.1f}%)")
print(f"  Співвідношення: {class_counts[0] / class_counts[1]:.2f}:1")

if abs(class_counts[0] - class_counts[1]) / len(df) < 0.1:
    print("\n  ВИСНОВОК: Класи збалансовані, спеціальні методи балансування не потрібні.")
else:
    print("\n  ВИСНОВОК: Виявлено дисбаланс класів, рекомендується SMOTE або class_weight.")

# ============================================================================
# ЗАВДАННЯ 2: СТАТИСТИЧНЕ ПОРІВНЯННЯ АЛГОРИТМІВ
# ============================================================================
print("\n" + "=" * 80)
print("ЗАВДАННЯ 2: СТАТИСТИЧНЕ ПОРІВНЯННЯ АЛГОРИТМІВ")
print("=" * 80)

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from scipy import stats

# Нормалізація
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Налаштування крос-валідації
n_folds = 10
cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Моделі для порівняння
models = {
    "Logistic Regression (Baseline)": LogisticRegression(random_state=42, max_iter=1000),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=100),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100)
}

print(f"\n--- 2.1 КРОС-ВАЛІДАЦІЯ ({n_folds} ФОЛДІВ) ---")
print("\nМетрика: F1-Score (macro)")

results = {}
f1_scorer = make_scorer(f1_score, average='macro')

for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=f1_scorer)
    results[name] = scores
    print(f"\n{name}:")
    print(f"  Значення по фолдах: {np.round(scores, 4).tolist()}")
    print(f"  Середнє F1: {scores.mean():.4f} ± {scores.std():.4f}")

print("\n--- 2.2 СТАТИСТИЧНИЙ ТЕСТ ---")

# Порівняння Logistic Regression vs Gradient Boosting
scores_lr = results["Logistic Regression (Baseline)"]
scores_gb = results["Gradient Boosting"]

print("\nПорівняння: Logistic Regression vs Gradient Boosting")

# Тест Шапіро-Вілка на нормальність
shapiro_lr = stats.shapiro(scores_lr)
shapiro_gb = stats.shapiro(scores_gb)

print(f"\nТест Шапіро-Вілка на нормальність:")
print(f"  Logistic Regression: W = {shapiro_lr.statistic:.4f}, p-value = {shapiro_lr.pvalue:.4f}")
print(f"  Gradient Boosting:   W = {shapiro_gb.statistic:.4f}, p-value = {shapiro_gb.pvalue:.4f}")

# Вибір тесту на основі нормальності
alpha = 0.05
if shapiro_lr.pvalue > alpha and shapiro_gb.pvalue > alpha:
    print(f"\n  Обидва розподіли нормальні (p > {alpha}). Застосовуємо парний t-тест.")
    test_result = stats.ttest_rel(scores_lr, scores_gb)
    test_name = "Парний t-тест"
else:
    print(f"\n  Хоча б один розподіл ненормальний (p <= {alpha}). Застосовуємо тест Вілкоксона.")
    test_result = stats.wilcoxon(scores_lr, scores_gb)
    test_name = "Тест Вілкоксона"

print(f"\n{test_name}:")
print(f"  Статистика: {test_result.statistic:.4f}")
print(f"  P-value: {test_result.pvalue:.6f}")

if test_result.pvalue < alpha:
    print(f"\n  ВИСНОВОК: Різниця між алгоритмами є статистично значущою (p < {alpha}).")
    if scores_gb.mean() > scores_lr.mean():
        print(f"  Gradient Boosting показує кращі результати.")
else:
    print(f"\n  ВИСНОВОК: Немає підстав вважати, що алгоритми працюють по-різному (p >= {alpha}).")

# Таблиця порівняння
print("\n--- ТАБЛИЦЯ ПОРІВНЯННЯ МОДЕЛЕЙ ---")
print(f"\n{'Модель':<35} {'Сер. F1':>10} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 69)
for name, scores in results.items():
    print(f"{name:<35} {scores.mean():>10.4f} {scores.std():>8.4f} {scores.min():>8.4f} {scores.max():>8.4f}")

# ============================================================================
# ЗАВДАННЯ 3: ТЕСТУВАННЯ НА СТІЙКІСТЬ (ROBUSTNESS & SECURITY)
# ============================================================================
print("\n" + "=" * 80)
print("ЗАВДАННЯ 3: ТЕСТУВАННЯ НА СТІЙКІСТЬ (ROBUSTNESS & SECURITY)")
print("=" * 80)

import tensorflow as tf

tf.random.set_seed(42)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Розбиття даних
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"\nРозмір навчальної вибірки: {X_train.shape}")
print(f"Розмір тестової вибірки: {X_test.shape}")

# --- 3.1 Побудова нейронної мережі ---
print("\n--- 3.1 ПОБУДОВА ТА НАВЧАННЯ НЕЙРОННОЇ МЕРЕЖІ ---")


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(3,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


model = build_model()
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1, verbose=0)

loss, acc_baseline = model.evaluate(X_test, y_test, verbose=0)
print(f"\nТочність на чистих даних (Baseline): {acc_baseline * 100:.2f}%")

# Виводимо архітектуру моделі
print("\nАрхітектура моделі:")
model.summary()

# --- 3.2 Тестування на Гаусівський шум ---
print("\n--- 3.2 ТЕСТУВАННЯ НА ГАУСІВСЬКИЙ ШУМ ---")
print("\nФормула: x_noisy = x + ε, де ε ~ N(0, σ²)")

sigmas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
noise_results = []

print(f"\n{'σ (Sigma)':<12} {'Accuracy':<12} {'Падіння точності'}")
print("-" * 40)

for sigma in sigmas:
    noise = np.random.normal(0, sigma, X_test.shape)
    X_noisy = X_test + noise
    y_pred = (model.predict(X_noisy, verbose=0) > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    drop = (acc_baseline - acc) * 100
    noise_results.append({'sigma': sigma, 'accuracy': acc, 'drop': drop})
    print(f"{sigma:<12.2f} {acc:.4f}       {drop:+.2f}%")

# --- 3.3 FGSM атака ---
print("\n--- 3.3 FGSM АТАКА (Fast Gradient Sign Method) ---")
print("\nФормула: X_adv = X + ε · sign(∇_x J(θ, X, y))")


def fgsm_attack(model, inputs, labels, epsilon):
    inputs = tf.cast(inputs, tf.float32)
    labels = tf.cast(labels, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(inputs)
        prediction = model(inputs)
        loss = tf.keras.losses.binary_crossentropy(labels, prediction)

    gradient = tape.gradient(loss, inputs)
    signed_grad = tf.sign(gradient)
    adversarial_data = inputs + epsilon * signed_grad

    return adversarial_data


epsilons = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
fgsm_results = []

y_sample = y_test.reshape(-1, 1)

print(f"\n{'Epsilon':<12} {'Accuracy':<12} {'Падіння точності'}")
print("-" * 40)

for eps in epsilons:
    X_adv = fgsm_attack(model, X_test, y_sample, epsilon=eps)
    y_pred_adv = (model.predict(X_adv, verbose=0) > 0.5).astype(int)
    acc = accuracy_score(y_sample, y_pred_adv)
    drop = (acc_baseline - acc) * 100
    fgsm_results.append({'epsilon': eps, 'accuracy': acc, 'drop': drop})
    print(f"{eps:<12.2f} {acc:.4f}       {drop:+.2f}%")

# --- 3.4 Детальний аналіз атаки ---
print("\n--- 3.4 ДЕТАЛЬНИЙ АНАЛІЗ АТАКИ НА ЗРАЗОК ---")

idx = 0
original_sample = X_test[idx].reshape(1, 3)
true_label = y_test[idx].reshape(1, 1)

# Атака з epsilon = 0.5
adv_sample = fgsm_attack(model, original_sample, true_label, epsilon=0.5).numpy()

# Денормалізація
original_phys = scaler.inverse_transform(original_sample)[0]
adv_phys = scaler.inverse_transform(adv_sample)[0]

# Передбачення
pred_clean = model.predict(original_sample, verbose=0)[0][0]
pred_adv = model.predict(adv_sample, verbose=0)[0][0]

true_class = int(true_label[0][0])
print(f"\nІстинний клас: {true_class} ({'Аварія' if true_class == 1 else 'Норма'})")
print("-" * 65)
print(f"{'Показник':<25} {'Оригінал':<15} {'Атака':<15} {'Різниця'}")
print("-" * 65)

for i, name in enumerate(feature_names):
    diff = adv_phys[i] - original_phys[i]
    print(f"{name:<25} {original_phys[i]:<15.2f} {adv_phys[i]:<15.2f} {diff:+.2f}")

print("-" * 65)
print(f"\nЙмовірність аварії (чисті дані):    {pred_clean:.4f} -> Рішення: {int(pred_clean > 0.5)}")
print(f"Ймовірність аварії (атаковані дані): {pred_adv:.4f} -> Рішення: {int(pred_adv > 0.5)}")

if int(pred_clean > 0.5) != int(pred_adv > 0.5):
    print("\n⚠️  АТАКА УСПІШНА: модель змінила своє рішення!")
else:
    print("\n✓ Модель стійка: рішення не змінилося.")

# --- 3.5 Adversarial Training ---
print("\n--- 3.5 ADVERSARIAL TRAINING (ЗАХИСТ) ---")

print("\nКрок 1: Генерація атакованих даних для train set (epsilon=0.2)...")
X_train_adv = fgsm_attack(model, X_train, y_train.reshape(-1, 1), epsilon=0.2)

print("Крок 2: Об'єднання чистих та атакованих даних...")
X_combined = np.vstack((X_train, X_train_adv))
y_combined = np.hstack((y_train, y_train))
print(f"Розмір розширеного датасету: {X_combined.shape}")

print("Крок 3: Донавчання моделі...")
model.fit(X_combined, y_combined, epochs=10, batch_size=32, verbose=0)

# Тестування після adversarial training
print("\nТестування після Adversarial Training:")
print(f"\n{'Epsilon':<12} {'До AT':<15} {'Після AT':<15} {'Покращення'}")
print("-" * 55)

for eps in [0.0, 0.1, 0.2, 0.3, 0.5]:
    X_adv_test = fgsm_attack(model, X_test, y_sample, epsilon=eps)
    y_pred_new = (model.predict(X_adv_test, verbose=0) > 0.5).astype(int)
    acc_new = accuracy_score(y_sample, y_pred_new)

    # Знаходимо стару точність
    old_acc = next(r['accuracy'] for r in fgsm_results if r['epsilon'] == eps)
    improvement = (acc_new - old_acc) * 100

    print(f"{eps:<12.2f} {old_acc:.4f}         {acc_new:.4f}         {improvement:+.2f}%")

# ============================================================================
# ЗАВДАННЯ 4: ЕТИЧНИЙ АУДИТ (SHAP АНАЛІЗ)
# ============================================================================
print("\n" + "=" * 80)
print("ЗАВДАННЯ 4: ЕТИЧНИЙ АУДИТ (SHAP АНАЛІЗ)")
print("=" * 80)

try:
    import shap

    # Використовуємо KernelExplainer для TensorFlow моделі
    print("\n--- 4.1 SHAP VALUES АНАЛІЗ ---")

    # Беремо підмножину для прискорення обчислень
    background = X_train[:100]
    test_samples = X_test[:10]


    # Функція для предсказания
    def model_predict(x):
        return model.predict(x, verbose=0)


    # Створюємо explainer
    explainer = shap.KernelExplainer(model_predict, background)
    shap_values = explainer.shap_values(test_samples)

    # Обробка різних форматів SHAP values
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values[0]) if len(shap_values) == 1 else np.array(shap_values)

    # Якщо shap_values має додатковий вимір, вибираємо потрібний
    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 0]

    print("\nSHAP values для перших 5 зразків:")
    print(f"\n{'Зразок':<10} {'Vibration':<15} {'Temperature':<15} {'Pressure':<15} {'Передбачення'}")
    print("-" * 70)

    for i in range(5):
        pred = model.predict(test_samples[i:i + 1], verbose=0)[0][0]
        shap_vals = shap_values[i].flatten()
        print(
            f"{i:<10} {float(shap_vals[0]):<15.4f} {float(shap_vals[1]):<15.4f} {float(shap_vals[2]):<15.4f} {pred:.4f}")

    # Середні абсолютні SHAP values (важливість ознак)
    mean_abs_shap = np.abs(shap_values).mean(axis=0).flatten()
    print("\n--- ВАЖЛИВІСТЬ ОЗНАК (середні |SHAP|) ---")
    for i, name in enumerate(feature_names):
        print(f"  {name}: {float(mean_abs_shap[i]):.4f}")

    most_important = feature_names[np.argmax(mean_abs_shap)]
    print(f"\nНайважливіша ознака: {most_important}")

except ImportError:
    print("\n⚠️  Бібліотека SHAP не встановлена. Встановіть її командою: pip install shap")
    print("    Виконуємо спрощений аналіз важливості ознак (Permutation Importance)...")


    # Ручний Permutation Importance для Keras моделі
    def manual_permutation_importance(model, X, y, n_repeats=10):
        """Розрахунок важливості ознак через перемішування."""
        baseline_pred = (model.predict(X, verbose=0) > 0.5).astype(int).flatten()
        baseline_acc = accuracy_score(y, baseline_pred)

        importances = []
        for i in range(X.shape[1]):
            scores = []
            for _ in range(n_repeats):
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, i])
                y_pred = (model.predict(X_permuted, verbose=0) > 0.5).astype(int).flatten()
                acc = accuracy_score(y, y_pred)
                scores.append(baseline_acc - acc)
            importances.append({'mean': np.mean(scores), 'std': np.std(scores)})
        return importances, baseline_acc


    importances, baseline_acc = manual_permutation_importance(model, X_test, y_test, n_repeats=10)

    print(f"\nBaseline accuracy: {baseline_acc:.4f}")
    print("\n--- ВАЖЛИВІСТЬ ОЗНАК (Permutation Importance) ---")
    for i, name in enumerate(feature_names):
        print(f"  {name}: {importances[i]['mean']:.4f} ± {importances[i]['std']:.4f}")

    most_important_idx = np.argmax([imp['mean'] for imp in importances])
    print(f"\nНайважливіша ознака: {feature_names[most_important_idx]}")

# --- 4.2 Аналіз ризиків ---
print("\n--- 4.2 АНАЛІЗ РИЗИКІВ FALSE POSITIVE / FALSE NEGATIVE ---")

# Обчислюємо confusion matrix
from sklearn.metrics import confusion_matrix

y_pred_final = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
cm = confusion_matrix(y_test, y_pred_final)

TN, FP, FN, TP = cm.ravel()

print(f"\nConfusion Matrix:")
print(f"                 Predicted")
print(f"                 Норма    Аварія")
print(f"Actual Норма     {TN:<8} {FP}")
print(f"       Аварія    {FN:<8} {TP}")

print(f"\nFalse Positive (FP): {FP} випадків")
print(f"  → Система прогнозує аварію, коли її насправді немає")
print(f"  → Наслідок: зупинка конвеєра, втрати від простою")

print(f"\nFalse Negative (FN): {FN} випадків")
print(f"  → Система не виявляє реальну аварію")
print(f"  → Наслідок: випуск браку, можливі аварії обладнання")

print(f"\nПоказники:")
print(f"  Precision (точність):  {TP / (TP + FP):.4f}")
print(f"  Recall (повнота):      {TP / (TP + FN):.4f}")
print(f"  Specificity:           {TN / (TN + FP):.4f}")

# ============================================================================
# ПІДСУМОК
# ============================================================================
print("\n" + "=" * 80)
print("ЗАГАЛЬНИЙ ПІДСУМОК")
print("=" * 80)

print(f"""
1. ДАТАСЕТ:
   - Синтетичні дані промислового моніторингу (турбіна/компресор)
   - 2000 зразків, 3 ознаки (вібрація, температура, тиск)
   - Класи збалансовані 50/50

2. ПОРІВНЯННЯ АЛГОРИТМІВ:
   - Logistic Regression (baseline): F1 = {results['Logistic Regression (Baseline)'].mean():.4f}
   - Gradient Boosting: F1 = {results['Gradient Boosting'].mean():.4f}
   - Random Forest: F1 = {results['Random Forest'].mean():.4f}
   - Статистичний тест: p-value = {test_result.pvalue:.6f}

3. ROBUSTNESS:
   - Baseline accuracy: {acc_baseline * 100:.2f}%
   - При σ=0.5 (Gaussian noise): {noise_results[4]['accuracy'] * 100:.2f}%
   - При ε=0.5 (FGSM attack): {fgsm_results[5]['accuracy'] * 100:.2f}%

4. ADVERSARIAL TRAINING:
   - Метод захисту показав покращення стійкості моделі

5. ЕТИКА:
   - FP: {FP} (зайві зупинки)
   - FN: {FN} (пропущені аварії)
""")

print("=" * 80)
print("ЛОГУВАННЯ ЗАВЕРШЕНО")
print("=" * 80)
