Розглянемо, як застосувати Target Encoding зі згладжуванням для стовпців `salesperson` та `source` за допомогою бібліотеки `category_encoders` у Python. Згладжування допомагає зменшити вплив невеликих груп на закодовані значення та мінімізувати ризик витоку даних.

Ось як це зробити:

```python
import pandas as pd
from category_encoders import TargetEncoder

# Завантажуємо дані
data = pd.read_csv('cleanest_data.csv')

# Визначаємо цільову змінну та категоріальні стовпці
y = data['is_successful']
salesperson_col = 'salesperson'
source_col = 'source'

# --- Обробка стовпця 'salesperson' ---
print("Унікальні значення до обробки 'salesperson':", data[salesperson_col].nunique())

# Ініціалізуємо TargetEncoder зі згладжуванням (параметр 'smoothing')
encoder_salesperson = TargetEncoder(cols=[salesperson_col], smoothing=10) # Можна налаштувати значення smoothing

# Застосовуємо кодування
data['salesperson_encoded'] = encoder_salesperson.fit_transform(data[salesperson_col], y)

print("Унікальні значення після обробки 'salesperson':", data['salesperson_encoded'].nunique())
print("\nПерші 5 значень 'salesperson' та їх закодовані значення:")
print(data[[salesperson_col, 'salesperson_encoded']].head())

# --- Обробка стовпця 'source' ---
print("\nУнікальні значення до обробки 'source':", data[source_col].nunique())

# Групуємо рідкісні значення в 'source' перед кодуванням
threshold = 5 # Визначаємо поріг для рідкісних значень
value_counts = data[source_col].value_counts()
rare_values = value_counts[value_counts <= threshold].index
data[source_col] = data[source_col].replace(rare_values, 'Rare_Source')

print("Унікальні значення після групування рідкісних в 'source':", data[source_col].nunique())

# Ініціалізуємо TargetEncoder для 'source'
encoder_source = TargetEncoder(cols=[source_col], smoothing=10) # Можна налаштувати значення smoothing

# Застосовуємо кодування
data['source_encoded'] = encoder_source.fit_transform(data[source_col], y)

print("Унікальні значення після обробки 'source':", data['source_encoded'].nunique())
print("\nПерші 5 значень 'source' та їх закодовані значення:")
print(data[[source_col, 'source_encoded']].head())

# Тепер DataFrame 'data' містить нові закодовані стовпці 'salesperson_encoded' та 'source_encoded'
# які можна використовувати для подальшого навчання моделі.
```

**Пояснення коду:**

1.  **Імпорт бібліотек:** Імпортуємо `pandas` для роботи з даними та `TargetEncoder` з `category_encoders` для застосування target encoding.
2.  **Завантаження даних та визначення змінних:** Завантажуємо твій CSV-файл та визначаємо цільову змінну (`y`) та назви категоріальних стовпців.
3.  **Обробка 'salesperson':**
    * Виводимо кількість унікальних значень до кодування для інформації.
    * Ініціалізуємо `TargetEncoder` для стовпця `salesperson`. Параметр `cols=[salesperson_col]` вказує, який стовпець кодувати.
    * **Параметр `smoothing=10`:** Це ключовий параметр для згладжування. Він контролює, наскільки середнє значення цільової змінної для всієї вибірки впливає на закодоване значення для кожної категорії. Вище значення `smoothing` призводить до більшого впливу загального середнього, особливо для категорій з невеликою кількістю спостережень. Ти можеш експериментувати з цим значенням.
    * Застосовуємо `fit_transform()` для навчання енкодера на даних та одночасного перетворення стовпця. **Важливо:** `fit()` навчає енкодер на основі зв'язку між категорією та цільовою змінною, а `transform()` застосовує це навчання для кодування даних. Об'єднання в `fit_transform()` зручно для початкового кодування.
    * Виводимо кількість унікальних значень після кодування (воно має відповідати кількості унікальних значень цільової змінної, тобто 2 у нашому випадку, оскільки закодовані значення є середніми успішності).
    * Виводимо перші кілька рядків, щоб побачити оригінальні та закодовані значення.
4.  **Обробка 'source':**
    * Виводимо кількість унікальних значень до обробки.
    * **Групування рідкісних значень:** Оскільки стовпець `source` має багато рідкісних значень, ми спочатку групуємо їх в одну категорію `'Rare_Source'`. Це допомагає зменшити кількість категорій та зробити кодування більш стабільним. Ми визначаємо поріг (`threshold=5`), і всі значення, що зустрічаються рідше або рівно цьому порогу, замінюються на `'Rare_Source'`. Ти можеш налаштувати цей поріг.
    * Виводимо кількість унікальних значень після групування.
    * Ініціалізуємо `TargetEncoder` для стовпця `source` з аналогічним параметром `smoothing`.
    * Застосовуємо `fit_transform()` для кодування стовпця `source` (вже з урахуванням згрупованих рідкісних значень).
    * Виводимо кількість унікальних закодованих значень та перші кілька рядків з оригінальними та закодованими значеннями.

**Важливі моменти щодо запобігання витоку даних:**

* **Розділення даних:** Перед застосуванням target encoding, **обов'язково розділити дані на тренувальний та тестовий набори.**
* **Навчання тільки на тренувальних даних:** Енкодер (`fit()`) **слід навчати тільки на тренувальному наборі**.
* **Застосування на тестових даних:** Після навчання енкодера на тренувальних даних, використовуй **той самий навчений енкодер (`transform()`) для кодування тестового набору.** Ніколи не навчайте енкодер окремо на тестових даних, оскільки це призведе до витоку інформації про цільову змінну з тестового набору в закодовані значення тренувального набору.

Ось приклад правильного застосування з розділенням даних:

```python
from sklearn.model_selection import train_test_split

# Розділяємо дані на тренувальний та тестовий набори
X = data.drop('is_successful', axis=1)
y = data['is_successful']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Кодування 'salesperson' ---
encoder_salesperson = TargetEncoder(cols=[salesperson_col], smoothing=10)
X_train['salesperson_encoded'] = encoder_salesperson.fit_transform(X_train[salesperson_col], y_train)
X_test['salesperson_encoded'] = encoder_salesperson.transform(X_test[salesperson_col])

# --- Кодування 'source' ---
threshold = 5
value_counts_train = X_train[source_col].value_counts()
rare_values_train = value_counts_train[value_counts_train <= threshold].index
X_train[source_col] = X_train[source_col].replace(rare_values_train, 'Rare_Source')
X_test[source_col] = X_test[source_col].replace(rare_values_train, 'Rare_Source') # Застосовуємо ті самі рідкісні категорії, що були визначені на тренувальному наборі

encoder_source = TargetEncoder(cols=[source_col], smoothing=10)
X_train['source_encoded'] = encoder_source.fit_transform(X_train[source_col], y_train)
X_test['source_encoded'] = encoder_source.transform(X_test[source_col])

print("\nПерші 5 рядків тренувального набору з закодованими значеннями:")
print(X_train[[salesperson_col, 'salesperson_encoded', source_col, 'source_encoded']].head())

print("\nПерші 5 рядків тестового набору з закодованими значеннями:")
print(X_test[[salesperson_col, 'salesperson_encoded', source_col, 'source_encoded']].head())
```

Тепер ми маємо закодовані стовпці в окремих тренувальному та тестовому наборах, що допоможе уникнути витоку даних під час навчання моделі. Треба експериментувати зі значенням параметра `smoothing` та порогом для групування рідкісних значень, щоб знайти оптимальні налаштування даних та моделі.