#!/usr/bin/env python
# coding: utf-8

# # Контрольна робота №1
# ## Аналіз даних з використанням бібліотеки Pandas
# 
# **Мета роботи:** Продемонструвати використання мови програмування Python та бібліотеки Pandas для обробки, отримання описової статистики та візуалізації даних.
# 
# **Набір даних:** Wine Quality (Red Wine) - UCI Machine Learning Repository
# 
# ---

# ## 1. Імпорт необхідних бібліотек
# 
# Для виконання аналізу даних необхідно імпортувати наступні бібліотеки:
# 
# - **Pandas** ($\textit{pandas}$) — основна бібліотека для маніпуляції та аналізу даних
# - **NumPy** ($\textit{numpy}$) — бібліотека для чисельних обчислень
# - **Matplotlib** ($\textit{matplotlib}$) — бібліотека для створення статичних візуалізацій
# - **Seaborn** ($\textit{seaborn}$) — високорівнева бібліотека для статистичної візуалізації

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Налаштування відображення
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("Бібліотеки успішно імпортовано!")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")


# ---
# ## 2. Завантаження набору даних
# 
# Для завантаження даних з CSV-файлу використовується функція `pd.read_csv()`. 
# 
# Набір даних Wine Quality містить фізико-хімічні властивості червоних вин та їх якість, оцінену експертами.
# 
# **Формат даних:** CSV з роздільником `;` (крапка з комою)

# In[2]:


# Завантаження набору даних
data = pd.read_csv('winequality-red.csv', sep=';')

print("Дані успішно завантажено!")
print(f"Розмірність датасету: {data.shape[0]} рядків × {data.shape[1]} стовпців")


# ---
# ## 3. Огляд вмісту та описова статистика
# 
# ### 3.1 Перегляд перших та останніх записів
# 
# Методи `head(n)` та `tail(n)` дозволяють переглянути перші та останні $n$ записів DataFrame відповідно.

# In[3]:


# Перегляд перших 5 записів
print("=" * 80)
print("ПЕРШІ 5 ЗАПИСІВ ДАТАСЕТУ:")
print("=" * 80)
data.head()


# In[4]:


# Перегляд останніх 5 записів
print("=" * 80)
print("ОСТАННІ 5 ЗАПИСІВ ДАТАСЕТУ:")
print("=" * 80)
data.tail()


# ### 3.2 Інформація про структуру даних
# 
# Метод `info()` надає загальну інформацію про DataFrame:
# - Кількість записів та стовпців
# - Типи даних кожного стовпця
# - Кількість ненульових значень
# - Використання пам'яті

# In[5]:


# Інформація про структуру датасету
print("=" * 80)
print("СТРУКТУРА ДАТАСЕТУ:")
print("=" * 80)
data.info()


# ### 3.3 Описова статистика
# 
# Метод `describe()` обчислює основні статистичні показники для числових стовпців:
# 
# | Показник | Формула | Опис |
# |----------|---------|------|
# | count | $n$ | Кількість спостережень |
# | mean | $\bar{x} = \frac{1}{n}\sum_{i=1}^{n}x_i$ | Середнє арифметичне |
# | std | $s = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2}$ | Стандартне відхилення |
# | min | $\min(x_1, x_2, ..., x_n)$ | Мінімальне значення |
# | 25% | $Q_1$ | Перший квартиль |
# | 50% | $Q_2$ (медіана) | Другий квартиль |
# | 75% | $Q_3$ | Третій квартиль |
# | max | $\max(x_1, x_2, ..., x_n)$ | Максимальне значення |

# In[6]:


# Описова статистика
print("=" * 80)
print("ОПИСОВА СТАТИСТИКА:")
print("=" * 80)
data.describe()


# In[7]:


# Додаткова статистика з округленням
print("\nОписова статистика (округлена):")
data.describe().round(3)


# ---
# ## 4. Обробка пропусків
# 
# Перевірка наявності пропущених значень є важливим етапом підготовки даних. Метод `isnull().sum()` підраховує кількість пропусків у кожному стовпці.
# 
# Для заповнення пропущених значень використовується метод `fillna()`, який може приймати:
# - Константне значення
# - Середнє ($\bar{x}$), медіану ($\tilde{x}$) або моду стовпця

# In[8]:


# Перевірка пропусків
print("=" * 80)
print("ПЕРЕВІРКА ПРОПУЩЕНИХ ЗНАЧЕНЬ:")
print("=" * 80)
missing_values = data.isnull().sum()
print(missing_values)
print(f"\nЗагальна кількість пропусків: {missing_values.sum()}")


# In[9]:


# Демонстрація заповнення пропусків (створення копії з штучними пропусками)
data_with_na = data.copy()

# Штучно створюємо пропуски для демонстрації
np.random.seed(42)
mask = np.random.random(len(data_with_na)) < 0.05
data_with_na.loc[mask, 'alcohol'] = np.nan

print(f"Кількість пропусків у стовпці 'alcohol': {data_with_na['alcohol'].isnull().sum()}")

# Заповнення пропусків середнім значенням
mean_alcohol = data_with_na['alcohol'].mean()
data_with_na['alcohol'] = data_with_na['alcohol'].fillna(mean_alcohol)

print(f"Після заповнення середнім ({mean_alcohol:.2f}): {data_with_na['alcohol'].isnull().sum()} пропусків")


# ---
# ## 5. Фільтрація рядків
# 
# Фільтрація даних здійснюється за допомогою логічних (булевих) індексів. Умова фільтрації створює масив булевих значень, який використовується для вибору відповідних рядків.
# 
# Математично це можна представити як:
# 
# $$
# \text{filtered\_data} = \{x_i : \text{condition}(x_i) = \text{True}\}
# $$

# In[10]:


# Фільтрація: вина з якістю >= 7 (високоякісні вина)
print("=" * 80)
print("ФІЛЬТРАЦІЯ: ВИСОКОЯКІСНІ ВИНА (quality >= 7)")
print("=" * 80)

high_quality_wines = data[data['quality'] >= 7]
print(f"Кількість високоякісних вин: {len(high_quality_wines)} ({len(high_quality_wines)/len(data)*100:.2f}%)")
high_quality_wines.head()


# In[11]:


# Фільтрація з кількома умовами (використання & та |)
print("=" * 80)
print("ФІЛЬТРАЦІЯ: ВИНА З ВИСОКИМ ВМІСТОМ АЛКОГОЛЮ ТА НИЗЬКОЮ КИСЛОТНІСТЮ")
print("=" * 80)

filtered_wines = data[(data['alcohol'] > 12) & (data['pH'] > 3.5)]
print(f"Кількість вин за умовами (alcohol > 12) AND (pH > 3.5): {len(filtered_wines)}")
filtered_wines.head()


# ---
# ## 6. Групування та агрегування
# 
# Метод `groupby()` дозволяє групувати дані за значеннями одного або кількох стовпців, після чого до кожної групи можна застосувати агрегатні функції:
# 
# - `mean()` — середнє арифметичне: $\bar{x} = \frac{1}{n}\sum_{i=1}^{n}x_i$
# - `sum()` — сума: $\sum_{i=1}^{n}x_i$
# - `count()` — кількість елементів: $n$
# - `min()`, `max()` — мінімум та максимум

# In[12]:


# Групування за якістю та обчислення середніх значень
print("=" * 80)
print("ГРУПУВАННЯ ЗА ЯКІСТЮ ВИНА:")
print("=" * 80)

quality_groups = data.groupby('quality').mean()
quality_groups.round(3)


# In[13]:


# Кількість вин кожної якості
print("\nРозподіл вин за якістю:")
quality_counts = data.groupby('quality').size()
print(quality_counts)


# ---
# ## 7. Сортування даних
# 
# Метод `sort_values()` дозволяє сортувати DataFrame за значеннями одного або кількох стовпців:
# 
# - `ascending=True` — сортування за зростанням
# - `ascending=False` — сортування за спаданням

# In[14]:


# Сортування за вмістом алкоголю (за спаданням)
print("=" * 80)
print("ТОП-10 ВИН ЗА ВМІСТОМ АЛКОГОЛЮ:")
print("=" * 80)

sorted_by_alcohol = data.sort_values('alcohol', ascending=False)
sorted_by_alcohol.head(10)


# In[15]:


# Сортування за кількома стовпцями
print("=" * 80)
print("СОРТУВАННЯ ЗА ЯКІСТЮ ТА АЛКОГОЛЕМ:")
print("=" * 80)

sorted_multi = data.sort_values(['quality', 'alcohol'], ascending=[False, False])
sorted_multi.head(10)


# ---
# ## 8. Зміна типів даних
# 
# Метод `astype()` дозволяє змінювати тип даних стовпця. Це важливо для:
# - Оптимізації використання пам'яті
# - Коректної обробки категоріальних змінних
# - Підготовки даних для моделювання

# In[16]:


# Перевірка поточних типів даних
print("=" * 80)
print("ПОТОЧНІ ТИПИ ДАНИХ:")
print("=" * 80)
print(data.dtypes)


# In[17]:


# Зміна типу даних стовпця 'quality' на категоріальний
data_copy = data.copy()
data_copy['quality'] = data_copy['quality'].astype('category')

print("=" * 80)
print("ПІСЛЯ ЗМІНИ ТИПУ 'quality' НА КАТЕГОРІАЛЬНИЙ:")
print("=" * 80)
print(f"Тип стовпця 'quality': {data_copy['quality'].dtype}")
print(f"Категорії: {data_copy['quality'].cat.categories.tolist()}")


# In[18]:


# Конвертація float64 в float32 для економії пам'яті
print("\nПорівняння використання пам'яті:")
print(f"До конвертації: {data.memory_usage(deep=True).sum() / 1024:.2f} KB")

data_optimized = data.copy()
for col in data_optimized.select_dtypes(include=['float64']).columns:
    data_optimized[col] = data_optimized[col].astype('float32')
    
print(f"Після конвертації: {data_optimized.memory_usage(deep=True).sum() / 1024:.2f} KB")


# ---
# ## 9. Додавання нового стовпця
# 
# Нові стовпці можуть бути створені шляхом арифметичних операцій над існуючими стовпцями. Pandas автоматично виконує поелементні операції.
# 
# Наприклад, для обчислення загальної кислотності:
# 
# $$
# \text{total\_acidity} = \text{fixed\_acidity} + \text{volatile\_acidity} + \text{citric\_acid}
# $$

# In[19]:


# Додавання нового стовпця - загальна кислотність
print("=" * 80)
print("ДОДАВАННЯ НОВОГО СТОВПЦЯ:")
print("=" * 80)

data['total_acidity'] = data['fixed acidity'] + data['volatile acidity'] + data['citric acid']

print("Новий стовпець 'total_acidity' додано!")
data[['fixed acidity', 'volatile acidity', 'citric acid', 'total_acidity']].head(10)


# In[20]:


# Додавання стовпця з категорією якості
def categorize_quality(quality):
    if quality <= 4:
        return 'Низька'
    elif quality <= 6:
        return 'Середня'
    else:
        return 'Висока'

data['quality_category'] = data['quality'].apply(categorize_quality)
print("\nРозподіл за категоріями якості:")
print(data['quality_category'].value_counts())


# ---
# ## 10. Збереження даних
# 
# Метод `to_csv()` зберігає DataFrame у файл формату CSV. Основні параметри:
# - `index=False` — не зберігати індекс
# - `sep=','` — роздільник (за замовчуванням кома)
# - `encoding='utf-8'` — кодування файлу

# In[21]:


# Збереження обробленого датасету
print("=" * 80)
print("ЗБЕРЕЖЕННЯ ДАНИХ:")
print("=" * 80)

output_file = 'winequality-red-processed.csv'
data.to_csv(output_file, index=False, sep=';', encoding='utf-8')

print(f"Дані успішно збережено у файл: {output_file}")
print(f"Кількість стовпців: {len(data.columns)}")
print(f"Кількість рядків: {len(data)}")


# ---
# ## 11. Візуалізація даних
# 
# Візуалізація є ключовим інструментом для розвідувального аналізу даних (EDA — Exploratory Data Analysis). Бібліотеки **Matplotlib** та **Seaborn** надають широкі можливості для створення інформативних графіків.

# ### 11.1 Гістограма розподілу якості вина

# In[22]:


# Гістограма розподілу якості
plt.figure(figsize=(10, 6))
plt.hist(data['quality'], bins=range(3, 10), edgecolor='black', alpha=0.7, color='steelblue')
plt.xlabel('Якість вина', fontsize=12)
plt.ylabel('Кількість зразків', fontsize=12)
plt.title('Розподіл якості червоного вина', fontsize=14)
plt.xticks(range(3, 9))
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.show()


# ### 11.2 Розподіл вмісту алкоголю

# In[23]:


# Гістограма з кривою густини для алкоголю
plt.figure(figsize=(10, 6))
sns.histplot(data['alcohol'], kde=True, bins=30, color='crimson')
plt.xlabel('Вміст алкоголю (%)', fontsize=12)
plt.ylabel('Частота', fontsize=12)
plt.title('Розподіл вмісту алкоголю у винах', fontsize=14)
plt.axvline(data['alcohol'].mean(), color='black', linestyle='--', label=f'Середнє: {data["alcohol"].mean():.2f}%')
plt.legend()
plt.tight_layout()
plt.show()


# ### 11.3 Boxplot для порівняння характеристик

# In[24]:


# Boxplot алкоголю за категоріями якості
plt.figure(figsize=(10, 6))
sns.boxplot(x='quality', y='alcohol', data=data, palette='viridis')
plt.xlabel('Якість вина', fontsize=12)
plt.ylabel('Вміст алкоголю (%)', fontsize=12)
plt.title('Залежність вмісту алкоголю від якості вина', fontsize=14)
plt.tight_layout()
plt.show()


# ### 11.4 Теплова карта кореляцій

# In[25]:


# Теплова карта кореляцій
plt.figure(figsize=(14, 10))
correlation_matrix = data.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5)
plt.title('Матриця кореляцій характеристик вина', fontsize=14)
plt.tight_layout()
plt.show()


# ### 11.5 Scatter plot (діаграма розсіювання)

# In[26]:


# Scatter plot: алкоголь vs якість
plt.figure(figsize=(10, 6))
plt.scatter(data['alcohol'], data['quality'], alpha=0.5, c=data['quality'], cmap='viridis')
plt.colorbar(label='Якість')
plt.xlabel('Вміст алкоголю (%)', fontsize=12)
plt.ylabel('Якість', fontsize=12)
plt.title('Залежність якості від вмісту алкоголю', fontsize=14)
plt.tight_layout()
plt.show()


# ---
# ## 12. Злиття DataFrame
# 
# Функція `pd.merge()` дозволяє об'єднувати два DataFrame за спільним стовпцем (або кількома стовпцями). Це аналогічно операції `JOIN` у SQL.
# 
# Типи злиття:
# - `inner` — залишаються тільки спільні ключі
# - `left` — всі рядки з лівого DataFrame
# - `right` — всі рядки з правого DataFrame
# - `outer` — всі рядки з обох DataFrame

# In[27]:


# Створення допоміжного DataFrame з категоріями якості
print("=" * 80)
print("ЗЛИТТЯ DATAFRAME:")
print("=" * 80)

quality_info = pd.DataFrame({
    'quality': [3, 4, 5, 6, 7, 8],
    'grade': ['F', 'D', 'C', 'B', 'A', 'A+'],
    'description': ['Дуже низька', 'Низька', 'Нижче середнього', 'Середня', 'Хороша', 'Відмінна']
})

print("Допоміжний DataFrame з описом якості:")
print(quality_info)


# In[28]:


# Злиття DataFrame
merged_data = pd.merge(data, quality_info, on='quality', how='left')

print("\nРезультат злиття (перші 10 рядків):")
merged_data[['fixed acidity', 'alcohol', 'quality', 'grade', 'description']].head(10)


# ---
# ## 13. Pivot Table (зведена таблиця)
# 
# Функція `pd.pivot_table()` створює зведену таблицю, яка дозволяє агрегувати дані за кількома вимірами. Це потужний інструмент для аналізу даних.
# 
# $$
# \text{Pivot Table}_{i,j} = f(\{x : \text{row} = i, \text{column} = j\})
# $$
# 
# де $f$ — агрегатна функція (mean, sum, count тощо).

# In[29]:


# Створення зведеної таблиці
print("=" * 80)
print("PIVOT TABLE:")
print("=" * 80)

# Створення категорій для pH
data['pH_category'] = pd.cut(data['pH'], bins=[0, 3.2, 3.4, 4.5], labels=['Низький', 'Середній', 'Високий'])

pivot_table = pd.pivot_table(
    data,
    values='alcohol',
    index='quality',
    columns='pH_category',
    aggfunc='mean'
)

print("Середній вміст алкоголю за якістю та рівнем pH:")
pivot_table.round(2)


# In[30]:


# Більш детальна зведена таблиця з кількома агрегатними функціями
pivot_detailed = pd.pivot_table(
    data,
    values=['alcohol', 'residual sugar', 'total_acidity'],
    index='quality',
    aggfunc={'alcohol': ['mean', 'std'], 
             'residual sugar': 'mean',
             'total_acidity': 'mean'}
)

print("\nДетальна зведена таблиця:")
pivot_detailed.round(3)


# ---
# ## 14. Розрахунки статистики
# 
# ### 14.1 Коефіцієнт кореляції Пірсона
# 
# Коефіцієнт кореляції Пірсона вимірює лінійну залежність між двома змінними:
# 
# $$
# r_{xy} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2} \cdot \sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}
# $$
# 
# де $r_{xy} \in [-1, 1]$:
# - $r = 1$ — ідеальна пряма залежність
# - $r = -1$ — ідеальна обернена залежність
# - $r = 0$ — відсутність лінійної залежності

# In[31]:


# Обчислення кореляції
print("=" * 80)
print("РОЗРАХУНКИ СТАТИСТИКИ:")
print("=" * 80)

numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
correlation_with_quality = data[numeric_cols].corrwith(data['quality']).sort_values(ascending=False)

print("Кореляція всіх ознак з якістю вина:")
print(correlation_with_quality.round(4))


# ### 14.2 Дисперсія та стандартне відхилення
# 
# **Дисперсія** ($\sigma^2$ або $s^2$) — міра розсіювання значень навколо середнього:
# 
# $$
# s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2
# $$
# 
# **Стандартне відхилення** ($\sigma$ або $s$) — квадратний корінь з дисперсії:
# 
# $$
# s = \sqrt{s^2} = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2}
# $$

# In[32]:


# Дисперсія
print("\nДисперсія числових стовпців:")
variance = data[numeric_cols].var()
print(variance.round(4))


# In[33]:


# Стандартне відхилення
print("\nСтандартне відхилення числових стовпців:")
std_dev = data[numeric_cols].std()
print(std_dev.round(4))


# In[34]:


# Коефіцієнт варіації
print("\nКоефіцієнт варіації (CV = std/mean * 100%):")
cv = (data[numeric_cols].std() / data[numeric_cols].mean() * 100).round(2)
print(cv)


# ---
# ## 15. Групування за кількома стовпцями
# 
# Метод `groupby()` може приймати список стовпців для створення ієрархічного групування. Це дозволяє аналізувати дані за кількома вимірами одночасно.

# In[35]:


# Групування за кількома стовпцями
print("=" * 80)
print("ГРУПУВАННЯ ЗА КІЛЬКОМА СТОВПЦЯМИ:")
print("=" * 80)

grouped_multi = data.groupby(['quality', 'quality_category']).agg({
    'alcohol': ['mean', 'std'],
    'fixed acidity': 'mean',
    'volatile acidity': 'mean'
}).round(3)

print(grouped_multi)


# In[36]:


# Групування за pH_category та quality
grouped_ph_quality = data.groupby(['pH_category', 'quality']).size().unstack(fill_value=0)
print("\nКількість вин за категоріями pH та якістю:")
print(grouped_ph_quality)


# ---
# ## 16. Функції застосовані до груп (agg, transform)
# 
# ### 16.1 Метод agg()
# 
# Метод `agg()` дозволяє застосовувати декілька агрегатних функцій одночасно та налаштовувати їх для різних стовпців.

# In[37]:


# Використання agg з декількома функціями
print("=" * 80)
print("МЕТОД AGG():")
print("=" * 80)

agg_result = data.groupby('quality').agg({
    'alcohol': ['min', 'max', 'mean', 'std', 'count'],
    'pH': ['min', 'max', 'mean'],
    'residual sugar': ['mean', 'median']
}).round(3)

print(agg_result)


# ### 16.2 Метод transform()
# 
# Метод `transform()` застосовує функцію до кожної групи і повертає результат з тією ж формою, що і вхідні дані. Це корисно для нормалізації даних всередині груп.

# In[38]:


# Використання transform для нормалізації
print("=" * 80)
print("МЕТОД TRANSFORM():")
print("=" * 80)

# Z-нормалізація алкоголю всередині кожної групи якості
data['alcohol_zscore_by_quality'] = data.groupby('quality')['alcohol'].transform(
    lambda x: (x - x.mean()) / x.std()
)

print("Перші 10 записів з Z-score алкоголю за групами якості:")
data[['alcohol', 'quality', 'alcohol_zscore_by_quality']].head(10)


# In[39]:


# Розрахунок відсотку від середнього групи
data['alcohol_pct_of_group_mean'] = data.groupby('quality')['alcohol'].transform(
    lambda x: x / x.mean() * 100
)

print("\nВміст алкоголю як % від середнього у групі:")
data[['alcohol', 'quality', 'alcohol_pct_of_group_mean']].head(10)


# ---
# ## 17. Створення стовпців за допомогою лямбда-функцій
# 
# Лямбда-функції (анонімні функції) в поєднанні з методом `apply()` дозволяють швидко створювати нові стовпці на основі складної логіки.
# 
# **Синтаксис лямбда-функції:**
# ```python
# lambda arguments: expression
# ```

# In[40]:


# Використання лямбда-функцій
print("=" * 80)
print("СТВОРЕННЯ СТОВПЦІВ З ЛЯМБДА-ФУНКЦІЯМИ:")
print("=" * 80)

# Приклад 1: Подвоєння значення
data['alcohol_doubled'] = data['alcohol'].apply(lambda x: x * 2)

# Приклад 2: Категоризація вмісту цукру
data['sugar_level'] = data['residual sugar'].apply(
    lambda x: 'Низький' if x < 2 else ('Середній' if x < 4 else 'Високий')
)

# Приклад 3: Логарифмічне перетворення
data['log_alcohol'] = data['alcohol'].apply(lambda x: np.log(x))

print("Нові стовпці створено:")
data[['alcohol', 'alcohol_doubled', 'residual sugar', 'sugar_level', 'log_alcohol']].head(10)


# In[41]:


# Застосування лямбда-функції до кількох стовпців
data['acidity_alcohol_ratio'] = data.apply(
    lambda row: row['total_acidity'] / row['alcohol'] if row['alcohol'] > 0 else 0, 
    axis=1
)

print("\nВідношення кислотності до алкоголю:")
data[['total_acidity', 'alcohol', 'acidity_alcohol_ratio']].head(10)


# In[42]:


# Розподіл за рівнем цукру
print("\nРозподіл вин за рівнем цукру:")
print(data['sugar_level'].value_counts())


# ---
# ## 18. Використання регулярних виразів для фільтрації
# 
# Регулярні вирази (regex) дозволяють виконувати складний пошук за шаблонами у текстових даних. Метод `str.contains()` використовується для фільтрації рядків за шаблоном.
# 
# **Основні символи регулярних виразів:**
# - `.` — будь-який символ
# - `*` — 0 або більше повторень
# - `+` — 1 або більше повторень
# - `^` — початок рядка
# - `$` — кінець рядка
# - `[abc]` — будь-який символ з набору

# In[43]:


# Демонстрація використання регулярних виразів
print("=" * 80)
print("ВИКОРИСТАННЯ РЕГУЛЯРНИХ ВИРАЗІВ:")
print("=" * 80)

# Фільтрація стовпців, що містять 'acid' у назві
acid_columns = [col for col in data.columns if 'acid' in col.lower()]
print(f"Стовпці, що містять 'acid': {acid_columns}")

# Відображення даних цих стовпців
data[acid_columns].head()


# In[44]:


# Фільтрація за текстовим стовпцем з регулярним виразом
# Знайдемо вина з високою якістю
print("\nФільтрація за категорією якості (містить 'Висока'):")
high_quality_regex = data[data['quality_category'].str.contains('Висока', na=False)]
print(f"Знайдено {len(high_quality_regex)} записів")
high_quality_regex.head()


# In[45]:


# Використання regex для знаходження стовпців за шаблоном
import re

pattern = re.compile(r'.*acid.*|.*sugar.*', re.IGNORECASE)
matching_columns = [col for col in data.columns if pattern.match(col)]
print(f"\nСтовпці, що відповідають шаблону (acid або sugar): {matching_columns}")


# ---
# ## Висновки
# 
# У даній контрольній роботі було продемонстровано основні можливості бібліотеки Pandas для аналізу даних:
# 
# 1. **Завантаження даних** — використання `pd.read_csv()` для імпорту CSV-файлів
# 2. **Огляд даних** — методи `head()`, `tail()`, `info()`, `describe()`
# 3. **Обробка пропусків** — перевірка та заповнення з використанням `fillna()`
# 4. **Фільтрація** — використання логічних індексів
# 5. **Групування та агрегування** — метод `groupby()` з агрегатними функціями
# 6. **Сортування** — метод `sort_values()`
# 7. **Зміна типів даних** — метод `astype()`
# 8. **Додавання стовпців** — арифметичні операції та функція `apply()`
# 9. **Збереження даних** — метод `to_csv()`
# 10. **Візуалізація** — використання Matplotlib та Seaborn
# 11. **Злиття DataFrame** — функція `pd.merge()`
# 12. **Pivot Table** — функція `pd.pivot_table()`
# 13. **Статистичні розрахунки** — `corr()`, `var()`, `std()`
# 14. **Групування за кількома стовпцями** — `groupby(['col1', 'col2'])`
# 15. **Функції до груп** — методи `agg()` та `transform()`
# 16. **Лямбда-функції** — метод `apply()` з анонімними функціями
# 17. **Регулярні вирази** — метод `str.contains()` для фільтрації
# 
# Набір даних Wine Quality дозволив на практиці застосувати всі ці методи для аналізу фізико-хімічних властивостей червоних вин та їх зв'язку з якістю.

# In[46]:


# Підсумкова інформація
print("=" * 80)
print("ПІДСУМОК:")
print("=" * 80)
print(f"Оброблено записів: {len(data)}")
print(f"Кількість стовпців (з новими): {len(data.columns)}")
print(f"\nСписок всіх стовпців:")
for i, col in enumerate(data.columns, 1):
    print(f"  {i}. {col}")

