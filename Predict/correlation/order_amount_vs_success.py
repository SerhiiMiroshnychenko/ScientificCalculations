import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
import datetime
from matplotlib import scale as mscale
from matplotlib.transforms import Transform
import matplotlib.transforms as mtransforms
from tabulate import tabulate

# Завантаження даних з файлу
df = pd.read_csv('cleaned_result.csv')

# Перетворення стовпця 'is_successful' на числовий тип (0 або 1), якщо це ще не зроблено
df['is_successful'] = df['is_successful'].astype(int)

# Розрахунок статистичних показників для різних груп
# Групуємо дані за ознакою is_successful та розраховуємо статистики для order_amount
stats = df.groupby('is_successful')['order_amount'].agg([
    ('Середнє', 'mean'),
    ('Медіана', 'median'),
    ('Стандартне відхилення', 'std'),
    ('Мінімум', 'min'),
    ('Максимум', 'max'),
    ('Квартиль 25%', lambda x: x.quantile(0.25)),
    ('Квартиль 75%', lambda x: x.quantile(0.75))
])

# Перетворюємо індекс на більш читабельні назви
stats.index = ['Неуспішні', 'Успішні']

# Отримуємо поточну дату і час для унікальних імен файлів
date_time = datetime.datetime.now().strftime("%Y%m%d")

# Зберігаємо статистики у CSV файл
stats.to_csv(f'order_amount_statistics_{date_time}.csv')

# Також розраховуємо кількість спостережень у кожній групі
count_by_group = df.groupby('is_successful').size()
count_by_group_df = pd.DataFrame({
    'Група': ['Неуспішні', 'Успішні', 'Всього'],
    'Кількість': [count_by_group[0], count_by_group[1], count_by_group.sum()]
})

# Створюємо форматовані таблиці
print("\n" + "="*80)
print("СТАТИСТИЧНИЙ АНАЛІЗ СУМ ЗАМОВЛЕНЬ".center(80))
print("="*80)

# Форматуємо та виводимо основну таблицю статистик
stats_table = stats.copy()
for col in stats_table.columns:
    if col == 'Середнє' or col == 'Стандартне відхилення' or col == 'Медіана':
        stats_table[col] = stats_table[col].map(lambda x: f"{x:,.2f}")
    else:
        stats_table[col] = stats_table[col].map(lambda x: f"{x:,.2f}")

print("\nОсновні статистичні показники сум замовлень за групами:")
print(tabulate(stats_table, headers='keys', tablefmt='grid', showindex=True))

# Виводимо кількість спостережень
print("\nРозподіл кількості спостережень за групами:")
print(tabulate(count_by_group_df, headers='keys', tablefmt='grid', showindex=False))

# Додаємо порівняльну статистику
comparison_data = []
for stat in stats.columns:
    ratio = stats.loc['Неуспішні', stat] / stats.loc['Успішні', stat] if stats.loc['Успішні', stat] != 0 else float('inf')
    diff = stats.loc['Неуспішні', stat] - stats.loc['Успішні', stat]
    comparison_data.append({
        'Статистика': stat,
        'Неуспішні': f"{stats.loc['Неуспішні', stat]:,.2f}",
        'Успішні': f"{stats.loc['Успішні', stat]:,.2f}",
        'Різниця': f"{diff:,.2f}",
        'Відношення': f"{ratio:.2f}"
    })

comparison_df = pd.DataFrame(comparison_data)
print("\nПорівняльний аналіз статистичних показників:")
print(tabulate(comparison_df, headers='keys', tablefmt='grid', showindex=False))

print("\nВисновок:")
if stats.loc['Неуспішні', 'Середнє'] > stats.loc['Успішні', 'Середнє']:
    print(f"⚠ Неуспішні замовлення мають БІЛЬШУ середню суму (в {stats.loc['Неуспішні', 'Середнє']/stats.loc['Успішні', 'Середнє']:.2f} рази)")
else:
    print(f"⚠ Успішні замовлення мають БІЛЬШУ середню суму (в {stats.loc['Успішні', 'Середнє']/stats.loc['Неуспішні', 'Середнє']:.2f} рази)")

print("="*80)

# Створюємо власну функцію трансформації для нелінійної шкали
class CustomScaleTransform(mtransforms.Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self, threshold=1e6, max_value=4.2e6):
        super().__init__()
        self.threshold = threshold  # поріг в 1 мільйон
        self.max_value = max_value  # максимальне значення на графіку (трохи більше 4 млн)

    def transform_non_affine(self, a):
        # Копіюємо масив, щоб не змінювати оригінал
        a = np.array(a, copy=True)

        # Застосовуємо трансформацію:
        # - Значення від 0 до threshold (1 млн) залишаються в діапазоні [0, 0.5]
        # - Значення від threshold до max_value (4.2 млн) масштабуються до діапазону [0.5, 1]
        mask = a <= self.threshold
        a[mask] = 0.5 * a[mask] / self.threshold
        a[~mask] = 0.5 + 0.5 * (a[~mask] - self.threshold) / (self.max_value - self.threshold)

        return a

    def inverted(self):
        return CustomScaleInverseTransform(self.threshold, self.max_value)

class CustomScaleInverseTransform(mtransforms.Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self, threshold, max_value):
        super().__init__()
        self.threshold = threshold
        self.max_value = max_value

    def transform_non_affine(self, a):
        a = np.array(a, copy=True)

        # Обернена трансформація
        mask = a <= 0.5
        a[mask] = a[mask] * 2 * self.threshold
        a[~mask] = self.threshold + (a[~mask] - 0.5) * 2 * (self.max_value - self.threshold)

        return a

    def inverted(self):
        return CustomScaleTransform(self.threshold, self.max_value)

class CustomScale(mscale.ScaleBase):
    name = 'custom'

    def __init__(self, axis, **kwargs):
        super().__init__(axis)
        self.threshold = kwargs.get('threshold', 1e6)
        self.max_value = kwargs.get('max_value', 4.2e6)

    def get_transform(self):
        return CustomScaleTransform(self.threshold, self.max_value)

    def set_default_locators_and_formatters(self, axis):
        # Встановлюємо розташування поділок на осі Y
        axis.set_major_locator(plt.FixedLocator([0, 0.25*1e6, 0.5*1e6, 0.75*1e6, 1e6, 2e6, 3e6, 4e6]))
        axis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x/1e6:.1f}' if x < 1e6 else f'{x/1e6:.0f}'))

# Реєструємо нашу кастомну шкалу правильним способом
mscale.register_scale(CustomScale)

# Використаємо альтернативний підхід із SymmetricalLogScale для нерівномірної шкали
plt.figure(figsize=(10, 6))
sns.stripplot(x='is_successful', y='order_amount', data=df, jitter=True, alpha=0.5)
plt.title('Scatter plot з jitter для суми замовлення за успішністю')
plt.xlabel('Успішність')
plt.ylabel('Сума замовлення (млн)')
plt.xticks([0, 1], ['Неуспішні', 'Успішні'])

try:
    # Спробуємо застосувати нашу кастомну шкалу
    plt.yscale('custom', threshold=1e6, max_value=4.2e6)
except Exception as e:
    print(f"Помилка при застосуванні кастомної шкали: {e}")
    print("Використовуємо альтернативний підхід із symlog...")
    # Якщо не вдалося, використовуємо вбудовану symlog шкалу
    plt.yscale('symlog', linthresh=1e6, linscale=0.5)
    plt.ylim(0, 4.2e6)

plt.grid(True, alpha=0.3)

# Додаємо горизонтальні лінії для візуального орієнтиру
plt.axhline(y=1e6, color='red', linestyle='--', alpha=0.5)
plt.text(1.05, 1.1e6, '1 млн (поділ шкали)', color='red', alpha=0.7)

# Зберігаємо графік у файл замість відображення
plt.savefig(f'scatter_plot_custom_scale_{date_time}.png', dpi=300, bbox_inches='tight')
plt.close()  # Закриваємо фігуру для звільнення пам'яті

plt.figure(figsize=(10, 6))
sns.violinplot(x='is_successful', y='order_amount', data=df, inner='box', palette='Set3')
plt.yscale('log')
plt.title('Violin Plot з логарифмічною шкалою')
plt.xlabel('Успішність')
plt.ylabel('Сума замовлення (млн, лог. шкала)')
plt.xticks([0, 1], ['Неуспішні', 'Успішні'])
# Зберігаємо графік у файл замість відображення
plt.savefig(f'violin_plot_{date_time}.png', dpi=300, bbox_inches='tight')
plt.close()  # Закриваємо фігуру для звільнення пам'яті

# Виводимо повідомлення про збереження графіків
print(f'Графіки збережено у поточній директорії з датою та часом: {date_time}')
