import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
import datetime

# Завантаження даних з файлу
df = pd.read_csv('cleaned_result.csv')

# Перетворення стовпця 'is_successful' на числовий тип (0 або 1), якщо це ще не зроблено
df['is_successful'] = df['is_successful'].astype(int)

# Отримуємо поточну дату і час для унікальних імен файлів
дата_час = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 3. Scatter plot з jitter
plt.figure(figsize=(10, 6))
sns.stripplot(x='is_successful', y='order_amount', data=df, jitter=True, alpha=0.5)
plt.title('Scatter plot з jitter для суми замовлення за успішністю')
plt.xlabel('Успішність')
plt.ylabel('Сума замовлення')
plt.xticks([0, 1], ['Неуспішні', 'Успішні'])
# Зберігаємо графік у файл замість відображення
plt.savefig(f'scatter_plot_{дата_час}.png', dpi=300, bbox_inches='tight')
plt.close()  # Закриваємо фігуру для звільнення пам'яті

plt.figure(figsize=(10, 6))
sns.violinplot(x='is_successful', y='order_amount', data=df, inner='box', palette='Set3')
plt.yscale('log')
plt.title('Violin Plot з логарифмічною шкалою')
plt.xlabel('Успішність')
plt.ylabel('Сума замовлення (млн, лог. шкала)')
plt.xticks([0, 1], ['Неуспішні', 'Успішні'])
# Зберігаємо графік у файл замість відображення
plt.savefig(f'violin_plot_{дата_час}.png', dpi=300, bbox_inches='tight')
plt.close()  # Закриваємо фігуру для звільнення пам'яті

# Виводимо повідомлення про збереження графіків
print(f'Графіки збережено у поточній директорії з датою та часом: {дата_час}')
