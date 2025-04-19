#!/usr/bin/env python
# coding: utf-8

# # Індивідуальне завдання 1

# ## Ідентифікації грубих промахів та лінійна регресія інформаційних даних в сфері дисертаційного дослідження

# ### Постановка задачі регресійного аналізу комунікаційних даних B2B компанії
# 
# Сучасні B2B компанії накопичують значні обсяги даних про взаємодію з клієнтами, які можуть містити цінну інформацію для покращення бізнес-процесів та підвищення ефективності роботи. Одним із ключових питань є розуміння взаємозв'язку між активністю комунікації з клієнтами та їхньою купівельною поведінкою.
# У дослідженні використовуються історичні дані B2B компанії, зібрані протягом тривалого періоду взаємодії з клієнтами. Дані представлені у форматі CSV та містять наступні ключові змінні:
# 
# - `partner_total_orders` — загальна кількість замовлень, здійснених кожним партнером
# - `partner_total_messages` — загальна кількість повідомлень, якими обмінялися партнер та менеджери компанії
# 
# Обидві змінні є кількісними та теоретично можуть містити викиди, які потенційно впливають на результати аналізу. Ці викиди можуть бути пов'язані як з особливостями поведінки окремих партнерів, так і з можливими помилками при фіксації даних.
# 
# ### Мета дослідження
# 
# Головною метою дослідження є виявлення та кількісна оцінка взаємозв'язку між інтенсивністю комунікації партнерів з менеджерами компанії та кількістю замовлень, що здійснюють ці партнери. Особливу увагу буде приділено впливу викидів на результати аналізу та стабільність отриманих моделей.
# 
# ### Завдання дослідження
# 
# 1. Побудувати лінійну регресійну модель залежності кількості замовлень партнера від кількості повідомлень без попереднього очищення даних.
# 
# 2. Провести аналіз даних на наявність викидів із використанням критерію γ, заснованого на розподілі Стьюдента, та здійснити їх видалення.
# 
# 3. Побудувати лінійну регресійну модель на основі очищених даних.
# 
# 4. Порівняти метрики якості обох моделей (коефіцієнт детермінації $R^2$, скоригований $R^2$, середньоквадратичну помилку RMSE, середню абсолютну помилку MAE, F-статистику) та оцінити вплив викидів на результати моделювання.
# 
# 5. Проаналізувати залишки моделей для перевірки виконання передумов лінійної регресії.
# 
# 6. Візуалізувати отримані результати за допомогою діаграм розсіювання, графіків розподілу залишків та порівняльних графіків регресійних ліній.
# 
# ### Методологія дослідження
# 
# Дослідження базуватиметься на застосуванні методів математичної статистики та економетрики. Для виявлення викидів буде використано статистичний підхід, що ґрунтується на критерії γ з критичним значенням $\gamma_p$, яке розраховується на основі розподілу Стьюдента. Для побудови регресійних моделей застосовуватиметься метод найменших квадратів (МНК).
# 
# Для оцінки якості моделей будуть розраховані ключові метрики: $R^2$, скоригований $R^2$, RMSE, MAE та F-статистика. Також буде проведено аналіз залишків для перевірки виконання передумов лінійної регресії: лінійності, незалежності, гомоскедастичності та нормальності.
# 
# ### Очікувані результати
# 
# В результаті дослідження очікується:
# 
# 1. Встановити наявність та силу статистичного зв'язку між кількістю повідомлень та кількістю замовлень партнерів.
# 
# 2. Визначити кількісний вплив викидів на параметри регресійної моделі та її якість.
# 
# 3. Отримати більш стабільну та надійну модель після видалення викидів, що краще відображатиме реальні взаємозв'язки між досліджуваними змінними.
# 
# 4. Сформулювати практичні рекомендації для менеджменту компанії щодо оптимізації комунікації з партнерами з метою збільшення кількості замовлень.
# 
# Результати дослідження матимуть практичну цінність для підрозділів продажів та клієнтської підтримки B2B компанії, оскільки дозволять краще зрозуміти вплив інтенсивності комунікації на обсяги продажів та розробити ефективніші стратегії взаємодії з партнерами.
# 

# ## Теоретичні основи регресійного аналізу з видаленням викидів
# 
# ### Виявлення та видалення викидів (грубих промахів)
# 
# #### Поняття викидів у статистичних даних
# 
# Викиди (англ. outliers) — це значення, які аномально відрізняються від більшості спостережень у наборі даних. Такі екстремальні значення можуть істотно спотворювати результати статистичного аналізу, особливо регресійного. Викиди можуть виникати внаслідок помилок вимірювання, помилок у записі даних, випадкових варіацій у вибірці або наявності спостережень з іншої генеральної сукупності.
# 
# #### Статистичне виявлення викидів за критерієм γ
# 
# Один з ефективних методів виявлення викидів базується на оцінці відхилення екстремальних значень від середнього значення вибірки відносно середньоквадратичного відхилення.
# 
# Для вибірки $X = \{x_1, x_2, ..., x_n\}$ розраховуються наступні статистичні характеристики. Середнє значення обчислюється за формулою $\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$. Дисперсія дорівнює $D = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2$. А середньоквадратичне відхилення розраховується як $\sigma = \sqrt{D}$.
# 
# Коефіцієнти $\gamma_1$ і $\gamma_2$ розраховуються для оцінки відхилення максимального та мінімального значень. Коефіцієнт $\gamma_1$ дорівнює $\gamma_1 = \frac{x_{max} - \bar{x}}{\sigma}$, а коефіцієнт $\gamma_2$ дорівнює $\gamma_2 = \frac{\bar{x} - x_{min}}{\sigma}$, де $x_{max}$ і $x_{min}$ — максимальне та мінімальне значення у вибірці.
# 
# #### Визначення критичного значення γₚ
# 
# Для прийняття рішення щодо наявності викидів необхідно порівняти отримані значення $\gamma_1$ і $\gamma_2$ з критичним значенням $\gamma_p$, яке залежить від розміру вибірки та рівня довіри.
# 
# Критичне значення $\gamma_p$ розраховується з використанням розподілу Стьюдента з $(n-2)$ ступенями свободи: $\gamma_p = \frac{t_{\alpha/(2n), n-2} \cdot (n-1)}{\sqrt{n(n-2) + t_{\alpha/(2n), n-2}^2}}$, де $t_{\alpha/(2n), n-2}$ — квантиль розподілу Стьюдента з рівнем значущості $\alpha/(2n)$ і $(n-2)$ ступенями свободи.
# 
# Правило виявлення викидів полягає в наступному. Якщо $\gamma_1 > \gamma_p$, то максимальне значення $x_{max}$ є викидом. Якщо $\gamma_2 > \gamma_p$, то мінімальне значення $x_{min}$ є викидом.
# 
# #### Ітеративний алгоритм видалення викидів
# 
# Процес видалення викидів доцільно проводити ітеративно. Спочатку визначають викиди за критерієм γ. Потім видаляють виявлені викиди з вибірки. Після цього перераховують статистичні характеристики для оновленої вибірки. Далі повторюють перевірку наявності викидів. І завершують процес, коли викиди відсутні або досягнуто максимальну кількість ітерацій.
# 
# ### Лінійний регресійний аналіз
# 
# #### Модель лінійної регресії
# 
# Лінійна регресія — це статистичний метод моделювання взаємозв'язку між залежною змінною $y$ та однією або кількома незалежними змінними $x$. Для випадку простої лінійної регресії (з однією незалежною змінною) модель має вигляд: $y = a_0 + a_1 x + \varepsilon$, де $a_0$ — вільний член (перетин з віссю $y$), $a_1$ — коефіцієнт нахилу (показує, як змінюється $y$ при зміні $x$ на одиницю), а $\varepsilon$ — випадкова помилка.
# 
# #### Метод найменших квадратів
# 
# Для оцінки параметрів $a_0$ і $a_1$ найчастіше використовується метод найменших квадратів (МНК), який мінімізує суму квадратів відхилень спостережуваних значень від передбачених моделлю: $S(a_0, a_1) = \sum_{i=1}^{n} (y_i - (a_0 + a_1 x_i))^2 \to \min$.
# 
# Для знаходження мінімуму функції $S(a_0, a_1)$ необхідно прирівняти до нуля її частинні похідні: $\frac{\partial S}{\partial a_0} = -2 \sum_{i=1}^{n} (y_i - a_0 - a_1 x_i) = 0$ та $\frac{\partial S}{\partial a_1} = -2 \sum_{i=1}^{n} x_i (y_i - a_0 - a_1 x_i) = 0$.
# 
# Розв'язання цієї системи рівнянь дає наступні оцінки параметрів: $a_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2} = \frac{cov(x,y)}{var(x)}$ та $a_0 = \bar{y} - a_1 \bar{x}$, де $\bar{x}$ і $\bar{y}$ — середні значення змінних $x$ і $y$.
# 
# #### Передбачені значення та залишки
# 
# Після оцінки параметрів $a_0$ і $a_1$ можна обчислити передбачені моделлю значення: $\hat{y}_i = a_0 + a_1 x_i$. Залишки (residuals) — це різниці між спостережуваними та передбаченими значеннями: $e_i = y_i - \hat{y}_i$. Аналіз залишків є важливим етапом перевірки адекватності регресійної моделі.
# 
# ### Оцінка якості регресійної моделі
# 
# #### Коефіцієнт детермінації
# 
# Коефіцієнт детермінації $R^2$ показує, яка частка дисперсії залежної змінної пояснюється моделлю: $R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$, де $SS_{res}$ — сума квадратів залишків, а $SS_{tot}$ — загальна сума квадратів. Значення $R^2$ приймає значення від 0 до 1, де 1 означає, що модель ідеально описує дані.
# 
# #### Скоригований коефіцієнт детермінації
# 
# Скоригований $R^2$ враховує кількість незалежних змінних у моделі та кількість спостережень: $R_{adj}^2 = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$, де $n$ — кількість спостережень, а $p$ — кількість незалежних змінних (для простої лінійної регресії $p=1$).
# 
# #### Середньоквадратична помилка
# 
# MSE (Mean Squared Error) — це середній квадрат залишків: $MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$. Корінь із середньоквадратичної помилки (RMSE) — це корінь із MSE: $RMSE = \sqrt{MSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$. RMSE має ті ж одиниці вимірювання, що й залежна змінна $y$.
# 
# #### Середня абсолютна помилка
# 
# MAE (Mean Absolute Error) — це середнє абсолютних значень залишків: $MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$.
# 
# #### F-статистика
# 
# F-статистика дозволяє перевірити загальну значущість регресійної моделі: $F = \frac{MS_{reg}}{MS_{res}} = \frac{SS_{reg}/p}{SS_{res}/(n-p-1)}$, де $SS_{reg}$ — сума квадратів регресії ($SS_{reg} = SS_{tot} - SS_{res}$), $MS_{reg}$ — середній квадрат регресії, а $MS_{res}$ — середній квадрат залишків. F-статистика має F-розподіл з $p$ і $(n-p-1)$ ступенями свободи. p-значення для F-статистики визначає ймовірність отримання такого значення F при справедливості нульової гіпотези (що всі коефіцієнти регресії рівні нулю).
# 
# #### Стандартні помилки та довірчі інтервали для коефіцієнтів
# 
# Стандартна помилка оцінки коефіцієнта $a_1$ розраховується за формулою $SE(a_1) = \sqrt{\frac{MS_{res}}{\sum_{i=1}^{n} (x_i - \bar{x})^2}}$. Стандартна помилка оцінки коефіцієнта $a_0$ розраховується як $SE(a_0) = \sqrt{MS_{res} \left( \frac{1}{n} + \frac{\bar{x}^2}{\sum_{i=1}^{n} (x_i - \bar{x})^2} \right)}$.
# 
# Довірчі інтервали для коефіцієнтів з рівнем довіри $(1-\alpha)$ визначаються наступним чином: $a_0 \pm t_{1-\alpha/2, n-2} \cdot SE(a_0)$ та $a_1 \pm t_{1-\alpha/2, n-2} \cdot SE(a_1)$, де $t_{1-\alpha/2, n-2}$ — квантиль розподілу Стьюдента з рівнем довіри $(1-\alpha/2)$ і $(n-2)$ ступенями свободи.
# 
# ### Аналіз залишків та діагностика моделі
# 
# #### Передумови лінійної регресії
# 
# Для коректного застосування лінійної регресії мають виконуватися певні передумови. По-перше, взаємозв'язок між $x$ і $y$ має бути лінійним. По-друге, залишки $e_i$ повинні бути незалежними. По-третє, має виконуватись умова гомоскедастичності, тобто дисперсія залишків повинна бути постійною і не залежати від $x$. По-четверте, залишки повинні мати нормальний розподіл.
# 
# #### Графіки для аналізу залишків
# 
# Для діагностики регресійної моделі використовуються різні графіки. Графік залишків проти передбачених значень дозволяє виявити нелінійність та гетероскедастичність. Q-Q plot використовується для перевірки нормальності розподілу залишків. Графік впливовості спостережень (leverage plot) допомагає виявити спостереження, що мають значний вплив на модель.
# 
# #### Діагностика впливових спостережень
# 
# Впливові спостереження можуть непропорційно впливати на параметри регресійної моделі. Для їх виявлення використовуються різні показники. Важелі (leverage) – це діагональні елементи матриці проекції $h_{ii}$. Нормовані залишки розраховуються за формулою $r_i = \frac{e_i}{\hat{\sigma}\sqrt{1-h_{ii}}}$. Відстань Кука визначається як $D_i = \frac{r_i^2}{p+1} \cdot \frac{h_{ii}}{1-h_{ii}}$.
# 
# #### Покроковий підхід до регресійного аналізу з видаленням викидів
# 
# Практичний аналіз даних з використанням лінійної регресії після видалення викидів включає кілька етапів. На початку здійснюється попередній аналіз даних, який полягає у візуалізації та розрахунку описових статистик. Наступний етап передбачає виявлення та видалення викидів за статистичними критеріями. Після цього відбувається побудова регресійної моделі на очищених даних. Далі проводиться оцінка якості моделі за допомогою різних метрик. Важливим етапом є діагностика моделі через аналіз залишків. Потім порівнюються моделі, побудовані на вихідних та очищених даних. І нарешті відбувається інтерпретація результатів та формулювання висновків.
# 
# Такий підхід дозволяє отримати більш надійні та стійкі оцінки параметрів регресійної моделі, які менш чутливі до наявності аномальних спостережень.
# 

# ## Розв'язання задачі з використанням Python

# In[1]:


"""
Комплексний скрипт для аналізу даних: виявлення викидів та регресійний аналіз.

Скрипт дозволяє:
1. Зчитувати два стовпці даних з CSV файлу
2. Автоматично виявляти та видаляти грубі промахи (викиди)
3. Будувати лінійну регресійну модель на очищених даних
4. Оцінювати якість моделі через статистичні метрики
5. Візуалізувати результати через різноманітні графіки
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# ###  Визначаємо допоміжні функції

# In[2]:


def read_csv_data(file_path, column_x, column_y, delimiter=','):
    """
    Зчитує два стовпці даних з CSV файлу

    Параметри:
    file_path (str): Шлях до CSV файлу
    column_x (str або int): Назва або індекс стовпця для X
    column_y (str або int): Назва або індекс стовпця для Y
    delimiter (str): Розділювач у CSV файлі

    Повертає:
    tuple: (x_data, y_data, df, x_name, y_name) - дані X та Y, датафрейм та назви стовпців
    """
    try:
        # Зчитування CSV файлу
        df = pd.read_csv(file_path, delimiter=delimiter)

        # Отримання даних з першого стовпця (X)
        if isinstance(column_x, int):
            if column_x < len(df.columns):
                x_data = df.iloc[:, column_x].values
                x_name = df.columns[column_x]
            else:
                raise ValueError(f"Стовпець з індексом {column_x} не існує в файлі. Доступні індекси: 0-{len(df.columns)-1}")
        else:
            if column_x in df.columns:
                x_data = df[column_x].values
                x_name = column_x
            else:
                raise ValueError(f"Стовпець '{column_x}' не знайдено в файлі. Доступні стовпці: {', '.join(df.columns)}")

        # Отримання даних з другого стовпця (Y)
        if isinstance(column_y, int):
            if column_y < len(df.columns):
                y_data = df.iloc[:, column_y].values
                y_name = df.columns[column_y]
            else:
                raise ValueError(f"Стовпець з індексом {column_y} не існує в файлі. Доступні індекси: 0-{len(df.columns)-1}")
        else:
            if column_y in df.columns:
                y_data = df[column_y].values
                y_name = column_y
            else:
                raise ValueError(f"Стовпець '{column_y}' не знайдено в файлі. Доступні стовпці: {', '.join(df.columns)}")

        print(f"Зчитано дані зі стовпців '{x_name}' (X) та '{y_name}' (Y), кількість значень: {len(x_data)}")

        # Перевірка, чи дані числові
        try:
            x_data = x_data.astype(float)
            y_data = y_data.astype(float)
        except ValueError:
            raise ValueError(f"Стовпці містять нечислові дані")

        return x_data, y_data, df, x_name, y_name

    except FileNotFoundError:
        raise FileNotFoundError(f"Файл '{file_path}' не знайдено")
    except Exception as e:
        raise Exception(f"Помилка при зчитуванні даних: {str(e)}")


# In[3]:


def print_header(text):
    """Функція виведення заголовка секції"""
    print("\n" + "="*60)
    print(text)
    print("="*60)


# In[4]:


def calculate_statistics(data):
    """
    Розрахунок статистичних характеристик для серії даних

    Параметри:
    data (numpy.ndarray): Масив даних

    Повертає:
    tuple: (n, x_mean, D, sigma) - кількість елементів, середнє, дисперсія, с.к.в.
    """
    n = len(data)
    x_mean = np.mean(data)  # середнє значення
    D = np.sum((data - x_mean)**2) / (n - 1)  # дисперсія
    sigma = np.sqrt(D)  # середньоквадратичне відхилення

    return n, x_mean, D, sigma


# In[5]:


def calculate_gamma(data, x_mean, sigma):
    """
    Розрахунок коефіцієнтів γ1 та γ2 для виявлення викидів

    Параметри:
    data (numpy.ndarray): Масив даних
    x_mean (float): Середнє значення
    sigma (float): Середньоквадратичне відхилення

    Повертає:
    tuple: (gamma1, gamma2) - коефіцієнти для аналізу
    """
    gamma1 = (np.max(data) - x_mean) / sigma
    gamma2 = (x_mean - np.min(data)) / sigma

    return gamma1, gamma2


# In[6]:


def get_critical_gamma(n, confidence=0.95):
    """
    Розрахунок критичного значення gamma_p для виявлення викидів
    з використанням розподілу Стьюдента

    Параметри:
    n (int): Кількість спостережень
    confidence (float): Рівень довіри

    Повертає:
    float: Критичне значення gamma_p
    """
    # Розрахунок критичного значення за формулою, що використовує
    # розподіл Стьюдента з (n-2) ступенями свободи
    t_critical = stats.t.ppf(1 - (1 - confidence) / (2 * n), n - 2)
    gamma_p = t_critical * (n - 1) / np.sqrt(n * (n - 2) + t_critical**2)

    return gamma_p


# In[7]:


def check_outliers(data, label, confidence=0.95):
    """
    Перевірка наявності викидів у серії даних

    Параметри:
    data (numpy.ndarray): Масив даних
    label (str): Назва стовпця для виведення
    confidence (float): Рівень довіри

    Повертає:
    tuple: (has_outliers, outlier_indices, stats) - чи є викиди, їх індекси, статистики
    """
    n, x_mean, D, sigma = calculate_statistics(data)
    gamma1, gamma2 = calculate_gamma(data, x_mean, sigma)

    # Розрахунок критичного значення gamma_p
    gamma_p = get_critical_gamma(n, confidence)

    has_outliers = False
    outlier_indices = []

    # print(f"{label}: γ1 = {gamma1:.3f}, γ2 = {gamma2:.3f}, γp = {gamma_p:.3f}")

    if gamma1 > gamma_p:
        has_outliers = True
        max_index = np.argmax(data)
        outlier_indices.append(max_index)
    #     print(f"{label}: γ1 > γp: Викид у максимальному значенні {data[max_index]:.3f} (індекс {max_index})")
    # else:
    #     print(f"{label}: γ1 <= γp: Немає викиду у максимальному значенні")

    if gamma2 > gamma_p:
        has_outliers = True
        min_index = np.argmin(data)
        outlier_indices.append(min_index)
    #     print(f"{label}: γ2 > γp: Викид у мінімальному значенні {data[min_index]:.3f} (індекс {min_index})")
    # else:
    #     print(f"{label}: γ2 <= γp: Немає викиду у мінімальному значенні")

    return has_outliers, outlier_indices, (n, x_mean, D, sigma)


# In[8]:


def plot_histograms(data_before, data_after, column_name):
    """
    Візуалізація гістограм до та після очищення

    Параметри:
    data_before (numpy.ndarray): Дані до очищення
    data_after (numpy.ndarray): Дані після очищення
    column_name (str): Назва стовпця для заголовків
    """
    # Налаштування загального стилю графіків
    plt.style.use('ggplot')
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'axes.grid': True,
        'axes.grid.which': 'both',
        'grid.alpha': 0.3,
        'figure.figsize': (10, 6),
        'figure.dpi': 120
    })
    plt.figure(figsize=(10, 6))

    # Додаткові налаштування для гістограм
    bin_params = {}
    if len(data_before) > 1000:
        # Автоматичний розрахунок оптимального числа бінів
        # Використовуємо правило Freedman-Diaconis
        data_range = np.max(data_before) - np.min(data_before)
        bin_width = 2 * stats.iqr(data_before) / (len(data_before) ** (1/3))
        n_bins = int(data_range / bin_width) if bin_width > 0 else 50
        n_bins = min(100, max(20, n_bins))  # Обмежуємо кількість бінів
        bin_params['bins'] = n_bins

    # Створюємо фігуру
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Визначення статистик для обох наборів даних
    mean_before = np.mean(data_before)
    std_before = np.std(data_before)
    mean_after = np.mean(data_after)
    std_after = np.std(data_after)

    # Добавляємо легенду з інформацією про початкові та очищені дані
    before_label = f'Початкові дані: $\\mu={mean_before:.2f}$, $\\sigma={std_before:.2f}$'
    after_label = f'Очищені дані: $\\mu={mean_after:.2f}$, $\\sigma={std_after:.2f}$'

    # Налаштування прозорості для кращого розрізнення
    alpha_val = 0.6

    # Накладені гістограми
    sns.histplot(data_before, kde=True, color='blue', alpha=alpha_val, label=before_label,
                 ax=ax, edgecolor='darkblue', linewidth=1.2, **bin_params)
    sns.histplot(data_after, kde=True, color='green', alpha=alpha_val, label=after_label,
                 ax=ax, edgecolor='darkgreen', linewidth=1.2, **bin_params)

    # Додаємо вертикальні лінії для середніх значень
    ax.axvline(mean_before, color='blue', linestyle='--', linewidth=2, alpha=0.9)
    ax.axvline(mean_after, color='green', linestyle='--', linewidth=2, alpha=0.9)

    # Оформлення графіка
    ax.set_title(f'{column_name} (порівняння гістограм)', fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel(column_name, fontsize=18, fontweight='bold')
    ax.set_ylabel('Частота', fontsize=18, fontweight='bold')

    # Покращення легенди
    legend = ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=14)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.0)

    # Покращення відображення сітки
    ax.grid(True, linestyle='--', alpha=0.7)

    # Покращення відображення меж графіка
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_edgecolor('black')

    # Встановлюємо межі по x для кращого відображення
    # Визначаємо межі на основі перцентилів даних
    data_all = np.concatenate([data_before, data_after])
    q_low, q_high = np.percentile(data_all, [1, 99])
    # Розширюємо межі на 10% в обидві сторони
    range_x = q_high - q_low
    ax.set_xlim([q_low - 0.1 * range_x, q_high + 0.1 * range_x])

    plt.tight_layout()
    plt.show()


# In[9]:


def plot_scatter_before_after(x_before, y_before, x_after, y_after, x_name, y_name):
    """
    Візуалізація діаграм розсіювання до та після очищення

    Параметри:
    x_before, y_before: Дані X і Y до очищення
    x_after, y_after: Дані X і Y після очищення
    x_name, y_name: Назви стовпців для заголовків
    """
    # Накладені діаграми розсіювання
    plt.figure(figsize=(10, 6))
    plt.scatter(x_before, y_before, alpha=0.5, color='blue', label='Початкові дані')
    plt.scatter(x_after, y_after, alpha=0.5, color='green', label='Очищені дані')
    plt.title('Порівняння даних до та після видалення викидів')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Діаграма викидів (точки, які були видалені)
    if len(x_before) != len(x_after):
        # Створюємо маску викидів
        outlier_mask = np.ones(len(x_before), dtype=bool)
        for i, x_val in enumerate(x_before):
            if x_val in x_after:
                # Перевіряємо також відповідне значення Y
                idx = np.where(x_after == x_val)[0]
                if len(idx) > 0 and y_before[i] in y_after[idx]:
                    outlier_mask[i] = False

        # Отримуємо тільки викиди
        x_outliers = x_before[outlier_mask]
        y_outliers = y_before[outlier_mask]

        if len(x_outliers) > 0:
            plt.figure(figsize=(10, 6))
            plt.scatter(x_before, y_before, alpha=0.3, color='blue', label='Всі дані')
            plt.scatter(x_outliers, y_outliers, alpha=0.7, color='red', label='Викиди')
            plt.title('Виявлені викиди')
            plt.xlabel(x_name)
            plt.ylabel(y_name)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()


# In[10]:


def remove_outliers_from_data(x_data, y_data, x_name, y_name, confidence=0.95, max_iterations=3):
    """
    Виявлення та видалення викидів з двох стовпців даних

    Параметри:
    x_data, y_data: Масиви даних X і Y
    x_name, y_name: Назви стовпців
    confidence: Рівень довіри
    max_iterations: Максимальна кількість ітерацій очищення

    Повертає:
    tuple: (x_clean, y_clean, outlier_indices) - очищені дані та індекси викидів
    """
    print_header("Аналіз та виявлення викидів")

    # Копіювання оригінальних даних
    x_clean = x_data.copy()
    y_clean = y_data.copy()

    # Створення маски для відстеження індексів (початково всі True)
    valid_indices = np.ones(len(x_data), dtype=bool)

    # Інформація про дані до очищення
    n_x, mean_x, var_x, std_x = calculate_statistics(x_data)
    n_y, mean_y, var_y, std_y = calculate_statistics(y_data)

    print(f"Початкові статистики X: n={n_x}, середнє={mean_x:.4f}, дисперсія={var_x:.4f}, с.к.в.={std_x:.4f}")
    print(f"Початкові статистики Y: n={n_y}, середнє={mean_y:.4f}, дисперсія={var_y:.4f}, с.к.в.={std_y:.4f}")

    # Створення таблиці для відстеження викидів
    outliers_data = {"Ітерація": [], "Змінна": [], "Індекс": [], "Значення": []}

    iteration = 0
    total_outliers = 0

    while iteration < max_iterations:
        iteration += 1
        # print(f"\nІтерація {iteration}:")

        # Перевірка наявності викидів у X та Y
        x_has_outliers, x_outlier_indices, x_stats = check_outliers(x_clean, f"X ({x_name})", confidence)
        y_has_outliers, y_outlier_indices, y_stats = check_outliers(y_clean, f"Y ({y_name})", confidence)

        # Якщо немає викидів в обох стовпцях, завершуємо цикл
        if not x_has_outliers and not y_has_outliers:
            print("Викиди не виявлені в обох стовпцях, завершуємо очищення")
            break

        # Збір індексів викидів
        current_outliers = set()

        # Додаємо викиди X
        for idx in x_outlier_indices:
            # Знаходимо оригінальний індекс
            orig_idx = np.where(valid_indices)[0][idx]
            current_outliers.add(orig_idx)
            outliers_data["Ітерація"].append(iteration)
            outliers_data["Змінна"].append(x_name)
            outliers_data["Індекс"].append(orig_idx)
            outliers_data["Значення"].append(x_data[orig_idx])

        # Додаємо викиди Y
        for idx in y_outlier_indices:
            # Знаходимо оригінальний індекс
            orig_idx = np.where(valid_indices)[0][idx]
            current_outliers.add(orig_idx)
            outliers_data["Ітерація"].append(iteration)
            outliers_data["Змінна"].append(y_name)
            outliers_data["Індекс"].append(orig_idx)
            outliers_data["Значення"].append(y_data[orig_idx])

        # Оновлюємо маску дійсних індексів
        for idx in current_outliers:
            valid_indices[idx] = False

        # Оновлюємо очищені дані
        x_clean = x_data[valid_indices]
        y_clean = y_data[valid_indices]

        total_outliers += len(current_outliers)
        # print(f"Знайдено {len(current_outliers)} викидів в ітерації {iteration}")
        # print(f"Загальна кількість видалених викидів: {total_outliers}")

        if len(current_outliers) == 0:
            print("Викиди не знайдені, завершуємо очищення")
            break

    # Підсумкові статистики після очищення
    if total_outliers > 0:
        n_x, mean_x, var_x, std_x = calculate_statistics(x_clean)
        n_y, mean_y, var_y, std_y = calculate_statistics(y_clean)

        print("\nСтатистики після очищення:")
        print(f"X: n={n_x}, середнє={mean_x:.4f}, дисперсія={var_x:.4f}, с.к.в.={std_x:.4f}")
        print(f"Y: n={n_y}, середнє={mean_y:.4f}, дисперсія={var_y:.4f}, с.к.в.={std_y:.4f}")
        print(f"Видалено {total_outliers} викидів, залишилось {len(x_clean)} спостережень")

        # Створення DataFrame з даними про викиди
        outliers_df = pd.DataFrame(outliers_data)
        if not outliers_df.empty:
            print("\nВидалені викиди:")
            print(outliers_df)
    else:
        print("\nВикиди не виявлені, дані залишаються без змін")

    # Повертаємо очищені дані та індекси видалених спостережень
    outlier_indices = np.where(~valid_indices)[0]
    return x_clean, y_clean, outlier_indices


# In[11]:


def calculate_regression_parameters(x, y):
    """
    Розраховує параметри лінійної регресії методом найменших квадратів.

    Параметри:
    x (numpy.ndarray): Масив незалежних змінних
    y (numpy.ndarray): Масив залежних змінних

    Повертає:
    tuple: (a0, a1) - коефіцієнти лінійної регресії y = a0 + a1*x
    """
    n = len(x)

    # Обчислення середніх значень
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Обчислення середнього значення квадратів x
    x2_mean = np.mean(x**2)

    # Обчислення середнього значення добутку x*y
    xy_mean = np.mean(x*y)

    # Розрахунок коефіцієнтів регресії
    a1 = (xy_mean - x_mean * y_mean) / (x2_mean - x_mean**2)
    a0 = y_mean - a1 * x_mean

    return a0, a1


# In[12]:


def calculate_regression_metrics(x, y, a0, a1):
    """
    Розраховує метрики якості лінійної регресійної моделі.

    Параметри:
    x (numpy.ndarray): Масив незалежних змінних
    y (numpy.ndarray): Масив залежних змінних
    a0 (float): Вільний член рівняння регресії
    a1 (float): Коефіцієнт нахилу

    Повертає:
    dict: Словник з метриками якості моделі
    """
    n = len(x)

    # Прогнозовані значення
    y_pred = a0 + a1 * x

    # Середні значення
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Залишки (похибки)
    residuals = y - y_pred

    # Сума квадратів залишків
    sse = np.sum(residuals**2)

    # Загальна сума квадратів
    sst = np.sum((y - y_mean)**2)

    # Сума квадратів регресії
    ssr = np.sum((y_pred - y_mean)**2)

    # Коефіцієнт детермінації R^2
    r_squared = ssr / sst

    # Скоригований R^2
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - 2)

    # Стандартна похибка регресії
    se_regression = np.sqrt(sse / (n - 2))

    # Коефіцієнт кореляції
    r_xy = np.sum((x - x_mean) * (y - y_mean)) / n
    r_xy /= (np.sqrt(np.sum((x - x_mean)**2) / n) * np.sqrt(np.sum((y - y_mean)**2) / n))

    # Стандартні похибки коефіцієнтів регресії
    se_a1 = se_regression / np.sqrt(np.sum((x - x_mean)**2))
    se_a0 = se_regression * np.sqrt(1/n + x_mean**2 / np.sum((x - x_mean)**2))

    # t-статистики для коефіцієнтів
    t_a0 = a0 / se_a0
    t_a1 = a1 / se_a1

    # F-статистика для загальної значущості моделі
    f_statistic = (ssr / 1) / (sse / (n - 2))

    # Усереднений коефіцієнт еластичності
    elasticity = a1 * np.mean(x) / np.mean(y)

    # Критичні значення для статистичних тестів
    alpha = 0.05  # Рівень значущості 5%
    t_critical = stats.t.ppf(1 - alpha/2, n - 2)
    f_critical = stats.f.ppf(1 - alpha, 1, n - 2)

    # Довірчі інтервали для коефіцієнтів регресії
    a0_lower = a0 - t_critical * se_a0
    a0_upper = a0 + t_critical * se_a0
    a1_lower = a1 - t_critical * se_a1
    a1_upper = a1 + t_critical * se_a1

    # Збереження результатів у словник
    metrics = {
        "n": n,
        "a0": a0,
        "a1": a1,
        "r_xy": r_xy,
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "se_regression": se_regression,
        "std_error": se_regression,  # Додано для сумісності з plot_regression_results
        "se_a0": se_a0,
        "se_a1": se_a1,
        "t_a0": t_a0,
        "t_a1": t_a1,
        "t_critical": t_critical,
        "f_statistic": f_statistic,
        "f_critical": f_critical,
        "elasticity": elasticity,
        "a0_ci": (a0_lower, a0_upper),
        "a1_ci": (a1_lower, a1_upper),
        "residuals": residuals,
        "y_pred": y_pred
    }

    return metrics


# In[13]:


def print_regression_results(x, y, x_name, y_name, metrics):
    """
    Виводить результати регресійного аналізу.

    Параметри:
    x (numpy.ndarray): Масив незалежних змінних
    y (numpy.ndarray): Масив залежних змінних
    x_name (str): Назва незалежної змінної
    y_name (str): Назва залежної змінної
    metrics (dict): Словник з метриками якості моделі
    """
    print_header("Результати регресійного аналізу")

    # Видобування метрик
    a0 = metrics["a0"]
    a1 = metrics["a1"]
    r_squared = metrics["r_squared"]
    adj_r_squared = metrics["adj_r_squared"]
    r_xy = metrics["r_xy"]
    se_regression = metrics["se_regression"]
    se_a0 = metrics["se_a0"]
    se_a1 = metrics["se_a1"]
    t_a0 = metrics["t_a0"]
    t_a1 = metrics["t_a1"]
    t_critical = metrics["t_critical"]
    f_statistic = metrics["f_statistic"]
    f_critical = metrics["f_critical"]
    elasticity = metrics["elasticity"]
    a0_ci = metrics["a0_ci"]
    a1_ci = metrics["a1_ci"]

    # Виведення рівняння регресії
    print(f"\nРівняння регресії: {y_name} = {a0:.4f} + {a1:.4f} * {x_name}")

    # Виведення коефіцієнтів та їх статистик
    print("\nКоефіцієнти регресії та їх статистична значущість:")
    print(f"a0 = {a0:.4f}, стандартна похибка = {se_a0:.4f}, t-статистика = {t_a0:.4f}")
    print(f"a1 = {a1:.4f}, стандартна похибка = {se_a1:.4f}, t-статистика = {t_a1:.4f}")
    print(f"Критичне значення t-статистики (α=0.05, df={len(x)-2}) = {t_critical:.4f}")

    # Висновок про значущість коефіцієнтів
    print("\nСтатистична значущість коефіцієнтів:")
    print(f"a0: {abs(t_a0) > t_critical}")
    print(f"a1: {abs(t_a1) > t_critical}")

    # Виведення довірчих інтервалів
    print("\nДовірчі інтервали для коефіцієнтів (95%):")
    print(f"a0: [{a0_ci[0]:.4f}, {a0_ci[1]:.4f}]")
    print(f"a1: [{a1_ci[0]:.4f}, {a1_ci[1]:.4f}]")

    # Виведення метрик якості моделі
    print("\nМетрики якості моделі:")
    print(f"Коефіцієнт кореляції: r = {r_xy:.4f}")
    print(f"Коефіцієнт детермінації: R² = {r_squared:.4f}")
    print(f"Скоригований R²: {adj_r_squared:.4f}")
    print(f"Стандартна похибка регресії: {se_regression:.4f}")

    # Виведення F-статистики
    print("\nF-статистика для загальної значущості моделі:")
    print(f"F = {f_statistic:.4f}, критичне значення = {f_critical:.4f}")
    print(f"Модель статистично значуща: {f_statistic > f_critical}")

    # Виведення коефіцієнту еластичності
    print(f"\nУсереднений коефіцієнт еластичності: {elasticity:.4f}")
    interpretation = ""  # Інтерпретація еластичності
    if abs(elasticity) < 0.5:
        interpretation = "низька еластичність (нееластичний зв'язок)"
    elif abs(elasticity) < 1:
        interpretation = "середня еластичність"
    else:
        interpretation = "висока еластичність (еластичний зв'язок)"
    print(f"Інтерпретація: {interpretation}")


# In[14]:


def compare_regression_models(x_before, y_before, x_after, y_after, metrics_before, metrics_after, x_name, y_name):
    """
    Візуалізує порівняння регресійних моделей до та після очищення викидів.

    Параметри:
    x_before (numpy.ndarray): Масив незалежних змінних до очищення
    y_before (numpy.ndarray): Масив залежних змінних до очищення
    x_after (numpy.ndarray): Масив незалежних змінних після очищення
    y_after (numpy.ndarray): Масив залежних змінних після очищення
    metrics_before (dict): Словник з метриками якості моделі до очищення
    metrics_after (dict): Словник з метриками якості моделі після очищення
    x_name (str): Назва незалежної змінної
    y_name (str): Назва залежної змінної
    """
    # Налаштування стилю
    plt.style.use('ggplot')
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.figsize': (10, 6),
        'figure.dpi': 120
    })

    # Видобування метрик для обох моделей
    a0_before = metrics_before["a0"]
    a1_before = metrics_before["a1"]
    r_squared_before = metrics_before["r_squared"]
    y_pred_before = metrics_before["y_pred"]

    a0_after = metrics_after["a0"]
    a1_after = metrics_after["a1"]
    r_squared_after = metrics_after["r_squared"]
    y_pred_after = metrics_after["y_pred"]

    # Порівняння ліній регресії на одному графіку
    fig, ax = plt.subplots(figsize=(10, 6))

    # Відображення точок даних до очищення
    ax.scatter(x_before, y_before, alpha=0.5, color='blue', s=30, label='Дані до очищення')

    # Відображення точок даних після очищення
    ax.scatter(x_after, y_after, alpha=0.7, color='green', s=40, label='Дані після очищення')

    # Формування меж для ліній регресії (для кращого відображення)
    all_x = np.concatenate([x_before, x_after])
    x_min, x_max = np.min(all_x), np.max(all_x)
    x_range = np.linspace(x_min, x_max, 100)

    # Обчислення прогнозних значень для плавної лінії
    y_range_before = a0_before + a1_before * x_range
    y_range_after = a0_after + a1_after * x_range

    # Додавання ліній регресії
    ax.plot(x_range, y_range_before, color='darkblue', linestyle='-', linewidth=2,
            label=f'Модель до: y = {a0_before:.4f} + {a1_before:.4f}x, R² = {r_squared_before:.4f}')

    ax.plot(x_range, y_range_after, color='darkgreen', linestyle='-', linewidth=2,
            label=f'Модель після: y = {a0_after:.4f} + {a1_after:.4f}x, R² = {r_squared_after:.4f}')

    # Оформлення графіка
    ax.set_title('Порівняння регресійних моделей до та після очищення викидів', fontsize=20, fontweight='bold')
    ax.set_xlabel(x_name, fontsize=16, fontweight='bold')
    ax.set_ylabel(y_name, fontsize=16, fontweight='bold')

    # Покращення легенди
    legend = ax.legend(fontsize=12, loc='upper left', frameon=True, framealpha=0.9)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.0)

    # Покращення відображення сітки
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    # Додатковий графік: Порівняння залишків
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    # Отримання залишків
    residuals_before = metrics_before["residuals"]
    residuals_after = metrics_after["residuals"]

    # Статистики залишків
    mean_before = np.mean(residuals_before)
    std_before = np.std(residuals_before)
    mean_after = np.mean(residuals_after)
    std_after = np.std(residuals_after)

    # Налаштування гістограм
    bin_params_before = {}
    bin_params_after = {}

    if len(residuals_before) > 1000:
        # Оптимальна кількість бінів для великих даних
        data_range = np.max(residuals_before) - np.min(residuals_before)
        bin_width = 2 * stats.iqr(residuals_before) / (len(residuals_before) ** (1/3))
        n_bins = int(data_range / bin_width) if bin_width > 0 else 50
        n_bins = min(100, max(30, n_bins))
        bin_params_before['bins'] = n_bins

    if len(residuals_after) > 1000:
        data_range = np.max(residuals_after) - np.min(residuals_after)
        bin_width = 2 * stats.iqr(residuals_after) / (len(residuals_after) ** (1/3))
        n_bins = int(data_range / bin_width) if bin_width > 0 else 50
        n_bins = min(100, max(30, n_bins))
        bin_params_after['bins'] = n_bins

    # Покращення відображення гістограми
    for i, (ax, residuals, title, color, mean_val, std_val, params) in enumerate([
        (axes[0], residuals_before, 'Залишки до очищення', 'royalblue', mean_before, std_before, bin_params_before),
        (axes[1], residuals_after, 'Залишки після очищення', 'forestgreen', mean_after, std_after, bin_params_after)
    ]):
        # Гістограма з кращим візуальним розподілом
        sns.histplot(residuals, kde=True, color=color, alpha=0.6, edgecolor='black', linewidth=1.0, ax=ax, **params)

        # Додавання вертикальної лінії середнього значення
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                   label=f'Середнє = {mean_val:.4f}')

        # Додавання нормального розподілу
        x_norm = np.linspace(np.percentile(residuals, 0.1), np.percentile(residuals, 99.9), 1000)
        y_norm = stats.norm.pdf(x_norm, mean_val, std_val)

        # Масштабування нормального розподілу
        bin_heights = [p.get_height() for p in ax.patches] if ax.patches else []
        max_height = max(bin_heights) if bin_heights else len(residuals) / 20
        scale_factor = max_height / (np.max(y_norm) if np.max(y_norm) > 0 else 1)

        # Додавання кривої нормального розподілу
        ax.plot(x_norm, y_norm * scale_factor, 'r-', alpha=0.7, linewidth=2,
                label=f'Нормальний розподіл')

        # Додавання статистичної інформації
        stats_text = (f'n = {len(residuals):,}\n'
                      f'\u03BC = {mean_val:.4f}\n'
                      f'\u03C3 = {std_val:.4f}')
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
                fontsize=14)

        # Налаштування зовнішнього вигляду
        ax.set_title(title, fontsize=20, fontweight='bold', pad=15)
        ax.set_xlabel('Залишки', fontsize=16, fontweight='bold')
        ax.set_ylabel('Частота', fontsize=16, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=12, loc='upper left')

        # Додавання горизонтальних ліній для стандартних відхилень
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_edgecolor('black')

        # Встановлення однакових меж по осі X для обох графіків
        # Визначення спільних меж на основі перцентилів
        all_residuals = np.concatenate([residuals_before, residuals_after])
        q_low, q_high = np.percentile(all_residuals, [0.5, 99.5])
        # Розширення меж на 10% для кращого відображення
        range_x = q_high - q_low
        xlim_min, xlim_max = q_low - 0.1 * range_x, q_high + 0.1 * range_x
        axes[i].set_xlim([xlim_min, xlim_max])

    # Спочатку застосовуємо tight_layout, щоб оптимізувати розміщення графіків
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Залишаємо місце для заголовка

    # Додаємо загальний заголовок після налаштування макету
    fig.suptitle('Порівняння розподілу залишків', fontsize=20, fontweight='bold', y=0.98)

    plt.show()


# In[15]:


def plot_regression_results(x, y, x_name, y_name, metrics):
    """
    Створює візуалізації результатів регресійного аналізу.

    Параметри:
    x (numpy.ndarray): Масив незалежних змінних
    y (numpy.ndarray): Масив залежних змінних
    x_name (str): Назва незалежної змінної
    y_name (str): Назва залежної змінної
    metrics (dict): Словник з метриками якості моделі
    """
    # Налаштування стилю
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
    })

    # Видобування метрик
    a0 = metrics["a0"]
    a1 = metrics["a1"]
    r_squared = metrics["r_squared"]
    residuals = metrics["residuals"]
    y_pred = metrics["y_pred"]
    std_error = metrics["std_error"]

    # Графік 1: Діаграма розсіювання з лінією регресії
    plt.figure(figsize=(10, 6))

    # Додаємо полігон довірчого інтервалу (якщо бажаєте)
    if len(x) > 2:  # Перевіряємо, що є достатньо точок для обчислення довірчого інтервалу
        # Сортуємо X та прогнозні значення для правильної побудови області
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_pred_sorted = y_pred[sort_idx]

        # Обчислюємо довірчий інтервал (приблизно)
        conf_interval = 1.96 * std_error  # 95% довірчий інтервал
        lower_bound = y_pred_sorted - conf_interval
        upper_bound = y_pred_sorted + conf_interval

        # Додаємо довірчий інтервал на графік
        plt.fill_between(x_sorted, lower_bound, upper_bound,
                         color='lightblue', alpha=0.3,
                         label='95% довірчий інтервал')

    # Додаємо точки даних та лінію регресії
    plt.scatter(x, y, alpha=0.7, color='blue', edgecolor='navy', s=50, label='Дані')
    plt.plot(x, y_pred, color='red', linewidth=2, label=f'y = {a0:.4f} + {a1:.4f}x')

    # Додаємо інформацію про модель
    equation_text = f'y = {a0:.4f} + {a1:.4f}x\nR² = {r_squared:.4f}\nСтандартна похибка = {std_error:.4f}'
    plt.annotate(equation_text, xy=(0.02, 0.97), xycoords='axes fraction',
                 ha='left', va='top', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    plt.title('Діаграма розсіювання з лінією регресії', fontweight='bold', pad=15)
    plt.xlabel(x_name, fontweight='bold')
    plt.ylabel(y_name, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', frameon=True, framealpha=0.9)
    plt.tight_layout()
    # plt.savefig('regression_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Графік 2: Залишки vs Прогнозовані значення
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.7, color='green', edgecolor='darkgreen', s=50)
    plt.axhline(y=0, color='r', linestyle='-', linewidth=1.5)

    # Додаємо горизонтальні лінії для стандартних відхилень
    std_residuals = np.std(residuals)
    plt.axhline(y=std_residuals, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    plt.axhline(y=-std_residuals, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    plt.axhline(y=2*std_residuals, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    plt.axhline(y=-2*std_residuals, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    # Додаємо зглажену лінію тренду для візуалізації паттернів
    if len(y_pred) > 10:  # Перевіряємо, що є достатньо точок
        try:
            from scipy.ndimage import gaussian_filter1d
            # Сортуємо за прогнозованими значеннями
            sort_idx = np.argsort(y_pred)
            y_pred_sorted = y_pred[sort_idx]
            residuals_sorted = residuals[sort_idx]
            # Застосовуємо згладжування
            smoothed = gaussian_filter1d(residuals_sorted, sigma=3)
            plt.plot(y_pred_sorted, smoothed, 'r--', alpha=0.5, linewidth=1.5)
        except ImportError:
            pass  # Якщо gaussian_filter1d недоступний, пропускаємо цей крок

    plt.title('Залишки vs Прогнозовані значення', fontweight='bold', pad=15)
    plt.xlabel('Прогнозовані значення', fontweight='bold')
    plt.ylabel('Залишки', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig('residuals_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Графік 3: Гістограма залишків
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

    # Налаштування гістограми
    bin_params = {}
    if len(residuals) > 1000:
        # Оптимальне число бінів для великих наборів даних
        data_range = np.max(residuals) - np.min(residuals)
        bin_width = 2 * stats.iqr(residuals) / (len(residuals) ** (1/3))
        n_bins = int(data_range / bin_width) if bin_width > 0 else 50
        n_bins = min(100, max(20, n_bins))  # Обмеження кількості бінів
        bin_params['bins'] = n_bins

    # Визначення статистик залишків
    mean_residuals = np.mean(residuals)
    std_residuals = np.std(residuals)
    std_error_mean = stats.sem(residuals)

    # Добавляємо гістограму з кращим форматуванням
    hist_color = '#440154'
    hist = sns.histplot(residuals, kde=False, color=hist_color, alpha=0.7,
                        edgecolor='black', linewidth=1.0, ax=ax, stat='count', **bin_params)

    # Додаємо нормальний розподіл для порівняння
    q_low, q_high = np.percentile(residuals, [0.1, 99.9])
    x_norm = np.linspace(q_low, q_high, 1000)
    y_norm = stats.norm.pdf(x_norm, mean_residuals, std_residuals)

    # Масштабування нормального розподілу до відповідної висоти гістограми
    # Використовуємо простіший метод масштабування, без використання np.diff
    bin_heights = [p.get_height() for p in ax.patches] if len(ax.patches) > 0 else []
    max_height = max(bin_heights) if bin_heights else len(residuals) / 20
    scale_factor = max_height / np.max(y_norm) if np.max(y_norm) > 0 else 1

    # Додаємо лінію нормального розподілу
    ax.plot(x_norm, y_norm * scale_factor, color='red', linewidth=2.5, alpha=0.7, label='Нормальний розподіл')

    # Додаємо вертикальну лінію для середнього значення
    ax.axvline(mean_residuals, color='red', linestyle='--', linewidth=2.0,
               label=f'Середнє значення: {mean_residuals:.4f}')

    # Налаштування зовнішнього вигляду графіка
    ax.set_title('Гістограма залишків', fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Залишки', fontsize=18, fontweight='bold')
    ax.set_ylabel('Частота', fontsize=18, fontweight='bold')

    # Додаємо текстову інформацію про статистику
    info_text = f'n = {len(residuals):,}\n'
    info_text += f'\u03BC = {mean_residuals:.4f}\n'
    info_text += f'\u03C3 = {std_residuals:.4f}'

    # Додаємо текстове поле з інформацією
    ax.text(0.97, 0.97, info_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
            fontsize=14)

    # Покращення легенди
    legend = ax.legend(loc='upper left', fontsize=14, frameon=True)
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_edgecolor('black')

    # Покращення сітки
    ax.grid(True, linestyle='--', alpha=0.6)

    # Покращення меж графіка
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_edgecolor('black')

    # Встановлюємо межі по x для кращого відображення
    # Використовуючи перцентилі замість просто min/max
    q_low, q_high = np.percentile(residuals, [0.5, 99.5])
    x_margin = (q_high - q_low) * 0.2  # 20% запас
    ax.set_xlim([q_low - x_margin, q_high + x_margin])

    plt.tight_layout()
    plt.show()

    # Графік 4: QQ-plot залишків з покращеним форматуванням
    plt.figure(figsize=(10, 6))

    # Використовуємо stats.probplot для створення QQ-plot
    (quantiles, ordered_residuals), (slope, intercept, r) = stats.probplot(residuals, dist="norm")

    # Малюємо точки QQ-plot з покращеним форматуванням
    plt.scatter(quantiles, ordered_residuals, color='darkblue', alpha=0.7, s=50)

    # Додаємо лінію очікуваного нормального розподілу
    line_x = np.array([quantiles.min(), quantiles.max()])
    line_y = intercept + slope * line_x
    plt.plot(line_x, line_y, 'r-', linewidth=2)

    # Додаємо інформацію про R корелології на QQ-plot
    plt.annotate(f'R = {r:.4f}', xy=(0.02, 0.97), xycoords='axes fraction',
                 ha='left', va='top', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    plt.title('QQ-plot залишків (перевірка нормальності)', fontweight='bold', pad=15)
    plt.xlabel('Теоретичні квантилі', fontweight='bold')
    plt.ylabel('Зразкові квантилі', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig('qq_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Графік 5: Залишки vs Незалежна змінна з вдосконаленим форматуванням
    plt.figure(figsize=(10, 6))
    plt.scatter(x, residuals, alpha=0.7, color='blue', edgecolor='navy', s=50)
    plt.axhline(y=0, color='r', linestyle='-', linewidth=1.5)

    # Додаємо горизонтальні лінії для стандартних відхилень
    plt.axhline(y=std_residuals, color='gray', linestyle='--', linewidth=1, alpha=0.7,
                label=f'+1 σ = {std_residuals:.4f}')
    plt.axhline(y=-std_residuals, color='gray', linestyle='--', linewidth=1, alpha=0.7,
                label=f'-1 σ = {-std_residuals:.4f}')

    # Додаємо локально зважену регресію (LOWESS) для виявлення тренду, якщо можливо
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        if len(x) > 10:  # Перевіряємо, що є достатньо точок
            # Сортуємо за x
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            residuals_sorted = residuals[sort_idx]
            # Застосовуємо LOWESS
            lowess_result = lowess(residuals_sorted, x_sorted, frac=0.3)
            plt.plot(lowess_result[:, 0], lowess_result[:, 1], 'g-',
                     alpha=0.7, linewidth=2, label='LOWESS тренд')
    except ImportError:
        pass  # Якщо lowess недоступний, пропускаємо цей крок

    plt.title(f'Залишки vs {x_name}', fontweight='bold', pad=15)
    plt.xlabel(x_name, fontweight='bold')
    plt.ylabel('Залишки', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    # plt.savefig('residuals_vs_x.png', dpi=300, bbox_inches='tight')
    plt.show()


# In[16]:


def analyze_linear_regression(x, y, x_name, y_name):
    """
    Проводить повний аналіз лінійної регресії.

    Параметри:
    x (numpy.ndarray): Масив незалежних змінних
    y (numpy.ndarray): Масив залежних змінних
    x_name (str): Назва незалежної змінної
    y_name (str): Назва залежної змінної

    Повертає:
    dict: Результати аналізу
    """
    print_header("Лінійний регресійний аналіз")

    # Розрахунок параметрів регресії
    a0, a1 = calculate_regression_parameters(x, y)
    print(f"Рівняння регресії: {y_name} = {a0:.4f} + {a1:.4f} * {x_name}")

    # Розрахунок метрик якості моделі
    metrics = calculate_regression_metrics(x, y, a0, a1)

    # Виведення повних результатів
    print_regression_results(x, y, x_name, y_name, metrics)

    # Візуалізація результатів
    plot_regression_results(x, y, x_name, y_name, metrics)

    return metrics


# In[17]:


# Отримання шляху до файлу
file_path = "cleanest_data.csv"

# Отримання інформації про стовпці
column_x = "partner_total_orders"  # Стовпець з незалежною змінною (X)
column_y = "partner_total_messages"  # Стовпець з залежною змінною (Y)


# Отримання розділювача
delimiter = ","  # Розділювач у CSV файлі

# Отримання рівня довіри
confidence = 0.95  # Рівень довіри

# Отримання кількості ітерацій
max_iterations = 1000000  # Максимальна кількість ітерацій для виявлення викидів


# In[18]:


# Зчитування даних
x_data, y_data, df, x_name, y_name = read_csv_data(file_path, column_x, column_y, delimiter)


# In[19]:


# Аналіз початкових даних
print_header("АНАЛІЗ ПОЧАТКОВИХ ДАНИХ")
results_before = analyze_linear_regression(x_data, y_data, x_name, y_name)


# ## Аналіз графіків

# ### Діаграма розсіювання з лінією регресії
# #### Опис графіка
# Представлений графік є діаграмою розсіювання з побудованою лінією регресії, що відображає взаємозв'язок між загальною кількістю замовлень клієнта (`partner_total_orders`, вісь X) та кількістю повідомлень, якими обмінялися партнери з менеджерами компанії (`partner_total_messages`, вісь Y).
# 
# Основні елементи графіка:
# - Сині точки відображають фактичні дані спостережень
# - Червона лінія представляє побудовану лінійну регресію
# - Блакитна область навколо лінії регресії показує 95% довірчий інтервал
# - У верхньому лівому куті наведено рівняння регресії, коефіцієнт детермінації та значення стандартної помилки
# 
# #### Кількісні характеристики моделі
# 
# Згідно з інформацією на графіку:
# - Рівняння регресії: y = 19.2445 + 8.8312x
# - Коефіцієнт детермінації: R² = 0.9448
# - Стандартна помилка: 322.3087
# 
# #### Аналіз результатів
# 
# #### Характер залежності
# 
# На діаграмі розсіювання чітко прослідковується виражена позитивна лінійна залежність між кількістю замовлень та кількістю повідомлень. Це означає, що зі збільшенням кількості замовлень спостерігається пропорційне зростання обсягу комунікації з партнерами.
# 
# #### Інтерпретація коефіцієнтів регресії
# 
# **Вільний член (19.2445)**: Означає, що навіть за відсутності замовлень (x = 0) прогнозується близько 19 повідомлень, що може відображати базовий рівень комунікації, необхідний для підтримки відносин з партнером. **Коефіцієнт нахилу (8.8312)**: Показує, що кожне додаткове замовлення асоціюється з приблизно 8.83 додатковими повідомленнями. Цей показник відображає середню інтенсивність комунікації, необхідну для обробки одного замовлення.
# 
# #### Статистична значущість моделі
# 
# Надзвичайно високий коефіцієнт детермінації (R² = 0.9448) свідчить про те, що модель пояснює близько 94.48% варіації в кількості повідомлень. Це вказує на дуже сильний взаємозв'язок між досліджуваними змінними та високу прогностичну здатність моделі.
# 
# #### Аналіз розподілу точок
# 
# На графіку можна спостерігати кілька особливостей розподілу даних: **Щільність точок**: Більша концентрація спостережень спостерігається в нижній лівій частині графіка, де кількість замовлень і повідомлень є відносно низькою. Це вказує на те, що більшість партнерів має невелику кількість замовлень і, відповідно, меншу інтенсивність комунікації. **Розсіювання точок**: З ростом значень на обох осях збільшується розкид точок навколо лінії регресії. Це може свідчити про наявність гетероскедастичності (непостійність дисперсії помилок) або про те, що високоактивні партнери демонструють більш різноманітні патерни комунікаційної поведінки. **Потенційні викиди**: На графіку помітні окремі точки, які значно відхиляються від загальної тенденції, особливо в області високих значень обох змінних. Ці потенційні викиди можуть впливати на параметри моделі.
# 
# #### Довірчий інтервал
# 
# Спостерігається розширення 95% довірчого інтервалу у міру збільшення значень по осі X, що є нормальним явищем і свідчить про те, що прогнози стають менш точними для партнерів з великою кількістю замовлень. Це може бути пов'язано як з меншою кількістю спостережень у цій області, так і з більшою варіабельністю комунікаційної поведінки таких партнерів.
# 
# #### Висновки та рекомендації
# 
# **Статистично значуща залежність**: Встановлено дуже сильний лінійний взаємозв'язок між кількістю замовлень та обсягом комунікації з партнерами.
# **Бізнес-застосування**: Модель може бути використана для: Прогнозування необхідних комунікаційних ресурсів на основі очікуваної кількості замовлень, виявлення партнерів з аномальними патернами комунікації (ті, що значно відхиляються від лінії регресії), оптимізації штату менеджерів з комунікації відповідно до прогнозованого обсягу взаємодії.
# **Подальший аналіз**: Рекомендується провести аналіз залишків для перевірки припущень регресійної моделі, виявити та дослідити потенційні викиди, які можуть спотворювати результати, розглянути можливість побудови окремих моделей для різних сегментів партнерів.
# **Обмеження інтерпретації**: Варто зазначити, що встановлений взаємозв'язок не обов'язково означає причинно-наслідкову залежність. Потрібні додаткові дослідження, щоб визначити, чи призводить більша кількість комунікації до збільшення замовлень, чи навпаки, більша кількість замовлень просто вимагає інтенсивнішої комунікації.

# ### Графік залишків відносно прогнозованих значень
# 
# #### Опис графіка
# 
# Представлений графік відображає залежність залишків регресійної моделі від прогнозованих значень. Це один із ключових діагностичних інструментів для оцінки адекватності лінійної регресійної моделі.
# 
# На графіку представлено:
# - Зелені точки - залишки моделі для різних спостережень
# - Червона горизонтальна лінія на рівні нуля - ідеальна лінія, що відповідає нульовим залишкам
# - Пунктирні горизонтальні лінії - ймовірно, межі стандартних відхилень залишків
# - Коричневі та рожеві точки - можливо, залишки для окремих категорій даних або точки з особливим статусом
# 
# #### Аналіз розподілу залишків
# 
# #### Відхилення від нульової лінії
# 
# На графіку спостерігається значне розсіювання залишків відносно нульової лінії. При цьому чітко видно декілька окремих "криволінійних структур", що формують своєрідні кластери або групи залишків. Ці структури мають вигляд вигнутих ліній, що поступово відхиляються від нульової лінії зі збільшенням прогнозованих значень.
# 
# #### Гетероскедастичність
# 
# Графік демонструє виражену гетероскедастичність - розсіювання залишків нерівномірне вздовж осі прогнозованих значень. В області низьких прогнозованих значень (до 2000) спостерігається значно менше розсіювання, тоді як при збільшенні прогнозованих значень розмах залишків істотно зростає, досягаючи максимальних відхилень (до ±2000) при високих прогнозованих значеннях.
# 
# #### Нелінійні патерни
# 
# Наявність криволінійних структур у розподілі залишків є сильним індикатором того, що в даних присутні нелінійні взаємозв'язки, які не враховані в лінійній моделі. Це може свідчити про потребу в нелінійній трансформації змінних, наявність пропущених важливих незалежних змінних, складну структуру даних, що потребує більш гнучкої моделі.
# 
# #### Аномальні спостереження
# 
# На графіку присутні окремі групи точок, які формують відособлені структури, особливо в верхній частині графіка (позитивні залишки) з прогнозованими значеннями від 4000 до 6000, а також у нижній частині (негативні залишки) для прогнозованих значень 6000-8000. Ці спостереження можуть представляти собою викиди або особливі підгрупи в даних, що не відповідають загальній тенденції.
# 
# #### Інтерпретація та висновки
# 
# #### Порушення передумов лінійної регресії
# 
# Характер розподілу залишків свідчить про порушення кількох ключових передумов лінійної регресії: **Лінійність** - явна присутність нелінійних патернів вказує на те, що лінійна модель не повністю описує взаємозв'язок між змінними. **Гомоскедастичність** - нерівномірне розсіювання залишків свідчить про порушення умови постійної дисперсії помилок. **Незалежність спостережень** - структурованість залишків може вказувати на наявність кластеризації або автокореляції в даних.
# 
# #### Проблема специфікації моделі
# 
# Виражена структурованість залишків свідчить про те, що поточна специфікація моделі не повністю відображає складність зв'язків у даних. Це може призводити до зміщених оцінок коефіцієнтів регресії та неточних прогнозів.
# 
# #### Наявність підгруп у даних
# 
# Формування окремих "ліній" або "треків" залишків може вказувати на те, що дані складаються з кількох підгруп з різними патернами взаємозв'язку між змінними. Ці підгрупи можуть відповідати різним сегментам клієнтів або типам взаємодії, які варто моделювати окремо.
# 
# #### Рекомендації для покращення моделі
# 
# **Видалення викидів** - ідентифікація та аналіз спостережень з найбільшими залишками, що можуть непропорційно впливати на параметри моделі.
# **Нелінійні трансформації** - розглянути можливість застосування логарифмічних, квадратичних або інших нелінійних трансформацій змінних.
# **Сегментація даних** - проаналізувати можливість розділення даних на більш однорідні групи і побудови окремих моделей для кожного сегмента.
# **Додаткові змінні** - включення в модель додаткових предикторів, які можуть покращити її пояснювальну здатність і зменшити систематичні патерни в залишках.
# **Альтернативні методи моделювання** - розглянути застосування більш гнучких методів регресійного аналізу, таких як поліноміальна регресія, сплайн-регресія або методи машинного навчання.
# 
# #### Висновок для бізнес-застосування
# 
# Аналіз графіка залишків вказує на те, що хоча лінійна регресійна модель може давати загальне уявлення про взаємозв'язок між кількістю замовлень та обсягом комунікації, вона не повністю відображає складність цих взаємозв'язків. Для прийняття більш точних бізнес-рішень рекомендується провести додатковий аналіз для виявлення різних сегментів партнерів з різними патернами взаємодії, розробити більш гнучкі моделі, що враховують нелінійні аспекти взаємозв'язку, дослідити контекстуальні фактори, що можуть пояснювати відхилення від лінійної тенденції
# Це дозволить отримати більш точні прогнози необхідних комунікаційних ресурсів та оптимізувати взаємодію з партнерами в різних сегментах.

# ### Гістограма залишків регресійної моделі
# 
# #### Опис графіка
# 
# Представлений графік є гістограмою розподілу залишків регресійної моделі. Гістограма візуалізує частотний розподіл залишків, що дозволяє оцінити їх відповідність нормальному розподілу — одній із ключових передумов класичної лінійної регресії.
# 
# На графіку зображено:
# - Фіолетові стовпчики — частотний розподіл залишків моделі
# - Червона крива — теоретична крива нормального розподілу
# - Вертикальна пунктирна лінія — середнє значення залишків
# - У правому верхньому куті — статистичні характеристики розподілу
# 
# #### Статистичні характеристики розподілу
# 
# Згідно з інформацією, наведеною на графіку:
# - Кількість спостережень (n): 86,794
# - Середнє значення залишків (μ): -0.0000
# - Стандартне відхилення (σ): 322.3050
# 
# #### Аналіз розподілу залишків
# 
# #### Центрованість розподілу
# 
# Середнє значення залишків дорівнює практично нулю (-0.0000), що є дуже хорошим показником. Це свідчить про те, що модель не має систематичної помилки прогнозування в один бік (переоцінки чи недооцінки) і відповідає важливій умові незміщеності оцінок у методі найменших квадратів.
# 
# #### Форма розподілу
# 
# Аналіз форми гістограми показує наступне: розподіл має чітко виражений симетричний характер, що візуально наближається до нормального, пік розподілу припадає на нульове значення залишків, спостерігається висока концентрація залишків у центральній частині розподілу (близько нуля), частота поступово зменшується в міру віддалення від центру в обидва боки
# 
# #### Відхилення від нормальності
# 
# Порівнюючи емпіричну гістограму з теоретичною кривою нормального розподілу, можна відзначити певні відхилення, а саме: центральний пік емпіричного розподілу є дещо вищим, ніж передбачає нормальна крива, що свідчить про лептокуртичність (гостровершинність) розподілу, на деяких ділянках гістограми помітні дрібні нерегулярності, які відхиляються від гладкої теоретичної кривої, в області далеких "хвостів" розподілу (значення більше ±1000) видно поодинокі спостереження, які можуть представляти потенційні викиди. Незважаючи на ці незначні відхилення, загалом розподіл залишків досить добре відповідає нормальному закону.
# 
# #### Діапазон залишків
# 
# Гістограма демонструє, що основна маса залишків знаходиться в діапазоні приблизно від -500 до +500. Залишки, що виходять за межі цього діапазону, трапляються значно рідше, хоча є окремі спостереження з відхиленнями до ±2000. Ці екстремальні значення можуть представляти аномальні випадки або вплив значних викидів у даних.
# 
# #### Інтерпретація результатів для моделі
# 
# #### Відповідність передумовам регресії
# 
# Аналіз гістограми залишків дозволяє зробити наступні висновки щодо відповідності регресійної моделі стандартним передумовам:
# **Нормальність розподілу залишків**: Ця передумова в цілому виконується, хоча є незначні відхилення у формі розподілу. Для великих вибірок (n = 86,794) такі відхилення зазвичай не створюють серйозних проблем для статистичного висновку.
# **Нульове середнє залишків**: Ця умова виконується ідеально, що свідчить про коректність специфікації моделі з точки зору її центрованості.
# **Однорідність дисперсії**: Гістограма не дає прямої інформації про гомоскедастичність, оскільки не показує залежність залишків від прогнозованих значень. Однак загальний розподіл залишків має досить чіткі межі, що може непрямо свідчити про відносну стабільність дисперсії.
# 
# #### Інтерпретація стандартного відхилення
# 
# Стандартне відхилення залишків (σ = 322.3050) відображає середній рівень похибки прогнозування моделі. Це означає, що в середньому прогнози моделі відхиляються від фактичних значень приблизно на ±322 одиниці. Для оцінки практичної значущості цього рівня помилки потрібно співвіднести його з масштабом досліджуваної змінної (partner_total_messages). Без додаткової інформації про діапазон цієї змінної складно точно оцінити відносну величину помилки.
# 
# #### Висновки та рекомендації
# 
# #### Висновки щодо якості моделі
# **Загальна адекватність моделі**: Розподіл залишків близький до нормального з нульовим середнім, що свідчить про загальну адекватність лінійної специфікації моделі.
# **Точність прогнозування**: Стандартне відхилення залишків дає уявлення про середню абсолютну помилку прогнозів моделі. Для визначення, чи є ця помилка прийнятною, необхідно співвіднести її з практичними вимогами до точності прогнозів.
# **Потенційні проблеми**: Наявність окремих екстремальних значень залишків вказує на можливість покращення моделі шляхом обробки викидів або введення додаткових предикторів.
# 
# #### Практичні рекомендації
# **Оцінка впливу викидів**: Провести додатковий аналіз спостережень з екстремальними значеннями залишків для визначення їх природи та потенційного впливу на параметри моделі.
# **Тестування трансформацій змінних**: Хоча розподіл залишків близький до нормального, деякі нелінійні трансформації предикторів можуть ще більше покращити цю відповідність.
# **Інтерпретація прогнозів**: При використанні моделі для прогнозування рекомендується враховувати довірчі інтервали, базуючись на стандартному відхиленні залишків.
# **Сегментація даних**: Якщо стандартне відхилення залишків вважається занадто високим для практичного застосування, варто розглянути можливість сегментації даних і побудови окремих моделей для різних груп партнерів.
# 
# Загалом, аналіз гістограми залишків підтверджує, що модель лінійної регресії є адекватним інструментом для опису взаємозв'язку між комунікацією та кількістю замовлень партнерів B2B компанії, хоча і має певний потенціал для подальшого вдосконалення.
# 

# ### QQ-plot залишків регресійної моделі
# 
# #### Опис графіка
# 
# Представлений графік є QQ-plot (Quantile-Quantile plot) залишків регресійної моделі. Це важливий діагностичний інструмент для перевірки відповідності розподілу залишків нормальному закону. QQ-plot порівнює квантилі емпіричного розподілу залишків з теоретичними квантилями нормального розподілу.
# 
# На графіку зображено:
# - Синя лінія з точок — емпіричні квантилі залишків
# - Червона пряма лінія — теоретична лінія, яка відповідає ідеальному нормальному розподілу
# - У лівому верхньому куті — коефіцієнт кореляції між емпіричними та теоретичними квантилями (R = 0.7763)
# 
# #### Аналіз відхилень від нормальності
# 
# #### Загальна відповідність нормальному розподілу
# 
# Коефіцієнт кореляції між емпіричними та теоретичними квантилями становить R = 0.7763. Це значення помірно високе, але все ж вказує на певні відхилення від нормального розподілу. Для ідеального нормального розподілу значення R наближалося б до 1.
# 
# #### S-подібна форма QQ-кривої
# 
# Графік демонструє виражену S-подібну форму, що є характерною ознакою розподілу з "важкими хвостами" (heavy-tailed distribution). Це означає, що розподіл залишків має більшу ймовірність екстремальних значень, ніж нормальний розподіл:
# **Центральна частина** (квантилі від -1 до 1): У цій області синя лінія майже горизонтальна і значно відхиляється від червоної теоретичної лінії. Це вказує на високу концентрацію залишків близько нуля, тобто на лептокуртичність (гостровершинність) розподілу.
# **Хвости розподілу** (квантилі < -2 та > 2): У цих областях емпірична лінія має значно крутіший нахил, ніж теоретична, що свідчить про більшу частоту великих за абсолютною величиною залишків.
# 
# #### Асиметрія та "сходинки" на графіку
# 
# На QQ-plot помітні горизонтальні "сходинки" або "плато", особливо в хвостах розподілу. Така структура може свідчити про дискретність або кластеризацію в даних, наявність значної кількості однакових або дуже близьких значень залишків, можливу присутність окремих підгруп у даних з різними характеристиками розподілу. Крім того, верхній правий хвіст (великі позитивні залишки) виглядає дещо довшим і крутішим, ніж нижній лівий (великі негативні залишки), що може вказувати на певну асиметричність розподілу з ухилом у бік позитивних залишків.
# 
# #### Інтерпретація для регресійної моделі
# 
# #### Порушення передумов лінійної регресії
# 
# QQ-plot явно демонструє відхилення від нормальності розподілу залишків, що є однією з ключових передумов класичної лінійної регресії. Це відхилення має специфічну форму:
# **Надмірна концентрація залишків близько нуля**: Може свідчити про те, що модель добре працює для "типових" спостережень, але має проблеми з описом нетипових випадків.
# **Важкі хвости розподілу**: Вказують на частіше, ніж передбачає нормальний розподіл, виникнення значних відхилень від прогнозованих значень, що може бути пов'язано з неврахованими факторами або нелінійними ефектами.
# 
# #### Вплив на статистичні висновки
# 
# Відхилення від нормальності може впливати на точність розрахунку довірчих інтервалів для коефіцієнтів регресії, надійність t-тестів та F-тестів для перевірки статистичної значущості, точність передбачуваних значень, особливо при екстраполяції. У великих вибірках (як у даному випадку) ці проблеми можуть бути менш критичними завдяки центральній граничній теоремі, але все одно варто враховувати їх при інтерпретації результатів.
# 
# #### Свідчення про структурні особливості даних
# 
# S-подібна форма QQ-plot з горизонтальними плато може вказувати на наявність суміші різних розподілів у вибірці, потенційну необхідність сегментації даних або створення категоріальних змінних, можливі проблеми з вимірюванням або збором даних, які призводять до кластеризації значень
# 
# #### Рекомендації для покращення моделі
# 
# #### Трансформація змінних
# 
# Відхилення від нормальності можна спробувати скоригувати через трансформацію змінних:
# - Логарифмічна трансформація (для даних з правою асиметрією)
# - Квадратний корінь (для даних з помірною правою асиметрією)
# - Box-Cox трансформація для визначення оптимального перетворення
# 
# #### Робастні методи оцінювання
# 
# Враховуючи важкі хвости розподілу, доцільно застосувати робастні методи регресійного аналізу, які менш чутливі до відхилень від нормальності:
# - М-оцінювачі (Huber, Tukey)
# - Квантильна регресія
# 
# #### Сегментація даних
# 
# Горизонтальні плато та загальна S-подібна форма QQ-plot можуть свідчити про доцільність сегментації даних:
# - Кластерний аналіз для виявлення природних груп
# - Створення окремих моделей для різних сегментів партнерів
# - Введення нових категоріальних предикторів, які відображають сегментацію
# 
# #### Видалення викидів та впливових спостережень
# 
# Важкі хвости розподілу можуть бути спричинені наявністю викидів:
# - Ідентифікація спостережень з екстремальними залишками
# - Аналіз впливовості спостережень (leverage, відстань Кука)
# - Вибіркове видалення або зважування спостережень при побудові моделі
# 
# #### Висновки для бізнес-застосування
# 
# Аналіз QQ-plot залишків дозволяє зробити важливі висновки для практичного застосування регресійної моделі у прогнозуванні комунікаційної активності партнерів B2B компанії:
# **Обережність при інтерпретації**: Хоча модель може бути корисною для загального розуміння тенденцій, певні відхилення від нормальності свідчать про необхідність обережної інтерпретації статистичних тестів та довірчих інтервалів.
# **Диференційований підхід до прогнозування**: Наявність різних патернів у розподілі залишків вказує на те, що для різних сегментів партнерів можуть бути доцільними різні моделі прогнозування комунікаційної активності.
# **Ідентифікація особливих випадків**: Важкі хвости розподілу вказують на наявність партнерів, взаємодія з якими значно відрізняється від типових патернів. Виявлення та аналіз таких випадків може дати цінні інсайти для розробки більш персоналізованих стратегій комунікації.
# **Підвищена увага до екстремальних прогнозів**: При використанні моделі для прогнозування комунікаційних ресурсів варто враховувати, що модель може систематично недооцінювати або переоцінювати необхідну інтенсивність комунікації для нетипових партнерів.
# 
# Загалом, QQ-plot свідчить про те, що хоча лінійна регресійна модель здатна вловити основні тенденції у взаємозв'язку між кількістю замовлень та обсягом комунікації, для отримання більш точних і надійних прогнозів доцільно розглянути більш гнучкі методи моделювання та сегментований підхід до аналізу партнерів.
# 

# ### Графік залишків відносно кількості замовлень партнера
# 
# #### Опис графіка
# 
# Представлений графік відображає залежність залишків регресійної моделі від незалежної змінної `partner_total_orders` (кількість замовлень партнера). Цей вид діагностичного графіка дозволяє оцінити, чи існує систематична залежність між похибками прогнозування та основним предиктором моделі.
# 
# На графіку зображено:
# - Сині точки — залишки моделі для різних значень кількості замовлень
# - Червона горизонтальна лінія на рівні нуля — ідеальна лінія, що відповідає відсутності систематичних відхилень
# - Пунктирні горизонтальні лінії — межі ±1σ (одне стандартне відхилення, σ = 322.3050)
# - Зелена лінія — LOWESS тренд (локально зважена поліноміальна регресія), що показує загальну тенденцію в залишках
# 
# #### Аналіз патернів залишків
# 
# #### Загальний розподіл залишків по осі X
# 
# Залишки демонструють неоднорідний і складний патерн розподілу вздовж осі `partner_total_orders`. Можна спостерігати декілька ключових особливостей:
# **Висока концентрація спостережень в лівій частині** (низькі значення кількості замовлень) — це відповідає тому, що більшість партнерів мають відносно невелику кількість замовлень.
# **Зменшення щільності точок** з ростом значень `partner_total_orders` — партнерів з великою кількістю замовлень значно менше.
# **Формування виразних "треків" або "ліній"** — точки на графіку утворюють чіткі криволінійні структури, що розходяться від початку координат подібно до променів. Така структура вказує на наявність підгруп у даних з різними патернами взаємозв'язку.
# 
# #### Тренди та систематичні відхилення
# **LOWESS тренд** (зелена лінія) демонструє слабке, але помітне систематичне відхилення від нульової лінії зі збільшенням кількості замовлень. Цей тренд має легкий позитивний нахил, що може свідчити про тенденцію до недооцінки кількості повідомлень для партнерів з великою кількістю замовлень.
# **Амплітуда відхилень** залишків від нульової лінії значно збільшується з ростом `partner_total_orders`, що є прямою ознакою гетероскедастичності — непостійності дисперсії помилок.
# **Асиметрія у розподілі залишків** — для деяких значень `partner_total_orders` спостерігається асиметрія в розподілі залишків, де позитивні відхилення мають більшу амплітуду, ніж негативні, або навпаки.
# 
# #### Структурні особливості
# **Виразні криволінійні "треки"** залишків вказують на те, що дані можуть бути неоднорідними і складатися з кількох груп партнерів, кожна з яких має власну функціональну залежність між кількістю замовлень та кількістю повідомлень.
# **Розширення діапазону залишків** з ростом `partner_total_orders` свідчить про те, що прогнозування стає менш точним для партнерів з більшою кількістю замовлень.
# **Наявність окремих віддалених груп точок** може вказувати на присутність підвибірок з особливими характеристиками або на вплив неврахованих факторів.
# 
# #### Інтерпретація для регресійної моделі
# 
# #### Порушення передумов лінійної регресії
# 
# Графік залишків відносно `partner_total_orders` чітко демонструє порушення низки ключових передумов лінійної регресії:
# **Гомоскедастичність** — умова постійної дисперсії залишків значно порушується, оскільки розсіювання залишків істотно зростає зі збільшенням значень предиктора.
# **Лінійність зв'язку** — наявність криволінійних структур вказує на те, що істинний взаємозв'язок між змінними може бути нелінійним або неоднаковим для різних підгруп даних.
# **Незалежність спостережень** — чіткі патерни в залишках свідчать про можливу кластеризацію даних або наявність неврахованих змінних, що впливають на залежну змінну.
# 
# #### Наслідки для якості прогнозування
# **Зміщеність прогнозів** — слабкий, але систематичний тренд у залишках (зелена лінія) вказує на потенційну зміщеність прогнозів: модель може систематично недооцінювати або переоцінювати кількість повідомлень для певних діапазонів кількості замовлень.
# **Непостійна точність прогнозів** — зростання розсіювання залишків з ростом `partner_total_orders` означає, що довірчі інтервали для прогнозів повинні бути ширшими для партнерів з більшою кількістю замовлень.
# **Невраховані взаємозв'язки** — наявність криволінійних структур у залишках вказує на те, що модель не враховує важливі закономірності у даних.
# 
# #### Рекомендації для покращення моделі
# 
# #### Трансформація змінних
# **Нелінійні трансформації предикторів** — спробувати логарифмічну, квадратичну або інші трансформації змінної `partner_total_orders` для кращого врахування нелінійних взаємозв'язків.
# **Зважені методи найменших квадратів** — застосувати зважування спостережень, обернено пропорційне до дисперсії залишків у відповідних діапазонах предиктора, для корекції гетероскедастичності.
# 
# #### Сегментація та удосконалення моделі
# **Кластерний аналіз** — ідентифікувати природні групи партнерів на основі їхніх характеристик та патернів взаємодії.
# **Сегментована регресія** — розробити окремі регресійні моделі для різних сегментів партнерів або застосувати кусково-лінійну регресію.
# **Додаткові предиктори** — включити до моделі додаткові змінні, які можуть пояснити виявлені патерни у залишках (наприклад, тип партнера, тривалість співпраці, галузь тощо).
# **Складніші моделі** — розглянути можливість застосування більш гнучких підходів, таких як поліноміальна регресія, регресія з взаємодіями між змінними, або методи машинного навчання.
# 
# #### Висновки для бізнес-застосування
# 
# Аналіз графіка залишків відносно кількості замовлень дозволяє зробити важливі висновки для бізнес-застосування моделі:
# **Диференційований підхід до партнерів** — чітко видно, що взаємозв'язок між кількістю замовлень та комунікаційною активністю не є однаковим для всіх партнерів. Компанії варто розробити сегментовану стратегію комунікації, яка враховує різні патерни взаємодії.
# **Увага до партнерів з великою кількістю замовлень** — збільшення розсіювання залишків для цієї групи вказує на те, що їхні потреби в комунікації можуть суттєво відрізнятися від "середніх" показників. Це може вимагати більш індивідуалізованого підходу.
# **Аналіз "треків" на графіку** — ідентифікація та аналіз партнерів, що формують окремі "треки" на графіку, може дати цінні інсайти щодо різних типів комунікаційної поведінки та допомогти у розробці більш цільових бізнес-стратегій.
# **Оцінка ресурсів для комунікації** — при плануванні комунікаційних ресурсів необхідно враховувати, що точність прогнозів знижується для партнерів з великою кількістю замовлень, і закладати відповідні резерви або застосовувати більш складні моделі прогнозування.
# 
# Загалом, графік залишків вказує на те, що хоча лінійна модель дає певне уявлення про взаємозв'язок між кількістю замовлень та обсягом комунікації, для практичного застосування необхідно враховувати наявність різних сегментів партнерів та потенційну нелінійність цього взаємозв'язку.

# In[20]:


# Виявлення та видалення викидів
x_clean, y_clean, outlier_indices = remove_outliers_from_data(x_data, y_data, x_name, y_name, confidence, max_iterations)


# In[21]:


# Візуалізація гістограм до та після очищення
plot_histograms(x_data, x_clean, x_name)


# ### Аналіз порівняльних гістограм кількості замовлень партнера до і після очищення даних
# 
# #### Опис графіка
# 
# Представлений графік демонструє порівняння розподілів кількості замовлень партнера (`partner_total_orders`) до та після процедури очищення даних від викидів. На графіку відображено:
# 
# - Сині стовпці — гістограма розподілу початкових даних
# - Зелені стовпці — гістограма розподілу очищених даних
# - Синя крива — апроксимація розподілу початкових даних
# - Вертикальні пунктирні лінії — імовірно, позначають середні значення відповідних розподілів
# - У легенді вказано статистичні характеристики обох розподілів:
#   - Початкові дані: μ = 83.70, σ = 151.01
#   - Очищені дані: μ = 64.38, σ = 94.95
# 
# #### Аналіз змін у розподілі даних
# 
# #### Зміни статистичних характеристик
# 
# Порівняння статистичних характеристик до та після очищення даних виявляє суттєві зміни:
# **Середнє значення** зменшилося з 83.70 до 64.38, що відповідає зниженню на 23.1%. Це вказує на те, що видалені викиди містили переважно спостереження з великими значеннями кількості замовлень.
# **Стандартне відхилення** зменшилося з 151.01 до 94.95, що відповідає зниженню на 37.1%. Це свідчить про значне підвищення однорідності даних після очищення.
# **Співвідношення σ/μ** (коефіцієнт варіації) зменшилося з 1.80 до 1.47, що вказує на підвищення відносної однорідності розподілу.
# 
# #### Зміни форми розподілу
# 
# Аналіз гістограм показує наступні зміни у формі розподілу:
# **Висота першого стовпця** (найнижчі значення кількості замовлень) зменшилася після очищення, що може вказувати на видалення деяких партнерів з дуже малим числом замовлень, які були ідентифіковані як статистичні викиди.
# **Правий "хвіст" розподілу** (область високих значень) став коротшим після очищення, що відображає видалення партнерів з аномально великою кількістю замовлень.
# **Пропорційний розподіл частот** змінився — очищені дані демонструють більшу концентрацію в середній частині розподілу порівняно з початковими даними.
# **Загальна форма розподілу** в обох випадках залишається яскраво вираженою правосторонньою (позитивною) асиметрією, де більшість значень сконцентрована в лівій частині, а правий "хвіст" розтягнутий у бік великих значень.
# 
# #### Діапазон значень
# 
# Хоча точний діапазон значень складно визначити безпосередньо з графіка, можна зробити такі спостереження:
# Максимальні значення на осі X сягають приблизно 600 для початкових даних, що вказує на наявність партнерів з дуже великою кількістю замовлень.
# Після очищення розподіл стає більш компактним, хоча все ще охоплює значний діапазон значень, що відображає природну варіативність у кількості замовлень серед партнерів.
# Видалення викидів, імовірно, зменшило максимальні значення `partner_total_orders` у наборі даних, що призвело до скорочення загального діапазону.
# 
# #### Інтерпретація результатів очищення даних
# 
# #### Вплив на характеристики вибірки
# **Підвищення репрезентативності**: Зменшення стандартного відхилення та середнього значення свідчить про те, що очищена вибірка краще представляє "типових" партнерів B2B компанії, без спотворення, що вносять екстремальні випадки.
# **Зменшення впливу екстремальних значень**: Видалення викидів зменшило вплив аномальних спостережень, які могли непропорційно впливати на результати статистичного аналізу.
# **Збереження загальної структури розподілу**: Незважаючи на видалення викидів, форма розподілу залишилася подібною, що вказує на те, що процедура очищення не спотворила базові характеристики даних.
# 
# #### Наслідки для регресійного аналізу
# **Підвищення стабільності оцінок**: Зменшення варіабельності даних може призвести до отримання більш стабільних і надійних оцінок коефіцієнтів регресії.
# **Потенційне покращення якості моделі**: Видалення викидів може покращити відповідність даних припущенням лінійної регресії, зокрема щодо нормальності розподілу залишків.
# **Можливе звуження області застосування**: Слід зауважити, що очищена модель може бути менш придатною для прогнозування поведінки партнерів з екстремальними значеннями кількості замовлень, які були виключені з аналізу.
# 
# #### Бізнес-імплікації спостережуваних змін
# 
# #### Розуміння клієнтської бази
# **Сегментація партнерів**: Значна різниця між статистичними характеристиками до та після очищення вказує на те, що компанія має справу з неоднорідною клієнтською базою, де є як "типові" партнери з відносно невеликою кількістю замовлень, так і "нетипові" з дуже великою активністю.
# **Диференційований підхід**: Виявлені відмінності обґрунтовують необхідність розробки диференційованих стратегій взаємодії для різних сегментів партнерів.
# 
# #### Прогнозування та планування
# **Вибір відповідної моделі**: При прогнозуванні комунікаційних потреб важливо враховувати, чи включені у вибірку для аналізу партнери з екстремальними значеннями, і відповідно обирати модель.
# **Розрахунок ресурсів**: Значне зменшення середнього значення та стандартного відхилення після очищення даних може вплинути на оцінку необхідних комунікаційних ресурсів при прогнозуванні для всієї клієнтської бази.
# 
# #### Рекомендації для подальшого аналізу
# **Порівняння регресійних моделей**: Доцільно порівняти параметри та якість регресійних моделей, побудованих на початкових та очищених даних, щоб оцінити вплив викидів на результати аналізу.
# **Деталізований аналіз викидів**: Дослідження спостережень, ідентифікованих як викиди, може дати цінну інформацію про характеристики та потреби специфічних сегментів партнерів.
# **Розгляд трансформацій даних**: Враховуючи виражену правосторонню асиметрію обох розподілів, можна розглянути можливість логарифмічної чи іншої трансформації змінної `partner_total_orders` для покращення відповідності даних припущенням регресійного аналізу.
# **Сегментований аналіз**: Розглянути можливість побудови окремих моделей для різних сегментів партнерів, визначених на основі кількості замовлень чи інших характеристик.
# 
# #### Висновки
# 
# Порівняння гістограм кількості замовлень партнера до та після очищення даних виявляє суттєвий вплив процедури видалення викидів на статистичні характеристики вибірки. Середнє значення та стандартне відхилення зменшилися на 23.1% та 37.1% відповідно, що вказує на значне підвищення однорідності даних.
# Хоча очищення даних потенційно покращує стабільність та надійність статистичних оцінок, важливо зберігати інформацію про видалені спостереження, оскільки вони можуть представляти важливі сегменти клієнтської бази з особливими потребами та патернами взаємодії.
# Для прийняття обґрунтованих бізнес-рішень рекомендується враховувати результати аналізу як початкових, так і очищених даних, а також розглянути можливість сегментованого підходу до моделювання взаємозв'язку між кількістю замовлень та комунікаційною активністю партнерів.
# 

# In[22]:


plot_histograms(y_data, y_clean, y_name)


# ### Аналіз порівняльних гістограм кількості повідомлень партнера до і після очищення даних
# 
# #### Опис графіка
# 
# Представлений графік демонструє порівняння розподілів кількості повідомлень партнера (`partner_total_messages`) до та після процедури очищення даних від викидів. На графіку відображено:
# 
# - Сині стовпці — гістограма розподілу початкових даних
# - Зелені стовпці — гістограма розподілу очищених даних
# - Синя крива — апроксимація розподілу початкових даних
# - Вертикальні пунктирні лінії — позначають середні значення відповідних розподілів
# - У легенді вказано статистичні характеристики обох розподілів:
#   - Початкові дані: μ = 758.38, σ = 1372.01
#   - Очищені дані: μ = 578.80, σ = 850.31
# 
# #### Аналіз змін у розподілі даних
# 
# #### Зміни статистичних характеристик
# 
# Порівняння статистичних характеристик до та після очищення даних виявляє значні зміни:
# **Середнє значення** зменшилося з 758.38 до 578.80, що відповідає зниженню на 23.7%. Це свідчить про те, що видалені викиди містили переважно спостереження з високою інтенсивністю обміну повідомленнями.
# **Стандартне відхилення** зменшилося з 1372.01 до 850.31, що відповідає зниженню на 38.0%. Це вказує на суттєве підвищення однорідності даних після очищення.
# **Співвідношення σ/μ** (коефіцієнт варіації) зменшилося з 1.81 до 1.47, що свідчить про покращення відносної однорідності розподілу.
# 
# #### Зміни форми розподілу
# 
# Аналіз гістограм показує наступні зміни у формі розподілу:
# **Висота першого стовпця** (найнижчі значення кількості повідомлень) зменшилася після очищення, хоча він залишається найвищим в обох розподілах. Це вказує на те, що більшість партнерів у обох випадках має відносно невелику кількість повідомлень.
# **Правий "хвіст" розподілу** (область високих значень) став значно коротшим після очищення, що відображає видалення партнерів з аномально великою кількістю повідомлень.
# **Загальна форма розподілу** в обох випадках демонструє виражену правосторонню (позитивну) асиметрію, що є типовим для комунікаційних даних, де більшість клієнтів має невелику кількість взаємодій, а невелика частка клієнтів — дуже велику.
# **Концентрація у нижньому діапазоні** — в обох розподілах спостерігається значна концентрація значень у нижній частині діапазону (до 1000 повідомлень), при цьому після очищення ця концентрація стає ще більш вираженою.
# 
# #### Діапазон значень
# 
# Аналіз масштабу осі X дозволяє зробити наступні спостереження щодо діапазону значень:
# Максимальні значення на осі X для початкових даних сягають приблизно 5000 повідомлень, що вказує на наявність партнерів з надзвичайно інтенсивною комунікацією.
# Після очищення максимальні значення, імовірно, не перевищують 3000, що свідчить про видалення спостережень з найбільшою кількістю повідомлень.
# Різниця між діапазонами значень до і після очищення є досить суттєвою, що вказує на значний вплив процедури видалення викидів на крайні значення розподілу.
# 
# #### Порівняння з гістограмами кількості замовлень
# 
# Порівнюючи цей графік з попереднім (гістограми `partner_total_orders`), можна відзначити:
# **Відносне зменшення середнього значення** приблизно однакове для обох змінних (близько 23-24%), що може свідчити про пропорційний зв'язок між кількістю замовлень та повідомлень у видалених викидах.
# **Коефіцієнт варіації** до і після очищення для обох змінних має схожі значення, але для кількості повідомлень спостерігається дещо більша варіативність, що може бути пов'язано з більшою гнучкістю та різноманітністю комунікаційних патернів.
# **Масштаб значень** для кількості повідомлень значно більший, ніж для кількості замовлень (тисячі проти сотень), що відображає природу цих показників — обробка одного замовлення часто вимагає багаторазового обміну повідомленнями.
# 
# #### Інтерпретація результатів очищення даних
# 
# #### Вплив на характеристики вибірки
# **Підвищення репрезентативності**: Значне зменшення стандартного відхилення та середнього значення свідчить про те, що очищена вибірка краще відображає комунікаційну поведінку "типових" партнерів B2B компанії.
# **Зменшення впливу надмірно активних партнерів**: Видалення викидів зменшило вплив партнерів з аномально високою комунікаційною активністю, які могли непропорційно впливати на результати аналізу.
# **Збереження загальної структури даних**: Незважаючи на видалення викидів, основні характеристики розподілу (правостороння асиметрія, висока концентрація в нижньому діапазоні) зберігаються, що свідчить про збереження базової структури даних.
# 
# #### Наслідки для регресійного аналізу
# **Стабілізація оцінок**: Значне зменшення стандартного відхилення може призвести до отримання більш стабільних і надійних оцінок коефіцієнтів регресії при моделюванні взаємозв'язку між кількістю замовлень та повідомлень.
# **Імовірне покращення нормальності розподілу залишків**: Видалення екстремальних значень може покращити відповідність залишків регресійної моделі нормальному розподілу, що є важливою передумовою класичної лінійної регресії.
# **Потенційне обмеження області застосування**: Очищена модель може гірше прогнозувати комунікаційну активність для партнерів з надзвичайно високою інтенсивністю взаємодії, які були виключені з аналізу.
# 
# #### Бізнес-імплікації виявлених змін
# 
# #### Розуміння комунікаційних патернів
# **Неоднорідна структура комунікації**: Значна різниця між початковими та очищеними даними вказує на наявність дуже неоднорідних патернів комунікації серед партнерів, де основна маса має відносно невелику інтенсивність повідомлень, а окремі партнери демонструють екстремальну активність.
# **Виявлення специфічних сегментів**: Партнери, видалені як викиди, можуть представляти специфічний сегмент клієнтської бази з унікальними комунікаційними потребами, який потребує окремого аналізу та стратегії обслуговування.
# 
# #### Оптимізація комунікаційних процесів
# **Планування ресурсів**: Значна різниця у статистичних характеристиках до і після очищення даних має бути врахована при плануванні комунікаційних ресурсів — загальна оцінка на основі очищених даних може призвести до недооцінки ресурсів, необхідних для обслуговування високоактивних партнерів.
# **Диференційовані комунікаційні стратегії**: Виявлена неоднорідність у кількості повідомлень обґрунтовує необхідність розробки диференційованих комунікаційних стратегій для різних сегментів партнерів — від мінімалістичного підходу для партнерів з низькою активністю до високоінтенсивного супроводу для найактивніших клієнтів.
# 
# #### Рекомендації для подальшого аналізу
# **Детальне дослідження викидів**: Окремий аналіз партнерів, виявлених як викиди, може дати цінні інсайти про їхні унікальні характеристики та потреби.
# **Аналіз співвідношення повідомлень до замовлень**: Розрахунок та аналіз середньої кількості повідомлень на одне замовлення для різних сегментів партнерів може виявити відмінності в ефективності комунікації.
# **Розгляд трансформацій даних**: Висока правостороння асиметрія розподілу створює передумови для застосування логарифмічної трансформації змінної `partner_total_messages` перед побудовою регресійних моделей.
# **Сегментований аналіз**: Розробка окремих моделей для різних сегментів партнерів, визначених за інтенсивністю комунікації, може дати більш точні прогнози та краще відображати реальні взаємозв'язки в кожному сегменті.
# 
# #### Висновки
# Аналіз порівняльних гістограм кількості повідомлень партнера до та після очищення даних виявляє значний вплив процедури видалення викидів на статистичні характеристики вибірки. Середнє значення зменшилося на 23.7%, а стандартне відхилення — на 38.0%, що свідчить про суттєве підвищення однорідності даних.
# Виражена правостороння асиметрія розподілу, яка зберігається навіть після очищення, вказує на фундаментальну особливість комунікаційної поведінки партнерів B2B компанії — більшість має відносно невелику кількість повідомлень, тоді як невелика частка демонструє дуже високу активність.
# Для прийняття обґрунтованих бізнес-рішень щодо комунікаційної стратегії важливо враховувати як "типову" поведінку, яка краще відображена в очищених даних, так і особливі потреби високоактивних партнерів, які могли бути видалені як статистичні викиди.
# 

# In[23]:


# Візуалізація діаграм розсіювання до та після очищення
plot_scatter_before_after(x_data, y_data, x_clean, y_clean, x_name, y_name)


# ### Аналіз порівняння даних до та після видалення викидів
# 
# #### Опис графіка
# 
# Представлений графік демонструє діаграму розсіювання, яка порівнює розподіл даних до та після процедури видалення викидів. На графіку відображено взаємозв'язок між кількістю замовлень партнера (`partner_total_orders`, вісь X) та кількістю повідомлень (`partner_total_messages`, вісь Y).
# 
# На графіку зображено:
# - Сині точки — початкові дані до очищення
# - Зелені точки — очищені дані після видалення викидів
# 
# #### Аналіз змін у наборі даних
# 
# #### Зміни у діапазоні значень
# **Обмеження діапазону X**: Після очищення максимальне значення `partner_total_orders` обмежується приблизно 450-500, тоді як початкові дані містили значення до 1200+. Це свідчить про те, що процедура видалення викидів ідентифікувала партнерів з надзвичайно великою кількістю замовлень як статистичні викиди.
# **Обмеження діапазону Y**: Максимальне значення `partner_total_messages` в очищених даних не перевищує ~5000, порівняно з ~11000 у початкових даних. Це вказує на істотне зменшення розкиду по вертикальній осі.
# **Загальне стиснення простору даних**: Загальний "простір" даних після очищення став набагато компактнішим, що відповідає зменшенню стандартних відхилень, виявлених раніше при аналізі гістограм.
# 
# #### Збереження структурних особливостей
# **Збереження "треків"**: Навіть після очищення зберігаються виразні криволінійні "треки" або "лінії" точок, що вказує на те, що ця структурна особливість є фундаментальною характеристикою даних, а не результатом впливу викидів.
# **Концентрація в нижньому лівому куті**: В обох наборах даних спостерігається висока концентрація точок у області низьких значень обох змінних, що підтверджує висновки, зроблені на основі гістограм.
# **Загальний патерн взаємозв'язку**: Загальний позитивний взаємозв'язок між кількістю замовлень та кількістю повідомлень зберігається як до, так і після очищення даних.
# 
# ### Видалені спостереження
# **Видалення "верхніх треків"**: Найбільш помітною зміною є видалення верхніх "треків" — груп точок, які демонстрували найвищі значення кількості повідомлень для відповідних значень кількості замовлень.
# **Видалення "далеких" спостережень**: Усі спостереження з кількістю замовлень більше ~450-500 були ідентифіковані як викиди і видалені з набору даних.
# **Селективне видалення по "треках"**: Цікаво відзначити, що алгоритм видалення викидів не просто "обрізав" дані по певних порогових значеннях, а виконав більш складний аналіз, що призвів до селективного видалення спостережень з різних "треків".
# 
# #### Інтерпретація для регресійного аналізу
# 
# #### Вплив на патерн взаємозв'язку
# **Зміна нахилу потенційної регресійної лінії**: Візуально можна припустити, що видалення верхніх "треків" та далеких спостережень може призвести до зменшення нахилу лінії регресії, оскільки було видалено точки з найбільш стрімким зростанням Y відносно X.
# **Зменшення гетероскедастичності**: Видалення спостережень з високими значеннями обох змінних, особливо тих, що демонстрували найбільший розкид по вертикальній осі, має потенціал значно зменшити гетероскедастичність у регресійній моделі.
# **Покращення лінійності**: Хоча криволінійні "треки" зберігаються, загальний патерн залежності в очищених даних може краще відповідати припущенню про лінійність, що є фундаментальним для лінійної регресії.
# 
# #### Потенційні переваги та обмеження очищеної моделі
# **Переваги**:
#    - Підвищена стабільність оцінок коефіцієнтів регресії
#    - Краща відповідність умовам гомоскедастичності
#    - Потенційно вища статистична значущість моделі
#    - Менша чутливість до окремих впливових спостережень
# **Обмеження**:
#    - Звужена область застосування — модель може втратити здатність прогнозувати для партнерів з дуже високою активністю
#    - Потенційна втрата інформації про важливі сегменти клієнтської бази
#    - Ризик "перевиховання" моделі для середньостатистичних випадків
# 
# #### Бізнес-імплікації
# 
# #### Сегментація клієнтської бази
# **Виявлення "нетипових" сегментів**: Графік чітко ілюструє наявність груп партнерів з унікальними характеристиками взаємодії, які були ідентифіковані як статистичні викиди. Ці групи можуть представляти особливо цінні або специфічні сегменти клієнтської бази.
# **Різні моделі комунікаційної поведінки**: Наявність різних "треків" на графіку, особливо тих, що були видалені як викиди, вказує на існування різних моделей комунікаційної поведінки серед партнерів. Такі відмінності можуть бути пов'язані з типом бізнесу партнера, складністю його операцій, або особливостями взаємодії.
# **Потенціал для таргетованого підходу**: Виявлені сегменти, зокрема ті, що були видалені як викиди, можуть потребувати розробки специфічних стратегій обслуговування, що відрізняються від підходу до "типових" партнерів.
# 
# #### Прогнозування та планування ресурсів
# **Двокомпонентний підхід до прогнозування**: Для ефективного планування комунікаційних ресурсів може бути доцільним застосування двокомпонентного підходу — використання очищеної моделі для "типових" партнерів і окремої стратегії для партнерів, які відповідають характеристикам видалених викидів.
# **Оцінка потреб у ресурсах**: Хоча видалені спостереження становлять відносно невелику частку загальної вибірки, вони відповідають партнерам з надзвичайно високими потребами в комунікації. Недооцінка ресурсів, необхідних для обслуговування цих партнерів, може призвести до значних проблем у рівні сервісу.
# **Ідентифікація "точок зростання"**: Партнери, що лежать на верхній межі "типових" даних (зелені точки, що наближаються до верхньої межі), можуть представляти потенційні "точки зростання" — клієнтів, які мають потенціал перейти у категорію високоактивних.
# 
# #### Рекомендації для подальшого аналізу
# **Порівняння регресійних моделей**: Побудувати та порівняти регресійні моделі на основі початкових та очищених даних, з особливою увагою до змін у коефіцієнтах, статистичній значущості та якості моделей.
# **Окремий аналіз видалених спостережень**: Провести детальний аналіз характеристик партнерів, чиї дані були ідентифіковані як викиди, для виявлення спільних рис та особливостей.
# **Аналіз "треків"**: Дослідити причини формування виразних криволінійних "треків" у даних. Це може включати аналіз додаткових характеристик партнерів, таких як сфера діяльності, тривалість співпраці, географічне положення тощо.
# **Кластерний аналіз**: Застосувати методи кластеризації для автоматичного виявлення природних груп партнерів на основі їхніх патернів взаємодії, що може дати більш нюансовану альтернативу бінарному поділу на "очищені дані" та "викиди".
# 
# #### Висновки
# 
# Порівняльний аналіз даних до та після видалення викидів наочно демонструє значний вплив процедури очищення на структуру даних. Очищені дані представляють більш компактний та однорідний набір, що потенційно краще відповідає вимогам класичної лінійної регресії.
# Водночас, видалені як викиди спостереження відображають важливі сегменти клієнтської бази з унікальними патернами взаємодії, які не повинні ігноруватися при розробці бізнес-стратегій. Наявність чітких "треків" як у початкових, так і в очищених даних вказує на існування структурованих закономірностей у взаємозв'язку між кількістю замовлень та обсягом комунікації.
# Оптимальний підхід до аналізу та прогнозування комунікаційної активності партнерів має поєднувати інсайти, отримані з обох наборів даних — враховувати стабільні закономірності, виявлені на очищених даних, та не втрачати з уваги специфічні потреби партнерів, виявлених як статистичні викиди.
# 

# ### Аналіз графіка виявлених викидів у даних
# 
# #### Опис графіка
# 
# Представлений графік відображає результати ідентифікації викидів у наборі даних. На діаграмі розсіювання показано взаємозв'язок між кількістю замовлень партнера (`partner_total_orders`, вісь X) та кількістю повідомлень (`partner_total_messages`, вісь Y).
# 
# На графіку зображено:
# - Сині точки — спостереження, які не були ідентифіковані як викиди (залишені в очищеному наборі даних)
# - Червоні точки — спостереження, ідентифіковані як статистичні викиди (видалені при очищенні даних)
# 
# #### Аналіз характеристик викидів
# 
# #### Патерни розподілу викидів
# **"Верхні треки"**: Найбільш помітною категорією викидів є точки, що формують верхні "треки" або криволінійні структури з високим співвідношенням кількості повідомлень до кількості замовлень. Ці треки починаються приблизно з `partner_total_orders` > 400 і демонструють стрімке зростання `partner_total_messages` зі збільшенням кількості замовлень.
# **Партнери з високою кількістю замовлень**: Практично всі спостереження з кількістю замовлень більше 450-500 були ідентифіковані як викиди, незалежно від кількості повідомлень.
# **Просторовий патерн**: Викиди формують чітку просторову структуру, розташовуючись на периферії основної маси даних та утворюючи виразні геометричні патерни (зокрема, криволінійні треки), а не випадково розподілені точки.
# 
# #### Статистичні особливості викидів
# **Інтенсивність комунікації**: Більшість викидів характеризується надзвичайно високою інтенсивністю комунікації — до 11000+ повідомлень, порівняно з максимумом близько 5000 для не-викидів.
# **Кількість замовлень**: Викиди включають спостереження з кількістю замовлень від ~400 до 1200+, що істотно перевищує середнє значення в ~64 замовлення для очищеного набору даних.
# **Пропорційне співвідношення**: Помітно, що в деяких "треках" викидів співвідношення кількості повідомлень до кількості замовлень значно вище, ніж у основної маси даних, що може вказувати на іншу модель комунікаційної поведінки.
# 
# #### Механізм ідентифікації викидів
# Судячи з патерну ідентифікованих викидів, алгоритм виявлення викидів працював не просто за абсолютними пороговими значеннями, а проводив складніший багатовимірний аналіз:
# Викиди виявлялися з урахуванням багатовимірного розподілу (обидві змінні).
# Процедура була чутливою до відхилень від загального патерну взаємозв'язку.
# Враховувалася як абсолютна величина значень, так і їх відносна позиція у розподілі.
# 
# #### Інтерпретація для регресійного аналізу
# 
# #### Вплив на параметри регресійної моделі
# **Нахил лінії регресії**: Видалення виявлених викидів, особливо тих, що формують "верхні треки" з високим співвідношенням `partner_total_messages` до `partner_total_orders`, імовірно призведе до зменшення нахилу лінії регресії, тобто до зменшення коефіцієнта при незалежній змінній.
# **Зміщення перетину**: Оскільки більшість викидів має високі значення обох змінних, їх видалення може призвести до невеликого збільшення вільного члена (перетину з віссю Y) регресійної моделі.
# **Стандартні похибки**: Видалення викидів має призвести до значного зменшення стандартних похибок оцінок коефіцієнтів регресії, підвищуючи таким чином їхню статистичну значущість.
# 
# #### Вплив на якість моделі
# **Покращення відповідності передумовам**: Видалення виявлених викидів, особливо тих, що формують високі "треки", має значно покращити відповідність залишків моделі передумовам лінійної регресії:
#    - Зменшення гетероскедастичності
#    - Покращення нормальності розподілу залишків
#    - Потенційне покращення лінійності взаємозв'язку
# **Підвищення R²**: Очікується, що коефіцієнт детермінації (R²) зросте після видалення викидів, оскільки модель краще відповідатиме більш однорідному набору даних.
# **Зниження RMSE**: Середньоквадратична похибка має значно зменшитися після видалення спостережень з екстремальними значеннями, які могли давати великі залишки.
# 
# #### Бізнес-аналіз виявлених викидів
# 
# #### Потенційні категорії "нетипових" партнерів
# 
# Аналіз патерну викидів дозволяє виділити кілька потенційних категорій "нетипових" партнерів:
# **"Високочастотні комунікатори"**: Партнери, що формують верхні "треки" з надзвичайно високою кількістю повідомлень відносно кількості замовлень. Вони можуть бути партнерами, які потребують інтенсивної підтримки, консультацій або мають складні запити.
# **"Великі замовники"**: Партнери з великою кількістю замовлень (> 500), які автоматично ідентифіковані як викиди незалежно від їхньої комунікаційної поведінки. Це, імовірно, великі клієнти, що генерують значний обсяг бізнесу.
# **"Гібридні випадки"**: Партнери, що демонструють одночасно високу кількість замовлень та непропорційно високу інтенсивність комунікації. Вони можуть представляти особливо цінних, але ресурсомістких клієнтів.
# 
# #### Імплікації для бізнес-процесів
# **Диференційоване обслуговування**: Виявлені як викиди партнери можуть потребувати спеціалізованого підходу до обслуговування, можливо, виділених менеджерів або спеціалізованих команд підтримки.
# **Аналіз причин високої інтенсивності комунікації**: Партнери з надзвичайно високим співвідношенням повідомлень до замовлень можуть вказувати на:
#    - Неефективність комунікаційних процесів
#    - Складність продуктів або послуг, що потребують інтенсивних консультацій
#    - Потенційні проблеми з якістю, що генерують додаткову комунікацію
# **Стратегічне значення**: Партнери, виявлені як викиди за кількістю замовлень, імовірно, мають стратегічне значення для бізнесу і потребують особливої уваги для збереження та розвитку відносин.
# 
# #### Рекомендації для подальшого аналізу
# **Деталізована класифікація викидів**: Провести кластерний аналіз виявлених викидів для ідентифікації більш тонких підкатегорій з різними патернами поведінки.
# **Додаткові змінні**: Включити в аналіз додаткові характеристики партнерів (галузь, розмір компанії, тривалість співпраці тощо) для кращого розуміння факторів, пов'язаних з ідентифікацією як викиду.
# **Часова динаміка**: Дослідити, як змінювалася комунікаційна поведінка партнерів, ідентифікованих як викиди, з часом — чи були вони завжди "нетиповими", чи стали такими внаслідок певних подій.
# **Окремі моделі**: Розглянути можливість побудови окремих регресійних моделей для різних категорій викидів, що може дати більш точні прогнози для цих специфічних сегментів.
# 
# #### Висновки
# 
# Графік виявлених викидів наочно демонструє структурований характер "нетипових" спостережень, які не є випадковими аномаліями, а формують чіткі патерни, що вказують на існування особливих сегментів партнерів з відмінними моделями взаємодії. 
# Видалення цих викидів, імовірно, значно покращить статистичні характеристики регресійної моделі, зробивши її більш відповідною класичним передумовам лінійної регресії. Водночас, важливо пам'ятати, що виявлені викиди представляють реальні спостереження з потенційно високою бізнес-цінністю, і їх вилучення з аналізу не має призводити до їх ігнорування в бізнес-стратегії.
# Оптимальний підхід має поєднувати статистичну строгість (побудова моделей на очищених даних) з бізнес-прагматизмом (спеціальні стратегії для обслуговування "нетипових" партнерів), що дозволить максимізувати як якість статистичного аналізу, так і практичну цінність його результатів.
# 

# In[24]:


# Аналіз очищених даних
if len(outlier_indices) > 0:
    print_header("АНАЛІЗ ДАНИХ ПІСЛЯ ВИДАЛЕННЯ ВИКИДІВ")
    results_after = analyze_linear_regression(x_clean, y_clean, x_name, y_name)
else:
    print("Викиди не виявлені, результати аналізу не змінилися")
    results_after = results_before


# ### Аналіз діаграми розсіювання з лінією регресії (після очищення даних)
# 
# #### Опис графіка
# 
# Представлений графік є діаграмою розсіювання з побудованою лінією регресії для очищеного набору даних (після видалення викидів). На графіку відображено взаємозв'язок між кількістю замовлень партнера (`partner_total_orders`, вісь X) та кількістю повідомлень, якими обмінялися партнери з менеджерами компанії (`partner_total_messages`, вісь Y).
# 
# На графіку зображено:
# - Сині точки — фактичні дані спостережень після видалення викидів
# - Червона лінія — побудована лінійна регресія
# - Блакитна область навколо лінії регресії — 95% довірчий інтервал
# - У верхньому лівому куті — параметри моделі: рівняння регресії, коефіцієнт детермінації та стандартна помилка
# 
# #### Кількісні характеристики моделі
# 
# Згідно з інформацією на графіку:
# - Рівняння регресії: y = 21.1406 + 8.6617x
# - Коефіцієнт детермінації: R² = 0.8355
# - Стандартна помилка: 216.0059
# 
# #### Аналіз регресійної моделі
# 
# #### Рівняння регресії
# **Вільний член (21.1406)**: Цей коефіцієнт вказує на те, що навіть за відсутності замовлень (x = 0) прогнозується приблизно 21 повідомлення. Це може інтерпретуватися як базовий рівень комунікації, необхідний для підтримки відносин з партнером, незалежно від наявності замовлень.
# **Коефіцієнт нахилу (8.6617)**: Це значення показує, що кожне додаткове замовлення асоціюється в середньому з приблизно 8.66 додатковими повідомленнями. Цей коефіцієнт відображає середню інтенсивність комунікації, необхідну для обробки одного замовлення в очищеному наборі даних.
# 
# #### Оцінка якості моделі
# **Коефіцієнт детермінації (R² = 0.8355)**: Цей показник свідчить про те, що модель пояснює близько 83.55% варіації в кількості повідомлень. Це вказує на дуже сильний взаємозв'язок між досліджуваними змінними та високу прогностичну здатність моделі.
# **Стандартна помилка (216.0059)**: Це середнє відхилення фактичних значень від прогнозованих. Для оцінки відносної величини цієї помилки корисно порівняти її з середнім значенням залежної змінної. Судячи з графіка, середнє значення `partner_total_messages` може бути в діапазоні 1000-2000, що робить стандартну помилку відносно невеликою у відсотковому вираженні.
# 
# #### Особливості розподілу даних на графіку
# 
# #### Структурні патерни
# **Виразні "треки"**: Найбільш помітною особливістю графіка є наявність численних криволінійних "треків" або "ліній", що формуються точками даних. Ці треки можуть відображати різні сегменти партнерів з відмінними патернами взаємодії або різні операційні процеси.
# **Щільність розподілу**: Спостерігається висока концентрація точок у нижній лівій частині графіка (низькі значення обох змінних), що вказує на те, що більшість партнерів має відносно невелику кількість замовлень і, відповідно, невелику кількість повідомлень.
# **Розширення розкиду**: З ростом значень обох змінних збільшується розкид точок навколо лінії регресії, що вказує на збільшення варіативності комунікаційної поведінки для партнерів з більшою кількістю замовлень.
# 
# #### Відхилення від лінійності
# **Криволінійні "треки"**: Більшість індивідуальних "треків" на графіку мають криволінійну форму, що може свідчити про нелінійний характер взаємозв'язку для окремих сегментів партнерів.
# **Загальний патерн**: Хоча лінійна модель демонструє високий R², візуальний аналіз графіка може вказувати на потенційну користь від застосування більш складних моделей, які могли б краще відповідати індивідуальним "трекам".
# 
# #### Порівняння з моделлю до очищення даних
# 
# Порівнюючи цей графік з аналогічним для неочищених даних (перший графік), можна відзначити наступні відмінності:
# **Зміна коефіцієнтів**:
#    - Вільний член збільшився з 19.2445 до 21.1406
#    - Коефіцієнт нахилу дещо зменшився з 8.8312 до 8.6617
# **Зміна R²**: R² зменшився з 0.9448 до 0.8355, що може здатися несподіваним, оскільки зазвичай очищення даних підвищує R²
# **Стандартна помилка**: Стандартна помилка зменшилася з 322.3087 до 216.0059, що відображає підвищення точності прогнозування
# **Діапазон даних**: Очищений набір даних обмежений значеннями `partner_total_orders` приблизно до 500 і `partner_total_messages` до 5000, тоді як неочищений набір містив значення до 1200+ і 11000+ відповідно
# 
# #### Інтерпретація змін після очищення даних
# 
# #### Зменшення R²
# 
# Зниження R² з 0.9448 до 0.8355 після очищення даних може здатися контрінтуїтивним, оскільки зазвичай видалення викидів покращує відповідність моделі. Це можна пояснити кількома факторами:
# **Структура викидів**: Видалені викиди могли мати дуже високу лінійну кореляцію між змінними, а їх видалення могло призвести до більшої відносної ваги "шуму" в очищених даних.
# **Зміна масштабу**: У неочищеному наборі даних дуже високі значення обох змінних могли "розтягувати" лінію регресії, штучно підвищуючи R² за рахунок ефекту масштабу.
# **Неоднорідність у "треках"**: Очищені дані зберігають виразні "треки", які свідчать про наявність кількох різних патернів залежності, що можуть гірше апроксимуватися єдиною лінійною моделлю.
# 
# #### Зменшення стандартної помилки
# 
# Значне зменшення стандартної помилки (на 33%) свідчить про суттєве підвищення точності прогнозування після видалення викидів. Це означає, що хоча модель тепер пояснює меншу частку загальної варіації (нижчий R²), її прогнози стали точнішими з точки зору абсолютної похибки.
# 
# #### Практичні висновки для бізнесу
# 
# #### Прогнозування комунікаційних потреб
# **Стабільність прогнозів**: Очищена модель дає більш стабільні та надійні прогнози кількості повідомлень для "типових" партнерів, які складають більшість клієнтської бази.
# **Базова комунікація**: Вільний член моделі (21.1406) дає уявлення про обсяг "базової" комунікації, необхідної для підтримки відносин з партнером незалежно від кількості замовлень.
# **Комунікація на одне замовлення**: Коефіцієнт нахилу (8.6617) дозволяє планувати комунікаційні ресурси пропорційно до очікуваної кількості замовлень.
# 
# #### Сегментація та індивідуальний підхід
# **Виявлення "треків"**: Чіткі "треки" на графіку вказують на необхідність сегментованого підходу до аналізу та прогнозування комунікаційних потреб різних груп партнерів.
# **Потенціал для нелінійних моделей**: Криволінійність "треків" свідчить про можливу користь від застосування більш складних, нелінійних моделей для окремих сегментів.
# **Стратегія диференціації**: Компанії варто розглянути можливість розробки диференційованих комунікаційних стратегій для різних патернів взаємодії, виявлених на графіку.
# 
# #### Рекомендації для подальшого аналізу
# **Сегментація "треків"**: Провести кластерний аналіз для ідентифікації та категоризації різних "треків" на графіку, що може виявити природні сегменти партнерів.
# **Нелінійні моделі**: Розглянути можливість побудови нелінійних моделей (поліноміальних, експоненційних тощо), які могли б краще відповідати криволінійним "трекам".
# **Аналіз додаткових факторів**: Дослідити, які додаткові характеристики партнерів (галузь, розмір, географія тощо) можуть пояснювати формування різних "треків" на графіку.
# **Порівняння прогностичної здатності**: Порівняти точність прогнозів моделей, побудованих на очищених та неочищених даних, на незалежній тестовій вибірці для визначення їхньої практичної цінності.
# 
# #### Висновки
# 
# Аналіз діаграми розсіювання з лінією регресії для очищеного набору даних показує, що існує сильний лінійний взаємозв'язок між кількістю замовлень партнера та обсягом комунікації (R² = 0.8355). Кожне додаткове замовлення асоціюється з приблизно 8.66 додатковими повідомленнями.
# Очищення даних призвело до зменшення R², але значно підвищило точність прогнозування, про що свідчить зменшення стандартної помилки. Наявність виразних криволінійних "треків" на графіку вказує на необхідність сегментованого підходу до аналізу та потенційну користь від застосування більш складних моделей.
# Для практичного застосування в бізнесі очищена лінійна модель надає надійну основу для прогнозування комунікаційних потреб "типових" партнерів, що дозволяє ефективніше планувати ресурси та оптимізувати процеси взаємодії з клієнтами.
# 

# ### Аналіз графіка залишків відносно прогнозованих значень (після очищення даних)
# 
# #### Опис графіка
# 
# Представлений графік відображає залежність залишків регресійної моделі від прогнозованих значень для очищеного набору даних (після видалення викидів). Цей діагностичний інструмент дозволяє оцінити якість лінійної регресійної моделі та перевірити виконання передумов регресійного аналізу після процедури очищення даних.
# 
# На графіку зображено:
# - Зелені точки — залишки моделі для різних прогнозованих значень
# - Червона горизонтальна лінія на рівні нуля — ідеальна лінія, що відповідає нульовим залишкам
# - Пунктирні горизонтальні лінії — імовірно, межі стандартного відхилення залишків
# - Коричневі та рожеві точки — можливо, залишки для окремих категорій даних або точки з особливим статусом
# 
# #### Аналіз розподілу залишків
# 
# #### Загальна структура розподілу
# **Формування "треків"**: Найбільш помітною особливістю графіка є наявність численних криволінійних "треків" або "ліній", що формуються залишками. Ці треки розходяться від початку координат, причому їх ширина збільшується зі зростанням прогнозованих значень.
# **Симетричність відносно нульової лінії**: Залишки розподілені відносно симетрично вище та нижче нульової лінії, що є позитивною ознакою з точки зору незміщеності оцінок.
# **Діапазон залишків**: Після очищення даних діапазон залишків становить приблизно від -1500 до +2000, що значно менше, ніж у неочищеному наборі даних, де він досягав ±3000.
# 
# #### Гетероскедастичність
# **Збільшення розкиду залишків**: Розкид залишків явно збільшується з ростом прогнозованих значень, що є чіткою ознакою гетероскедастичності (непостійності дисперсії помилок).
# **Характер розширення**: Розширення розкиду відбувається структуровано — більшість "треків" розходяться віялоподібно, зберігаючи свою внутрішню структуру.
# **Зони концентрації**: Спостерігається висока концентрація залишків у зоні малих прогнозованих значень (до 1000) та поступове розрідження з ростом прогнозованих значень.
# 
# #### Нелінійні патерни
# **Криволінійність "треків"**: Більшість "треків" залишків мають криволінійну форму, що вказує на потенційні нелінійні залежності, які не були враховані лінійною моделлю.
# **Систематичні відхилення**: Деякі "треки" систематично відхиляються від нульової лінії, що свідчить про наявність невідомих факторів, які структурують дані у групи з різними патернами залежності.
# **Відсутність горизонтального тренду**: На відміну від класичної картини невідповідності лінійної моделі, на графіку не спостерігається єдиного горизонтального тренду залишків (наприклад, U-подібної форми), що ускладнює застосування простих нелінійних трансформацій для покращення моделі.
# 
# #### Порівняння з графіком залишків до очищення даних
# У порівнянні з аналогічним графіком для неочищеного набору даних (другий графік) можна відзначити наступні зміни:
# **Зменшення абсолютного діапазону залишків**: Максимальні відхилення залишків значно зменшилися після очищення даних, що свідчить про підвищення точності прогнозування.
# **Збереження структурних особливостей**: Незважаючи на видалення викидів, основні структурні особливості розподілу залишків (формування "треків", гетероскедастичність) зберігаються, що вказує на їх глибинну природу в даних.
# **Більш виражена симетрія**: В очищеному наборі даних розподіл залишків є більш симетричним відносно нульової лінії, що відповідає поліпшенню відповідності передумові про нормальний розподіл похибок.
# 
# #### Інтерпретація для регресійної моделі
# 
# #### Порушення передумов лінійної регресії
# Графік залишків після очищення даних продовжує демонструвати порушення кількох ключових передумов лінійної регресії:
# **Гомоскедастичність**: Умова постійної дисперсії залишків все ще суттєво порушується, хоча й меншою мірою, ніж до очищення даних.
# **Лінійність**: Наявність криволінійних "треків" вказує на те, що істинна залежність між змінними може бути нелінійною або що різні сегменти даних можуть вимагати різних моделей.
# **Незалежність спостережень**: Формування чітких "треків" може свідчити про наявність прихованих факторів, які групують спостереження.
# 
# #### Наслідки для статистичних висновків
# **Надійність p-значень**: Порушення передумов, особливо гетероскедастичності, може призвести до неточних стандартних помилок коефіцієнтів регресії та, відповідно, ненадійних p-значень для перевірки статистичної значущості.
# **Довірчі інтервали**: Ширина довірчих інтервалів прогнозів має збільшуватися з ростом прогнозованих значень через збільшення розкиду залишків, що важливо враховувати при інтерпретації результатів.
# **Якість прогнозів**: Прогнози моделі будуть більш точними для спостережень з низькими прогнозованими значеннями і потенційно менш надійними для високих значень.
# 
# #### Імплікації для бізнес-аналізу
# 
# #### Сегментація даних
# **Виявлення природних сегментів**: Чіткі "треки" на графіку залишків вказують на наявність природних сегментів у даних, які можуть відповідати різним групам партнерів з відмінними моделями комунікаційної поведінки.
# **Підхід до моделювання**: Замість єдиної лінійної моделі для всіх даних, може бути доцільним розробити окремі моделі для кожного ідентифікованого сегмента або застосувати методи, що враховують групову структуру даних.
# 
# #### Точність прогнозування
# **Змінна точність**: Бізнес-користувачам моделі слід враховувати, що точність прогнозів буде варіюватися залежно від сегмента партнерів та обсягу їхніх замовлень.
# **Планування комунікаційних ресурсів**: При плануванні ресурсів для комунікації з партнерами необхідно закладати більший "буфер" для партнерів з великою кількістю замовлень, оскільки прогнози для них мають більшу невизначеність.
# 
# #### Виявлення особливих випадків
# **Аналіз екстремальних залишків**: Спостереження з найбільшими позитивними або негативними залишками представляють випадки, де комунікаційна поведінка суттєво відхиляється від загальної тенденції. Аналіз таких випадків може виявити специфічні фактори, що впливають на інтенсивність комунікації.
# **Оптимізація процесів**: Виявлення партнерів, які систематично потребують більше або менше комунікації, ніж передбачено моделлю, може дати цінні інсайти для оптимізації процесів взаємодії.
# 
# #### Рекомендації для покращення моделі
# 
# #### Трансформації змінних
# **Логарифмічна трансформація**: Враховуючи характер гетероскедастичності, можна спробувати логарифмічну трансформацію змінних для стабілізації дисперсії, хоча наявність чітких "треків" може обмежити ефективність цього підходу.
# **Зважені методи найменших квадратів**: Для коригування гетероскедастичності можна застосувати зважене оцінювання, де ваги обернено пропорційні до дисперсії залишків для відповідних прогнозованих значень.
# 
# #### Альтернативні підходи до моделювання
# **Сегментована регресія**: Побудова окремих лінійних моделей для кластерів даних, виявлених на основі "треків" залишків.
# **Нелінійні моделі**: Застосування поліноміальної регресії, сплайн-регресії або інших нелінійних методів для кращого врахування криволінійних патернів у даних.
# **Методи машинного навчання**: Для даних зі складною структурою, як у цьому випадку, можуть бути корисними такі методи, як випадковий ліс або градієнтний бустинг, які краще справляються з нелінійними взаємозв'язками та взаємодіями.
# 
# #### Аналіз прихованих факторів
# **Додаткові предиктори**: Пошук та включення в модель додаткових змінних, які можуть пояснити формування "треків" у залишках (тип партнера, галузь, тривалість співпраці тощо).
# **Взаємодії між змінними**: Дослідження потенційних взаємодій між кількістю замовлень та іншими характеристиками партнерів, які можуть пояснити наявність різних патернів залежності.
# 
# #### Висновки
# Аналіз графіка залишків відносно прогнозованих значень для очищеного набору даних показує, що хоча видалення викидів покращило певні аспекти моделі (зменшення абсолютного розкиду залишків, краща симетрія), фундаментальні структурні особливості даних зберігаються. Наявність чітких "треків" залишків та гетероскедастичність вказують на складну структуру взаємозв'язку між кількістю замовлень та обсягом комунікації партнерів.
# Для отримання більш точних та надійних прогнозів рекомендується розглянути можливість сегментації даних, застосування нелінійних методів моделювання або включення додаткових предикторів, які можуть пояснити спостережувані патерни у залишках.
# З практичної точки зору, нинішня лінійна модель на основі очищених даних може бути корисною для загального розуміння взаємозв'язку та орієнтовного планування ресурсів, але її обмеження щодо гетероскедастичності та нелінійних патернів слід враховувати при інтерпретації результатів та прийнятті бізнес-рішень.
# 

# ### Аналіз гістограми залишків регресійної моделі (після очищення даних)
# 
# #### Опис графіка
# 
# Представлений графік є гістограмою розподілу залишків регресійної моделі після процедури очищення даних від викидів. Гістограма візуалізує частотний розподіл залишків, що дозволяє оцінити їх відповідність нормальному розподілу — одній із ключових передумов класичної лінійної регресії.
# 
# На графіку зображено:
# - Фіолетові стовпчики — частотний розподіл залишків моделі
# - Червона крива — теоретична крива нормального розподілу
# - Вертикальна пунктирна лінія — середнє значення залишків
# - У правому верхньому куті — статистичні характеристики розподілу
# 
# #### Статистичні характеристики розподілу
# 
# Згідно з інформацією, наведеною на графіку:
# - Кількість спостережень (n): 84,273
# - Середнє значення залишків (μ): 0.0000
# - Стандартне відхилення (σ): 216.0033
# 
# #### Аналіз розподілу залишків
# 
# #### Центрованість розподілу
# 
# Середнє значення залишків дорівнює точно нулю (0.0000), що є ідеальним показником. Це свідчить про те, що модель не має систематичної помилки прогнозування в один бік (переоцінки чи недооцінки) і відповідає важливій умові незміщеності оцінок у методі найменших квадратів.
# 
# #### Форма розподілу
# Аналіз форми гістограми показує:
# - Розподіл має чітко виражений симетричний характер, що візуально наближається до нормального
# - Пік розподілу чітко збігається з нульовим значенням залишків
# - Спостерігається надзвичайно висока концентрація залишків у центральній частині розподілу
# - Частота стрімко зменшується в міру віддалення від центру в обидва боки
# 
# #### Відхилення від нормальності
# Порівнюючи емпіричну гістограму з теоретичною кривою нормального розподілу, можна відзначити такі особливості:
# - Центральний пік емпіричного розподілу є значно вищим, ніж передбачає нормальна крива, що свідчить про яскраво виражену лептокуртичність (гостровершинність) розподілу
# - "Хвости" емпіричного розподілу тонші, ніж у теоретичного нормального розподілу, тобто крайні значення залишків зустрічаються рідше, ніж передбачає нормальний розподіл
# - Проте на діапазоні приблизно від -500 до +500 помітні невеликі "горби", які виходять за межі теоретичної кривої нормального розподілу, що може свідчити про наявність змішаного розподілу
# 
# #### Діапазон залишків
# 
# Гістограма демонструє, що основна маса залишків знаходиться в діапазоні приблизно від -500 до +500, що відповідає приблизно ±2.3σ. На графіку видно, що залишки практично не виходять за межі ±1000, що свідчить про відсутність значних екстремальних значень після процедури очищення даних.
# 
# #### Порівняння з гістограмою залишків до очищення даних
# Порівнюючи цей графік з аналогічним для неочищеного набору даних (третій графік), можна відзначити наступні зміни:
# **Кількість спостережень**: Зменшилася з 86,794 до 84,273, що означає, що було видалено 2,521 спостереження (близько 2.9% даних).
# **Стандартне відхилення**: Зменшилося з 322.3050 до 216.0033, що відповідає зниженню на 33%. Це значне покращення точності прогнозування.
# **Форма розподілу**: 
#    - Центральний пік став ще більш вираженим після очищення даних
#    - Зменшилася кількість екстремальних значень у "хвостах" розподілу
#    - Загальний діапазон залишків значно звузився
# 
# #### Інтерпретація результатів для моделі
# 
# #### Відповідність передумовам регресії
# Аналіз гістограми залишків дозволяє зробити наступні висновки щодо відповідності регресійної моделі стандартним передумовам:
# **Нормальність розподілу залишків**: Хоча розподіл має форму, близьку до дзвоноподібної та симетричну відносно нуля, він демонструє виражену лептокуртичність (загостреність), що є відхиленням від класичного нормального розподілу. Втім, для великих вибірок (n = 84,273) такі відхилення зазвичай не створюють серйозних проблем для статистичного висновку.
# **Нульове середнє залишків**: Ця умова виконується ідеально, що свідчить про коректність специфікації моделі з точки зору її центрованості.
# **Однорідність дисперсії**: Гістограма не дає прямої інформації про гомоскедастичність, проте значне зменшення стандартного відхилення після очищення даних може вказувати на покращення у цьому аспекті.
# 
# #### Інтерпретація стандартного відхилення
# Стандартне відхилення залишків (σ = 216.0033) відображає середній рівень похибки прогнозування моделі. Його значне зменшення порівняно з неочищеним набором даних (з 322.3050 до 216.0033) свідчить про суттєве підвищення точності прогнозів моделі.
# З огляду на те, що середнє значення `partner_total_messages` для очищеного набору даних становить 578.80 (як було визначено на основі попередніх графіків), стандартне відхилення залишків відповідає приблизно 37.3% від середнього значення. Це досить високий показник, який вказує на те, що навіть після очищення даних прогнози моделі мають значну невизначеність.
# 
# #### Форма розподілу та її імплікації
# Лептокуртичний розподіл залишків із "товстішими хвостами" ніж у нормального розподілу має кілька важливих імплікацій:
# **Ймовірність екстремальних значень**: Висока концентрація залишків біля нуля в поєднанні з відносно товстими хвостами вказує на те, що модель дуже точна для більшості спостережень, але має окремі випадки із значними помилками прогнозування.
# **Неоднорідність даних**: Такий тип розподілу може свідчити про наявність кількох підгруп у даних з різними характеристиками взаємозв'язку між змінними.
# **Можливі пропущені змінні**: Патерн розподілу може вказувати на наявність важливих предикторів, не включених до моделі, які могли б пояснити систематичні відхилення у певних сегментах даних.
# 
# #### Порівняння ефективності очищення даних
# 
# #### Покращення точності моделі
# Зменшення стандартного відхилення залишків на 33% є дуже значним покращенням точності моделі. Це свідчить про ефективність процедури видалення викидів для підвищення якості прогнозування.
# 
# #### Вплив на форму розподілу
# Цікаво відзначити, що після очищення даних розподіл залишків став ще більш лептокуртичним, з більш вираженим центральним піком. Це може вказувати на те, що видалені викиди не були просто випадковими аномаліями, а представляли структуровані підгрупи даних з відмінними характеристиками.
# 
# #### Компроміс між точністю та повнотою
# Видалення близько 2.9% спостережень призвело до значного покращення точності моделі, що свідчить про високу ефективність процедури очищення. Водночас, важливо пам'ятати, що видалені спостереження можуть представляти реальні сегменти клієнтської бази з особливими характеристиками.
# 
# #### Рекомендації для бізнес-застосування
# 
# #### Практичні висновки
# **Висока прогностична здатність для "типових" випадків**: Модель, побудована на очищених даних, демонструє високу точність прогнозування для більшості партнерів, що дозволяє ефективно планувати комунікаційні ресурси для основної клієнтської бази.
# **Обережність щодо екстремальних прогнозів**: Наявність "товстих хвостів" у розподілі залишків вказує на те, що для деяких партнерів прогнози можуть мати значні відхилення від фактичних значень. Це слід враховувати при плануванні ресурсів.
# **Потенціал для сегментованого підходу**: Форма розподілу залишків може свідчити про наявність кількох різних сегментів партнерів, для яких можуть бути доцільними окремі моделі прогнозування.
# 
# #### Рекомендації для вдосконалення аналізу
# **Розбиття на сегменти**: Провести кластерний аналіз для ідентифікації природних груп партнерів і розробити окремі моделі для кожного сегмента.
# **Додаткові предиктори**: Включити до моделі додаткові змінні, які можуть краще пояснити варіацію в інтенсивності комунікації, особливо для випадків із значними залишками.
# **Робастні методи оцінювання**: Враховуючи лептокуртичний розподіл залишків, варто розглянути застосування робастних методів регресійного аналізу, які менш чутливі до відхилень від нормальності.
# **Моніторинг новий спостережень**: При використанні моделі для прогнозування комунікаційних потреб нових партнерів важливо відстежувати, чи не є вони потенційними викидами, для яких прогнози можуть бути менш надійними.
# 
# #### Висновки
# Аналіз гістограми залишків регресійної моделі після очищення даних свідчить про значне покращення точності прогнозування, що відображається у зменшенні стандартного відхилення залишків на 33%. Розподіл залишків є симетричним із середнім значенням точно на нулі, що відповідає важливим передумовам регресійного аналізу.
# Водночас, виражена лептокуртичність розподілу та наявність "товстих хвостів" вказують на потенційну неоднорідність даних та можливість подальшого вдосконалення моделі через сегментацію або включення додаткових предикторів.
# Загалом, процедура очищення даних виявилася ефективною для підвищення якості моделі, хоча форма розподілу залишків свідчить про наявність структурних особливостей у даних, які можуть бути предметом подальшого дослідження.
# 

# ### Аналіз QQ-plot залишків регресійної моделі після очищення даних
# 
# #### Опис графіка
# 
# Представлений графік є QQ-plot (Quantile-Quantile plot) залишків регресійної моделі після процедури очищення даних від викидів. Цей графік є важливим діагностичним інструментом для перевірки відповідності розподілу залишків нормальному закону розподілу, що є однією з ключових передумов класичної лінійної регресії.
# 
# На графіку зображено:
# - Сині точки — емпіричні квантилі залишків моделі
# - Червона пряма лінія — теоретична лінія, яка відповідає ідеальному нормальному розподілу
# - У лівому верхньому куті — коефіцієнт кореляції між емпіричними та теоретичними квантилями (R = 0.8169)
# 
# #### Аналіз відхилень від нормальності
# 
# #### Загальна відповідність нормальному розподілу
# Коефіцієнт кореляції між емпіричними та теоретичними квантилями становить R = 0.8169. Це значення відображає помірно високу, але не ідеальну відповідність нормальному розподілу. Для ідеального нормального розподілу значення R наближалося б до 1.
# Порівнюючи з аналогічним графіком для неочищеного набору даних (де було R = 0.7763), спостерігається покращення цього показника після видалення викидів, що вказує на більшу відповідність залишків нормальному розподілу.
# 
# #### S-подібна форма QQ-кривої
# Графік демонструє виражену S-подібну форму, що є характерною ознакою розподілу з "важкими хвостами" (heavy-tailed distribution). Це означає, що розподіл залишків має більшу ймовірність екстремальних значень, ніж нормальний розподіл.
# **Центральна частина** (квантилі від -1 до 1): У цій області спостерігається майже горизонтальне плато, де синя лінія відхиляється від червоної теоретичної лінії. Це вказує на високу концентрацію залишків близько нуля, тобто на лептокуртичність (гостровершинність) розподілу, що підтверджується результатами аналізу гістограми.
# **Хвости розподілу** (квантилі < -2 та > 2): У цих областях емпірична лінія має значно крутіший нахил, ніж теоретична, що свідчить про більшу частоту великих за абсолютною величиною залишків, ніж передбачає нормальний розподіл.
# 
# #### Асиметрія та "сходинки" на графіку
# На QQ-plot присутні певні горизонтальні "сходинки" або "плато", особливо в хвостах розподілу. Ця структура може свідчити про:
# - Дискретність або кластеризацію в даних
# - Наявність однакових або близьких значень залишків
# - Присутність окремих підгруп у даних з різними характеристиками розподілу
# На графіку помітна асиметрія між верхнім правим та нижнім лівим хвостами. Верхній правий хвіст (великі позитивні залишки) виглядає довшим і крутішим, ніж нижній лівий (великі негативні залишки), що вказує на певну асиметричність розподілу з ухилом у бік позитивних залишків.
# 
# #### Порівняння з QQ-plot до очищення даних
# Порівнюючи цей графік з аналогічним для неочищеного набору даних (четвертий графік), можна зробити такі спостереження:
# **Підвищення коефіцієнта кореляції**: R збільшився з 0.7763 до 0.8169, що вказує на покращення відповідності нормальному розподілу після видалення викидів.
# **Зменшення діапазону залишків**: Після очищення даних діапазон залишків на QQ-plot скоротився з приблизно ±2000 до ±1500, що свідчить про видалення найбільш екстремальних значень.
# **Збереження S-подібної форми**: Незважаючи на очищення даних, основна структурна особливість QQ-plot — S-подібна форма — зберігається, що вказує на фундаментальну характеристику даних, а не на вплив окремих викидів.
# **Більш плавна крива**: QQ-plot після очищення демонструє більш плавну криву з менш вираженими "стрибками", що може свідчити про більшу однорідність даних.
# 
# #### Інтерпретація для регресійної моделі
# 
# #### Порушення передумов лінійної регресії
# QQ-plot продовжує демонструвати відхилення від нормальності розподілу залишків, хоча й менш виражене, ніж до очищення даних. Це відхилення має специфічну форму:
# **Лептокуртичність**: Висока концентрація залишків близько нуля (горизонтальне плато в центральній частині) свідчить про те, що модель досить точна для більшості "типових" спостережень.
# **"Важкі хвости"**: Крутіший нахил у хвостах розподілу вказує на частіші, ніж передбачає нормальний розподіл, значні відхилення від прогнозованих значень.
# 
# #### Вплив на статистичні висновки
# Відхилення від нормальності може впливати на:
# - Точність довірчих інтервалів для коефіцієнтів регресії
# - Надійність t-тестів та F-тестів для перевірки статистичної значущості
# - Якість передбачень, особливо для спостережень, що лежать далеко від середніх значень
# Втім, оскільки розмір вибірки дуже великий (понад 84 тисячі спостережень), центральна гранична теорема дозволяє зробити висновок, що ці проблеми не є критичними для загальної надійності моделі.
# 
# #### Структурні особливості даних
# S-подібна форма QQ-plot з горизонтальними плато може вказувати на:
# - Наявність суміші різних розподілів у вибірці
# - Потребу в сегментації даних або створенні категоріальних змінних
# - Можливі особливості вимірювання або збору даних
# 
# #### Порівняння ефективності очищення даних
# 
# #### Покращення нормальності розподілу
# Збільшення коефіцієнта кореляції на QQ-plot з 0.7763 до 0.8169 (на 5.2%) свідчить про помітне, хоча й не радикальне, покращення відповідності розподілу залишків нормальному закону після очищення даних.
# 
# #### Збереження структурних особливостей
# Важливим спостереженням є те, що основні структурні особливості QQ-plot (S-подібна форма, лептокуртичність, "важкі хвости") зберігаються навіть після видалення викидів. Це вказує на те, що ці характеристики є фундаментальними властивостями даних, а не наслідком впливу окремих аномальних спостережень.
# 
# #### Компроміс між покращенням моделі та повнотою даних
# Хоча очищення даних призвело до певного покращення нормальності розподілу залишків, зберігається питання, чи не втрачено важливу інформацію про специфічні сегменти партнерів при видаленні викидів.
# 
# #### Рекомендації для покращення моделі
# 
# #### Трансформація змінних
# Відхилення від нормальності можна спробувати скоригувати через трансформацію змінних:
# - Логарифмічна трансформація (для даних з правою асиметрією)
# - Квадратний корінь (для даних з помірною правою асиметрією)
# - Box-Cox трансформація для визначення оптимального перетворення
# 
# #### Робастні методи оцінювання
# Враховуючи "важкі хвости" розподілу залишків, доцільно застосувати робастні методи регресійного аналізу, які менш чутливі до відхилень від нормальності:
# - М-оцінювачі (Huber, Tukey)
# - Квантильна регресія
# 
# #### Сегментація даних
# S-подібна форма QQ-plot може свідчити про доцільність сегментації даних:
# - Кластерний аналіз для виявлення природних груп
# - Створення окремих моделей для різних сегментів партнерів
# - Введення нових категоріальних предикторів, які відображають сегментацію
# 
# #### Висновки для бізнес-застосування
# Аналіз QQ-plot залишків після очищення даних дозволяє зробити важливі висновки для практичного застосування регресійної моделі у прогнозуванні комунікаційної активності партнерів B2B компанії:
# **Покращення, але не ідеальність**: Хоча очищення даних покращило відповідність залишків нормальному розподілу, зберігаються певні відхилення, які вказують на необхідність диференційованого підходу до аналізу та прогнозування.
# **Висока точність для "типових" партнерів**: Лептокуртичність розподілу свідчить про високу точність моделі для більшості "типових" партнерів, що є позитивним аспектом для планування комунікаційних ресурсів.
# **Обережність з крайніми випадками**: "Важкі хвости" розподілу вказують на те, що для деяких партнерів модель може давати значні помилки прогнозування. Це потребує більш гнучкого підходу до планування ресурсів для нетипових випадків.
# **Потенціал для сегментації**: Структура QQ-plot підтверджує наявність різних патернів взаємодії серед партнерів, що обґрунтовує доцільність розробки специфічних комунікаційних стратегій для різних сегментів.
# Загалом, хоча очищена лінійна модель демонструє кращу відповідність нормальному розподілу залишків, ніж неочищена, зберігаються структурні особливості, які вказують на потенційну користь від застосування більш складних методів моделювання та сегментованого підходу до аналізу партнерів.
# 

# ### Аналіз графіка залежності залишків від кількості замовлень після очищення даних
# 
# #### Опис графіка
# 
# Представлений графік відображає залежність залишків регресійної моделі від кількості замовлень партнерів (`partner_total_orders`) після процедури очищення даних від викидів. Цей графік є важливим діагностичним інструментом, який дозволяє оцінити наявність систематичних відхилень прогнозованих значень від фактичних у залежності від величини предиктора.
# 
# На графіку зображено:
# - Горизонтальна вісь: кількість замовлень партнера (`partner_total_orders`), в діапазоні від 0 до приблизно 500
# - Вертикальна вісь: залишки регресійної моделі, в діапазоні від -1500 до +2000
# - Сині точки: окремі спостереження
# - Чорна горизонтальна лінія на нульовому рівні: ідеальна ситуація, коли модель прогнозує точно
# - Горизонтальні пунктирні лінії: межі одного стандартного відхилення (±1σ = ±216.0033)
# - Зелена крива: LOWESS (Locally Weighted Scatterplot Smoothing) тренд, який показує локально зважену лінію тренду
# 
# #### Аналіз структури залишків
# 
# #### Гетероскедастичність
# Графік демонструє яскраво виражену гетероскедастичність (нерівномірність дисперсії залишків). Спостерігається чітка "віялоподібна" структура, при якій:
# Для партнерів з низькою кількістю замовлень (приблизно до 50) дисперсія залишків відносно мала
# З ростом кількості замовлень дисперсія залишків суттєво збільшується, досягаючи максимальних значень для партнерів з кількістю замовлень понад 300
# Така структура свідчить про порушення передумови гомоскедастичності, яка є основоположною для класичної лінійної регресії. Гетероскедастичність призводить до:
# - Неефективності оцінок методу найменших квадратів
# - Помилкової оцінки стандартних помилок коефіцієнтів
# - Некоректних висновків про статистичну значущість
# 
# #### Нелінійні патерни
# На графіку чітко простежуються групи точок, що утворюють криволінійні траєкторії, особливо в області високих значень `partner_total_orders`. Ці "промені" точок, що відхиляються від нульової лінії як у позитивному, так і в негативному напрямках, свідчать про:
# Можливу нелінійну залежність між змінними, яку не вдалося вловити лінійною моделлю
# Наявність структурованих підгруп партнерів з різними патернами взаємозв'язку між замовленнями та повідомленнями
# Імовірність того, що модель недооцінює кількість повідомлень для одних груп партнерів і переоцінює для інших
# Особливо помітні кілька "наборів траєкторій":
# - Висхідні криві в області позитивних залишків (верхня частина графіка)
# - Низхідні криві в області негативних залишків (нижня частина графіка)
# 
# #### LOWESS тренд
# Зелена лінія LOWESS тренду показує загальну тенденцію середніх значень залишків на різних рівнях `partner_total_orders`. Ідеально вона мала б збігатися з горизонтальною лінією на нульовому рівні, проте спостерігаються незначні відхилення:
# Для малих значень `partner_total_orders` (до ~50) тренд коливається близько нуля
# У середньому діапазоні (від ~50 до ~300) спостерігається незначне зниження тренду
# Для великих значень (понад 300) тренд знову наближається до нуля
# Ці коливання LOWESS тренду є значно менш вираженими, ніж до очищення даних, що свідчить про певне покращення лінійності моделі.
# 
# #### Порівняння з графіком до очищення даних
# Порівнюючи з аналогічним графіком до очищення даних (графік 5), можна відзначити такі зміни:
# **Зменшення загального діапазону залишків**:
#    - До очищення: приблизно від -3000 до +3000
#    - Після очищення: приблизно від -1500 до +2000
#    Це свідчить про видалення найбільш екстремальних відхилень і загальне покращення точності моделі.
# **Збереження структури гетероскедастичності**:
#    - "Віялоподібна" форма розсіювання залишків залишається, хоча й у менш вираженому вигляді
#    - Дисперсія залишків усе ще збільшується з ростом кількості замовлень
#    Це вказує на те, що гетероскедастичність є структурною особливістю даних, а не наслідком окремих викидів.
# **Зменшення нахилу LOWESS тренду**:
#    - До очищення LOWESS тренд мав більш виражені відхилення від нуля
#    - Після очищення тренд ближчий до горизонтальної лінії
#    Це свідчить про покращення лінійності моделі після видалення викидів.
# **Збереження нелінійних патернів**:
#    - Крайні патерни, що утворюють криволінійні траєкторії, залишаються присутніми
#    - Структуровані групи точок продовжують виокремлюватися, хоча й з меншою амплітудою
#    Це підтверджує, що нелінійні структури є фундаментальною характеристикою даних, а не артефактом викидів.
# 
# #### Аналіз статистичних характеристик
# 
# #### Стандартне відхилення залишків
# Горизонтальні пунктирні лінії на графіку позначають межі одного стандартного відхилення (±1σ = ±216.0033). Порівнюючи з аналогічним показником до очищення даних (±1σ = ±323.3741), спостерігаємо:
# **Зменшення стандартного відхилення на 33.2%** (з 323.3741 до 216.0033)
# **Звуження діапазону "типових" залишків**
# Це суттєве покращення свідчить про підвищення точності прогнозування моделі після очищення даних.
# 
# #### Екстремальні значення
# Незважаючи на очищення даних, графік все ще демонструє наявність значень, що перевищують ±3σ:
# - Максимальні позитивні залишки сягають приблизно +1800
# - Максимальні негативні залишки сягають приблизно -1400
# Ці значення виходять за межі ±3σ (±648.0099), що свідчить про "важкі хвости" розподілу залишків та підтверджує висновки, зроблені при аналізі QQ-plot.
# 
# #### Інтерпретація для регресійної моделі
# 
# #### Порушення передумов лінійної регресії
# Графік залишків відносно `partner_total_orders` виявляє два ключові порушення передумов класичної лінійної регресії:
# **Гетероскедастичність**: Нерівномірність дисперсії залишків у залежності від величини предиктора призводить до неефективності оцінок методом найменших квадратів та потенційно некоректних висновків про статистичну значущість.
# **Нелінійність**: Наявність криволінійних патернів на графіку залишків вказує на те, що лінійна функціональна форма моделі може не повністю відображати складність взаємозв'язку між змінними.
# 
# #### Інтерпретація для різних груп партнерів
# 
# Аналіз графіка дозволяє зробити важливі висновки про точність прогнозування для різних груп партнерів:
# **Партнери з малою кількістю замовлень** (до ~50):
#    - Відносно низька дисперсія залишків
#    - Висока точність прогнозування
#    - Можливе незначне систематичне відхилення (залежно від коливань LOWESS тренду)
# **Партнери з середньою кількістю замовлень** (від ~50 до ~200):
#    - Помірна дисперсія залишків
#    - Задовільна точність прогнозування
#    - Потенційні патерни нелінійності
# **Партнери з великою кількістю замовлень** (понад 200):
#    - Висока дисперсія залишків
#    - Значно нижча точність прогнозування
#    - Виражені групові патерни, що утворюють криволінійні траєкторії
# 
# #### Вплив на якість прогнозування
# Виявлені особливості графіка залишків мають такі наслідки для якості прогнозування:
# **Нерівномірна надійність прогнозів**: Довірчі інтервали прогнозів мають суттєво відрізнятися в залежності від обсягу замовлень партнера.
# **Потенційні систематичні помилки**: Наявність криволінійних патернів вказує на потенційні систематичні помилки прогнозування для певних груп партнерів.
# **Обмежена екстраполяція**: Модель може бути ненадійною при екстраполяції на партнерів з кількістю замовлень, що виходить за межі наявних даних.
# 
# #### Рекомендації для покращення моделі
# 
# #### Підходи до боротьби з гетероскедастичністю
# **Зважений метод найменших квадратів (WLS)**:
#    - Застосування вагових коефіцієнтів, обернено пропорційних до дисперсії залишків
#    - Використання різних вагових функцій для різних діапазонів `partner_total_orders`
# **Трансформація залежної змінної**:
#    - Логарифмічна трансформація `partner_total_messages`
#    - Трансформація Box-Cox для оптимального перетворення
# **Робастні методи оцінювання**:
#    - М-оцінювачі (Huber, Tukey)
#    - Метод найменших модулів (LAD)
# 
# #### Врахування нелінійності
# **Поліноміальна регресія**:
#    - Включення квадратичного та/або кубічного члена для `partner_total_orders`
#    - Вибір оптимального степеня полінома на основі інформаційних критеріїв
# **Сплайн-регресія**:
#    - Використання кусково-поліноміальних функцій
#    - Природні сплайни або В-сплайни з оптимізованим розміщенням вузлів
# **Нелінійні моделі**:
#    - Моделі з дробово-раціональними функціями
#    - Експоненційні або логістичні моделі
# 
# #### Сегментація даних
# Враховуючи виражені групові патерни, доцільно розглянути:
# **Кластеризацію партнерів**:
#    - Застосування методів кластерного аналізу для виявлення природних груп
#    - Побудова окремих моделей для кожного кластера
# **Моделі зі зміною режиму**:
#    - Переключення між різними режимами моделювання в залежності від порогових значень `partner_total_orders`
#    - Використання моделей регресії зі структурними розривами
# **Змішані моделі**:
#    - Включення випадкових ефектів для різних груп партнерів
#    - Ієрархічне моделювання з урахуванням групової структури
# 
# #### Висновки для бізнес-застосування
# Аналіз графіка залишків відносно кількості замовлень після очищення даних дозволяє зробити важливі висновки для практичного застосування регресійної моделі у прогнозуванні комунікаційної активності партнерів B2B компанії:
# **Диференційована точність прогнозування**:
#    - Висока точність для партнерів з малою кількістю замовлень
#    - Помірна точність для середнього сегмента
#    - Знижена точність для партнерів з великою кількістю замовлень
# Практична рекомендація: розробка різних стратегій планування комунікаційних ресурсів з урахуванням сегмента партнера та очікуваної точності прогнозу.
# **Необхідність сегментованого підходу**:
#    - Чіткі групові патерни свідчать про різні моделі взаємодії для різних типів партнерів
#    - Ефективність комунікації може значно відрізнятися між сегментами
# Практична рекомендація: розробка таргетованих комунікаційних стратегій для різних категорій партнерів з урахуванням їхніх специфічних патернів взаємодії.
# **Вдосконалення прогностичної моделі**:
#    - Очищення даних покращило загальну точність моделі (зменшення стандартного відхилення на 33.2%)
#    - Однак структурні особливості (гетероскедастичність, нелінійність) зберігаються
# Практична рекомендація: постійне вдосконалення моделі прогнозування з використанням більш складних методів, що враховують виявлені структурні особливості.
# **Операційні наслідки**:
#    - Для партнерів з великою кількістю замовлень необхідний більший резерв комунікаційних ресурсів через підвищену невизначеність прогнозу
#    - Для партнерів з малою та середньою кількістю замовлень можливе більш точне планування
# Практична рекомендація: впровадження адаптивної системи планування ресурсів з урахуванням ступеня невизначеності прогнозу для різних сегментів партнерів.
# 
# Загалом, графік залишків після очищення даних демонструє суттєве покращення статистичних характеристик моделі, але водночас вказує на необхідність застосування більш складних методів моделювання та сегментованого підходу до аналізу партнерської бази. Це дозволить забезпечити максимальну ефективність комунікаційної стратегії B2B компанії та оптимальне використання її ресурсів.

# In[25]:


# Порівняння результатів до та після очищення
if len(outlier_indices) > 0:
    print_header("ПОРІВНЯННЯ РЕЗУЛЬТАТІВ ДО ТА ПІСЛЯ ОЧИЩЕННЯ")
    print(f"Кількість спостережень до очищення: {len(x_data)}")
    print(f"Кількість спостережень після очищення: {len(x_clean)}")
    print(f"Видалено викидів: {len(outlier_indices)}")
    print("\nРівняння регресії:")
    print(f"До:    {y_name} = {results_before['a0']:.4f} + {results_before['a1']:.4f} * {x_name}")
    print(f"Після: {y_name} = {results_after['a0']:.4f} + {results_after['a1']:.4f} * {x_name}")
    print("\nКоефіцієнт детермінації R²:")
    print(f"До:    {results_before['r_squared']:.4f}")
    print(f"Після: {results_after['r_squared']:.4f}")
    print(f"Зміна: {results_after['r_squared'] - results_before['r_squared']:.4f}")


# In[26]:


# Візуалізація порівняння моделей до та після очищення
print_header("ВІЗУАЛІЗАЦІЯ ПОРІВНЯННЯ МОДЕЛЕЙ")
# Викликаємо функцію порівняння моделей
compare_regression_models(x_data, y_data, x_clean, y_clean, results_before, results_after, x_name, y_name)


# ### Аналіз порівняння регресійних моделей до та після очищення даних
# 
# #### Опис графіка
# 
# Представлений графік демонструє порівняння двох регресійних моделей: однієї, побудованої на основі оригінального набору даних, та іншої — після процедури очищення даних від викидів. Цей графік є узагальнюючим, оскільки дозволяє візуально оцінити вплив очищення даних на характеристики регресійної моделі та розподіл спостережень.
# 
# На графіку зображено:
# Горизонтальна вісь: кількість замовлень партнера (`partner_total_orders`), в діапазоні від 0 до приблизно 1300
# Вертикальна вісь: кількість повідомлень партнера (`partner_total_messages`), в діапазоні від 0 до приблизно 12000
# Сині точки: спостереження до очищення даних
# Зелені точки: спостереження після очищення даних
# Синя лінія: лінія регресії до очищення даних (y = 19.2445 + 8.8312x, R² = 0.9448)
# Зелена лінія: лінія регресії після очищення даних (y = 21.1406 + 8.6617x, R² = 0.9355)
# 
# #### Аналіз параметрів регресійних моделей
# 
# #### Коефіцієнти регресії
# **Вільний член (intercept)**:
# До очищення: a₀ = 19.2445
# Після очищення: a₀ = 21.1406
# Різниця: +1.8961 (+9.85%)
# Збільшення вільного члена означає, що після очищення даних базовий рівень комунікації (кількість повідомлень при нульовій кількості замовлень) оцінюється як вищий. Це може відображати більш точну оцінку "фонової" комунікації, яка не пов'язана безпосередньо з процесом обробки замовлень.
# **Коефіцієнт нахилу (slope)**:
# До очищення: a₁ = 8.8312
# Після очищення: a₁ = 8.6617
# Різниця: -0.1695 (-1.92%)
# Незначне зменшення коефіцієнта нахилу вказує на те, що після очищення даних оцінка кількості повідомлень на одне замовлення стала дещо нижчою. Однак, ця зміна є мінімальною, що свідчить про стабільність основного взаємозв'язку між кількістю замовлень та кількістю повідомлень.
# 
# #### Коефіцієнт детермінації (R²)
# До очищення: R² = 0.9448
# Після очищення: R² = 0.9355
# Різниця: -0.0093 (-0.98%)
# Незначне зменшення коефіцієнта детермінації після очищення даних може здатися парадоксальним, оскільки зазвичай очікується, що видалення викидів має покращити якість моделі. Однак це можна пояснити такими факторами:
# **Зміна в структурі даних**: Після очищення залишилися спостереження, які мають більш складну структуру взаємозв'язку, що не повністю вловлюється лінійною моделлю.
# **Видалення інформативних спостережень**: Деякі віддалені точки, які були видалені як "викиди", могли насправді надавати важливу інформацію про взаємозв'язок між змінними для певних сегментів партнерів.
# **Вужчий діапазон даних**: Зменшення діапазону пояснюючої змінної після очищення даних може приводити до зниження R², навіть якщо модель стає більш надійною.
# 
# Незважаючи на незначне зменшення R², очищена модель може бути більш надійною та стабільною, оскільки вона менше залежить від екстремальних значень, які могли непропорційно впливати на оцінки параметрів.
# 
# #### Аналіз розподілу даних до та після очищення
# 
# #### Зміна діапазону даних
# **Діапазон `partner_total_orders`**:
# До очищення: від 0 до приблизно 1300
# Після очищення: від 0 до приблизно 400
# Скорочення: приблизно на 69%
# **Діапазон `partner_total_messages`**:
# До очищення: від 0 до приблизно 12000
# Після очищення: від 0 до приблизно 4500
# Скорочення: приблизно на 63%
# 
# Суттєве зменшення діапазону обох змінних після очищення даних свідчить про видалення спостережень з екстремально високими значеннями, які могли бути викидами або представляти атипових партнерів.
# 
# #### Зміна щільності розподілу
# **Щільність точок**:
# До очищення: точки розподілені більш розріджено, особливо в області високих значень
# Після очищення: точки концентруються в більш компактній області, створюючи більш щільну хмару
# **Патерни розподілу**:
# До очищення: помітні окремі "промені" або "траєкторії" точок, особливо у верхній частині графіка
# Після очищення: структура більш однорідна, хоча певні патерни все ще простежуються
# 
# Зміна щільності розподілу вказує на те, що очищення даних призвело до формування більш гомогенної вибірки, яка, однак, все ще зберігає певну структурну неоднорідність.
# 
# #### Вплив на гетероскедастичність
# Незважаючи на очищення даних, візуально все ще спостерігається віялоподібне розширення розсіювання точок із зростанням `partner_total_orders`, що свідчить про збереження гетероскедастичності. Це підтверджує висновок про те, що гетероскедастичність є структурною характеристикою даних, а не наслідком окремих викидів.
# 
# #### Інтерпретація змін параметрів моделі
# 
# #### Зміна вільного члена
# Збільшення вільного члена на 9.85% (з 19.2445 до 21.1406) після очищення даних можна інтерпретувати так:
# **Базова комунікація**: Очищена модель оцінює вищий базовий рівень комунікації, який не залежить від кількості замовлень. Це може відображати постійну комунікацію, пов'язану з підтримкою відносин з партнерами, інформаційними повідомленнями тощо.
# **Корекція зміщення**: Видалення викидів могло усунути зміщення оцінки вільного члена, яке було спричинене непропорційним впливом екстремальних значень.
# **Структурні зміни у вибірці**: Зміна складу вибірки після очищення могла призвести до того, що в ній стали переважати партнери з вищим базовим рівнем комунікації.
# 
# #### Зміна коефіцієнта нахилу
# Незначне зменшення коефіцієнта нахилу на 1.92% (з 8.8312 до 8.6617) свідчить про стабільність основного взаємозв'язку між кількістю замовлень та кількістю повідомлень. Однак навіть ця незначна зміна має певні наслідки:
# **Ефективність комунікації**: Після очищення даних модель передбачає дещо меншу кількість повідомлень на одне замовлення, що може вказувати на вищу ефективність комунікації у "типових" партнерів порівняно з атиповими випадками, які були видалені.
# **Нелінійність залежності**: Невелика зміна нахилу може також відображати нелінійну природу залежності, яка стає більш помітною після видалення викидів.
# 
# #### Зміна коефіцієнта детермінації
# Незначне зменшення R² на 0.98% (з 0.9448 до 0.9355) у контексті проведених змін можна інтерпретувати так:
# **Вплив видалення інформативних спостережень**: Деякі спостереження з високими значеннями змінних, які були видалені як викиди, могли сильно підтримувати лінійну модель, збільшуючи R².
# **Зменшення "розтягуючого ефекту"**: Видалення спостережень з екстремальними значеннями зменшує штучне збільшення R², яке може виникати через розширення діапазону даних.
# **Зниження чутливості до шуму**: Очищена модель, хоча й має дещо нижчий R², може бути менш чутливою до шуму в даних і тому більш надійною для прогнозування.
# 
# Важливо зазначити, що обидві моделі мають дуже високий коефіцієнт детермінації (>0.93), що свідчить про сильний лінійний взаємозв'язок між змінними незалежно від процедури очищення даних.
# 
# #### Оцінка якості очищення даних
# 
# #### Ідентифікація та видалення викидів
# Графік дозволяє візуально оцінити результати процедури очищення даних:
# **Видалені спостереження**: Всі точки, що знаходяться у верхній правій частині графіка (приблизно з `partner_total_orders` > 400), були класифіковані як викиди і видалені.
# **Дискримінаційна здатність**: Процедура очищення чітко дискримінувала "типових" та "атипових" партнерів, формуючи більш гомогенну вибірку.
# **Збереження структури даних**: Важливо, що процедура очищення зберегла основну структуру розподілу даних, не спотворюючи її.
# 
# #### Компроміс між точністю та повнотою даних
# Графік наочно демонструє компроміс, який доводиться робити при очищенні даних:
# **Втрата інформації про атипових партнерів**: Видалення викидів призвело до втрати інформації про партнерів з надзвичайно високою кількістю замовлень та повідомлень, які можуть представляти важливий сегмент клієнтської бази.
# **Покращення надійності для "типових" випадків**: Натомість модель стала більш надійною для прогнозування "типових" випадків, які становлять більшість клієнтської бази.
# **Обмеження діапазону прогнозування**: Очищена модель має обмежений діапазон надійного прогнозування, що необхідно враховувати при її застосуванні.
# 
# #### Практичні висновки для бізнес-застосування
# 
# #### Рекомендації для використання моделей
# **Вибір моделі залежно від мети**:
# Для загального аналізу та візуалізації тенденцій: обидві моделі дають схожі результати
# Для прогнозування "типових" партнерів: перевага очищеної моделі (зелена лінія)
# Для прогнозування партнерів з високою активністю: необхідна оригінальна модель (синя лінія) або розробка спеціалізованих моделей
# **Комбінований підхід**:
# Використання очищеної моделі як базової
# Додаткове коригування для партнерів з високою активністю
# Розробка системи "прапорців" для ідентифікації потенційно атипових випадків
# **Встановлення діапазону надійності**:
# Для очищеної моделі: надійний прогноз до `partner_total_orders` ≈ 400
# Для оригінальної моделі: ширший діапазон, але нижча точність для "типових" випадків
# 
# #### Покращення операційної ефективності
# **Оптимізація комунікаційних процесів**:
# На основі коефіцієнта нахилу: в середньому 8.6-8.8 повідомлень на одне замовлення
# Розробка стандартизованих комунікаційних шаблонів для скорочення кількості повідомлень
# Диференціація підходів залежно від обсягу замовлень партнера
# **Планування ресурсів**:
# Використання очищеної моделі для базового планування комунікаційних ресурсів
# Додатковий резерв для партнерів з високою активністю
# Врахування вищого базового рівня комунікації, виявленого в очищеній моделі
# **Моніторинг ефективності**:
# Відстеження співвідношення `partner_total_messages` / `partner_total_orders`
# Ідентифікація партнерів, які суттєво відхиляються від прогнозованих значень
# Аналіз причин відхилень для вдосконалення моделі та комунікаційних процесів
# 
# #### Стратегічні рекомендації
# **Сегментація партнерської бази**:
# Виділення сегмента "висококомунікаційних" партнерів (вище лінії регресії)
# Виділення сегмента "низькокомунікаційних" партнерів (нижче лінії регресії)
# Розробка таргетованих стратегій для кожного сегмента
# **Оптимізація комунікаційної стратегії**:
# Для партнерів з низькою активністю: акцент на базову комунікацію (вільний член моделі)
# Для партнерів з середньою активністю: стандартизований підхід на основі лінійної залежності
# Для партнерів з високою активністю: індивідуалізований підхід з урахуванням особливостей
# **Розвиток аналітичних можливостей**:
# Впровадження більш складних регресійних моделей (поліноміальних, сплайнових)
# Розробка системи раннього виявлення змін у комунікаційних патернах
# Інтеграція даних про комунікацію з іншими бізнес-показниками
# 
# #### Порівняльна оцінка моделей
# 
# #### Статистичне порівняння
# **Зміни в параметрах**:
# Вільний член: збільшення на 9.85%
# Коефіцієнт нахилу: зменшення на 1.92%
# R²: зменшення на 0.98%
# Ці зміни є статистично значущими, але не змінюють кардинально характер моделі, що свідчить про надійність виявленого лінійного взаємозв'язку.
# **Відносність змін**:
# Більша зміна вільного члена порівняно з коефіцієнтом нахилу
# Мінімальна зміна R² відносно його високого абсолютного значення
# Суттєва зміна в діапазоні даних
# Відносність змін вказує на те, що очищення даних більше вплинуло на оцінку базового рівня комунікації, ніж на основний взаємозв'язок між кількістю замовлень та повідомлень.
# 
# #### Візуальне порівняння
# **Нахил ліній регресії**:
# Візуально лінії майже паралельні
# Незначна різниця в нахилі помітна лише на великих значеннях
# В діапазоні очищених даних лінії дуже близькі
# **Щільність точок відносно ліній**:
# До очищення: більший розкид, особливо для високих значень
# Після очищення: компактніше розташування навколо лінії регресії
# В обох випадках присутні патерни, що відхиляються від лінійної залежності
# **Загальна оцінка**:
# Очищена модель: краща для "більшості" спостережень
# Оригінальна модель: ширший діапазон застосування, але потенційно менша точність
# 
# #### Висновки та рекомендації
# Порівняльний аналіз регресійних моделей до та після очищення даних дозволяє зробити такі висновки:
# **Надійність лінійної залежності**:
# Високі значення R² (>0.93) для обох моделей підтверджують сильну лінійну залежність між кількістю замовлень та кількістю повідомлень
# Незначні зміни в коефіцієнті нахилу свідчать про стабільність цієї залежності
# **Вплив очищення даних**:
# Видалення викидів значно звузило діапазон даних
# Очищення даних призвело до більш надійної оцінки параметрів для "типових" партнерів
# Втім, виникає ризик втрати важливої інформації про атипових партнерів
# **Практичні рекомендації**:
# Використання очищеної моделі для планування комунікаційних ресурсів для "типових" партнерів
# Розробка окремої стратегії для партнерів з високою активністю
# Моніторинг та адаптація підходу залежно від змін у структурі партнерської бази
# **Напрямки подальшого дослідження**:
# Аналіз причин високої комунікаційної активності у партнерів з великою кількістю замовлень
# Розробка нелінійних моделей для кращого врахування патернів у даних
# Дослідження змін взаємозв'язку між замовленнями та комунікацією в часі
# 
# Загалом, порівняння регресійних моделей демонструє, що очищення даних є корисним інструментом для отримання більш надійних оцінок параметрів моделі, але повинно застосовуватися з урахуванням можливої втрати важливої інформації. Для ефективного бізнес-застосування доцільно використовувати гібридний підхід, який поєднує переваги обох моделей та враховує сегментацію партнерської бази.

# ### Аналіз порівняння розподілу залишків до та після очищення даних
# 
# #### Опис графіка
# 
# Представлений графік складається з двох гістограм, які відображають розподіл залишків регресійної моделі до очищення даних (ліва частина) та після очищення даних (права частина). Цей графік дозволяє безпосередньо оцінити вплив процедури очищення даних на статистичні характеристики залишків та їхню відповідність нормальному розподілу, що є важливою передумовою для класичного регресійного аналізу.
# 
# На графіку зображено:
# - Дві гістограми залишків: синя (до очищення) та зелена (після очищення)
# - Криві нормального розподілу (червоні лінії), накладені на кожну гістограму
# - Вертикальні пунктирні червоні лінії, що позначають середні значення
# - Статистичні характеристики кожного розподілу: обсяг вибірки (n), середнє значення (μ) та стандартне відхилення (σ)
# 
# #### Аналіз статистичних характеристик
# 
# #### Обсяг вибірки (n)
# До очищення: n = 86,794
# Після очищення: n = 84,273
# Різниця: -2,521 спостереження (-2.91%)
# Відносно невелике зменшення обсягу вибірки свідчить про те, що процедура очищення даних була достатньо консервативною і видалила лише 2.91% спостережень, які було класифіковано як викиди. Це позитивний аспект, оскільки зберігається більшість інформації, закладеної в оригінальних даних.
# 
# #### Середнє значення (μ)
# До очищення: μ = -0.0000
# Після очищення: μ = 0.0000
# Різниця: практично відсутня
# Як і очікувалося, середнє значення залишків як до, так і після очищення даних є практично нульовим. Це є наслідком математичних властивостей методу найменших квадратів, який гарантує, що сума залишків дорівнює нулю. Збереження цієї властивості після очищення даних свідчить про коректність процедури.
# 
# #### Стандартне відхилення (σ)
# До очищення: σ = 322.3050
# Після очищення: σ = 216.0033
# Різниця: -106.3017 (-32.98%)
# Суттєве зменшення стандартного відхилення є найважливішою зміною, що спостерігається після очищення даних. Зменшення σ на приблизно 33% свідчить про значне підвищення точності моделі — прогнозовані значення стали ближчими до фактичних після видалення викидів.
# 
# #### Аналіз форми розподілу
# 
# #### Центральна тенденція
# Обидва розподіли демонструють сильну центральну тенденцію з високим піком біля нуля, що вказує на лептокуртичність (гостровершинність) розподілу. Однак після очищення даних:
# **Висота піка**: Максимальна частота у центральному стовпці гістограми збільшилася з приблизно 30,000 до приблизно 34,000, що свідчить про ще більшу концентрацію залишків навколо нуля після очищення.
# **Ширина центральної області**: Візуально ширина центральної частини гістограми (де зосереджена більшість спостережень) зменшилася, що корелює зі зменшенням стандартного відхилення.
# Ці зміни вказують на те, що після очищення даних модель стала більш точною для більшості спостережень, що зосереджені навколо центральної частини розподілу.
# 
# #### Симетричність
# Обидва розподіли візуально симетричні відносно нуля, що є позитивною характеристикою, оскільки симетричність розподілу залишків вказує на відсутність систематичного зміщення прогнозів моделі в якомусь одному напрямку.
# Втім, можна помітити певні особливості:
# - До очищення: спостерігається незначна асиметрія з дещо більшою масою в області негативних залишків
# - Після очищення: розподіл став ще більш симетричним
# Ця незначна зміна у симетрії може свідчити про те, що оригінальна модель мала тенденцію дещо переоцінювати значення для деяких спостережень, і ця тенденція була частково скоригована після очищення даних.
# 
# #### Відповідність нормальному розподілу
# Червоні криві на гістограмах представляють теоретичний нормальний розподіл з відповідними параметрами (μ та σ). Порівнюючи емпіричні розподіли з теоретичними, можна зробити такі спостереження:
# **До очищення**:
# Емпіричний розподіл має значно вищий та гостріший пік у центрі, ніж теоретичний нормальний розподіл
# Хвости емпіричного розподілу "важчі", ніж у нормального розподілу
# Спостерігається суттєве відхилення від нормальності
# **Після очищення**:
# Розбіжність між емпіричним та теоретичним розподілами залишається, але змінюється її характер
# Пік емпіричного розподілу став ще гострішим відносно теоретичного
# Хвости стали легшими, але все ще відрізняються від теоретичних
# Це свідчить про те, що хоча очищення даних наблизило розподіл залишків до нормального в термінах зменшення "важкості" хвостів, воно також посилило лептокуртичність розподілу, що є відхиленням від нормальності іншого характеру.
# 
# #### Хвости розподілу
# Хвости розподілу, особливо їхня "важкість", є важливою характеристикою для оцінки надійності статистичних висновків:
# **До очищення**:
# Хвости простягаються приблизно від -1500 до +1500
# Спостерігаються окремі стовпці гістограми в далеких хвостах, що відповідають екстремальним залишкам
# **Після очищення**:
# Діапазон хвостів звузився, але не радикально (приблизно від -1000 до +1000)
# Екстремальні значення в далеких хвостах стали рідшими
# Ця зміна є очікуваною, оскільки процедура очищення даних зазвичай спрямована на видалення спостережень з екстремальними залишками. Однак важливо відзначити, що навіть після очищення розподіл залишків все ще має "важчі" хвости, ніж нормальний розподіл, що підтверджується результатами аналізу QQ-plot.
# 
# #### Порівняльний аналіз впливу очищення даних
# 
# ### Зміна точності моделі
# Найбільш значущою зміною після очищення даних є зменшення стандартного відхилення залишків на 32.98% (з 322.3050 до 216.0033). Це вказує на:
# **Підвищення загальної точності**: Середня абсолютна помилка прогнозування зменшилася
# **Зниження дисперсії помилок**: Прогнози стали більш стабільними та надійними
# **Покращення якості моделі**: Видалення викидів дозволило краще виявити основний взаємозв'язок між змінними
# У практичному сенсі це означає, що після очищення даних модель дає значно точніші прогнози для переважної більшості спостережень.
# 
# #### Зміна форми розподілу
# Окрім зміни стандартного відхилення, спостерігаються також зміни у формі розподілу:
# **Посилення лептокуртичності**: Центральний пік став ще вищим та гострішим після очищення даних, що вказує на ще більшу концентрацію залишків близько нуля.
# **Зменшення "важкості" хвостів**: Хоча хвости все ще "важчі", ніж у нормального розподілу, після очищення вони стали менш вираженими.
# **Покращення симетрії**: Розподіл став більш симетричним, що свідчить про зменшення систематичних відхилень у прогнозах.
# Ці зміни мають двоякий характер для оцінки нормальності розподілу: з одного боку, зменшення "важкості" хвостів та покращення симетрії наближає розподіл до нормального; з іншого боку, посилення лептокуртичності віддаляє його від нормального розподілу, який має мезокуртичну форму.
# 
# #### Збереження розміру вибірки
# Важливим позитивним аспектом процедури очищення є збереження 97.09% оригінальних спостережень. Це означає, що:
# **Процедура очищення була таргетованою**: Вона видалила лише найбільш проблемні спостереження
# **Збереження статистичної потужності**: Розмір вибірки залишається дуже великим, що забезпечує надійність статистичних висновків
# **Мінімальна втрата інформації**: Очищення не призвело до значної втрати інформації, закладеної в оригінальних даних
# Це є свідченням ефективності обраного методу очищення даних, який досяг суттєвого покращення моделі при мінімальній втраті даних.
# 
# #### Інтерпретація з погляду регресійного аналізу
# 
# #### Відповідність передумовам регресійного аналізу
# Класична лінійна регресія базується на припущенні, що залишки мають нормальний розподіл. Порівняльний аналіз розподілів дозволяє оцінити, наскільки це припущення виконується:
# **До очищення даних**: Розподіл залишків суттєво відхиляється від нормального через гостровершинність та "важкі" хвости, що ставить під сумнів валідність статистичних висновків.
# **Після очищення даних**: Хоча розподіл залишків став ближчим до нормального в термінах зменшення "важкості" хвостів, він став ще більш лептокуртичним, що все ще є відхиленням від нормальності.
# Однак, враховуючи дуже великий розмір вибірки (n > 84,000), центральна гранична теорема дозволяє певною мірою "пом'якшити" вимогу нормальності, особливо для тестування гіпотез щодо параметрів моделі.
# 
# #### Вплив на довірчі інтервали та прогнози
# Зміни в розподілі залишків мають такі наслідки для статистичних висновків:
# **Довірчі інтервали для коефіцієнтів**:
# До очищення: Потенційно ненадійні через відхилення від нормальності
# Після очищення: Більш надійні завдяки зменшенню стандартного відхилення, але все ще вимагають обережності через лептокуртичність
# **Предиктивні інтервали**:
# До очищення: Широкі та потенційно неточні
# Після очищення: Значно звужені (пропорційно до зменшення σ), що дозволяє робити більш точні прогнози
# **Надійність p-значень**:
# До очищення: Потенційно викривлені через відхилення від нормальності
# Після очищення: Більш надійні, хоча все ще вимагають обережної інтерпретації
# 
# Важливо зазначити, що для регресійної моделі з таким великим розміром вибірки статистична значущість коефіцієнтів майже гарантована, тому більшу практичну цінність становить оцінка точності прогнозів, яка суттєво покращилася після очищення даних.
# 
# #### Інтерпретація лептокуртичності
# Виражена лептокуртичність розподілу залишків, яка посилилася після очищення даних, має важливу інтерпретацію для регресійної моделі:
# **Висока точність для "типових" випадків**: Концентрація залишків навколо нуля свідчить про те, що модель дуже точно прогнозує значення для більшості "типових" спостережень.
# **Наявність "аномальних" випадків**: Хоча їх стало менше після очищення, все ще присутні спостереження, для яких модель дає значні помилки прогнозування.
# **Потенційна неоднорідність даних**: Лептокуртичність може вказувати на те, що дані представляють суміш різних підгруп з різними характеристиками взаємозв'язку між змінними.
# Ця інтерпретація узгоджується з візуальними патернами, виявленими при аналізі графіка залишків відносно прогнозованих значень та змінної-предиктора.
# 
# ### Рекомендації щодо подальшого вдосконалення моделі
# 
# #### Робастні методи оцінювання
# Враховуючи, що навіть після очищення даних розподіл залишків все ще відрізняється від нормального, доцільно розглянути застосування робастних методів регресійного аналізу:
# **М-оцінювачі**: Методи, які надають менших ваг спостереженням з великими залишками, можуть бути ефективними для даних з лептокуртичним розподілом залишків.
# **Квантильна регресія**: Цей метод менш чутливий до відхилень від нормальності та може надати більш повну картину взаємозв'язку між змінними на різних рівнях розподілу.
# **Зважений метод найменших квадратів**: Застосування вагових коефіцієнтів, обернено пропорційних до дисперсії залишків, може покращити ефективність оцінок у випадку гетероскедастичності.
# 
# #### Трансформація даних
# Альтернативним підходом до подолання відхилень від нормальності є трансформація змінних:
# **Логарифмічна трансформація**: Для залежної змінної та/або предиктора може бути ефективною для зменшення лептокуртичності розподілу залишків.
# **Box-Cox трансформація**: Підбір оптимального параметра трансформації може наблизити розподіл залишків до нормального.
# **Нелінійні перетворення**: Включення квадратичних, кубічних або інших нелінійних членів може покращити лінійність взаємозв'язку та, як наслідок, нормальність розподілу залишків.
# 
# #### Сегментація даних
# Враховуючи потенційну неоднорідність даних, доцільно розглянути сегментований підхід до моделювання:
# **Кластерний аналіз**: Виявлення природних груп у даних і побудова окремих моделей для кожного кластера.
# **Моделі зі змінною структурою**: Включення індикаторних змінних або інтеракційних членів для різних сегментів партнерів.
# **Ієрархічні моделі**: Врахування групової структури даних через випадкові ефекти або інші методи багаторівневого моделювання.
# 
# #### Висновки для бізнес-застосування
# Порівняльний аналіз розподілу залишків до та після очищення даних дозволяє зробити такі висновки для практичного застосування регресійної моделі у прогнозуванні комунікаційної активності партнерів B2B компанії:
# 
# #### Підвищення надійності прогнозів
# Зменшення стандартного відхилення залишків на 32.98% після очищення даних має прямий вплив на надійність прогнозів:
# **Звуження довірчих інтервалів**: Для очищеної моделі 95% довірчий інтервал для прогнозованого значення буде приблизно на 33% вужчим, ніж для оригінальної моделі.
# **Підвищення точності планування**: Більш точні прогнози дозволяють ефективніше планувати комунікаційні ресурси та робоче навантаження менеджерів.
# **Зменшення ризику помилкових рішень**: Менша невизначеність у прогнозах знижує ризик прийняття неоптимальних бізнес-рішень.
# 
# #### Інтерпретація форми розподілу залишків
# Лептокуртичність розподілу залишків, яка посилилася після очищення даних, має важливу бізнес-інтерпретацію:
# **Висока передбачуваність для більшості партнерів**: Концентрація залишків навколо нуля свідчить про те, що для більшості "типових" партнерів модель дає дуже точні прогнози, що дозволяє стандартизувати підходи до комунікації.
# **Наявність "особливих" партнерів**: Хоча їх стало менше після очищення, все ще присутні партнери, для яких стандартний підхід до комунікації може бути неоптимальним, що вимагає індивідуалізованого підходу.
# **Сегментація партнерської бази**: Форма розподілу підтверджує доцільність сегментації партнерів на групи з різними комунікаційними патернами та розробки таргетованих стратегій для кожного сегмента.
# 
# #### Практичні рекомендації
# На основі порівняльного аналізу розподілу залишків можна сформулювати такі практичні рекомендації:
# **Використання очищеної моделі для основного сегмента**: Для більшості партнерів (центральна частина розподілу) рекомендується використовувати модель на основі очищених даних, яка забезпечує вищу точність прогнозування.
# **Спеціальний підхід до атипових партнерів**: Для партнерів, щодо яких прогнози мають значні залишки, доцільно розробити окремі підходи до комунікації, можливо, з використанням альтернативних моделей.
# **Моніторинг та адаптація**: Регулярний моніторинг розподілу залишків дозволить виявляти зміни у взаємозв'язку між замовленнями та комунікацією та своєчасно адаптувати модель.
# **Подальша сегментація**: Дослідження причин лептокуртичності розподілу залишків може привести до виявлення важливих сегментаційних факторів, що дозволять покращити як модель, так і бізнес-процеси.
# 
# #### Загальна оцінка ефективності очищення даних
# Порівняльний аналіз розподілу залишків дозволяє зробити узагальнену оцінку ефективності процедури очищення даних:
# **Позитивні аспекти**:
# Суттєве зменшення стандартного відхилення залишків (на 32.98%)
# Збереження більшості оригінальних спостережень (97.09%)
# Покращення симетричності розподілу
# Зменшення "важкості" хвостів
# **Нейтральні аспекти**:
# Збереження центральної тенденції (μ ≈ 0)
# Збереження загальної форми розподілу
# **Потенційно проблемні аспекти**:
# Посилення лептокуртичності розподілу
# Збереження відхилення від нормальності, хоча й іншого характеру
# 
# Загалом, процедура очищення даних була ефективною, оскільки вона досягла основної мети — підвищення точності моделі при мінімальній втраті інформації. Однак, залишкові відхилення від нормальності свідчать про те, що простої лінійної моделі може бути недостатньо для повного врахування складності взаємозв'язку між кількістю замовлень та кількістю повідомлень у різних сегментах партнерської бази B2B компанії.
# Подальше вдосконалення моделі може рухатися у напрямку сегментованого підходу, нелінійних трансформацій або робастних методів оцінювання, що дозволить ще більше підвищити точність прогнозів та, як наслідок, ефективність бізнес-процесів компанії.

# # Висновки
# ## Аналіз первинних даних
# 
# Початковий набір даних містив 86 794 спостереження, що є достатньо великою вибіркою для отримання статистично надійних результатів. Статистичні характеристики змінних до очищення даних представлені у таблиці 1.
# 
# **Таблиця 1. Статистичні характеристики змінних до очищення даних**
# 
# | Змінна | Кількість спостережень | Середнє значення | Дисперсія | Стандартне відхилення |
# |--------|------------------------|------------------|-----------|------------------------|
# | partner_total_orders | 86 794 | 83.6963 | 22 804.7030 | 151.0123 |
# | partner_total_messages | 86 794 | 758.3827 | 1 882 421.7473 | 1 372.0138 |
# 
# Первинний аналіз даних показав значну варіативність як у кількості замовлень, так і в кількості повідомлень, що може свідчити про неоднорідність партнерської бази та наявність різних патернів комунікаційної поведінки серед партнерів.
# 
# ## Результати регресійного аналізу оригінального набору даних
# 
# На основі оригінального набору даних було побудовано лінійну регресійну модель, результати якої представлені у таблиці 2.
# 
# **Таблиця 2. Результати регресійного аналізу до очищення даних**
# 
# | Параметр | Значення | Стандартна похибка | t-статистика | Статистична значущість |
# |----------|----------|---------------------|--------------|-------------------------|
# | a₀ (вільний член) | 19.2445 | 1.2508 | 15.3855 | Так |
# | a₁ (коефіцієнт нахилу) | 8.8312 | 0.0072 | 1 218.9955 | Так |
# 
# Рівняння регресії має вигляд:
# ```
# partner_total_messages = 19.2445 + 8.8312 * partner_total_orders
# ```
# 
# Довірчі інтервали для коефіцієнтів (95%):
# - a₀: [16.7929, 21.6961]
# - a₁: [8.8170, 8.8454]
# 
# Метрики якості моделі:
# - Коефіцієнт кореляції (r): 0.9720
# - Коефіцієнт детермінації (R²): 0.9448
# - Скоригований R²: 0.9448
# - Стандартна похибка регресії: 322.3087
# 
# F-статистика для загальної значущості моделі становить 1 485 950.1166 при критичному значенні 3.8416, що підтверджує статистичну значущість моделі в цілому.
# 
# Усереднений коефіцієнт еластичності становить 0.9746, що свідчить про середню еластичність кількості повідомлень по відношенню до кількості замовлень.
# 
# ## Аналіз та виявлення викидів
# 
# Процедура виявлення викидів виявила 2 521 спостереження, які було класифіковано як викиди та видалено з набору даних. Це становить лише 2.91% від загальної кількості спостережень, що свідчить про відносно консервативний підхід до очищення даних.
# 
# Фрагмент таблиці видалених викидів представлено нижче:
# 
# **Таблиця 3. Фрагмент даних видалених викидів**
# 
# | Ітерація | Змінна | Індекс | Значення |
# |----------|--------|--------|----------|
# | 1 | partner_total_orders | 86755 | 1307.0 |
# | 1 | partner_total_messages | 86722 | 11481.0 |
# | 2 | partner_total_orders | 86555 | 1306.0 |
# | 2 | partner_total_messages | 86635 | 11479.0 |
# | 3 | partner_total_orders | 86127 | 1305.0 |
# | ... | ... | ... | ... |
# | 1329 | partner_total_orders | 65990 | 539.0 |
# | 1329 | partner_total_messages | 48506 | 4828.0 |
# | 1330 | partner_total_orders | 66738 | 539.0 |
# | 1330 | partner_total_messages | 53766 | 4828.0 |
# | 1331 | partner_total_messages | 79362 | 4828.0 |
# 
# Аналіз викидів показує, що видалено було переважно спостереження з надзвичайно високими значеннями як кількості замовлень (понад 539), так і кількості повідомлень (понад 4828). Це свідчить про те, що процедура очищення даних була спрямована на видалення атипових спостережень, які могли непропорційно впливати на оцінки параметрів регресійної моделі.
# 
# ## Результати регресійного аналізу після очищення даних
# 
# Після видалення викидів було перераховано регресійну модель на основі очищеного набору даних, що містив 84 273 спостереження. Результати оновленої моделі представлені у таблиці 4.
# 
# **Таблиця 4. Результати регресійного аналізу після очищення даних**
# 
# | Параметр | Значення | Стандартна похибка | t-статистика | Статистична значущість |
# |----------|----------|---------------------|--------------|-------------------------|
# | a₀ (вільний член) | 21.1406 | 0.8990 | 23.5154 | Так |
# | a₁ (коефіцієнт нахилу) | 8.6617 | 0.0078 | 1 105.2737 | Так |
# 
# Рівняння регресії після очищення даних має вигляд:
# ```
# partner_total_messages = 21.1406 + 8.6617 * partner_total_orders
# ```
# 
# Довірчі інтервали для коефіцієнтів (95%):
# - a₀: [19.3786, 22.9027]
# - a₁: [8.6463, 8.6770]
# 
# Метрики якості моделі:
# - Коефіцієнт кореляції (r): 0.9672
# - Коефіцієнт детермінації (R²): 0.9355
# - Скоригований R²: 0.9355
# - Стандартна похибка регресії: 216.0059
# 
# F-статистика для загальної значущості моделі становить 1 221 629.9983 при критичному значенні 3.8416, що підтверджує статистичну значущість моделі в цілому.
# 
# Усереднений коефіцієнт еластичності становить 0.9635, що свідчить про середню еластичність кількості повідомлень по відношенню до кількості замовлень.
# 
# ## Порівняльний аналіз результатів до та після очищення даних
# 
# Порівняння статистичних характеристик змінних до та після очищення даних представлено у таблиці 5.
# 
# **Таблиця 5. Порівняння статистичних характеристик змінних**
# 
# | Змінна | Етап | Кількість спостережень | Середнє значення | Дисперсія | Стандартне відхилення |
# |--------|------|------------------------|------------------|-----------|------------------------|
# | partner_total_orders | До | 86 794 | 83.6963 | 22 804.7030 | 151.0123 |
# | partner_total_orders | Після | 84 273 | 64.3826 | 9 015.3509 | 94.9492 |
# | partner_total_messages | До | 86 794 | 758.3827 | 1 882 421.7473 | 1 372.0138 |
# | partner_total_messages | Після | 84 273 | 578.8023 | 723 033.0680 | 850.3135 |
# 
# Порівняння основних параметрів регресійних моделей представлено у таблиці 6.
# 
# **Таблиця 6. Порівняння параметрів регресійних моделей**
# 
# | Параметр | До очищення | Після очищення | Абсолютна зміна | Відносна зміна (%) |
# |----------|-------------|----------------|-----------------|----------------------|
# | a₀ (вільний член) | 19.2445 | 21.1406 | +1.8961 | +9.85 |
# | a₁ (коефіцієнт нахилу) | 8.8312 | 8.6617 | -0.1695 | -1.92 |
# | Коефіцієнт кореляції (r) | 0.9720 | 0.9672 | -0.0048 | -0.49 |
# | Коефіцієнт детермінації (R²) | 0.9448 | 0.9355 | -0.0093 | -0.98 |
# | Стандартна похибка регресії | 322.3087 | 216.0059 | -106.3028 | -32.98 |
# 
# Порівняльний аналіз показує, що очищення даних призвело до таких змін:
# 
# Середнє значення кількості замовлень зменшилося на 23.08% (з 83.6963 до 64.3826), а стандартне відхилення зменшилося на 37.12% (з 151.0123 до 94.9492). Середнє значення кількості повідомлень зменшилося на 23.68% (з 758.3827 до 578.8023), а стандартне відхилення зменшилося на 38.02% (з 1372.0138 до 850.3135). Ці зміни свідчать про видалення спостережень з високими значеннями обох змінних.
# 
# Вільний член регресійного рівняння збільшився на 9.85% (з 19.2445 до 21.1406), а коефіцієнт нахилу зменшився на 1.92% (з 8.8312 до 8.6617). Доцільно зауважити, що обидва 95% довірчі інтервали для коефіцієнтів після очищення даних стали вужчими, що свідчить про підвищення точності оцінок.
# 
# Коефіцієнт детермінації (R²) зменшився на 0.98% (з 0.9448 до 0.9355), однак стандартна похибка регресії, яка безпосередньо характеризує точність прогнозування, зменшилася на значні 32.98% (з 322.3087 до 216.0059). Це є суттєвим покращенням точності моделі, яке перекриває незначне зменшення R².
# 
# ## Теоретичне пояснення результатів
# 
# Спостережене зменшення коефіцієнта детермінації при одночасному зменшенні стандартної похибки регресії після очищення даних вимагає теоретичного пояснення, оскільки на перший погляд це може здатися парадоксальним.
# 
# Коефіцієнт детермінації (R²) вимірює частку загальної варіації залежної змінної, що пояснюється регресійною моделлю. Він розраховується за формулою R² = 1 - (сума квадратів залишків / загальна сума квадратів). Очищення даних призвело до видалення спостережень з екстремально високими значеннями обох змінних, що зменшило загальну суму квадратів більшою мірою, ніж суму квадратів залишків. Це може пояснювати незначне зменшення R².
# 
# Стандартна похибка регресії, з іншого боку, є оцінкою стандартного відхилення залишків і безпосередньо характеризує точність прогнозування моделі. Її значне зменшення (на 32.98%) свідчить про суттєве підвищення точності моделі після очищення даних.
# 
# Зміна коефіцієнтів регресії також має теоретичне пояснення. Збільшення вільного члена (a₀) на 9.85% може вказувати на те, що після видалення викидів базовий рівень комунікації (кількість повідомлень при нульовій кількості замовлень) оцінюється як вищий. Це може відображати більш точну оцінку "фонової" комунікації, яка не пов'язана безпосередньо з процесом обробки замовлень.
# 
# Незначне зменшення коефіцієнта нахилу (a₁) на 1.92% свідчить про стабільність основного взаємозв'язку між кількістю замовлень та кількістю повідомлень. Однак навіть ця незначна зміна може вказувати на те, що після очищення даних оцінка кількості повідомлень на одне замовлення стала дещо нижчою, що може відображати вищу ефективність комунікації у "типових" партнерів порівняно з атиповими випадками, які були видалені.
# 
# ## Доцільність очищення даних
# Питання доцільності очищення даних у регресійному аналізі є багатоаспектним і залежить від конкретних цілей дослідження та характеру даних. У даному випадку аналіз результатів дозволяє зробити такі висновки щодо доцільності очищення даних:
# 
# Перевагами очищення даних у проведеному дослідженні є:
# Значне зменшення стандартної похибки регресії на 32.98%, що суттєво підвищує точність прогнозування моделі для більшості спостережень.
# Звуження довірчих інтервалів для коефіцієнтів регресії, що підвищує надійність оцінок параметрів моделі.
# Зменшення впливу атипових спостережень, які можуть непропорційно впливати на оцінки параметрів моделі через квадратичну природу методу найменших квадратів.
# 
# Водночас, очищення даних має певні недоліки:
# Незначне зменшення коефіцієнта детермінації (R²) на 0.98%, що може свідчити про деяку втрату пояснювальної сили моделі.
# Видалення 2.91% спостережень, які можуть нести важливу інформацію про атипових партнерів, що становлять значний сегмент клієнтської бази з точки зору обсягу замовлень та комунікації.
# Потенційне обмеження діапазону застосування моделі, оскільки очищена модель може бути менш надійною для прогнозування поведінки партнерів з характеристиками, близькими до видалених спостережень.
# 
# Враховуючи вищенаведені аргументи, очищення даних у даному дослідженні можна вважати доцільним у контексті підвищення точності та надійності моделі для більшості "типових" партнерів. Однак для повної картини рекомендується також зберігати оригінальну модель для аналізу та прогнозування поведінки атипових партнерів або розробити окремі моделі для різних сегментів партнерської бази.
# 
# ## Доцільність застосування лінійної регресії
# Питання доцільності застосування лінійної регресії для аналізу наданих даних також потребує детального розгляду:
# Аргументами на користь застосування лінійної регресії є:
# Високі значення коефіцієнта детермінації як до (R² = 0.9448), так і після (R² = 0.9355) очищення даних, що свідчить про сильний лінійний взаємозв'язок між кількістю замовлень та кількістю повідомлень.
# Висока статистична значущість як окремих коефіцієнтів регресії (t-статистика >> критичного значення), так і моделі в цілому (F-статистика >> критичного значення).
# Простота інтерпретації лінійної моделі, де коефіцієнт нахилу безпосередньо показує, скільки додаткових повідомлень асоціюється з кожним додатковим замовленням.
# Водночас, є фактори, які можуть ставити під сумнів доцільність застосування лінійної регресії:
# Графічний аналіз залишків моделі показує наявність гетероскедастичності (нерівномірності дисперсії залишків залежно від значення предиктора), що є порушенням однієї з ключових передумов класичної лінійної регресії.
# Аналіз розподілу залишків вказує на відхилення від нормальності, що може впливати на надійність статистичних висновків, особливо для малих вибірок (хоча у даному випадку вибірка дуже велика, що пом'якшує цю проблему).
# Спостереження нелінійних патернів на графіках залишків вказує на можливість того, що лінійна функціональна форма моделі може не повністю відображати складність взаємозв'язку між змінними.
# 
# Враховуючи ці аргументи, можна зробити висновок, що лінійна регресія є доцільним інструментом для першого етапу аналізу наданих даних, оскільки вона забезпечує високу пояснювальну силу та статистичну значущість. Однак для повнішого врахування виявлених особливостей даних (гетероскедастичність, нелінійні патерни) рекомендується розглянути можливість застосування більш складних моделей, таких як зважений метод найменших квадратів, поліноміальна регресія, сплайн-регресія або сегментована регресія.
# 
# ## Висновки
# Проведене дослідження взаємозв'язку між кількістю замовлень партнерів B2B компанії та кількістю комунікаційних повідомлень дозволяє зробити такі висновки:
# Встановлено наявність сильного позитивного лінійного взаємозв'язку між кількістю замовлень партнера та кількістю повідомлень, якими він обмінюється з менеджерами компанії. Коефіцієнт детермінації (R² > 0.93) свідчить про те, що варіація кількості замовлень пояснює понад 93% варіації кількості повідомлень.
# Визначено, що кожне додаткове замовлення партнера асоціюється в середньому з 8.66-8.83 додатковими повідомленнями, що є важливим показником для планування комунікаційних ресурсів компанії.
# Виявлено, що партнери мають певний базовий рівень комунікації (19.24-21.14 повідомлень), який не пов'язаний безпосередньо з замовленнями і може відображати загальну інформаційну взаємодію.
# Процедура очищення даних призвела до видалення 2.91% спостережень, які було класифіковано як викиди, що дозволило значно підвищити точність моделі для більшості "типових" партнерів, про що свідчить зменшення стандартної похибки регресії на 32.98%.
# Виявлено певні обмеження лінійної регресійної моделі, зокрема, наявність гетероскедастичності та відхилень від нормальності розподілу залишків, що вказує на потенційну користь від застосування більш складних методів моделювання.
# 
# Загалом, результати дослідження підтверджують наявність стабільного та передбачуваного взаємозв'язку між кількістю замовлень та комунікаційною активністю партнерів, що створює надійну основу для планування та оптимізації комунікаційних процесів B2B компанії.
# 
# ## Рекомендації
# На основі проведеного дослідження можна сформулювати такі практичні рекомендації:
# Використовувати виявлений коефіцієнт нахилу (8.66-8.83 повідомлень на одне замовлення) як базовий показник для планування комунікаційних ресурсів та робочого навантаження менеджерів.
# Застосовувати диференційований підхід до різних сегментів партнерів, враховуючи, що модель має різну точність для партнерів з різною кількістю замовлень:
# Для партнерів з малою кількістю замовлень (до ~50) модель демонструє високу точність, що дозволяє точно планувати ресурси.
# Для партнерів з середньою кількістю замовлень (від ~50 до ~200) модель має задовільну точність, що вимагає певного запасу ресурсів.
# Для партнерів з великою кількістю замовлень (понад 200) модель має нижчу точність, що вимагає індивідуалізованого підходу та більшого резерву ресурсів.
# 
# Розробити систему моніторингу співвідношення `partner_total_messages / partner_total_orders` для ідентифікації партнерів, які суттєво відхиляються від прогнозованих значень, що може вказувати на проблеми в комунікаційному процесі або особливі потреби партнера.
# Розглянути можливість вдосконалення моделі через:
# Застосування зваженого методу найменших квадратів для боротьби з гетероскедастичністю.
# Включення додаткових предикторів, які можуть впливати на комунікаційну активність.
# Розробку нелінійних моделей (поліноміальна регресія, сплайн-регресія) для кращого врахування виявлених нелінійних патернів.
# Сегментацію партнерської бази та розробку окремих моделей для різних сегментів.
# Проводити періодичне оновлення моделі з урахуванням нових даних та зміни патернів комунікації, що дозволить підтримувати високу точність прогнозування та ефективність планування ресурсів.
# 
# Загалом, результати дослідження надають цінну інформацію для оптимізації комунікаційних процесів B2B компанії та підвищення ефективності взаємодії з партнерами, що може призвести до покращення якості обслуговування та підвищення задоволеності партнерів.
# 

# In[ ]:




