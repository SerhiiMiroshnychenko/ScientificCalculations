# Моделювання процесу охолодження залізнорудної агломераційної шихти

## Теоретична частина

### 1. Фізична постановка задачі

Досліджується процес охолодження стандартного шару залізнорудної агломераційної шихти після завершення процесу агломерації. Розглядається поперечний зріз шару з наступними характеристиками:

**Геометричні параметри:**
- Висота шару: 400 мм
- Ширина шару: 2500 мм

**Фізичні параметри:**
- Початкова температура: 400°C (температура після завершення агломерації)
- Кінцева температура: 20°C (температура навколишнього середовища)
- Коефіцієнт температуропровідності: 1×10⁻⁶ м²/с

Коефіцієнт температуропровідності (або коефіцієнт теплопровідності) залізнорудного агломерату залежить від його складу, структури, пористості та інших фізико-хімічних властивостей. Цей коефіцієнт визначає здатність матеріалу проводити тепло і розраховується за формулою:

$$ a = \frac{\lambda}{\rho \cdot c_p} $$

де:
- $a$ — коефіцієнт температуропровідності, [м²/с]
- $\lambda$ — коефіцієнт теплопровідності, [Вт/(м·К)]
- $\rho$ — густина матеріалу, [кг/м³]
- $c_p$ — питома теплоємність, [Дж/(кг·К)]

Для залізнорудного агломерату значення коефіцієнта температуропровідності може коливатися в межах 0,5–1,5·10⁻⁶ м²/с, оскільки агломерат має складну структуру і містить різні компоненти (залізо, оксиди, домішки тощо). Точне значення залежить від конкретного складу та умов вимірювання.

В даній моделі використовується значення $a = 1 \cdot 10^{-6}$ м²/с, що знаходиться в середині діапазону типових значень для залізнорудного агломерату.

### 2. Математична модель

#### 2.1 Рівняння теплопровідності

Процес охолодження описується двовимірним рівнянням теплопровідності Фур'є:

$$ \frac{\partial T}{\partial t} = a \left(\frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2}\right) $$

де:
- $T(x,y,t)$ - температура в точці $(x,y)$ в момент часу $t$
- $a$ - коефіцієнт температуропровідності [м²/с]
- $t$ - час [с]
- $x, y$ - просторові координати [м]

#### 2.2 Граничні умови

У задачі використовуються граничні умови першого роду (умови Діріхле):
- На всіх границях підтримується постійна температура навколишнього середовища $T = 20°C$

#### 2.3 Початкові умови

В початковий момент часу температура у всіх точках шару дорівнює температурі після агломерації:
- $T(x,y,0) = 400°C$ для всіх $(x,y)$ всередині області

### 3. Припущення моделі

1. Розглядається двовимірний випадок, оскільки зміни температури в поперечному вимірі (перпендикулярно до зрізу) вважаються незначними
2. Теплообмін відбувається лише з навколишнім середовищем через границі шару
3. Властивості матеріалу (коефіцієнт температуропровідності) вважаються постійними
4. Внутрішні джерела тепла відсутні

## Практична реалізація

### 1. Метод розв'язання

Для розв'язання задачі використовується комбінований підхід:
1. Метод кінцевих різниць для просторової дискретизації
2. Метод Рунге-Кутти 4-5 порядку для інтегрування за часом

#### 1.1 Просторова дискретизація

Використовується рівномірна прямокутна сітка:
- Крок по просторовим координатам: $\Delta x = \Delta y = 0.01$ м
- Кількість точок по ширині: 250
- Кількість точок по висоті: 40

Апроксимація просторових похідних:
$$ \frac{\partial^2 T}{\partial x^2} \approx \frac{T_{i+1,j} - 2T_{i,j} + T_{i-1,j}}{(\Delta x)^2} $$
$$ \frac{\partial^2 T}{\partial y^2} \approx \frac{T_{i,j+1} - 2T_{i,j} + T_{i,j-1}}{(\Delta y)^2} $$

#### 1.2 Часова дискретизація

Для інтегрування за часом використовується адаптивний метод Рунге-Кутти 4-5 порядку (`RK45`), реалізований у функції `solve_ivp` з бібліотеки `scipy.integrate`. Переваги методу:
1. Автоматичний вибір кроку за часом
2. Контроль локальної похибки
3. Висока точність розв'язку
4. Стійкість для жорстких систем

### 2. Особливості реалізації

#### 2.1 Оптимізація обчислень

1. Векторизація обчислень за допомогою NumPy
2. Ефективна робота з масивами без явних циклів
3. Перевикористання масивів для економії пам'яті

#### 2.2 Критерії зупинки

Розрахунок припиняється при виконанні однієї з умов:
1. Досягнення цільової температури (20±1°C) у всіх точках
2. Перевищення максимального часу розрахунку (500000 с ≈ 5.8 діб)

#### 2.3 Візуалізація результатів

Результати представляються у вигляді:
1. Контурних графіків розподілу температури для ключових моментів часу:
   - Початковий стан
   - 1/3 загального часу
   - 2/3 загального часу
   - Кінцевий стан
2. Текстових повідомлень про:
   - Поточну мінімальну, середню та максимальну температуру
   - Загальний час охолодження

### 3. Висновки та рекомендації

1. Час охолодження значно залежить від розмірів шару та коефіцієнта температуропровідності
2. Охолодження відбувається нерівномірно:
   - Краї охолоджуються швидше через прямий контакт з навколишнім середовищем
   - Центральна частина охолоджується повільніше через необхідність відведення тепла через сусідні шари
3. Для прискорення охолодження можна рекомендувати:
   - Зменшення товщини шару
   - Забезпечення кращого теплообміну з навколишнім середовищем
   - Використання примусового охолодження
