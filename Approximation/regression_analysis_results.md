# Аналіз залежності температури від вмісту вуглецю

## Експериментальні дані

### Вихідні дані
| Висота (мм) | Температура (°C) | Вміст вуглецю (%) |
|-------------|------------------|-------------------|
| 400         | 1000            | 3.58              |
| 300         | 980             | 3.77              |
| 200         | 1350            | 3.84              |
| 100         | 1385            | 3.92              |

### Статистичний аналіз
Коефіцієнт кореляції між температурою та вмістом вуглецю: **0.8074**

## Регресійний аналіз

### Порівняння моделей

1. **Поліноміальна регресія 3-го ступеня** (R² = 1.0000)
   ```
   T = -156046.93C³ + 1766899.71C² - 6662875.72C + 8368658.14
   ```

2. **Поліноміальна регресія 2-го ступеня** (R² = 0.7840)
   ```
   T = 5532.18C² - 40181.21C + 73934.35
   ```

3. **Експоненціальна регресія** (R² = 0.6824)
   ```
   T = 16.58 * exp(1.13C)
   ```

4. **Степенева регресія** (R² = 0.6753)
   ```
   T = 4.39 * C^4.20
   ```

5. **Лінійна регресія** (R² = 0.6519)
   ```
   T = 1215.13C - 3411.41
   ```

6. **Логарифмічна регресія** (R² = 0.6441)
   ```
   T = 4515.63 * ln(C) - 4820.27
   ```

## Висновки

1. **Загальна кореляція**
   - Коефіцієнт кореляції 0.8074 вказує на сильний позитивний зв'язок між температурою та вмістом вуглецю
   - Це підтверджує значущість впливу вмісту вуглецю на температурний режим

2. **Вибір оптимальної моделі**
   - Поліноміальні регресії 3-го та 4-го ступеня показують ідеальне наближення (R² = 1.0000)
   - Однак, враховуючи малу кількість експериментальних точок (4), такі високі значення R² можуть свідчити про перенавчання моделі

3. **Практичні рекомендації**
   - Для практичного використання рекомендується поліноміальна регресія 2-го ступеня:
     ```
     T = 5532.18C² - 40181.21C + 73934.35
     ```
   - Ця модель забезпечує:
     * Достатньо високу точність (R² = 0.7840)
     * Кращу стійкість до похибок вимірювань
     * Більш надійні результати при інтерполяції

4. **Обмеження аналізу**
   - Невелика кількість експериментальних точок (4)
   - Можлива наявність прихованих факторів впливу
   - Потенційна нестабільність моделей високих порядків

## Рекомендації для подальших досліджень

1. Збільшити кількість експериментальних точок для підвищення надійності регресійного аналізу
2. Провести додаткові експерименти в проміжних точках для верифікації обраної моделі
3. Дослідити вплив інших факторів на температурний режим
4. Розглянути можливість створення багатофакторної моделі