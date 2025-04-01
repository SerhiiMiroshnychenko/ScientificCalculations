## 3. Побудова лінійної парної регресійної моделі для залежності між змінними

### 3.1 Класичний підхід (Підхід 1)

Класичний підхід передбачає використання математичних формул та покрокове обчислення всіх необхідних параметрів, подібно до того, як це робиться в MathCad.

#### Математичні формули класичного підходу

Для розрахунку коефіцієнтів регресії за класичним підходом спочатку обчислюються середні значення:

$$y_{ср} = \frac{1}{n} \cdot \sum_{i=0}^{n-1} Y_i$$

$$x_{ср} = \frac{1}{n} \cdot \sum_{i=0}^{n-1} X_i$$

$$x2_{ср} = \frac{1}{n} \cdot \sum_{i=0}^{n-1} X_i^2$$

$$xy_{ср} = \frac{1}{n} \cdot \sum_{i=0}^{n-1} X_i \cdot Y_i$$

Коефіцієнти регресії обчислюються за формулами:

$$a_1 = \frac{xy_{ср} - x_{ср} \cdot y_{ср}}{x2_{ср} - x_{ср}^2}$$

$$a_0 = y_{ср} - a_1 \cdot x_{ср}$$

#### Генерація даних

```python
# Параметри моделі
a0 = 3.4  # Вільний член
a1 = 2.7  # Коефіцієнт при x
noise_range = 1.10  # Діапазон випадкового шуму

# Генерація даних
n = 51  # Кількість точок
x, y, y_true = generate_data(a0, a1, noise_range, n)

# Виведення фрагменту даних
print("Фрагмент згенерованих даних:")
df_sample = pd.DataFrame({'x': x[:10], 'y': y[:10], 'y_true': y_true[:10]})
print(tabulate(df_sample, headers='keys', tablefmt='psql', showindex=True))
```

#### Побудова лінійної регресії за класичним підходом

```python
# Розрахунок середніх значень
n = len(x)
x_mean = np.mean(x)
y_mean = np.mean(y)
x2_mean = np.mean(x**2)
xy_mean = np.mean(x * y)

# Розрахунок коефіцієнтів регресії за класичним підходом
b1_classic = (xy_mean - x_mean * y_mean) / (x2_mean - x_mean**2)
b0_classic = y_mean - b1_classic * x_mean

# Виведення середніх значень та коефіцієнтів
print("Середні значення (як у MathCad):")
print(f"Середнє значення y (y_ср): {y_mean:.4f}")
print(f"Середнє значення x (x_ср): {x_mean:.4f}")
print(f"Середнє значення x² (x2_ср): {x2_mean:.4f}")
print(f"Середнє значення xy (xy_ср): {xy_mean:.4f}")

print("\nРозрахунок коефіцієнтів (як у MathCad):")
print(f"a₁ = (xy_ср - x_ср·y_ср) / (x2_ср - x_ср²) = {b1_classic:.4f}")
print(f"a₀ = y_ср - a₁·x_ср = {b0_classic:.4f}")

# Розрахунок прогнозних значень
y_pred_classic = b0_classic + b1_classic * x

print(f"\nСправжнє рівняння: y = {a0} + {a1}*x + випадковий шум в діапазоні [-{noise_range}, {noise_range}]")
print(f"Оцінене рівняння регресії: y = {b0_classic:.4f} + {b1_classic:.4f}*x")
```

#### Розрахунок коефіцієнта кореляції та тестування статистичної значущості

```python
# Коефіцієнт кореляції за формулою з MathCad
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
r_classic = numerator / denominator

# Тестування статистичної значущості коефіцієнта кореляції
t_calc = r_classic * np.sqrt(n - 2) / np.sqrt(1 - r_classic**2)
t_crit = stats.t.ppf(0.975, n - 2)  # Критичне значення t-статистики для двостороннього тесту

print("\nСтатистичні показники:")
print(f"Коефіцієнт кореляції (r): {r_classic:.4f}")
print(f"t-статистика: t_розрах = {t_calc:.4f}, t_крит = {t_crit:.4f}")
print(f"  Висновок: t_розрах {'>' if t_calc > t_crit else '<'} t_крит, зв'язок {'статистично значущий' if t_calc > t_crit else 'статистично незначущий'}")
```

#### Розрахунок коефіцієнта детермінації та стандартної похибки регресії

```python
# Коефіцієнт детермінації за формулою з MathCad
SST = np.sum((y - y_mean)**2)  # Загальна сума квадратів
SSR = np.sum((y_pred_classic - y_mean)**2)  # Сума квадратів регресії
R2_classic = SSR / SST  # Формула: R2 = ∑(y2(Xᵢ)-y_ср)² / ∑(Yᵢ-y_ср)²

# Стандартна похибка регресії за формулою з MathCad
SSE = np.sum((y - y_pred_classic)**2)  # Сума квадратів залишків
S_classic = np.sqrt(SSE / (n - 2))  # Формула: S = sqrt(1/(n-2) · ∑(Yᵢ-y2(Xᵢ))²)

print(f"Коефіцієнт детермінації (R²): {R2_classic:.4f}")
print(f"  Якість моделі: {'модель вважається точною' if R2_classic > 0.8 else 'модель незадовільна' if R2_classic < 0.5 else 'модель задовільна'}")
print(f"Стандартна похибка регресії (S): {S_classic:.4f}")
```

#### F-тест для оцінки статистичної значущості моделі

```python
# F-тест за формулою з MathCad: F_сп = R2·(n-2)/(1-R2)
F_calc = R2_classic * (n - 2) / (1 - R2_classic)
F_crit = stats.f.ppf(0.95, 1, n - 2)

print(f"F-статистика: F_розрах = {F_calc:.4f}, F_крит = {F_crit:.4f}")
print(f"  Висновок: F_розрах {'>' if F_calc > F_crit else '<'} F_крит, {'існує' if F_calc > F_crit else 'не існує'} лінійна регресія між показниками x і y")
```

#### Розрахунок довірчих інтервалів для коефіцієнтів регресії

```python
# Розрахунок довірчих інтервалів для коефіцієнтів регресії за формулами з MathCad

# Сума квадратів Х
sum_x2 = np.sum(x**2)

# Сума квадратів відхилень Y від середнього
sum_y_minus_mean_squared = np.sum((y - y_mean)**2)

# Сума квадратів відхилень X від середнього
sum_x_minus_mean_squared = np.sum((x - x_mean)**2)

# Стандартні похибки коефіцієнтів за формулами з MathCad
S_a0 = np.sqrt((sum_x2 * sum_y_minus_mean_squared) / (n * (n - 2) * sum_x_minus_mean_squared))
S_a1 = np.sqrt(sum_y_minus_mean_squared / ((n - 2) * sum_x_minus_mean_squared))

# Критичне значення t-статистики
t_crit = stats.t.ppf(0.975, n - 2)

# Довірчі інтервали
a0_lower = b0_classic - t_crit * S_a0
a0_upper = b0_classic + t_crit * S_a0
a1_lower = b1_classic - t_crit * S_a1
a1_upper = b1_classic + t_crit * S_a1

print("\nДовірчі інтервали для коефіцієнтів (95%):")
print(f"Стандартна похибка a₀ (S_a0): {S_a0:.4f}")
print(f"Стандартна похибка a₁ (S_a1): {S_a1:.4f}")
print(f"Вільний член (a₀): [{a0_lower:.4f}, {a0_upper:.4f}]")
print(f"Коефіцієнт при x (a₁): [{a1_lower:.4f}, {a1_upper:.4f}]")
```

#### Розрахунок коефіцієнта еластичності

```python
# Розрахунок коефіцієнта еластичності за формулою з MathCad: ε = a₁·(x_ср/y_ср)
elasticity = b1_classic * (x_mean / y_mean)

print(f"\nУсереднений коефіцієнт еластичності: {elasticity:.4f}")
print(f"При зміні x на 1%, y в середньому змінюється на {elasticity*100:.2f}%")

# Перевірка на входження справжніх значень
print("\nПеревірка чи справжні значення входять в довірчі інтервали:")
print(f"   - Справжнє значення a0 = {a0} {'входить' if a0_lower <= a0 <= a0_upper else 'НЕ входить'} в довірчий інтервал [{a0_lower:.4f}, {a0_upper:.4f}]")
print(f"   - Справжнє значення a1 = {a1} {'входить' if a1_lower <= a1 <= a1_upper else 'НЕ входить'} в довірчий інтервал [{a1_lower:.4f}, {a1_upper:.4f}]")
```

#### Побудова таблиці результатів

```python
# Побудова таблиці результатів
results = pd.DataFrame({
    'x': x,
    'y фактичні': y,
    'y прогнозні': y_pred_classic,
    'Залишки': y - y_pred_classic
})

print("\nФрагмент таблиці з результатами:")
print(tabulate(results.head(10), headers='keys', tablefmt='psql', showindex=True))
```

#### Візуалізація моделі і залишків

```python
# Створення папки для збереження графіків, якщо її не існує
output_dir = 'mathcad_method_plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Графік 1: Лінія регресії
plt.figure(figsize=(10, 8))
plt.scatter(x, y, color='blue', label='Фактичні дані')
plt.plot(x, y_pred_classic, color='red', label=f'Регресія: y = {b0_classic:.4f} + {b1_classic:.4f}*x')
plt.plot(x, y_true, '--', color='green', label=f'Справжня функція: y = {a0} + {a1}*x')

# Довірчі інтервали для прогнозних значень
x_sorted = np.sort(x)
y_fitted = b0_classic + b1_classic * x_sorted

# Довірчі інтервали для середніх значень
se_mean = S_classic * np.sqrt(1/n + (x_sorted - x_mean)**2 / sum_x_minus_mean_squared)
ci_mean_lower = y_fitted - t_crit * se_mean
ci_mean_upper = y_fitted + t_crit * se_mean

# Інтервали передбачення для індивідуальних значень
se_pred = S_classic * np.sqrt(1 + 1/n + (x_sorted - x_mean)**2 / sum_x_minus_mean_squared)
pi_lower = y_fitted - t_crit * se_pred
pi_upper = y_fitted + t_crit * se_pred

plt.fill_between(x_sorted, ci_mean_lower, ci_mean_upper, color='red', alpha=0.1, label='95% довірчий інтервал')
plt.fill_between(x_sorted, pi_lower, pi_upper, color='blue', alpha=0.1, label='95% інтервал передбачення')
plt.title('Лінійна регресійна модель (класичний підхід)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig(f'{output_dir}/1_regression_model_mathcad.png', dpi=300, bbox_inches='tight')
plt.show()

# Графік 2: Залишки
plt.figure(figsize=(10, 8))
plt.scatter(x, y - y_pred_classic, color='orange')
plt.axhline(y=0, color='r', linestyle='-')
plt.title('Залишки vs x (класичний підхід)')
plt.xlabel('x')
plt.ylabel('Залишки')
plt.grid(True)
plt.savefig(f'{output_dir}/2_residuals_mathcad.png', dpi=300, bbox_inches='tight')
plt.show()

# Графік 3: Розподіл залишків
plt.figure(figsize=(10, 8))
sns.histplot(y - y_pred_classic, kde=True, color='skyblue')
plt.title('Розподіл залишків (класичний підхід)')
plt.xlabel('Залишки')
plt.ylabel('Частота')
plt.savefig(f'{output_dir}/3_residuals_distribution_mathcad.png', dpi=300, bbox_inches='tight')
plt.show()

# Графік 4: QQ-графік залишків
plt.figure(figsize=(10, 8))
stats.probplot(y - y_pred_classic, dist="norm", plot=plt)
plt.title('QQ-графік залишків (класичний підхід)')
plt.tight_layout()
plt.savefig(f'{output_dir}/4_qq_plot_mathcad.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nГрафіки збережено у директорію: {output_dir}")
print(f"1. Лінійна регресійна модель: {output_dir}/1_regression_model_mathcad.png")
print(f"2. Графік залишків: {output_dir}/2_residuals_mathcad.png")
print(f"3. Розподіл залишків: {output_dir}/3_residuals_distribution_mathcad.png")
print(f"4. QQ-графік залишків: {output_dir}/4_qq_plot_mathcad.png")
```

### Інтерпретація результатів класичного підходу

Аналізуючи результати класичного підходу, ми можемо зробити такі висновки:

1. **Рівняння регресії**: y = 3.3013 + 2.7005*x
   - Це рівняння показує очікувану зміну залежної змінної y при зміні незалежної змінної x.
   - Коефіцієнт нахилу 2.7005 дуже близький до справжнього значення 2.7.

2. **Коефіцієнти регресії**:
   - Вільний член b₀ = 3.3013 - значення y, коли x = 0.
   - Коефіцієнт при x, b₁ = 2.7005 - показує, що при збільшенні x на 1 одиницю, y в середньому збільшується на 2.7005 одиниць.

3. **Статистичні тести**:
   - Коефіцієнт кореляції r = 0.9967 показує дуже сильний зв'язок між змінними.
   - Коефіцієнт детермінації R² = 0.9935 означає, що модель пояснює 99.35% варіації y.
   - Тести статистичної значущості (t-тест та F-тест) підтверджують, що модель є статистично значущою.

4. **Довірчі інтервали**:
   - Для вільного члена: [-1.2116, 7.8143] - досить широкий інтервал.
   - Для коефіцієнта при x: [1.9227, 3.4782] - також досить широкий.
   - Справжні значення параметрів (a₀ = 3.4, a₁ = 2.7) входять у відповідні довірчі інтервали.

5. **Характеристика якості моделі**:
   - Стандартна похибка регресії S = 0.6561 - показник середньої величини залишків.
   - Коефіцієнт еластичності 0.8035 показує, що при зміні x на 1%, y в середньому змінюється на 80.35%.

Загалом, класичний підхід до лінійної регресії надає детальну інформацію про якість моделі та робить прозорим процес обчислення, що дозволяє краще зрозуміти суть регресійного аналізу.
