# Розрахунок висоти димової труби нагрівальної печі

Розрахувати висоту димової труби нагрівальної печі за умов:
температура димових газів 850 °С, сумарний втрати тиску (опір руху) в трубі 310 Па.
Прийняти, що щільність димових газів за нормальних умов приблизно дорівнює щільності повітря і становить $1.29$ кг/м$^3$.

## Умови задачі
- Температура димових газів: $850 °С$
- Сумарні втрати тиску (опір руху) в трубі: $310$ Па
- Щільність димових газів за нормальних умов: $1.29$ кг/м$^3$ (приблизно дорівнює щільності повітря)

## Теоретичне обґрунтування

### Фізичні основи природної тяги

Принцип роботи димової труби базується на фізичному явищі природної тяги, яка виникає внаслідок різниці густин між нагрітими димовими газами всередині труби та зовнішнім повітрям. Ця різниця густин створює різницю тисків, яка забезпечує рух газів вгору по трубі.

Природна тяга — це результат дії архімедової сили, що діє на димові гази з меншою густиною, які знаходяться в середовищі з більшою густиною (атмосферне повітря). Згідно з принципом Архімеда, на тіло, занурене в рідину або газ, діє виштовхувальна сила, рівна вазі витісненого ним об'єму рідини або газу.

### Формування статичного тиску

У випадку димової труби виникає статичний тиск, який виражається через різницю висот стовпів повітря ззовні та стовпа димових газів всередині труби. Математично це можна виразити так:

$$\Delta p = h \times g \times (\rho_{пов} - \rho_{газ})$$

де:
- $\Delta p$ - величина створюваної тяги, Па;
- $h$ - висота димової труби, м;
- $g$ - прискорення вільного падіння, $9.81$ м/с$^2$;
- $\rho_{пов}$ - густина зовнішнього повітря, кг/м$^3$;
- $\rho_{газ}$ - густина димових газів при робочій температурі, кг/м$^3$.

### Вплив температури на тягу

Температура газів відіграє критичну роль у створенні тяги, оскільки вона безпосередньо впливає на густину газів. Згідно з законами термодинаміки, при підвищенні температури густина газу зменшується, що збільшує різницю густин між димовими газами та атмосферним повітрям, тим самим посилюючи тягу.

### Баланс тяги та опору системи

Для забезпечення нормальної роботи системи, створювана тяга повинна бути достатньою для подолання всіх опорів системи. Ці опори включають:

1. **Аеродинамічний опір каналів** — опір, що виникає при русі газів по димоходах та газоходах печі
2. **Місцеві опори** — опір, що виникає в місцях зміни напрямку потоку, перетину або при проходженні через різні пристрої (шибери, засувки)
3. **Опір на вході та виході** — виникає при зміні швидкості та напрямку потоку на вході в димохід та при виході з труби в атмосферу

Таким чином, необхідну висоту труби можна визначити, прирівнявши створювану тягу до сумарних втрат тиску:

$$h \times g \times (\rho_{пов} - \rho_{газ}) = \sum \Delta p_{опору}$$

де $\sum \Delta p_{опору}$ — сума всіх аеродинамічних опорів системи.

## Термодинамічні закономірності зміни густини газів

### Закон ідеального газу та залежність густини від температури

Зміна густини газів з температурою підпорядковується закону ідеального газу. За умови постійного тиску, густина газу обернено пропорційна його абсолютній температурі:

$$\rho_{газ} = \rho_0 \times \frac{T_0}{T}$$

де:
- $\rho_0$ - густина газів за нормальних умов, кг/м$^3$;
- $T_0$ - температура за нормальних умов, $273$ К;
- $T$ - абсолютна температура газів, К.

Ця залежність є фундаментальною для розрахунку тяги, оскільки визначає різницю густин між димовими газами та атмосферним повітрям.

### Вплив вологості та складу димових газів

На практиці склад димових газів і наявність в них вологи також впливають на їх густину. Водяна пара має меншу молекулярну масу порівняно з повітрям, тому вологі димові гази мають меншу густину, що посилює тягу. Однак у спрощених розрахунках цим ефектом часто нехтують.

## Розрахунок густини димових газів при підвищеній температурі

Підставляючи значення для нашої задачі:

$$\rho_{газ} = 1.29 \times \frac{273.15}{273.15 + 850} = 0.314\;кг/м^3$$

## Розрахунок висоти димової труби

Тепер визначимо необхідну висоту димової труби, підставивши отримані значення в рівняння тяги:

$$310 = h \times 9.81 \times (1.29 - 0.314)$$

$$h = \frac{310}{9.574} = 32.378\;м$$

## Практичні аспекти проектування димових труб

### Коефіцієнт запасу тяги

У практичному проектуванні часто вводять коефіцієнт запасу тяги (зазвичай 1.2-1.5), щоб компенсувати можливі відхилення від розрахункових умов, таких як:
- Зміни атмосферного тиску
- Коливання температури зовнішнього повітря
- Зміни режиму роботи печі
- Забруднення внутрішніх поверхонь димоходів

### Конструктивні обмеження

При проектуванні димової труби необхідно також враховувати:
- Вплив вітрових навантажень на стійкість конструкції
- Температурні деформації матеріалів
- Вимоги будівельних норм щодо мінімальної висоти труб над дахами будівель
- Екологічні вимоги щодо розсіювання забруднюючих речовин в атмосфері

## Висновок

Таким чином, для забезпечення необхідної тяги, що дозволить подолати опір системи в $310$ Па при температурі димових газів $850 °С$, необхідно спроектувати димову трубу висотою $32.378$ м, яку можна округлити до $32.4$ м для практичного застосування.

Ця висота забезпечить достатню природну тягу для ефективного відведення продуктів згоряння з нагрівальної печі в атмосферу, підтримуючи оптимальний режим роботи теплотехнічного обладнання.

При цьому, в залежності від конкретних умов експлуатації та вимог безпеки, може бути доцільним застосування коефіцієнта запасу, що призведе до збільшення розрахункової висоти труби.