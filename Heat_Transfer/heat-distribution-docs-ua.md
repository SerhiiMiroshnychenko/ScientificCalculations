# Документація аналізу розподілу тепла

## Огляд
Зображення представляє візуалізацію та математичний метод для аналізу розподілу тепла по двовимірній сітці з використанням методу скінченних різниць. Воно складається з трьох основних компонентів:
1. Візуалізація теплової карти
2. Діаграма шаблону скінченних різниць
3. Математична формула для розрахунку температури

## Візуалізація теплової карти
Ліва частина зображення показує кольорову теплову карту з такими характеристиками:
- Розміри сітки: 9x9 одиниць
- Шкала температур: 45-77 одиниць (показано на кольоровій шкалі)
- Кольорова схема:
  - Червоний: найвищі температури (~77 одиниць)
  - Жовтий: середньо-високі температури (~69 одиниць)
  - Зелений: середні температури (~61 одиниця)
  - Світло-синій: середньо-низькі температури (~57 одиниць)
  - Темно-синій: найнижчі температури (~45 одиниць)
- Температурний градієнт: Показує чіткий перехід від високих температур зліва до нижчих температур справа

## Шаблон скінченних різниць
Верхня права частина показує діаграму шаблону скінченних різниць, що включає:
- Точки сітки позначені хрестиками (×)
- Центральна точка: Ti,j
- Суміжні точки:
  - Верхня: Ti,j+1
  - Нижня: Ti,j-1
  - Ліва: Ti-1,j
  - Права: Ti+1,j
- Позначення кроку сітки:
  - Δx: горизонтальний крок
  - Δy: вертикальний крок

## Формула розрахунку температури
Математична формула для розрахунку температури в будь-якій точці (i,j):

```
Ti,j = (Ti-1,j + Ti+1,j + Ti,j+1 + Ti,j-1) / 4
```

Ця формула представляє:
- Середню температуру чотирьох сусідніх точок
- Дискретизовану форму рівняння Лапласа для теплопровідності
- Числовий метод розв'язання стаціонарного розподілу тепла

## Застосування
Цей метод часто використовується в:
- Аналізі теплопередачі
- Числових розв'язках диференціальних рівнянь у частинних похідних
- Моделюванні розподілу температури
- Застосуваннях теплотехніки

Візуалізація демонструє, як цей числовий метод може ефективно моделювати температурні градієнти та теплові потоки у двовимірній області.
