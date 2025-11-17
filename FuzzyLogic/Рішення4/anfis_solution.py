from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from skanfis.fs import FS, LinguisticVariable, TriangleFuzzySet
from skanfis import scikit_anfis


# Батьківська директорія для результатів (відповідно до вимоги)
ROOT_DIR = Path(r"D:\PROJECTs\MY\ScientificCalculations\SC\ScientificCalculations\FuzzyLogic\Рішення")

# Шлях до Excel-файла з даними (матриця залежності у(x1, x2))
EXCEL_PATH = Path(r"D:\PROJECTs\MY\ScientificCalculations\SC\ScientificCalculations\FuzzyLogic\Рішення\data2.xls")


def load_and_transform_excel(excel_path: Path) -> np.ndarray:
    """Завантажує матрицю data2 з Excel та перетворює її у масив xxT (x1, x2, y).

    Аналог MATLAB-коду:
        p = data2(2:end,1);
        M = data2(1,2:end);
        tabT = data2(2:end,2:end);
        [max_i, max_j] = size(tabT);
        xxT = zeros(max_j*max_i, 3);
        for i = 1:max_i
            for j = 1:max_j
                xxT(i+(j-1)*max_i,1) = p(i);
                xxT(i+(j-1)*max_i,2) = M(j);
                xxT(i+(j-1)*max_i,3) = tabT(i,j);
            end
        end
    """

    # Завантаження Excel без заголовків, як «цифрову матрицю»
    df = pd.read_excel(excel_path, header=None)
    data = df.values

    # Перший стовпець, рядки з 2-го до кінця – вхід x1
    p = data[1:, 0].astype(float)
    # Перший рядок, стовпці з 2-го до кінця – вхід x2
    M = data[0, 1:].astype(float)
    # Внутрішня таблиця – вихідні значення y
    tabT = data[1:, 1:].astype(float)

    max_i, max_j = tabT.shape
    xxT = np.zeros((max_i * max_j, 3), dtype=float)

    # Відтворення формули індексації i + (j-1)*max_i
    index = 0
    for j in range(max_j):
        for i in range(max_i):
            xxT[index, 0] = p[i]
            xxT[index, 1] = M[j]
            xxT[index, 2] = tabT[i, j]
            index += 1

    return xxT


def _build_triangular_terms(var_min: float, var_max: float, n_terms: int, prefix: str):
    """Будує список трикутних термів на відрізку [var_min, var_max].

    Кількість термів n_terms відповідає кількості функцій належності
    (наприклад, 5 – як у налаштуванні Grid partition в MATLAB).
    """

    # Створюємо вузли розбиття для термів
    grid = np.linspace(var_min, var_max, n_terms)

    terms = []
    for k in range(n_terms):
        if k == 0:
            a = grid[0]
            b = grid[0]
            c = grid[1]
        elif k == n_terms - 1:
            a = grid[-2]
            b = grid[-1]
            c = grid[-1]
        else:
            a = grid[k - 1]
            b = grid[k]
            c = grid[k + 1]

        term_name = f"{prefix}_{k + 1}"
        terms.append(TriangleFuzzySet(a=a, b=b, c=c, term=term_name))

    return terms


def build_fs_for_two_inputs(xxT: np.ndarray, n_terms: int = 5) -> FS:
    """Створює нечітку систему FS для двох входів і одного виходу.

    Для кожного входу будується n_terms трикутних функцій належності.
    Тип функцій належності в MATLAB-прикладі – gaussmf, але у примітці
    дозволено обирати інші типи залежно від об'єкта. Тут використано
    трикутні МФ для простоти та сумісності з прикладом scikit-anfis.
    """

    x1 = xxT[:, 0]
    x2 = xxT[:, 1]

    x1_min, x1_max = float(np.min(x1)), float(np.max(x1))
    x2_min, x2_max = float(np.min(x2)), float(np.max(x2))

    fs = FS()

    # Побудова лінгвістичних змінних для X1 та X2
    x1_terms = _build_triangular_terms(x1_min, x1_max, n_terms, prefix="X1")
    x2_terms = _build_triangular_terms(x2_min, x2_max, n_terms, prefix="X2")

    fs.add_linguistic_variable("X1", LinguisticVariable(x1_terms, concept="Вхід X1"))
    fs.add_linguistic_variable("X2", LinguisticVariable(x2_terms, concept="Вхід X2"))

    # Вихідна змінна Y – Сугено-подібна, задамо одну лінійну функцію від X1 та X2
    # Коефіцієнти цієї функції навчатимуться під час оптимізації моделі.
    fs.set_output_function("OUT", "a1*X1 + a2*X2 + c")

    # Правила типу: IF (X1 IS X1_k) AND (X2 IS X2_m) THEN (Y IS OUT)
    rules = []
    for k in range(n_terms):
        for m in range(n_terms):
            term1 = f"X1_{k + 1}"
            term2 = f"X2_{m + 1}"
            rule = f"IF (X1 IS {term1}) AND (X2 IS {term2}) THEN (Y IS OUT)"
            rules.append(rule)

    fs.add_rules(rules)

    return fs


def train_anfis_model(fs: FS, xxT: np.ndarray, n_epochs: int = 50):
    """Навчає ANFIS-модель на основі заданої нечіткої системи FS та даних xxT.

    Формат xxT: стовпці [x1, x2, y]. Модель навчається у гібридному режимі
    (градієнтний спуск + метод найменших квадратів), аналогічно MATLAB ANFIS.
    """

    # Для ручно визначеної FS scikit-anfis очікує єдиний масив [x1, x2, y]
    train_data = xxT.astype(float)

    model = scikit_anfis(
        fs,
        description="ANFIS_Excel_2in1out",
        epoch=n_epochs,
        hybrid=True,
    )

    model.fit(train_data)

    return model


def visualize_results(model, xxT: np.ndarray, output_dir: Path) -> None:
    """Будує основні візуалізації, аналогічні MATLAB-прикладу.

    1) Порівняння фактичних та змодельованих значень (розсіювання).
    2) Поверхню закону керування y(x1, x2) за даними моделі.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    X = xxT[:, :2]
    y_true = xxT[:, 2]

    # Прогноз моделі на навчальній вибірці
    y_pred = model.predict(X).reshape(-1)

    # 1. Розсіювання фактичних та змодельованих значень
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolor="k")
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="y = ŷ")
    plt.xlabel("Фактичні значення y")
    plt.ylabel("Прогнозовані значення ŷ")
    plt.title("ANFIS: порівняння фактичних та змодельованих значень")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "scatter_true_vs_pred.png", dpi=150)
    plt.close()

    # 2. Поверхня закону керування y(x1, x2)
    x1 = X[:, 0]
    x2 = X[:, 1]
    x1_min, x1_max = float(np.min(x1)), float(np.max(x1))
    x2_min, x2_max = float(np.min(x2)), float(np.max(x2))

    grid_size = 40
    x1_grid = np.linspace(x1_min, x1_max, grid_size)
    x2_grid = np.linspace(x2_min, x2_max, grid_size)
    X1_mesh, X2_mesh = np.meshgrid(x1_grid, x2_grid)

    grid_points = np.column_stack((X1_mesh.ravel(), X2_mesh.ravel()))
    y_grid = model.predict(grid_points).reshape(X1_mesh.shape)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X1_mesh, X2_mesh, y_grid, cmap="viridis", edgecolor="none", alpha=0.8)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("y")
    ax.set_title("ANFIS: поверхня закону керування y(x1, x2)")
    plt.tight_layout()
    plt.savefig(output_dir / "surface_y_x1_x2.png", dpi=150)
    plt.close()


def interactive_prediction(model) -> None:
    """Простий інтерфейс для ручного введення двох вхідних параметрів.

    Користувач може ввести значення x1 та x2 і отримати вихід y.
    Для завершення роботи достатньо натиснути Enter без введення значення.
    """

    print("\nІнтерактивний режим: обчислення вихідного значення y за заданими x1 та x2.")
    print("Для виходу натисніть Enter, не вводячи значення.")

    while True:
        raw_x1 = input("Введіть значення x1 (або Enter для завершення): ").strip()
        if raw_x1 == "":
            break

        raw_x2 = input("Введіть значення x2 (або Enter для завершення): ").strip()
        if raw_x2 == "":
            break

        try:
            x1_val = float(raw_x1.replace(",", "."))
            x2_val = float(raw_x2.replace(",", "."))
        except ValueError:
            print("Помилка: необхідно вводити числові значення. Спробуйте ще раз.\n")
            continue

        x_input = np.array([[x1_val, x2_val]], dtype=float)
        y_out = model.predict(x_input).reshape(-1)[0]
        print(f"Результат ANFIS: y = {y_out:.6f}\n")


def main():
    """Основна функція виконання скрипта.

    1) Завантаження та трансформація даних з Excel (data2.xls).
    2) Побудова нечіткої системи FS для двох входів.
    3) Навчання ANFIS-моделі в гібридному режимі.
    4) Побудова візуалізацій (розсіювання, поверхня закону керування).
    5) Інтерактивний режим для ручного введення x1, x2.
    """

    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Не знайдено файл з даними: {EXCEL_PATH}")

    # Каталог для збереження результатів
    results_dir = ROOT_DIR / "ANFIS_Excel_2in1out"

    print("Крок 1. Завантаження та трансформація даних з Excel...")
    xxT = load_and_transform_excel(EXCEL_PATH)
    print(f"Розмір масиву xxT: {xxT.shape[0]} рядків, {xxT.shape[1]} стовпців.")

    print("Крок 2. Побудова нечіткої системи (FS) для двох входів...")
    fs = build_fs_for_two_inputs(xxT, n_terms=5)

    print("Крок 3. Навчання ANFIS-моделі (гібридний режим)...")
    model = train_anfis_model(fs, xxT, n_epochs=50)

    print("Крок 4. Побудова візуалізацій (розсіювання та поверхня закону керування)...")
    visualize_results(model, xxT, results_dir)
    print(f"Графіки збережено у каталозі: {results_dir}")

    print("Крок 5. Інтерактивна перевірка моделі за довільними значеннями x1 та x2.")
    interactive_prediction(model)


if __name__ == "__main__":
    main()
