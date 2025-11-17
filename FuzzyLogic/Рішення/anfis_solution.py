from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from skanfis.fs import FS, LinguisticVariable, GaussianFuzzySet
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


def _build_gaussian_terms(var_min: float, var_max: float, n_terms: int, prefix: str):
    """Будує список гаусовських термів на відрізку [var_min, var_max].

    Кількість термів n_terms відповідає кількості функцій належності
    (наприклад, 5 – як у налаштуванні Grid partition в MATLAB).

    Центри гаусів розміщуються рівномірно по діапазону,
    σ вибирається так, щоб сусідні гауси перекривались,
    подібно до прикладу MATLAB.
    """

    # Центри гаусових МФ
    centers = np.linspace(var_min, var_max, n_terms)

    # Оцінимо стандартне відхилення як відстань між центрами, поділену на 2
    if n_terms > 1:
        delta = centers[1] - centers[0]
        sigma = float(delta / 2.0)
    else:
        sigma = float((var_max - var_min) / 2.0) if var_max > var_min else 1.0

    terms = []
    for k, c in enumerate(centers, start=1):
        term_name = f"{prefix}_{k}"
        terms.append(GaussianFuzzySet(mu=float(c), sigma=sigma, term=term_name))

    return terms


def build_fs_for_two_inputs(xxT: np.ndarray, n_terms: int = 5) -> FS:
    """Створює нечітку систему FS для двох входів і одного виходу.

    Для кожного входу будується n_terms гаусовських функцій належності
    (аналог gaussmf у MATLAB-прикладі) з рівномірно розташованими
    центрами по діапазону даних.
    """

    x1 = xxT[:, 0]
    x2 = xxT[:, 1]

    x1_min, x1_max = float(np.min(x1)), float(np.max(x1))
    x2_min, x2_max = float(np.min(x2)), float(np.max(x2))

    fs = FS()

    # Побудова лінгвістичних змінних для X1 та X2 з гаусовськими МФ
    x1_terms = _build_gaussian_terms(x1_min, x1_max, n_terms, prefix="X1")
    x2_terms = _build_gaussian_terms(x2_min, x2_max, n_terms, prefix="X2")

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


def _gaussian_mf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Обчислює значення гаусовської функції належності для масиву x."""

    if sigma <= 0:
        return (x == mu).astype(float)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


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


def visualize_training_data(xxT: np.ndarray, output_dir: Path) -> None:
    """Будує графік Training Data: вихід y від індексу точки (аналог рис. 3.4)."""

    output_dir.mkdir(parents=True, exist_ok=True)

    indices = np.arange(xxT.shape[0])
    y_true = xxT[:, 2]

    plt.figure(figsize=(7, 5))
    plt.scatter(indices, y_true, edgecolor="k", facecolors="none")
    plt.xlabel("data set index")
    plt.ylabel("Output")
    plt.title("Training Data (output vs index)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "training_data_index.png", dpi=150)
    plt.close()


def visualize_training_vs_fis(model, xxT: np.ndarray, output_dir: Path) -> None:
    """Будує графік Training data vs FIS output (аналог рис. 3.8)."""

    output_dir.mkdir(parents=True, exist_ok=True)

    indices = np.arange(xxT.shape[0])
    X = xxT[:, :2]
    y_true = xxT[:, 2]
    y_pred = model.predict(X).reshape(-1)

    plt.figure(figsize=(7, 5))
    # Навчальні дані – кружки
    plt.scatter(indices, y_true, c="k", marker="o", label="Training data")
    # Вихід FIS – зірочки
    plt.scatter(indices, y_pred, c="r", marker="*", label="FIS output")
    plt.xlabel("Index")
    plt.ylabel("Output")
    plt.title("Training data (o) and FIS output (*) vs index")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_vs_fis_index.png", dpi=150)
    plt.close()


def visualize_training_error(model, output_dir: Path) -> None:
    """Будує графік Training Error vs Epochs, якщо доступна історія втрат.

    Якщо модель не зберігає історію помилки, графік не будується.
    """

    # Підтримуємо кілька можливих назв атрибутів з історією помилки
    history = None
    if hasattr(model, "loss_history_"):
        history = getattr(model, "loss_history_")
    elif hasattr(model, "loss_history"):
        history = getattr(model, "loss_history")

    if history is None:
        # Немає даних про помилку по епохах – пропускаємо побудову
        return

    history = np.asarray(history, dtype=float).ravel()
    if history.size == 0:
        return

    epochs = np.arange(1, history.size + 1)

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history, "b*-")
    plt.xlabel("Epochs")
    plt.ylabel("Training Error")
    plt.title("Training Error vs Epochs")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "training_error_epochs.png", dpi=150)
    plt.close()


def visualize_membership_functions(xxT: np.ndarray, n_terms: int, output_dir: Path) -> None:
    """Будує графіки функцій приналежності для X1 та X2.

    Використовуються гаусовські МФ, побудовані на інтервалах значень X1 та X2
    з рівномірно розташованими центрами (аналог налаштувань у FIS editor).
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    x1 = xxT[:, 0]
    x2 = xxT[:, 1]

    x1_min, x1_max = float(np.min(x1)), float(np.max(x1))
    x2_min, x2_max = float(np.min(x2)), float(np.max(x2))

    # Відтворюємо параметри гаусовських МФ так само, як у _build_gaussian_terms
    x1_centers = np.linspace(x1_min, x1_max, n_terms)
    x2_centers = np.linspace(x2_min, x2_max, n_terms)

    if n_terms > 1:
        sigma1 = float((x1_centers[1] - x1_centers[0]) / 2.0)
        sigma2 = float((x2_centers[1] - x2_centers[0]) / 2.0)
    else:
        sigma1 = float((x1_max - x1_min) / 2.0) if x1_max > x1_min else 1.0
        sigma2 = float((x2_max - x2_min) / 2.0) if x2_max > x2_min else 1.0

    # Графік МФ для X1
    x_axis1 = np.linspace(x1_min, x1_max, 400)
    plt.figure(figsize=(7, 5))
    for k, mu in enumerate(x1_centers, start=1):
        term_name = f"X1_{k}"
        y_vals = _gaussian_mf(x_axis1, mu, sigma1)
        plt.plot(x_axis1, y_vals, label=term_name)
    plt.xlabel("X1")
    plt.ylabel("Ступінь належності")
    plt.title("Функції приналежності для змінної X1")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "mf_X1.png", dpi=150)
    plt.close()

    # Графік МФ для X2
    x_axis2 = np.linspace(x2_min, x2_max, 400)
    plt.figure(figsize=(7, 5))
    for k, mu in enumerate(x2_centers, start=1):
        term_name = f"X2_{k}"
        y_vals = _gaussian_mf(x_axis2, mu, sigma2)
        plt.plot(x_axis2, y_vals, label=term_name)
    plt.xlabel("X2")
    plt.ylabel("Ступінь належності")
    plt.title("Функції приналежності для змінної X2")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "mf_X2.png", dpi=150)
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
    model = train_anfis_model(fs, xxT, n_epochs=100)

    # Обчислення RMSE на навчальній вибірці (аналог повідомлення Matlab про training RMSE)
    X_train = xxT[:, :2]
    y_train_true = xxT[:, 2]
    y_train_pred = model.predict(X_train).reshape(-1)
    rmse = float(np.sqrt(np.mean((y_train_true - y_train_pred) ** 2)))
    print(f"Мінімальний training RMSE (оцінка на навченій моделі) = {rmse:.6f}")

    # Контрольна точка для порівняння з Matlab: x1 = 5, x2 = 21.4
    test_point = np.array([[5.0, 21.4]], dtype=float)
    test_y = model.predict(test_point).reshape(-1)[0]
    print(f"Контрольна точка (x1=5, x2=21.4): прогнозоване y = {test_y:.6f}")

    print("Крок 4. Побудова візуалізацій (Training data, FIS output, помилка, МФ та поверхня)...")
    visualize_training_data(xxT, results_dir)
    visualize_training_vs_fis(model, xxT, results_dir)
    visualize_training_error(model, results_dir)
    visualize_results(model, xxT, results_dir)
    visualize_membership_functions(xxT, n_terms=5, output_dir=results_dir)
    print(f"Графіки збережено у каталозі: {results_dir}")

    print("Крок 5. Інтерактивна перевірка моделі за довільними значеннями x1 та x2.")
    interactive_prediction(model)


if __name__ == "__main__":
    main()
