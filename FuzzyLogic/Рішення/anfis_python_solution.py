import itertools
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


plt.style.use("seaborn-v0_8")

ROOT_DIR = Path(
    r"D:\PROJECTs\MY\ScientificCalculations\SC\ScientificCalculations\FuzzyLogic\Рішення"
)


@dataclass
class GaussianMF:
    """Гаусова функція приналежності з навчуваними параметрами."""

    mean: float
    sigma: float

    def __post_init__(self) -> None:
        self.sigma = max(float(self.sigma), 1e-3)
        self.mean = float(self.mean)

    def response(self, x: np.ndarray) -> np.ndarray:
        sigma_safe = self.sigma + 1e-12
        return np.exp(-0.5 * ((x - self.mean) / sigma_safe) ** 2)


class ANFISModel:
    """Мінімалістична реалізація ANFIS з двома входами та ґратковою ініціалізацією."""

    def __init__(
        self,
        n_inputs: int = 2,
        mfs_per_input: int = 5,
        premise_learning_rate: float = 5e-2,
        epochs: int = 10,
        min_sigma: float = 1e-2,
        error_tolerance: float = 0.0,
    ) -> None:
        if n_inputs != 2:
            raise ValueError("Ця реалізація підтримує рівно два входи")
        self.n_inputs = n_inputs
        self.mfs_per_input = mfs_per_input
        self.premise_learning_rate = premise_learning_rate
        self.epochs = epochs
        self.min_sigma = min_sigma
        self.error_tolerance = error_tolerance
        self.input_mfs: List[List[GaussianMF]] = []
        self.rules: List[Tuple[int, ...]] = []
        self.consequents: np.ndarray | None = None
        self.training_history: List[float] = []

    # --- допоміжні методи ініціалізації ---
    def _init_memberships(self, data: np.ndarray) -> None:
        self.input_mfs = []
        for col in range(self.n_inputs):
            col_min = float(np.min(data[:, col]))
            col_max = float(np.max(data[:, col]))
            centers = np.linspace(col_min, col_max, self.mfs_per_input)
            step = (col_max - col_min) / max(self.mfs_per_input - 1, 1)
            sigma = max(step / math.sqrt(2), self.min_sigma)
            self.input_mfs.append([GaussianMF(c, sigma) for c in centers])
        self.rules = list(itertools.product(range(self.mfs_per_input), repeat=self.n_inputs))

    def _compute_memberships(self, x: np.ndarray) -> List[np.ndarray]:
        membership_values: List[np.ndarray] = []
        for inp_idx in range(self.n_inputs):
            responses = [mf.response(x[:, inp_idx]) for mf in self.input_mfs[inp_idx]]
            membership_values.append(np.stack(responses, axis=1))
        return membership_values

    def _rule_strengths(self, membership_values: List[np.ndarray]) -> np.ndarray:
        n_samples = membership_values[0].shape[0]
        n_rules = len(self.rules)
        strengths = np.ones((n_samples, n_rules))
        for rule_idx, rule in enumerate(self.rules):
            for inp_idx, mf_idx in enumerate(rule):
                strengths[:, rule_idx] *= membership_values[inp_idx][:, mf_idx]
        return strengths

    def _rule_outputs(self, x: np.ndarray) -> np.ndarray:
        assert self.consequents is not None
        linear_part = x @ self.consequents[:, : self.n_inputs].T
        biases = self.consequents[:, -1]
        return linear_part + biases

    def _predict_with_strengths(
        self, x: np.ndarray, strengths: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        eps = 1e-9
        normalized = strengths / (np.sum(strengths, axis=1, keepdims=True) + eps)
        rule_outputs = self._rule_outputs(x)
        y_hat = np.sum(normalized * rule_outputs, axis=1)
        return y_hat, normalized

    def _update_consequents(
        self, x: np.ndarray, y: np.ndarray, normalized_strengths: np.ndarray
    ) -> None:
        n_samples = x.shape[0]
        n_rules = len(self.rules)
        feats = np.zeros((n_samples, n_rules * (self.n_inputs + 1)))
        for rule_idx in range(n_rules):
            base = rule_idx * (self.n_inputs + 1)
            weight = normalized_strengths[:, rule_idx]
            for inp_idx in range(self.n_inputs):
                feats[:, base + inp_idx] = weight * x[:, inp_idx]
            feats[:, base + self.n_inputs] = weight
        coeffs, *_ = np.linalg.lstsq(feats, y, rcond=None)
        self.consequents = coeffs.reshape(n_rules, self.n_inputs + 1)

    def _update_memberships(
        self,
        x: np.ndarray,
        y: np.ndarray,
        membership_values: List[np.ndarray],
        strengths: np.ndarray,
        predictions: np.ndarray,
    ) -> None:
        eps = 1e-9
        sum_strengths = np.sum(strengths, axis=1, keepdims=True) + eps
        rule_outputs = self._rule_outputs(x)
        dy_dwk = (rule_outputs - predictions[:, None]) / sum_strengths
        errors = predictions - y
        n_samples = x.shape[0]
        step = self.premise_learning_rate / max(n_samples, 1)
        for inp_idx in range(self.n_inputs):
            mf_matrix = membership_values[inp_idx]
            for mf_idx, mf in enumerate(self.input_mfs[inp_idx]):
                mu = mf_matrix[:, mf_idx]
                mu_safe = np.where(mu < eps, eps, mu)
                x_col = x[:, inp_idx]
                sigma_safe = mf.sigma + eps
                dmu_dmean = mu * (x_col - mf.mean) / (sigma_safe**2)
                dmu_dsigma = mu * (
                    ((x_col - mf.mean) ** 2) / (sigma_safe**3) - 1.0 / sigma_safe
                )
                influence = np.zeros(n_samples)
                for rule_idx, rule in enumerate(self.rules):
                    if rule[inp_idx] != mf_idx:
                        continue
                    influence += dy_dwk[:, rule_idx] * (
                        strengths[:, rule_idx] / mu_safe
                    )
                grad_mean = np.sum(errors * influence * dmu_dmean)
                grad_sigma = np.sum(errors * influence * dmu_dsigma)
                mf.mean -= step * grad_mean
                mf.sigma = max(mf.sigma - step * grad_sigma, self.min_sigma)

    # --- публічні методи ---
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if not self.input_mfs:
            self._init_memberships(x)
        n_rules = len(self.rules)
        if self.consequents is None:
            self.consequents = np.zeros((n_rules, self.n_inputs + 1))
        for epoch in range(1, self.epochs + 1):
            membership_values = self._compute_memberships(x)
            strengths = self._rule_strengths(membership_values)
            preds, normalized = self._predict_with_strengths(x, strengths)
            self._update_consequents(x, y, normalized)
            preds, _ = self._predict_with_strengths(x, strengths)
            rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
            self.training_history.append(rmse)
            self._update_memberships(x, y, membership_values, strengths, preds)
            print(f"Епоха {epoch:02d}: RMSE = {rmse:.5f}")
            if self.error_tolerance > 0 and rmse <= self.error_tolerance:
                print(
                    "Досягнуто задану точність, навчання достроково завершено."
                )
                break

        # фінальна перебудова наслідків для актуальних функцій приналежності
        membership_values = self._compute_memberships(x)
        strengths = self._rule_strengths(membership_values)
        _, normalized = self._predict_with_strengths(x, strengths)
        self._update_consequents(x, y, normalized)

    def predict(self, x: np.ndarray) -> np.ndarray:
        membership_values = self._compute_memberships(x)
        strengths = self._rule_strengths(membership_values)
        preds, _ = self._predict_with_strengths(x, strengths)
        return preds

    def describe_rules(self, x_labels: Sequence[str]) -> List[str]:
        descriptions = []
        for idx, rule in enumerate(self.rules):
            antecedents = []
            for inp_idx, mf_idx in enumerate(rule):
                mf = self.input_mfs[inp_idx][mf_idx]
                antecedents.append(
                    f"{x_labels[inp_idx]} is Gauss(mean={mf.mean:.3f}, sigma={mf.sigma:.3f})"
                )
            cons = self.consequents[idx]
            consequents = f"y = {cons[0]:.3f}*{x_labels[0]} + {cons[1]:.3f}*{x_labels[1]} + {cons[2]:.3f}"
            descriptions.append(f"Rule {idx + 1}: IF {' AND '.join(antecedents)} THEN {consequents}")
        return descriptions


def load_grid_from_excel(path: Path) -> pd.DataFrame:
    """Завантаження даних та збереження структури матриці."""
    df = pd.read_excel(path, header=None)
    if df.isna().all(axis=None):
        raise ValueError("У файлі не знайдено числових даних")
    return df


def matrix_to_triplets(df: pd.DataFrame) -> pd.DataFrame:
    """Перетворення матриці у формат [x1, x2, y]."""
    x1_values = df.iloc[1:, 0].to_numpy(dtype=float)
    x2_values = df.iloc[0, 1:].to_numpy(dtype=float)
    response_matrix = df.iloc[1:, 1:].to_numpy(dtype=float)
    rows = []
    for i, x1 in enumerate(x1_values):
        for j, x2 in enumerate(x2_values):
            rows.append((x1, x2, response_matrix[i, j]))
    return pd.DataFrame(rows, columns=["x1", "x2", "y"])


# --- побудова графіків ---
def plot_heatmap(df: pd.DataFrame, save_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.iloc[1:, 1:].astype(float), cmap="viridis", cbar_kws={"label": "y"})
    plt.title("Матриця вихідних значень")
    plt.xlabel("x2")
    plt.ylabel("x1")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_memberships(model: ANFISModel, data: np.ndarray, labels: Sequence[str], save_path: Path) -> None:
    fig, axes = plt.subplots(1, model.n_inputs, figsize=(12, 4))
    for idx, ax in enumerate(np.atleast_1d(axes)):
        xs = np.linspace(np.min(data[:, idx]), np.max(data[:, idx]), 500)
        for mf in model.input_mfs[idx]:
            ax.plot(xs, mf.response(xs), label=f"μ(mean={mf.mean:.2f}, σ={mf.sigma:.2f})")
        ax.set_title(f"Функції приналежності для {labels[idx]}")
        ax.set_xlabel(labels[idx])
        ax.set_ylabel("μ")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_learning_curve(history: Sequence[float], save_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(history) + 1), history, marker="o")
    plt.title("Динаміка RMSE під час навчання")
    plt.xlabel("Епоха")
    plt.ylabel("RMSE")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="ідеальна відповідність")
    plt.xlabel("Спостережене y")
    plt.ylabel("Прогнозоване y")
    plt.title("Порівняння фактичних та модельних значень")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_surface(
    x1: np.ndarray,
    x2: np.ndarray,
    y_true: np.ndarray,
    model: ANFISModel,
    save_path: Path,
) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    X_grid, Y_grid = np.meshgrid(
        np.linspace(x1.min(), x1.max(), 40),
        np.linspace(x2.min(), x2.max(), 40),
    )
    grid_points = np.column_stack((X_grid.ravel(), Y_grid.ravel()))
    Z_pred = model.predict(grid_points).reshape(X_grid.shape)

    ax1.plot_trisurf(x1, x2, y_true, cmap="viridis", linewidth=0.2, alpha=0.9)
    ax1.set_title("Поверхня за даними")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_zlabel("y")

    ax2.plot_surface(X_grid, Y_grid, Z_pred, cmap="plasma", alpha=0.9)
    ax2.set_title("Поверхня ANFIS")
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_zlabel("y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def prompt_user(model: ANFISModel) -> None:
    print("\nВведіть значення двох входів (порожній рядок для завершення).")
    while True:
        raw = input("x1, x2 = ")
        if not raw.strip():
            break
        try:
            parts = [float(val) for val in raw.replace(";", ",").split(",")]
            if len(parts) != 2:
                raise ValueError
        except ValueError:
            print("Будь ласка, введіть два числа через кому.")
            continue
        x_arr = np.array(parts).reshape(1, -1)
        pred = model.predict(x_arr)[0]
        print(f"ANFIS прогноз y = {pred:.4f}")


def main() -> None:
    data_path = ROOT_DIR / "data2.xls"
    solution_dir = ROOT_DIR / "Рішення"
    output_dir = solution_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    solution_dir.mkdir(parents=True, exist_ok=True)

    print(f"Завантаження даних з {data_path}")
    matrix_df = load_grid_from_excel(data_path)
    triplet_df = matrix_to_triplets(matrix_df)
    triplet_path = solution_dir / "transformed_dataset.csv"
    triplet_df.to_csv(triplet_path, index=False)
    print(f"Перетворені дані збережено у {triplet_path}")

    x = triplet_df[["x1", "x2"]].to_numpy()
    y = triplet_df["y"].to_numpy()

    model = ANFISModel(mfs_per_input=5, epochs=10)
    model.fit(x, y)

    predictions = model.predict(x)
    rmse = float(np.sqrt(np.mean((predictions - y) ** 2)))
    mae = float(np.mean(np.abs(predictions - y)))
    r2 = 1 - float(np.sum((y - predictions) ** 2) / np.sum((y - y.mean()) ** 2))
    print(f"Підсумкові метрики: RMSE={rmse:.4f}, MAE={mae:.4f}, R^2={r2:.4f}")

    plot_heatmap(matrix_df, output_dir / "heatmap.png")
    plot_memberships(model, x, ["x1", "x2"], output_dir / "membership_functions.png")
    plot_learning_curve(model.training_history, output_dir / "learning_curve.png")
    plot_predictions(y, predictions, output_dir / "prediction_scatter.png")
    plot_surface(x[:, 0], x[:, 1], y, model, output_dir / "surfaces.png")

    print("Графіки збережено у директорії", output_dir)

    print("\nФормульовані правила:")
    for line in model.describe_rules(["x1", "x2"]):
        print(line)

    prompt_user(model)


if __name__ == "__main__":
    main()
