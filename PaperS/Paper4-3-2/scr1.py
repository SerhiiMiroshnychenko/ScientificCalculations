import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Dataset 1
cm1_baseline = np.array([[8258, 1352],
                         [496, 15933]])
cm1_optuna = np.array([[8439, 1171],
                       [623, 15806]])
cm1_best = np.array([[8447, 1163],
                     [616, 15813]])

# Dataset 2
cm2_baseline = np.array([[8460, 1150],
                         [575, 15854]])  # restored
cm2_optuna = np.array([[8276, 1334],
                       [456, 15973]])
cm2_best = np.array([[8273, 1337],
                     [472, 15957]])

# Plotting
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

cms = [
    cm1_baseline, cm1_optuna, cm1_best,
    cm2_baseline, cm2_optuna, cm2_best
]
titles = [
    "Dataset 1\nAll features (default)",
    "Dataset 1\nAll features + Optuna",
    "Dataset 1\nBest features (Greedy, 16) + Optuna",
    "Dataset 2\nAll features (default, restored)",
    "Dataset 2\nAll features + Optuna",
    "Dataset 2\nBest features (Reversed Greedy, 16) + Optuna"
]

for i, ax in enumerate(axes.flat):
    disp = ConfusionMatrixDisplay(confusion_matrix=cms[i], display_labels=["Negative", "Positive"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(titles[i])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

plt.tight_layout()
plt.savefig("confusion_matrices_comparison.png")
# plt.show()  # Only save, do not show