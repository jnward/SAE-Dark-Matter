# %%

import pickle
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import FixedLocator, FixedFormatter
from utils import BASE_DIR


def confidence_interval(successes, total, z=1.96):
    p = successes / total
    se = np.sqrt(p * (1 - p) / total)
    return z * se


def process_results(results):
    total_per_quantile = [0 for _ in range(11)]
    total_correct_per_quantile = [0 for _ in range(11)]
    for quantile_positives, quantile_totals in results[0]:
        for i in range(11):
            total_per_quantile[i] += quantile_totals[i]
            total_correct_per_quantile[i] += quantile_positives[i]

    total_negative = 0
    total_negative_correct = 0
    for negative_positives, negative_totals in results[1]:
        total_negative += negative_totals
        total_negative_correct += negative_positives

    average_per_quantile = [
        total_correct_per_quantile[i] / total_per_quantile[i] for i in range(1, 11)
    ]
    average_negative = total_negative_correct / total_negative

    ci_per_quantile = [
        confidence_interval(total_correct_per_quantile[i], total_per_quantile[i])
        for i in range(1, 11)
    ]
    ci_negative = confidence_interval(total_negative_correct, total_negative)

    x = ["Not"] + ["Q" + str(i) for i in range(1, 11)]
    y = [1 - average_negative] + average_per_quantile[::-1]
    ci = [ci_negative] + ci_per_quantile[::-1]

    return x, y, ci


# Create two subplots with the new figure size
fig, ax = plt.subplots(1, 1, figsize=(2.75, 2))

topk_data = None

results = [
    pickle.load(
        open(
            f"{BASE_DIR}/feature_evals/sae-ckpts/nonlinear_error/model.layers.12/results.pkl",
            "rb",
        )
    ),
    pickle.load(
        open(
            f"{BASE_DIR}/feature_evals/sae-ckpts/linear_error/model.layers.12/results.pkl",
            "rb",
        )
    ),
]
names = ["SAE trained on nonlinear error", "SAE trained on linear error"]


cmap = get_cmap("viridis")
color_step = 1.0 / 7
colors = [cmap(2 * color_step), cmap(5 * color_step)]

for results, name, color in zip(results, names, colors):
    x, y, ci = process_results(results)
    ax.errorbar(
        x,
        y,
        yerr=ci,
        fmt="-o",
        capsize=5,
        markersize=5,
        label=name,
        color=color,
        alpha=0.8,
    )

ax.set_xlabel("Decile", fontsize=8)
ax.set_ylabel("Accuracy", fontsize=8)
ax.grid(True, which="both", ls="--", linewidth=0.5)
ax.legend(loc="upper center", prop={"size": 5.5})

# Set y-axis limits and ticks
ax.set_ylim(0.3, 0.85)  # Adjust these values as needed

# Adjust tick label size
ax.tick_params(axis="both", labelsize=6.5)

# Remove minor ticks
ax.minorticks_off()

# Tilt all x-axis labels
for tick in ax.get_xticklabels():
    tick.set_rotation(45)
    tick.set_va("top")

os.makedirs("plots", exist_ok=True)
plt.savefig("plots/detection.pdf", bbox_inches="tight")
plt.show()
# %%
