#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = Path("/reports_qs")  # change if needed

def load_hist(name):
    path = OUT_DIR / f"{name}_history.json"
    with path.open() as fh:
        hist = json.load(fh)
    return np.array(hist["drift_mean"]), np.array(hist["drift_std"])

baseline_mean, baseline_std = load_hist("baseline")
qat_mean, qat_std = load_hist("qat")
qs_mean, qs_std = load_hist("q_sentinel")
epochs = np.arange(1, len(qs_mean) + 1)

plt.rcParams.update({
    "figure.dpi": 300,
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})

colors = {"baseline": "#d62728", "qat": "#ff7f0e", "q_sentinel": "#1f77b4"}
markers = {"baseline": "o", "qat": "s", "q_sentinel": "D"}

fig, ax = plt.subplots(figsize=(5.5, 3.6), dpi=300)

def plot_curve(mean, std, label, color):
    ax.plot(epochs, mean, color=color, marker=markers[label], linewidth=2.0, label=label.upper())
    ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.25)

plot_curve(baseline_mean, baseline_std, "baseline", colors["baseline"])
plot_curve(qat_mean, qat_std, "qat", colors["qat"])
plot_curve(qs_mean, qs_std, "q_sentinel", colors["q_sentinel"])

ax.set_xlabel("Epoch")
ax.set_ylabel(r"Mean Fidelity Drift $(1 - F_t)$")
ax.set_title("Hilbert-Space Drift Across Training")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="upper right")

plt.tight_layout()
fig.savefig(OUT_DIR / "hilbert_fidelity_drift.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "hilbert_fidelity_drift.png", bbox_inches="tight")
plt.close(fig)
print(f"Hilbert drift figure saved to {OUT_DIR}")
