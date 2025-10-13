"""
Command-line experiment runner for baseline, QAT, and Quantum-Sentinel.
Saves histories, robustness metrics, training dynamics, and QC-poison scatter
to a writable output directory.
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from data_handler import create_real_data_loaders
from quantum_classical_hybrid import ModelConfig, create_hybrid_model
from qs_sentinel import (
    QATConfig,
    QSentinelConfig,
    QuantumAdversarialTrainer,
    QuantumSentinelTrainer,
    evaluate_model,
)

LOGGER = logging.getLogger(__name__)


def _setup_logging():  # optional nicer logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

EPS_GRID = (0.01, 0.02, 0.03, 0.04, 0.05, 0.10)
def run_baseline(loaders, model_cfg, device, epochs):
    model = create_hybrid_model(model_cfg)
    cfg = QATConfig(epochs=epochs)
    cfg.adv.enabled = False     # clean training
    trainer = QuantumAdversarialTrainer(model, cfg, device)
    history = trainer.train(loaders["train"], loaders["val"])
    metrics = evaluate_model(
        model,
        loaders["test"],
        epsilons=EPS_GRID,  # -> epsilon grid
        device=device,
        poison_cfg=None,
        train_loader=None,
        snapshot=None,
    )
    return history, metrics


def run_qat(loaders, model_cfg, device, epochs):
    model = create_hybrid_model(model_cfg)
    cfg = QATConfig(epochs=epochs)
    trainer = QuantumAdversarialTrainer(model, cfg, device)
    history = trainer.train(loaders["train"], loaders["val"])
    metrics = evaluate_model(
        model,
        loaders["test"],
        epsilons=EPS_GRID,
        device=device,
        poison_cfg=None,
        train_loader=None,
        snapshot=None,
    )
    return history, metrics


def run_qsentinel(loaders, model_cfg, device, epochs):
    model = create_hybrid_model(model_cfg)
    cfg = QSentinelConfig(epochs=epochs)
    trainer = QuantumSentinelTrainer(
        model,
        num_classes=model_cfg.num_classes,
        n_qubits=model_cfg.n_qubits,
        cfg=cfg,
        device=device,
        enable_qst=True,
        enable_qdbr=True,
    )
    history, snapshot = trainer.train(loaders["train"], loaders["val"])
    metrics = evaluate_model(
        model,
        loaders["test"],
        epsilons=EPS_GRID,
        device=device,
        poison_cfg=cfg.poison_eval,
        train_loader=loaders["train"],
        snapshot=snapshot,
    )
    return history, metrics


def save_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2))


def plot_training_dynamics(histories, out_dir: Path):
    plt.rcParams.update({
        "figure.dpi": 300,
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    })
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), dpi=300)

    colors = {"baseline": "#1f77b4", "qat": "#ff7f0e", "q_sentinel": "#2ca02c"}
    markers = {"baseline": "o", "qat": "s", "q_sentinel": "D"}

    for name, hist in histories.items():
        epochs = range(1, len(hist["train_loss"]) + 1)
        axes[0].plot(epochs, hist["train_loss"], color=colors[name], marker=markers[name], linestyle="-", label=f"{name} train")
        axes[0].plot(epochs, hist["val_loss"],   color=colors[name], marker=markers[name], linestyle="--", label=f"{name} val")
        axes[1].plot(epochs, [100*x for x in hist["train_acc"]], color=colors[name], marker=markers[name], linestyle="-", label=f"{name} train")
        axes[1].plot(epochs, [100*x for x in hist["val_acc"]],   color=colors[name], marker=markers[name], linestyle="--", label=f"{name} val")

    axes[0].set_title("Training vs. Validation Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    axes[1].set_title("Training vs. Validation Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    axes[0].legend(loc="upper right"); axes[1].legend(loc="lower right")
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "training_dynamics.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "training_dynamics.png", bbox_inches="tight")
    plt.close(fig)


def plot_qc_poison_scatter(df: pd.DataFrame, out_dir: Path):
    plt.rcParams.update({
        "figure.dpi": 300,
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    })
    records = []
    for scenario, row in df.iterrows():
        for col in row.index:
            if col.startswith("qc_poison_acc_eps"):
                eps = float(col.split("eps")[1])
                acc = float(row[col])
                linf = float(row.get(f"qc_poison_param_linf_eps{eps}", float("nan")))
                records.append({"scenario": scenario, "epsilon": eps, "acc": acc, "linf": linf})
    scatter_df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(5.0, 4.0), dpi=300)
    markers = {"baseline": "o", "qat": "s", "q_sentinel": "D"}
    colors = {"baseline": "#d62728", "qat": "#ff7f0e", "q_sentinel": "#1f77b4"}

    for scenario, group in scatter_df.groupby("scenario"):
        ax.scatter(group["linf"], [100*x for x in group["acc"]], marker=markers.get(scenario, "o"),
                   color=colors.get(scenario, "k"), s=60, label=scenario.upper())
        for _, row in group.iterrows():
            ax.annotate(f"{row['epsilon']:.2f}", (row["linf"], 100*row["acc"]), textcoords="offset points", xytext=(4, 4), fontsize=8)

    ax.set_xlabel(r"Mean $L_\infty$ Parameter Shift")
    ax.set_ylabel("QC-Poison Accuracy (%)")
    ax.set_title("QC-Poison: Accuracy vs. Parameter Drift")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left")

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "qc_poison_scatter.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "qc_poison_scatter.png", bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_vs_epsilon(results_df: pd.DataFrame, out_dir: Path):
    eps = (0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10)
    def curve(row, prefix):
        vals = [row[f"{prefix}_acc_eps{eps[i]}"] if i else row["clean_acc"] for i in range(len(eps))]
        return np.array(vals) * 100

    colors = {"baseline": "#1f77b4", "qat": "#ff7f0e", "q_sentinel": "#2ca02c"}
    markers = {"baseline": "o", "qat": "s", "q_sentinel": "D"}

    plt.rcParams.update({
        "figure.dpi": 300,
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    })
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    for scenario in results_df.index:
        row = results_df.loc[scenario]
        fgsm_curve = curve(row, "fgsm")
        pgd_curve = curve(row, "pgd")
        ax.plot(eps, fgsm_curve, color=colors[scenario], marker=markers[scenario], linestyle="-", label=f"{scenario.upper()} (FGSM)")
        ax.plot(eps, pgd_curve, color=colors[scenario], marker=markers[scenario], linestyle="--", label=f"{scenario.upper()} (PGD)")

    ax.set_xlabel(r"Epsilon ($\epsilon$)")
    ax.set_ylabel("Robust Accuracy (%)")
    ax.set_xlim(0.0, 0.10); ax.set_ylim(0, 100)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="lower left")
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "accuracy_vs_epsilon.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "accuracy_vs_epsilon.png", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output-dir", type=str, default="/reports_qs")
    parser.add_argument("--soi-dir", type=str, default="data/soi")
    parser.add_argument("--cwi-dir", type=str, default="data/cwi")
    args = parser.parse_args()

    _setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    loaders = create_real_data_loaders(
        soi_dir=args.soi_dir,
        cwi_dir=args.cwi_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        augment=True,
        balance_classes=True,
        num_workers=0,
    )

    model_cfg = ModelConfig(
        resnet_pretrained=False,
        feature_dim=256,
        n_qubits=6,
        n_layers=4,
        num_classes=2,
        input_channels=3,
        image_size=args.image_size,
        dropout_rate=0.3,
    )

    histories = {}
    results = {}

    LOGGER.info("Running baseline scenario")
    hist, metrics = run_baseline(loaders, model_cfg, device, args.epochs)
    histories["baseline"] = hist
    results["baseline"] = metrics

    LOGGER.info("Running QAT scenario")
    hist, metrics = run_qat(loaders, model_cfg, device, args.epochs)
    histories["qat"] = hist
    results["qat"] = metrics

    LOGGER.info("Running Q-Sentinel scenario")
    hist, metrics = run_qsentinel(loaders, model_cfg, device, args.epochs)
    histories["q_sentinel"] = hist
    results["q_sentinel"] = metrics

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, hist in histories.items():
        save_json(out_dir / f"{name}_history.json", hist)
    df = pd.DataFrame(results).T
    df.to_csv(out_dir / "robustness_metrics.csv")
    plot_training_dynamics(histories, out_dir)
    plot_qc_poison_scatter(df, out_dir)
    plot_accuracy_vs_epsilon(df, out_dir)
    LOGGER.info("Saved reports to %s", out_dir.resolve())


if __name__ == "__main__":
    main()
