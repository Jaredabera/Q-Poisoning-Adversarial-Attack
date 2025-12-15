# Q-SHIELD: Towards Adversarial Robustness for Quantum-Classical xApps in Intelligent O-RAN
This repository contains the research code, experiments, and visual assets for **Q-SHIELD**, a layered defense stack that hardens hybrid quantum–classical (QC) xApps against RF interference attacks in Open RAN (O-RAN) deployments. The project benchmarks clean training, adversarial training (QAT), and the proposed Q-SHIELD approach across gradient-based (QC-FGSM/PGD) and circuit-poisoning threats, demonstrating superior robustness with minimal loss in clean accuracy.

> 🔬 **Core idea:** Combine quantum state tomography (QST), fidelity-aware drift-balancing regularization (Q-DBR), and sentinel snapshots with cosine-drift auto-repair to maintain a wide quantum decision margin under adaptive adversaries.

---
<img width="1111" height="504" alt="Q-SHIELD github" src="https://github.com/user-attachments/assets/e75c85e4-3a0d-44c9-9539-a061007b16bc" />


## Highlights
- **Hybrid QC xApp** built with PyTorch, fusing classical ResNet feature extractors and 6-qubit variational circuits (`quantum_classical_hybrid.py`).
- **Defense suite** including baseline, QAT, and Q-SHIELD trainers with configurable adversaries (see `qs_sentinel.py`, `defense_utils.py`).
- **Reproducible pipeline** (`qsentinel_train_eval.py`) that trains each scenario, evaluates robustness across an epsilon grid, and generates publication-ready plots/tables.
- **Comprehensive visualization assets** in `figures/`, `spectrogram-dataset/soi/`, and `reports_qs/` for inclusion in papers, slides, and posters.
---


## Repository Layout

```
.
├── data/                         # Clean SOI/CWI spectrogram data (placeholder paths)
├── quantum_classical_hybrid.py   # Hybrid model definition + configuration dataclass
├── data_handler.py               # Data loader utilities and augmentations
├── defense_utils.py              # Adversarial example generation + Q-DBR objective
├── qs_sentinel.py                # Trainers, configs, evaluation helpers
├── qsentinel_train_eval.py       # End-to-end experiment runner + reporting
├── figures/                      # Baseline learning curve and other figures
├── spectrogram-dataset/soi/      # Robustness plots (accuracy vs ε, ASR, etc.)
├── reports_qs/                   # Generated CSV/plot outputs (after running pipeline)
├── env.yml / requirements.txt    # Conda + pip dependency specifications
└── README.md                     # This overview
```

---

## Environment Setup

```bash
# 1) Clone your fork
git clone https://github.com/<user>/Q-SHIELD.git
cd Q-SHIELD

# 2) Create the environment (preferred: Conda)
conda env create -f env.yml
conda activate qsentinel

# 3) Alternatively (pip)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The experiments expect access to spectrogram datasets for signal-of-interest (SOI) and co-channel interference (CWI). Update the default paths in `qsentinel_train_eval.py` or provide `--soi-dir` and `--cwi-dir` on the CLI.

---

## Quickstart: Reproduce the Benchmark

```bash
python qsentinel_train_eval.py \
    --epochs 25 \
    --batch-size 16 \
    --image-size 224 \
    --soi-dir data/soi \
    --cwi-dir data/cwi \
    --output-dir reports_qs
```

This command:
- Trains the **baseline**, **QAT**, and **Q-SHIELD** models on the provided spectrogram data.
- Evaluates clean accuracy plus QC-FGSM, QC-PGD, and QC-Poison worst-case accuracy/ASR across ε ∈ {0.01,…,0.10}.
- Exports JSON histories, `robustness_metrics.csv`, and plots (`training_dynamics.*`, `accuracy_vs_epsilon.*`, `qc_poison_scatter.*`) into `reports_qs/`.

Use the saved `robustness_metrics.csv` for tables or load it into notebooks (`QuantumHybrid_Train.ipynb`, `quantum_classical_hybrid.ipynb`) for deeper analysis.

## Key Results

| Scenario | Clean Acc. | FGSM Acc<sub>min</sub> | FGSM ASR | PGD Acc<sub>min</sub> | PGD ASR | QC-Poison Acc<sub>min</sub> / ‖Δθ‖<sub>∞</sub> |
|----------|------------|------------------------|----------|------------------------|---------|-----------------------------------------------|
| Baseline | 0.982 | 0.558 | 0.442 | **0.160** | 0.639 | **0.708** / 3.1×10⁻² |
| QAT (PGD)| 0.948 | 0.804 | 0.196 | 0.692 | 0.308 | 0.611 / 2.6×10⁻² |
| **Q-SHIELD** | **0.974** | **0.847** | **0.153** | **0.736** | **0.264** | **0.708 / 1.8×10⁻²** |

- Q-SHIELD narrows the worst-case gap by ≈4–5 percentage points beyond QAT while halving Hilbert-space drift under QC-Poison.
- Clean accuracy remains near 98%, demonstrating that robustness does not compromise nominal performance.
- Figures `spectrogram-dataset/soi/naccuracy_vs_epsilon_ieee.pdf` and `spectrogram-dataset/soi/asr_qcpoison_comparison.pdf` visualize the collapse of undefended models versus the flat response of Q-SHIELD.

---

## Citing This Work


# Quantum State Poisoning-Adversarial Attacks 
## Overview
This project implements a quantum state poisoning algorithm designed to explore adversarial attacks in quantum machine learning using the **PennyLane** framework. The algorithm modifies quantum states through angle phase-shifting, demonstrating how small perturbations can impact the performance of quantum models.
## AE Structure and VQC Layer
![QI-attacks1-qtn_Poisoning 1-1](https://github.com/user-attachments/assets/d9249b24-77d0-4c7c-8f2e-798c220fc8b7)

## Features
- **Quantum Circuit Implementation**: Utilizes PennyLane for customizable quantum circuit design.
- **Angle Phase-Shifting**: Applies perturbations to rotation angles to create adversarial states.
- **Fidelity Evaluation**: Computes fidelity between original and adversarial states to assess attack effectiveness.
- **Dynamic Perturbation Adjustment**: Updates perturbations based on fidelity thresholds to refine attacks.
- The core perturbation is modeled as a quantum phase transition.
  ![VQc-1](https://github.com/user-attachments/assets/41f55af2-0274-4da5-bf6e-9a2a2869431a)
## Algorithm
The core algorithm is structured as follows:
1. **Initialization**: Prepares the initial quantum state.
2. **Layer and Qubit Iteration**: Iterates through each layer and qubit to apply perturbations.
3. **Adversarial Angle Calculation**: Modifies angles to create adversarial perturbations.
4. **State Update and Fidelity Calculation**: Updates the quantum state and computes fidelity.
5. **Perturbation Adjustment**: Dynamically adjusts perturbations based on fidelity results.

Refer to the algorithm section in the code for detailed implementation.

The dataset can be found here: https://www.nextgwirelesslab.org/datasets
## Requirements
- Python 3.8.20
- PennyLane
- Tensorflow

## Installation
Clone the repository and install the required packages:
```bash
git clone https://github.com/Jaredabera/quantum-state-poisoning.git
```
There are 2 conda environment discription files for easily reconstructing the dependencies.
```bash
conda env create -f <env-file>
```
This technique opens new avenues for:

- Studying adversarial robustness in quantum-classical hybrid systems
- Understanding the relationship between quantum phenomena and adversarial examples
- Developing quantum-hybrid defenses against classical attacks
## Citing This Work
BibTeX: "@INPROCEEDINGS{11140504,
  author={Ergu, Yared Abera and Nguyen, Van-Linh and Lin, Po-Ching and Hwang, Ren-Hung},
  booktitle={2025 IEEE International Conference on Machine Learning for Communication and Networking (ICMLCN)}, 
  title={Q-Poison: Quantum Adversarial Attacks against QML-driven Interference Classification in O-RAN}, 
  year={2025},
  volume={},
  number={},
  pages={1-6},
  keywords={Quantum computing;Accuracy;Systematics;Perturbation methods;Open RAN;Interference;Computer architecture;Logic gates;Quantum state;Stability analysis;Hybrid quantum-classical networks;adversarial attacks;Q-Poison;Open Radio Access Networks;classification},
  doi={10.1109/ICMLCN64995.2025.11140504}}"
