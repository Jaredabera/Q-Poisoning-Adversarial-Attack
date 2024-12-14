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
![updated_psq_with_all-1](https://github.com/user-attachments/assets/39a0d10e-4005-47b6-a383-8cd9c26396ac)
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
