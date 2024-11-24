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
![updated_psq_with_all-1](https://github.com/user-attachments/assets/39a0d10e-4005-47b6-a383-8cd9c26396ac)
## Algorithm
The core algorithm is structured as follows:
1. **Initialization**: Prepares the initial quantum state.
2. **Layer and Qubit Iteration**: Iterates through each layer and qubit to apply perturbations.
3. **Adversarial Angle Calculation**: Modifies angles to create adversarial perturbations.
4. **State Update and Fidelity Calculation**: Updates the quantum state and computes fidelity.
5. **Perturbation Adjustment**: Dynamically adjusts perturbations based on fidelity results.

Refer to the algorithm section in the code for detailed implementation.
"""
Quantum State Poisoning (QSP) Attack Implementation

This module implements a novel quantum state poisoning attack against quantum-classical hybrid
machine learning models. The attack perturbs input features by simulating quantum phase shifts
and non-linear transformations to generate adversarial examples.

Author: [Your Name]
License: MIT
Version: 1.0.0
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def quantum_state_poisoning_attack(
    model: torch.nn.Module,
    X_batch: torch.Tensor,
    y_batch: torch.Tensor,
    epsilon: float = 0.1,
    max_depth: int = 5
) -> torch.Tensor:
    """
    Performs a quantum state poisoning attack on the input data.
    
    This attack generates adversarial examples by applying quantum-inspired perturbations
    to the input features. The perturbations are based on quantum phase shifts and
    non-linear transformations, with adaptive adjustment based on model response.
    
    Args:
        model (torch.nn.Module): Target model to attack
        X_batch (torch.Tensor): Input batch of shape (batch_size, input_size)
        y_batch (torch.Tensor): Target labels
        epsilon (float, optional): Initial perturbation magnitude. Defaults to 0.1
        max_depth (int, optional): Maximum number of quantum circuit layers. Defaults to 5
        
    Returns:
        torch.Tensor: Adversarially perturbed input batch, clamped to [0,1]
        
    Example:
        >>> model = QuantumNeuralNetwork()
        >>> X_batch = torch.rand(32, 10)
        >>> y_batch = torch.randint(0, 2, (32,))
        >>> X_adv = quantum_state_poisoning_attack(model, X_batch, y_batch)
    """
    # Initialize adversarial examples
    X_batch_adv = X_batch.clone().detach().requires_grad_(True)
    batch_size, input_size = X_batch.shape
    
    # Iterate through quantum circuit layers
    for layer in range(max_depth):
        for qubit in range(input_size):
            # Calculate phase-shifting perturbations
            theta = np.pi * layer / max_depth
            phi = 2 * np.pi * qubit / input_size
            
            # Compute non-linear perturbation transformations
            theta_adv = theta + epsilon * np.sin(theta) + (epsilon**2) * (np.sin(theta)**2)
            phi_adv = phi + epsilon * np.cos(phi) + (epsilon**2) * (np.cos(phi)**2)
            
            # Apply quantum-inspired perturbation to specific feature/qubit
            X_batch_adv[:, qubit] += epsilon * (np.sin(theta_adv) + np.cos(phi_adv))
        
        # Adaptive perturbation adjustment based on model response
        with torch.no_grad():
            output = model(X_batch_adv).squeeze(1)
            loss = F.binary_cross_entropy(output, y_batch)
            
            # Increase perturbation if loss is too small
            if loss > epsilon:
                epsilon *= 1.1
    
    return torch.clamp(X_batch_adv.detach(), 0, 1)
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
