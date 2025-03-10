# Molecular Graph Neural Network Training Module

This repository contains a comprehensive training module for a graph-based neural network designed for molecular graph data. The module supports data loading, model training, hyperparameter optimization, evaluation, and visualization.

## Features

- **DSSA Model**: Implements `DSSA` for SA scores data.
- **Training Pipeline**: Includes k-fold cross-validation, early stopping, and learning rate scheduling.
- **Custom Loss Functions**: Supports `WeightedMSELoss`, `FocalLoss`, and `CombinedLoss` (BCE + Focal Loss).
- **Data Augmentation**: Subgraph augmentation, node/edge dropout, and feature masking.
- **Hyperparameter Optimization**: Uses Optuna for tuning learning rate, dropout, and weight decay.
- **Evaluation Metrics**: Computes accuracy, precision, recall, F1, ROC-AUC, PR-AUC, sensitivity, specificity, MSE, MAE, and R2.
- **Visualization**: Generates ROC curves, precision-recall curves, confusion matrices, and target distribution plots.
- **Memory Optimization**: Efficient GPU memory management and data prefetching.
- **Ensemble Learning**: Combines models from k-fold cross-validation for improved performance.
- **Error Handling**: Robust debugging tools for data batches, model forward passes, and SMILES integration.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mrmoon900/DSSA.git
   cd molecular-gnn-training


Install dependencies:

bash
Copy
pip install -r requirements.txt
Usage
To train the model:

bash
Copy
python train.py
To evaluate the model:

bash
Copy
python test.py
License
This project is licensed under the MIT License. See LICENSE for details.
