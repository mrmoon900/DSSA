
# DSSA: Dual Stream Synthetic Accessibility Prediction

This repository contains the implementation of a dDual Stream Synthetic Accessibility .

- **DSSA Model**: Implements modle architecture.
- **Data Preprocessing**: Tools for loading and preprocessing molecular datasets.
- **Training Pipeline**: Includes k-fold cross-validation, early stopping, and learning rate scheduling.
- **Evaluation Metrics**: Computes accuracy, precision, recall, F1-score, ROC-AUC, and PR-AUC.
- **Visualization**: Generates ROC curves, precision-recall curves, and confusion matrices.
- **Hyperparameter Optimization**: Supports Optuna for tuning hyperparameters.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mrmoon900/DSSA.git
   cd DSSA
   ```

2. **Set up a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install PyTorch and PyTorch Geometric**:
   Follow the official installation instructions for [PyTorch](https://pytorch.org/get-started/locally/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

## Usage

### Training the Model

To train the model, run:
```bash
python train.py --num-epochs 100 --learning-rate 0.001 --batch-size 256
```

### Hyperparameter Optimization

To perform hyperparameter optimization using Optuna, use the `--hp-search` flag:
```bash
python train.py --hp-search
```

### Evaluating the Model

After training, evaluate the model on the test set:
```bash
python test.py
```

### Visualizing Results

The following visualizations are generated and saved in the `results` directory:
- **Confusion Matrices**
- **ROC Curves**
- **Precision-Recall Curves**
- **Target Distribution Plots**

## Directory Structure

```
DSSA/
├── data/                    # Directory for datasets
├── models/                  # Saved models
├── results/                 # Training results and visualizations
├── src/                     # Source code
│   ├── data_preprocess.py   # Data loading and preprocessing
│   ├── molecular_model.py   # Model definition
│   ├── train.py             # Training script
│   └── utils.py             # Utility functions
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── LICENSE                  # License file
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project uses [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/).
- Inspired by various works in molecular graph neural networks.
---
---

### **`requirements.txt`**

Here’s a sample `requirements.txt` file listing the dependencies for your project:

```plaintext
torch>=2.0.0
torch-geometric>=2.3.0
torch-scatter>=2.1.0
torch-sparse>=0.6.16
torch-cluster>=1.6.0
torch-spline-conv>=1.2.0
numpy>=1.21.0
scikit-learn>=1.0.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.62.0
optuna>=3.0.0
rdkit>=2022.03.0
```

---

### **How to Use These Files**

1. **Save the `README.md`**:
   - Copy the content of the `README.md` above and save it in your project's root directory.

2. **Save the `requirements.txt`**:
   - Copy the content of the `requirements.txt` above and save it in your project's root directory.

3. **Install Dependencies**:
   - Run the following command to install the dependencies:
     ```bash
     pip install -r requirements.txt
     ```

4. **Push to GitHub**:
   - Add, commit, and push the files to your GitHub repository:
     ```bash
     git add README.md requirements.txt
     git commit -m "Add README and requirements file"
     git push origin main
     ```

---

### **Key Notes**
- The `requirements.txt` file ensures that users can easily install all necessary dependencies.
- The `README.md` file provides a clear and concise overview of your project, making it easier for others to understand and use.

Let me know if you need further assistance!
