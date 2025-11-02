# Advanced Dimensionality Reduction

_Course: Advanced Data Analysis and Machine Learning (ADAML)_
_Author: Chamath Wijerathne_
_Date: 02-11-2025_

## 🎯 Overview

This assignment explores and compares **linear** and **non-linear** dimensionality-reduction methods, and applies a **Self-Organizing Map (SOM)** for visualizing high-dimensional data.

The work is divided into two major tasks:

1. **Task 1 – Comparing Linear and Non-Linear Dimensionality Reduction**

   - Methods: PCA vs t-SNE
   - Dataset: Bike Sharing Rental (OpenML #42712)
   - Includes a regression model (MLP Regressor) to compare predictive performance after reduction.

2. **Task 2 – Visualizing with Self-Organizing Map (SOM)**
   - Dataset: MNIST-784 handwritten digits (OpenML #554)
   - Uses MiniSom to train a 20×20 map, visualize U-Matrix, and analyze digit topology.

## 📂 Folder Contents

```
Advanced_Dimensionality_Reduction/
├── Compare_PCA_and_t-SNE.ipynb # Task 1 notebook
├── handwritten_digits_with_SOM.ipynb # Task 2 notebook
└── README.md # this file
```

## Task 1 – Comparing Linear and Non-Linear Dimensionality Reduction

### Objective

To compare how **PCA (Principal Component Analysis)** and **t-SNE (t-Distributed Stochastic Neighbor Embedding)** represent the structure of the Bike Sharing dataset, and to evaluate how these representations affect a downstream regression model predicting total bike rentals (`count`).

### Methods & Workflow

1. **Data Loading & Inspection**

   - `dataset.arff` (Bike Sharing Rental) from OpenML #42712.
   - Target variable: `count`.

2. **Exploratory Data Analysis**

   - Summary statistics, correlation heatmap, target distribution.

3. **Pre-processing**

   - **Numeric features:** standardized with `StandardScaler`.
   - **Cyclical features:** `month` and `hour` encoded using sine & cosine transformations to preserve circular continuity.
   - **Categorical features:** one-hot encoded (`year`, `season`, `holiday`, `workingday`, `weather`).
   - Implemented via `ColumnTransformer`.

4. **Linear DR – PCA**

   - Reduced to 2 components for visualization and 95 % variance version for MLP.
   - Explained variance ratio ≈ 40 %.
   - Scatter plot colored by `count`.

5. **Non-Linear DR – t-SNE**

   - Applied on standardized features.
   - Parameters: `perplexity = 35`, `learning_rate = 200`, `random_state = 42`.
   - 2-D embedding visualized with Seaborn.

6. **Prediction Model (MLP Regressor)**

   - Trained on:  
     a. Raw standardized data  
     b. PCA-reduced data (95 % variance)  
     c. t-SNE embedding
   - Evaluated using R², RMSE, and MAE.

7. **Comparison & Interpretation**
   - PCA captures global variance and retains predictive information.
   - t-SNE preserves local neighborhood structure for visualization but is less suitable for regression.

### Key Results

| Model            | R² (Test) | RMSE (Test) | MAE (Test) | Observation                                           |
| :--------------- | :-------: | :---------: | :--------: | :---------------------------------------------------- |
| MLP • Raw Data   |  ≈ 0.999  |   ≈ 0.20    |   ≈ 0.15   | Best overall performance.                             |
| MLP • PCA (95 %) | ≈ 0.9999  |   ≈ 0.74    |   ≈ 0.51   | Slight drop after dim. reduction.                     |
| MLP • t-SNE      |  ≈ 0.67   |    ≈ 101    |    ≈ 76    | Preserves local clusters but not regression accuracy. |

### Conclusions

- **PCA** is effective for compression and predictive tasks (global structure).
- **t-SNE** is ideal for visualizing local patterns (non-linear structure).
- Proper feature scaling and cyclic encoding significantly improve interpretability.

---

## Task 2 – Visualizing with Self-Organizing Map (SOM)

### Objective

To use a Self-Organizing Map to project the 784-dimensional MNIST handwritten digits dataset onto a 2-D grid and visualize the topological relationships among digits.

### Methods & Workflow

1. **Data Loading**

   - `mnist_784` from OpenML #554 (downloaded automatically).
   - 10 000 samples used for efficient training.
   - Pixel values normalized to [0, 1].

2. **Training the SOM**

   - Library: `MiniSom`.
   - Grid size: 20 × 20 neurons (400 nodes).
   - Parameters: `σ = 1.0`, `learning_rate = 0.3`, `iterations = 25 000`.

3. **Visualizations**

   - **U-Matrix:** Euclidean distance map showing cluster boundaries.
   - **Class Map:** dominant digit labels with numbers overlayed.
   - **Prototype Visualization:** neuron weight vectors as 28×28 images.

4. **Interpretation**
   - Bright lines in the U-Matrix represent high dissimilarity between clusters.
   - Neighboring neurons encode visually similar digits (e.g., 3 ↔ 8, 4 ↔ 9).
   - SOM learns a topological ordering that maps continuous variations in digit shape.

### Key Results

- **U-Matrix:** clearly separated regions for different digit groups.
- **Class Map:** distinct colored zones for digits 0–9 with smooth transitions.
- **Prototypes:** show representative digit patterns learned by neurons.

### Conclusions

- SOM effectively reduces 784-D input to a 2-D grid while preserving neighborhood structure.
- It provides interpretable visualizations of clusters and feature continuity.
- Compared to PCA and t-SNE, SOM offers a topologically ordered, prototype-based view of non-linear data.

---

## Environment Setup

Tested with:

- Python 3.10 +
- Jupyter Notebook / VS Code

### Required Packages

```bash
pip install numpy pandas scikit-learn matplotlib seaborn minisom scipy
```
