# Customer Churn Prediction using Neural Networks


A binary classification project that predicts whether a bank customer will leave (churn) using a fully connected **Artificial Neural Network (ANN)** built with TensorFlow and Keras.

---

## Problem Statement

Banks lose revenue when customers close their accounts. This model helps predict which customers are likely to churn so the bank can take proactive action to retain them.

---

## Dataset

**Churn Modelling Dataset** — [Kaggle](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)

- 10,000 customer records
- 14 features (Geography, Age, Balance, Credit Score, etc.)
- Target: `Exited` (1 = churned, 0 = stayed)

---

## Model Architecture

```
Input Layer  →  11 features
Hidden Layer 1  →  11 neurons, Sigmoid activation
Hidden Layer 2  →  11 neurons, Sigmoid activation
Output Layer  →  1 neuron, Sigmoid activation (binary output)
```

- **Loss:** Binary Crossentropy
- **Optimizer:** Adam
- **Epochs:** 100
- **Batch Size:** 50

---

## Workflow

1. Load and explore dataset
2. Drop irrelevant columns (`RowNumber`, `CustomerId`, `Surname`)
3. One-hot encode categorical columns (`Geography`, `Gender`)
4. Train/Test split (80/20)
5. Feature scaling with `StandardScaler`
6. Build and train ANN
7. Evaluate with accuracy score
8. Plot training loss and accuracy curves

---

## Results

| Metric | Value |
|--------|-------|
| Accuracy | ~86% |

---

## Tech Stack

- Python 3.x
- TensorFlow / Keras
- Pandas, NumPy
- Scikit-learn
- Matplotlib

---

## How to Run

```bash
pip install tensorflow pandas scikit-learn matplotlib
jupyter notebook
```

Open `Customer_Churn_pred_using_neural_networks.ipynb` and run all cells.

Or run directly on **Google Colab** — no setup needed.

---

*Part of my Deep Learning series — [github.com/FawadAhmad-bilal](https://github.com/FawadAhmad-bilal)*
