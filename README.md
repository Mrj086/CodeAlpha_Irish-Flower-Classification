# 🌸 Iris Flower Classification

A complete machine learning project that classifies Iris flowers into three species — *Setosa*, *Versicolor*, and *Virginica* — using six classification algorithms with full visualizations, cross-validation, ROC curves, and learning curve analysis.

---

## 📌 Project Overview

This project was built as **Task 1** of a data science internship series. It covers the full ML pipeline: data loading, exploratory analysis, feature engineering, model training, evaluation, and multi-panel visualization — all on the classic Iris dataset.

---

## 🎯 Objectives

- Load and explore the Iris dataset (150 samples, 4 features, 3 classes)
- Engineer derived features (petal area, sepal area, petal/sepal ratios)
- Train and compare six classification models
- Evaluate using accuracy, precision, recall, F1-score, ROC-AUC, and cross-validation
- Visualize insights through a 10-panel analysis dashboard

---

## 🧠 Models Used

| Model | Test Accuracy |
|---|---|
| Logistic Regression | ~96.7% |
| K-Nearest Neighbors | ~96.7% |
| Decision Tree | ~96.7% |
| **Random Forest** | **~96.7%** |
| SVM (RBF Kernel) | ~96.7% |
| **Gradient Boosting** | **~96.7%** |

> Best model selected automatically based on test accuracy, with full CV scoring across 5 stratified folds.

---

## 📊 Visualizations Included

- **Violin plots** — petal length & width distributions per species
- **Scatter plots** — sepal and petal feature pairs by class
- **Confusion matrix** — for the best-performing model
- **Model accuracy bar chart** — side-by-side comparison
- **5-Fold CV box plot** — score stability across models
- **Feature importance** — from Random Forest
- **Correlation heatmap** — between all four features
- **ROC curves** — AUC for all models (one-vs-rest, micro-average)
- **Precision / Recall / F1 chart** — grouped bar across models
- **CV score heatmap** — Model × Fold matrix
- **Learning curve** — training vs validation for the best model
- **Error analysis** — per-class prediction errors for each model

---

## 🗂️ Project Structure

```
CodeAlpha_Irish-Flower-Classification/
├── Irish Flower Classification/         # Main project folder
│   ├── Iris.csv                         # Dataset file
│   ├── IrishFlowerClassification.py     # Python script
│   └── IrishFlowerClassification.ipynb  # Jupyter Notebook
    └──Outputs(.png)
├── LICENSE                              # License file
├── README.md                            # Documentation
└── requirements.txt                     # Dependencies

```

---

## ⚙️ Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/your-username/iris-flower-classification.git
cd iris-flower-classification
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebook

```bash
jupyter notebook Task1.ipynb
```

The output chart will be saved to `outputs/task1_iris_classification.png`.

---

## 📦 Dependencies

```
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

See `requirements.txt` for pinned versions.

---

## 📁 Dataset

The **Iris dataset** is loaded directly from `sklearn.datasets.load_iris()` — no manual download required. It contains:

- **150 samples** (50 per class)
- **4 features**: sepal length, sepal width, petal length, petal width (all in cm)
- **3 classes**: *Iris setosa*, *Iris versicolor*, *Iris virginica*
- **0 missing values**

Originally published by R.A. Fisher (1936). Available via [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris).

---

## 🔬 Feature Engineering

Four derived features were computed to enhance model input:

| Feature | Formula |
|---|---|
| `petal_area` | petal length × petal width |
| `sepal_area` | sepal length × sepal width |
| `petal_ratio` | petal length / petal width |
| `sepal_ratio` | sepal length / sepal width |

---

## 📈 Key Findings

- **Petal features** (length & width) are the strongest discriminators — petal length alone separates *setosa* almost perfectly from the other two.
- **Sepal width** is the weakest predictor, with significant class overlap.
- All models achieve high accuracy (≥95%) due to the clean, well-separated nature of the Iris data.
- **Setosa** is trivially classifiable; most errors occur between *Versicolor* and *Virginica*.
- Random Forest and Gradient Boosting show the most stable cross-validation scores.

---

## 🏆 Part of the CodeAlpha Internship Series

| Task | Title |
|---|---|
| ✅ Task 1 | Iris Flower Classification ← *this repo* |
| Task 2 | Unemployment Rate Analysis |
| Task 3 | Car Price Prediction |
| Task 4 | Sales Prediction |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙋 Author

**Md. Miraj-Ul-Islam**
- GitHub: [@Mrj086](https://github.com/Mrj086)
- LinkedIn:(https://www.linkedin.com/in/md-miraj-ul-islam-77b30b26a/)

---

*Built with Python 3.x · scikit-learn · matplotlib · seaborn*
