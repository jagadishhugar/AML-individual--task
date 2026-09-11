# Iris Flower Classification Using Machine Learning

## 📌 Project Overview

This project focuses on the classification of iris flowers into three different species using machine learning algorithms. The **Iris Flower Dataset** is used to train and evaluate three supervised machine learning classification algorithms:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)

The main objective of this project is to compare the performance of these algorithms using suitable evaluation metrics and identify the best-performing algorithm.

---

## 🎯 Objectives

The major objectives of this project are:

- To understand and analyze the Iris dataset.
- To perform basic data preprocessing.
- To explore the dataset using statistical analysis and visualization.
- To divide the dataset into training and testing sets.
- To train three different machine learning classification models.
- To evaluate the models using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix
- To compare the performance of the three algorithms.
- To identify the best-performing machine learning algorithm.

---

## 📊 Dataset

The project uses the **Iris Flower Dataset**.

The dataset contains **150 observations** belonging to three species of iris flowers:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

Each observation contains four numerical features:

| Feature | Description |
|---|---|
| Sepal Length | Length of the sepal |
| Sepal Width | Width of the sepal |
| Petal Length | Length of the petal |
| Petal Width | Width of the petal |

### Dataset Statistics

- Total samples: **150**
- Number of features: **4**
- Number of classes: **3**
- Samples per class: **50**
- Training samples: **120**
- Testing samples: **30**

---

## 🧠 Machine Learning Algorithms

### 1. Logistic Regression

Logistic Regression is a supervised classification algorithm that estimates the probability of an observation belonging to a particular class.

In this project, Logistic Regression uses the four flower measurements to predict the species of the iris flower.

**Advantages:**

- Simple and easy to understand
- Computationally efficient
- Performs well on linearly separable data
- Provides a strong baseline classification model

---

### 2. Decision Tree

Decision Tree is a supervised learning algorithm that makes predictions using a sequence of decision rules.

The algorithm divides the dataset based on feature values until it reaches a suitable classification decision.

**Advantages:**

- Easy to understand
- Easy to visualize
- Does not require feature scaling
- Can capture nonlinear relationships

---

### 3. K-Nearest Neighbors (KNN)

K-Nearest Neighbors is an instance-based classification algorithm. It predicts the class of a new observation based on the classes of its nearest training observations.

In this project, KNN uses the standardized feature values and `k = 5` neighbors.

**Advantages:**

- Simple to implement
- Effective for small datasets
- Does not require complex model assumptions
- Performs well when similar observations belong to the same class

---

## 🔄 Project Workflow

```text
                 Iris Dataset
                      |
                      ↓
              Data Inspection
                      |
                      ↓
             Data Preprocessing
                      |
                      ↓
       Exploratory Data Analysis
                      |
                      ↓
             Train-Test Split
                      |
                      ↓
              Feature Scaling
                      |
          ┌───────────┼───────────┐
          ↓           ↓           ↓
     Logistic      Decision      KNN
     Regression      Tree
          ↓           ↓           ↓
          └───────────┼───────────┘
                      ↓
               Model Prediction
                      |
                      ↓
             Model Evaluation
                      |
                      ↓
             Performance Comparison
                      |
                      ↓
          Best Model Identification
```

---

## ⚙️ Data Preprocessing

The following preprocessing steps are performed:

1. Load the Iris dataset.
2. Inspect the dataset structure.
3. Check the number of samples and features.
4. Separate input features and target variable.
5. Split the data into training and testing sets.
6. Apply feature scaling for Logistic Regression and KNN.
7. Keep the test dataset separate for final evaluation.

The dataset is divided using an **80:20 train-test split**.

```text
Training Data: 80% = 120 samples
Testing Data:  20% = 30 samples
```

---

## 📈 Exploratory Data Analysis

Exploratory Data Analysis is performed to understand the characteristics of the dataset.

The analysis includes:

- Dataset shape
- Feature statistics
- Feature distributions
- Relationship between features
- Class distribution
- Feature correlation
- Visualization of the dataset

Petal length and petal width are particularly useful features for distinguishing the different iris species.

---

## 🧪 Model Evaluation

The models are evaluated using the following metrics:

### Accuracy

Measures the percentage of correctly classified observations.

### Precision

Measures how many observations predicted as a particular class actually belong to that class.

### Recall

Measures how many observations belonging to a particular class were correctly identified.

### F1-Score

Provides a combined measure of precision and recall.

### Confusion Matrix

Shows the number of correct and incorrect predictions for each class.

---

## 📊 Performance Comparison

The three algorithms are compared using the same testing dataset.

### Example Results

| Algorithm | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 96.67% | 96.83% | 96.67% | 96.67% |
| Decision Tree | 93.33% | 94.44% | 93.33% | 93.33% |
| KNN | **100.00%** | **100.00%** | **100.00%** | **100.00%** |

> **Note:** The exact values may vary depending on the train-test split, preprocessing, and model parameters. The values generated by the notebook should be used as the final project results.

---

## 🏆 Best-Performing Algorithm

Based on the example experimental results, **K-Nearest Neighbors (KNN)** is the best-performing algorithm.

KNN achieved:

```text
Accuracy  : 100%
Precision : 100%
Recall    : 100%
F1-Score  : 100%
```

The strong performance of KNN can be attributed to the relatively clear separation between the iris species in the feature space.

However, the best algorithm should ultimately be selected using the actual results produced when the notebook is executed.

---

## 🛠️ Technologies Used

The project is implemented using Python.

### Programming Language

- Python 3

### Libraries

- NumPy
- Pandas
- Matplotlib
- Scikit-learn

### Development Environment

- Jupyter Notebook
- Google Colab / JupyterLab

---

## 📁 Project Structure

```text
Iris-ML-Classification/
│
├── Iris_ML_Algorithm_Comparison.ipynb
│
├── README.md
│
└── results/
    └── model_comparison_results.csv
```

---

## ▶️ How to Run the Project

### Step 1: Install Python

Make sure Python 3 is installed on your system.

### Step 2: Install Required Libraries

Open a terminal or command prompt and run:

```bash
pip install numpy pandas matplotlib scikit-learn
```

### Step 3: Open Jupyter Notebook

Run:

```bash
jupyter notebook
```

### Step 4: Open the Notebook

Open:

```text
Iris_ML_Algorithm_Comparison.ipynb
```

### Step 5: Run the Cells

Run the notebook cells sequentially from top to bottom.

The notebook will:

- Load the dataset
- Preprocess the data
- Train the models
- Generate predictions
- Calculate evaluation metrics
- Display confusion matrices
- Generate the accuracy comparison graph
- Identify the best-performing algorithm

---

## 🔍 Expected Output

After running the notebook, the following outputs will be generated:

```text
Dataset Shape: (150, 4)

Training samples: 120
Testing samples: 30

Feature scaling completed successfully.

All three models trained successfully.

Predictions generated successfully.
```

The notebook will also display:

- Model performance comparison table
- Confusion matrices
- Classification reports
- Accuracy comparison graph
- Best-performing algorithm

---

## 📌 Results Summary

The experiment demonstrates that all three machine learning algorithms can successfully classify iris flowers.

The general performance ranking in the example experiment is:

```text
1. KNN                 → 100.00%
2. Logistic Regression → 96.67%
3. Decision Tree       → 93.33%
```

KNN provides the highest classification performance for the selected train-test split.

---

## 🔮 Future Improvements

The project can be extended in several ways:

- Apply cross-validation.
- Perform hyperparameter tuning.
- Test additional algorithms such as:
  - Random Forest
  - Support Vector Machine
  - Naive Bayes
  - Gradient Boosting
- Compare training and prediction time.
- Perform feature selection.
- Use a larger real-world classification dataset.
- Deploy the best-performing model as a web application.
- Create an interactive prediction interface.

---

## 📚 References

1. Fisher, R. A. (1936). *The Use of Multiple Measurements in Taxonomic Problems*. Annals of Eugenics.

2. Géron, A. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly Media.

3. James, G., Witten, D., Hastie, T., & Tibshirani, R. *An Introduction to Statistical Learning*. Springer.

4. Scikit-learn documentation and machine learning library resources.

5. Python documentation.

---

## 👨‍💻 Project Type

**Individual Machine Learning Project**

### Problem Type

**Supervised Multi-Class Classification**

### Dataset

**Iris Flower Dataset**

### Algorithms

**Logistic Regression | Decision Tree | KNN**

### Best Model

**K-Nearest Neighbors (KNN)**

---

## ✅ Conclusion

This project successfully demonstrates the implementation and comparison of three supervised machine learning classification algorithms using the Iris Flower Dataset. The models were trained using the same dataset and evaluated using multiple performance metrics.

The experiment shows that machine learning algorithms can effectively classify iris flowers based on sepal and petal measurements. Among the tested algorithms, KNN achieved the best performance in the example experiment.

The project provides practical experience with the complete machine learning workflow, including data preprocessing, exploratory data analysis, model training, prediction, evaluation, performance comparison, and model selection.