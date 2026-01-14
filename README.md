# Machine Learning Final Term – Telkom University

This repository contains my work for the **Final Term Examination (UAS)** of the  
**Machine Learning** course at **Telkom University**.

The project focuses on building complete **end-to-end Machine Learning and Deep Learning pipelines**, following the official instructions provided by the lecturer and teaching assistants.

---

## 👩‍🎓 Student Information

- **Name:** Agatha Kinanthi Pramdriswara Truly Amorta  
- **NIM:** 1103223212  
- **Program:** Computer Engineering  
- **Institution:** Telkom University  

---

## 🎯 Project Objectives

This final term project aims to strengthen practical understanding of modern Machine Learning systems through real-world case studies, including:

- Data preprocessing & feature engineering  
- Handling missing values & class imbalance  
- Model training & basic hyperparameter tuning  
- Performance evaluation & result interpretation  
- Comparison between traditional ML models and Deep Learning models  

---

## 🧪 Project Tasks Overview

This repository contains three main projects:

### 1. Fraud Detection (Classification)
- Predict whether an online transaction is fraudulent (`isFraud`).
- Includes data cleaning, imbalance handling, model training, and evaluation.
- Uses metrics such as Accuracy, Precision, Recall, F1-score, ROC-AUC.

### 2. Regression Task
- Predict a continuous target value from audio features.
- Includes preprocessing, feature selection, model comparison.
- Evaluated using MSE, RMSE, MAE, and R².

### 3. Image Classification (Deep Learning)
- Build a complete CNN-based image classification pipeline.
- Includes data augmentation, CNN modeling, and evaluation.
- Applies both custom CNN and transfer learning approaches.

---

## 🗂 Repository Structure

```text
Machine-Learning-Finalterm
├── datasets
│   ├── fraud
│   ├── regression
│   └── images
│       └── fish_dataset
├── notebooks
│   ├── UAS_FraudDetection.ipynb
│   ├── UAS_Regression.ipynb
│   └── UAS_ImageClassification.ipynb
├── .gitignore
├── LICENSE
└── README.md
```
---

## 🚀 How to Run the Notebooks

Due to the large dataset sizes and computational requirements, all notebooks are designed to be executed in **Google Colab**.

### Step 1 – Open Notebook in Colab

You can open each notebook directly from GitHub:

- **Fraud Detection:**  
  https://colab.research.google.com/github/kinanpta/Machine-Learning-Finalterm/blob/main/notebooks/UAS_FraudDetection.ipynb

- **Regression:**  
  https://colab.research.google.com/github/kinanpta/Machine-Learning-Finalterm/blob/main/notebooks/UAS_Regression.ipynb

- **Image Classification:**  
  https://colab.research.google.com/github/kinanpta/Machine-Learning-Finalterm/blob/main/notebooks/UAS_ImageClassification.ipynb

### Step 2 – Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 3 - Load Dataset Example

```python
import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/Machine-Learning-Finalterm-Datasets/train_transaction.csv')
df.head()
```

---

## 🛠 Tools and Libraries

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- TensorFlow or Keras
- Google Colab

--- 

## 🧾License

This project is licensed under the MIT License.
You are free to use, modify, and distribute this project for educational and research purposes.

---

<p align="center"> <i>Final Term Project • Machine Learning • Telkom University</i> </p>
