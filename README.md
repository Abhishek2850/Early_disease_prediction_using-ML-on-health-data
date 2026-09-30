# ❤️ Heart Disease Prediction System

## 📌 Project Overview

This project is a Machine Learning-based Heart Disease Prediction System that predicts whether a person is at risk of heart disease based on various health parameters.

The model was built using Python and Scikit-Learn and deployed with Streamlit for an interactive user experience.

---

## 🚀 Features

* Data Cleaning and Preprocessing
* Exploratory Data Analysis (EDA)
* Feature Engineering
* One-Hot Encoding for Categorical Variables
* Feature Scaling using StandardScaler
* K-Nearest Neighbors (KNN) Classification Model
* Interactive Streamlit Web Application
* Real-time Heart Disease Prediction

---

## 📊 Dataset Features

The model uses the following input features:

* Age
* Resting Blood Pressure (RestingBP)
* Cholesterol
* Fasting Blood Sugar (FastingBS)
* Maximum Heart Rate (MaxHR)
* Oldpeak
* Gender
* Chest Pain Type
* Resting ECG
* Exercise-Induced Angina
* ST Slope

Target Variable:

* HeartDisease (0 = No Disease, 1 = Disease)

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Streamlit
* Pickle

---

## 📂 Project Structure

```text
Heart-Disease-Prediction/
│
├── app.py
├── Heart.ipynb
├── knn_heart_model.pkl
├── heart_scaler.pkl
├── heart_columns.pkl
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Heart-Disease-Prediction.git
```

Move to project directory:

```bash
cd Heart-Disease-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 🌐 Deployment

The application is deployed using Streamlit Cloud.

After deployment, access the application through the generated Streamlit URL.

---

## 📈 Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Encoding
5. Feature Scaling
6. Model Training (KNN)
7. Model Evaluation
8. Model Serialization using Pickle
9. Streamlit Deployment

---

## 🎯 Future Improvements

* Compare multiple ML algorithms
* Hyperparameter tuning
* Probability-based risk prediction
* Enhanced UI/UX design
* Cloud database integration

---

## 👨‍💻 Author

Ankit Rai

Aspiring Data Analyst & Machine Learning Enthusiast
