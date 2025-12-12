# 🍷 Red Wine Quality Prediction
A Machine Learning project that predicts the quality of red wine based on its physicochemical properties using a Random Forest Regressor and a Streamlit web interface.

---

## 📖 Overview
This system evaluates wine quality by analyzing attributes such as acidity, sulphates, pH, alcohol percentage, and chlorides. The model is trained on the UCI Wine Quality dataset and deployed through an interactive Streamlit application where users can input wine features and obtain instant predictions.

---

## 🚀 Features
- Random Forest Regressor for accurate prediction  
- Streamlit UI for real-time user input  
- Preprocessing using StandardScaler  
- Model saved as `.pkl` for reuse  
- Simple, fast, and user-friendly interface  

---

## 📊 Dataset
**Source:** UCI Machine Learning Repository  
**Records:** 1599 samples  
**Features:**  
- Fixed acidity  
- Volatile acidity  
- Citric acid  
- Residual sugar  
- Chlorides  
- Free & Total sulfur dioxide  
- Density  
- pH  
- Sulphates  
- Alcohol  

**Target:** Quality score (0–10)

---

## 🧠 Model Training Workflow
1. Load dataset  
2. Clean & preprocess data  
3. Normalize features  
4. Train Random Forest Regressor  
5. Evaluate using RMSE & R²  
6. Save `model.pkl` and `scaler.pkl`  

---

## ▶️ How to Run

### Install dependencies
```bash
pip install -r requirements.txt
