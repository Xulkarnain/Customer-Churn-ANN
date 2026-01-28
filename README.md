# Customer Churn Prediction using Artificial Neural Network (ANN)
## Live preview : [Click Here](https://customer-churn-ann-mu5tfdyrqt84ytvr7gksav.streamlit.app/)
## 📌 Project Overview
Customer churn is a major business problem where customers discontinue a service.  
This project builds an **end-to-end deep learning system** to predict customer churn using an **Artificial Neural Network (ANN)** and deploys it as an interactive **Streamlit web application**.

The model is optimized not just for accuracy, but for **recall of churn customers**, which is critical for real-world retention strategies.

---

## 📊 Dataset
- **Dataset**: Telco Customer Churn Dataset
- **Samples**: ~7,000 customers
- **Target Variable**: `Churn` (Yes / No)
- **Feature Types**:
  - Numerical: tenure, MonthlyCharges, TotalCharges
  - Categorical: Contract, InternetService, PaymentMethod, etc.

---

## 🔍 Exploratory Data Analysis (EDA)
Key insights from EDA:
- Customers with **short tenure** are more likely to churn
- **Month-to-month contracts** show significantly higher churn
- **Fiber optic users** and **electronic check payments** have higher churn rates
- Higher **monthly charges** correlate with increased churn

Both categorical and numerical features were analyzed using:
- Count plots
- Distribution plots
- Box plots
- Correlation heatmaps

---

## 🧠 Feature Engineering & Preprocessing
A robust preprocessing pipeline was built using **scikit-learn**:

- Dropped non-informative identifier (`customerID`)
- Converted `TotalCharges` safely using `pd.to_numeric`
- Binary encoding for binary categorical features
- One-hot encoding for multi-class categorical features
- Standard scaling for numerical features
- Implemented using `ColumnTransformer` and `Pipeline` to prevent data leakage

The preprocessing pipeline is saved and reused during inference.

---

## 🤖 Model Architecture (ANN)

# Customer Churn Prediction using Artificial Neural Network (ANN)

## 📌 Project Overview
Customer churn is a major business problem where customers discontinue a service.  
This project builds an **end-to-end deep learning system** to predict customer churn using an **Artificial Neural Network (ANN)** and deploys it as an interactive **Streamlit web application**.

The model is optimized not just for accuracy, but for **recall of churn customers**, which is critical for real-world retention strategies.

---

## 📊 Dataset
- **Dataset**: Telco Customer Churn Dataset
- **Samples**: ~7,000 customers
- **Target Variable**: `Churn` (Yes / No)
- **Feature Types**:
  - Numerical: tenure, MonthlyCharges, TotalCharges
  - Categorical: Contract, InternetService, PaymentMethod, etc.

---

## 🔍 Exploratory Data Analysis (EDA)
Key insights from EDA:
- Customers with **short tenure** are more likely to churn
- **Month-to-month contracts** show significantly higher churn
- **Fiber optic users** and **electronic check payments** have higher churn rates
- Higher **monthly charges** correlate with increased churn

Both categorical and numerical features were analyzed using:
- Count plots
- Distribution plots
- Box plots
- Correlation heatmaps

---

## 🧠 Feature Engineering & Preprocessing
A robust preprocessing pipeline was built using **scikit-learn**:

- Dropped non-informative identifier (`customerID`)
- Converted `TotalCharges` safely using `pd.to_numeric`
- Binary encoding for binary categorical features
- One-hot encoding for multi-class categorical features
- Standard scaling for numerical features
- Implemented using `ColumnTransformer` and `Pipeline` to prevent data leakage

The preprocessing pipeline is saved and reused during inference.

---

## 🤖 Model Architecture (ANN)

Input Layer (23 features)
↓
Dense (32 units, ReLU)
↓
Dropout (0.3)
↓
Dense (16 units, ReLU)
↓
Dropout (0.3)
↓
Dense (1 unit, Sigmoid)


### Training Details
- Optimizer: Adam
- Loss Function: Binary Cross-Entropy
- Regularization: Dropout + EarlyStopping
- Validation Strategy: Hold-out validation split

---

## 📈 Model Evaluation

### Initial Model
- Accuracy: ~80%
- Churn Recall: ~58%

### Optimized Model (Threshold Tuning)
- **Churn Recall improved to ~74%**
- Reduced missed churn customers by ~39%
- Acceptable trade-off with precision and accuracy

> In churn prediction, identifying churn-prone customers is more valuable than maximizing accuracy.

---

## 🌐 Deployment with Streamlit
The trained model and preprocessing pipeline are deployed using **Streamlit**.

### Features:
- Interactive user input form
- Real-time churn probability prediction
- Business-friendly output (Churn / No Churn)
- Uses saved model and preprocessing pipeline for consistency

### Run the App Locally
```bash
streamlit run app.py


Customer-Churn-ANN/
│
├── data/
│   └── churn.csv
│
├── models/
│   ├── churn_ann_final.keras
│   └── preprocessor.pkl
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── fe.ipynb
│   └── 03_ANN_Model.ipynb
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md

🛠️ Tech Stack

Python

Pandas, NumPy

Scikit-learn

TensorFlow / Keras

Streamlit

Matplotlib, Seaborn