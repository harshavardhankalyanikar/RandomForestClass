# 🏥 KNN Classification: Diabetes Prediction

A comprehensive machine learning project implementing **K-Nearest Neighbors (KNN) Classification** with hyperparameter tuning to predict diabetes in patients based on various medical measurements.

## 📋 Project Overview

This project uses the **Pima Indian Diabetes Dataset** to build a classification model that predicts whether a patient has diabetes (binary classification: 0 or 1).

**Key Improvements:**
- ✅ Converted from **KNN Regression → KNN Classification**
- ✅ Implemented **GridSearchCV** for hyperparameter tuning
- ✅ Comprehensive data cleaning and EDA
- ✅ Multiple evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- ✅ Interactive Streamlit web application for real-time predictions

## 📁 Project Structure

```
KNNclass/
├── notebooks/
│   └── diabetes_knn_classification.ipynb    # Main analysis notebook
├── data/
│   └── diabetes.csv                          # Dataset (768 samples, 9 features)
├── models/
│   ├── knn_classifier.pkl                   # Trained KNN classifier model
│   ├── scaler.pkl                            # Feature scaler
│   ├── feature_names.pkl                    # Feature column names
│   └── hyperparameters.pkl                  # Best hyperparameters & metrics
├── app.py                                    # Streamlit web application
├── style.css                                 # Custom styling
├── requirements.txt                          # Python dependencies
└── README.md                                 # This file
```

## 🔬 Dataset Information

**File:** `data/diabetes.csv`

**Features (8):**
1. **Pregnancies** - Number of pregnancies
2. **Glucose** - Plasma glucose concentration (mg/dL)
3. **BloodPressure** - Diastolic blood pressure (mmHg)
4. **SkinThickness** - Triceps skin fold thickness (mm)
5. **Insulin** - 2-Hour serum insulin (mu U/ml)
6. **BMI** - Body Mass Index (kg/m²)
7. **DiabetesPedigreeFunction** - Diabetes pedigree function
8. **Age** - Age in years

**Target:** `Outcome` (0 = No Diabetes, 1 = Diabetes)

**Statistics:**
- Total Samples: 768
- Positive Cases (Diabetes): ~35%
- Negative Cases (No Diabetes): ~65%

## 🚀 Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Jupyter Notebook

Navigate to the notebooks folder and run the analysis:

```bash
jupyter notebook notebooks/diabetes_knn_classification.ipynb
```

The notebook will:
- Load and explore the diabetes dataset
- Perform data cleaning and EDA
- Train KNN classifier with hyperparameter tuning
- Evaluate the model with multiple metrics
- Save trained model and artifacts

### 3. Run the Streamlit Application

```bash
streamlit run app.py
```

The web app provides:
- 📊 Dataset preview and statistics
- ⚙️ Optimized hyperparameter display
- 📈 Model performance metrics
- 🎯 Confusion matrix visualization
- 📊 ROC curve analysis
- 🔮 Real-time diabetes risk prediction

## 🔍 Model Details

### Algorithm: K-Nearest Neighbors Classification

**Hyperparameter Tuning Results:**

The model uses GridSearchCV to find optimal hyperparameters across:

| Parameter | Values Tested |
|-----------|---------------|
| `n_neighbors` | [3, 5, 7, 9, 11, 13, 15, 17, 19, 21] |
| `weights` | ['uniform', 'distance'] |
| `metric` | ['euclidean', 'manhattan', 'minkowski'] |

**Total Combinations:** 60

### Data Preprocessing

1. **Data Cleaning:**
   - Remove duplicate rows
   - Replace zero values with median (for medical measurements)

2. **Feature Scaling:**
   - StandardScaler normalization
   - Mean = 0, Standard Deviation = 1

3. **Train-Test Split:**
   - 80% training, 20% testing
   - Stratified split to maintain class balance

## 📊 Model Performance Metrics

After hyperparameter tuning:

| Metric | Value |
|--------|-------|
| **Training Accuracy** | ~0.80 |
| **Testing Accuracy** | ~0.78 |
| **Precision** | ~0.72 |
| **Recall** | ~0.62 |
| **F1 Score** | ~0.67 |
| **ROC-AUC** | ~0.82 |

*Note: Actual values depend on the random seed and data preprocessing*

## 📈 Exploratory Data Analysis (EDA)

The notebook includes:
- ✅ Target variable distribution (count & percentage)
- ✅ Feature distributions by outcome
- ✅ Correlation matrix heatmap
- ✅ Box plots and histograms
- ✅ Statistical summary

## 🎯 Classification Metrics Explained

- **Accuracy:** Overall correctness of predictions
- **Precision:** Of predicted positive cases, how many were correct
- **Recall:** Of actual positive cases, how many were correctly predicted
- **F1 Score:** Harmonic mean of precision and recall
- **ROC-AUC:** Measure of model's ability to distinguish between classes

## 🔮 Making Predictions

### Using the Streamlit App:
1. Open the app: `streamlit run app.py`
2. Enter patient measurements in the prediction section
3. Click "🏥 Predict Diabetes Risk"
4. Get instant prediction with confidence score

### Using Python Code:

```python
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('models/knn_classifier.pkl')
scaler = joblib.load('models/scaler.pkl')

# Prepare input (example values)
input_data = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])

# Scale features
input_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_scaled)[0]
probabilities = model.predict_proba(input_scaled)[0]

print(f"Prediction: {prediction}")  # 0 or 1
print(f"Probabilities: No Diabetes={probabilities[0]:.2%}, Diabetes={probabilities[1]:.2%}")
```

## 📝 Files Generated After Running Notebook

After executing the notebook, the following files are created in the `models/` directory:

1. **knn_classifier.pkl** - Trained KNN classifier model
2. **scaler.pkl** - StandardScaler for feature normalization
3. **feature_names.pkl** - List of feature column names
4. **hyperparameters.pkl** - Best hyperparameters and performance metrics

## 🔧 Technologies Used

- **Python 3.8+**
- **scikit-learn** - Machine learning algorithms
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib & seaborn** - Data visualization
- **streamlit** - Web application framework
- **joblib** - Model serialization

## 📚 Key References

- [scikit-learn KNN Documentation](https://scikit-learn.org/stable/modules/neighbors.html)
- [GridSearchCV Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [Pima Indian Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

## 💡 Future Enhancements

- [ ] Implement other classification algorithms (Logistic Regression, SVM, Random Forest)
- [ ] Feature engineering and selection
- [ ] Cross-validation for more robust evaluation
- [ ] Model interpretability (SHAP values, feature importance)
- [ ] Deployment on cloud platforms (AWS, Azure, Google Cloud)
- [ ] API endpoint for predictions

## ⚠️ Important Notes

1. **Medical Disclaimer:** This model is for educational purposes and should NOT be used as a medical diagnosis tool
2. **Data Privacy:** Handle patient data responsibly and ensure HIPAA/GDPR compliance
3. **Model Limitations:** KNN is a simple algorithm; consider ensemble methods for production use
4. **Imbalanced Data:** Dataset has class imbalance; consider SMOTE or class weights for improvement

## 🤝 Contributing

Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests
- Share feedback

## 📄 License

This project is open source and available for educational and research purposes.

---

**Last Updated:** May 2024  
**Version:** 1.0  
**Status:** Complete ✅
