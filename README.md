# 🏥 Random Forest Classification: Diabetes Prediction

A comprehensive machine learning project implementing **Random Forest Classification** with hyperparameter tuning to predict diabetes in patients based on various medical measurements.

## 📋 Project Overview

This project uses the **Pima Indian Diabetes Dataset** to build a classification model that predicts whether a patient has diabetes (binary classification: 0 or 1).

**Key Features:**
- ✅ Implemented **Random Forest Classification** with ensemble of decision trees
- ✅ **GridSearchCV** for hyperparameter tuning across estimator count, tree depth, and feature sampling
- ✅ Comprehensive data cleaning and EDA
- ✅ Multiple evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- ✅ Feature importance and Out-of-Bag (OOB) error visualizations
- ✅ Interactive Streamlit web application for real-time predictions

## 📁 Project Structure

```
RandomForestClass/
├── notebooks/
│   └── diabetes_RF_classification.ipynb        # Main analysis notebook
├── data/
│   └── diabetes.csv                             # Dataset (768 samples, 9 features)
├── models/
│   ├── random_forest_classifier.pkl                        # Trained Random Forest model
│   ├── scaler.pkl                               # Feature scaler
│   ├── feature_names.pkl                        # Feature column names
│   └── hyperparameters.pkl                      # Best hyperparameters & metrics
├── app.py                                        # Streamlit web application
├── style.css                                     # Custom styling
├── requirements.txt                              # Python dependencies
└── README.md                                     # This file
```

## 🔬 Dataset Information

**File:** `data/diabetes.csv`

**Features (8):**
1. **Pregnancies** – Number of pregnancies
2. **Glucose** – Plasma glucose concentration (mg/dL)
3. **BloodPressure** – Diastolic blood pressure (mmHg)
4. **SkinThickness** – Triceps skin fold thickness (mm)
5. **Insulin** – 2-Hour serum insulin (mu U/ml)
6. **BMI** – Body Mass Index (kg/m²)
7. **DiabetesPedigreeFunction** – Diabetes pedigree function
8. **Age** – Age in years

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

```bash
jupyter notebook notebooks/diabetes_RF_classification.ipynb
```

The notebook will:
- Load and explore the diabetes dataset
- Perform data cleaning and EDA
- Build a Random Forest Classifier with hyperparameter tuning
- Evaluate the model with multiple metrics
- Plot OOB error curve and feature importances
- Save trained model and artifacts

### 3. Run the Streamlit Application

```bash
streamlit run app.py
```

The web app provides:
- 📊 Dataset preview and statistics
- 🌲 Random Forest algorithm overview
- ⚙️ Optimized hyperparameter display
- 📈 Model performance metrics
- 🎯 Confusion matrix visualization
- 📊 ROC curve analysis
- 📉 OOB error vs number of trees curve
- 🔑 Feature importance bar chart
- 🔮 Real-time diabetes risk prediction

## 🔍 Model Details

### Algorithm: Random Forest Classification

Random Forest builds a large number of Decision Trees in **parallel**, each trained on a random bootstrap sample of the data and a random subset of features at each split (bagging + feature randomness). Final prediction is the **majority vote** across all trees, which reduces variance and prevents overfitting compared to a single Decision Tree.

```
Bootstrap Sample 1 ──▶ Decision Tree 1 ──▶ Prediction 1 ─┐
Bootstrap Sample 2 ──▶ Decision Tree 2 ──▶ Prediction 2 ─┤
Bootstrap Sample 3 ──▶ Decision Tree 3 ──▶ Prediction 3 ─┤──▶ Majority Vote ──▶ Final Prediction
        ...                   ...                ...      ─┤
Bootstrap Sample N ──▶ Decision Tree N ──▶ Prediction N ─┘
```

**Splitting Criterion:** `gini` impurity  
**Feature Sampling:** `max_features` features randomly considered at each split  
**Bootstrap:** Enabled — each tree trains on ~63% of samples; remaining ~37% form the Out-of-Bag set for internal validation

### Hyperparameter Tuning (GridSearchCV)

| Parameter | Values Tested | Description |
|-----------|---------------|-------------|
| `n_estimators` | [50, 100, 150, 200, 250] | Number of trees in the forest |
| `max_depth` | [None, 5, 10, 15] | Maximum depth of each tree; `None` grows fully |
| `max_features` | ['sqrt', 'log2', 0.5] | Features considered at each split |
| `min_samples_split` | [2, 5, 10] | Minimum samples to split an internal node |
| `min_samples_leaf` | [1, 2, 4] | Minimum samples required at a leaf node |

**Total Combinations:** 540  
**CV Strategy:** 5-fold cross-validation, scoring on accuracy

> **Tip:** `n_estimators` and `max_features` are the most impactful parameters. More trees generally improve stability but with diminishing returns; monitor the OOB error curve to find the sweet spot.

### Data Preprocessing

1. **Data Cleaning:**
   - Remove duplicate rows
   - Replace physiologically impossible zero values with column medians (Glucose, BloodPressure, SkinThickness, Insulin, BMI)

2. **Feature Scaling:**
   - StandardScaler normalization (mean = 0, std = 1)
   - Fitted on training data only; applied to test data

3. **Train-Test Split:**
   - 80% training, 20% testing
   - Stratified split to maintain class balance

## 📊 Model Performance Metrics

After hyperparameter tuning:

| Metric | Value |
|--------|-------|
| **Training Accuracy** | ~0.97 |
| **Testing Accuracy** | ~0.81 |
| **Precision** | ~0.76 |
| **Recall** | ~0.68 |
| **F1 Score** | ~0.72 |
| **ROC-AUC** | ~0.87 |

*Note: Actual values depend on GridSearchCV results and random seed.*

## 📈 Exploratory Data Analysis (EDA)

The notebook includes:
- ✅ Target variable distribution (count & percentage)
- ✅ Feature distributions by outcome (diabetes vs no diabetes)
- ✅ Correlation matrix heatmap
- ✅ Box plots and histograms
- ✅ Statistical summary

## 🌲 Random Forest-Specific Visualizations

### OOB Error Curve
Plots Out-of-Bag classification error as `n_estimators` increases, revealing:
- The minimum number of trees needed for stable performance
- Convergence point beyond which adding more trees yields no benefit

### Feature Importance
Random Forest aggregates Gini-based importance across all trees, weighted by sample proportion. The bar chart highlights the most influential features (typically Glucose, BMI, and Age) with confidence intervals across trees.

## 🎯 Classification Metrics Explained

- **Accuracy** – Overall correctness of predictions
- **Precision** – Of predicted positive cases, how many were correct
- **Recall** – Of actual positive cases, how many were correctly predicted
- **F1 Score** – Harmonic mean of precision and recall
- **ROC-AUC** – Model's ability to distinguish between classes

## 🔮 Making Predictions

### Using the Streamlit App:
1. Open the app: `streamlit run app.py`
2. Enter patient measurements in the prediction section
3. Click **🏥 Predict Diabetes Risk**
4. Get instant prediction with confidence score from the Random Forest

### Using Python Code:

```python
import joblib
import numpy as np

# Load model and scaler
model  = joblib.load('models/rf_classifier.pkl')
scaler = joblib.load('models/scaler.pkl')

# Prepare input (example values)
input_data = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])

# Scale features
input_scaled = scaler.transform(input_data)

# Make prediction
prediction    = model.predict(input_scaled)[0]
probabilities = model.predict_proba(input_scaled)[0]

print(f"Prediction: {prediction}")  # 0 or 1
print(f"No Diabetes: {probabilities[0]:.2%}  |  Diabetes: {probabilities[1]:.2%}")
```

## 📝 Files Generated After Running Notebook

After executing the notebook, the following files are created in the `models/` directory:

1. **rf_classifier.pkl** – Trained Random Forest model
2. **scaler.pkl** – StandardScaler for feature normalization
3. **feature_names.pkl** – List of feature column names
4. **hyperparameters.pkl** – Best hyperparameters, OOB scores, and performance metrics

## 🔧 Technologies Used

- **Python 3.11+**
- **scikit-learn** – RandomForestClassifier, GridSearchCV, StandardScaler
- **pandas** – Data manipulation and analysis
- **numpy** – Numerical computing
- **matplotlib & seaborn** – Data visualization
- **streamlit** – Web application framework
- **joblib** – Model serialization

## 📚 Key References

- [scikit-learn RandomForestClassifier Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [GridSearchCV Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [Pima Indian Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- [Breiman (2001) – Random Forests, Machine Learning Journal](https://link.springer.com/article/10.1023/A:1010933404324)

## 💡 Future Enhancements

- [ ] Compare Random Forest vs Gradient Boosting vs XGBoost
- [ ] Implement Extremely Randomized Trees (`ExtraTreesClassifier`) for speed comparison
- [ ] Address class imbalance with SMOTE or `class_weight='balanced'`
- [ ] Add SHAP values for individual prediction explanations
- [ ] Deployment on cloud platforms (AWS, Azure, Google Cloud)
- [ ] REST API endpoint for predictions

## ⚠️ Important Notes

1. **Medical Disclaimer:** This model is for educational purposes and should NOT be used as a medical diagnosis tool
2. **Data Privacy:** Handle patient data responsibly and ensure HIPAA/GDPR compliance
3. **Training Time:** Large grids with high `n_estimators` are computationally intensive; reduce grid size or use `RandomizedSearchCV` for faster iteration
4. **Imbalanced Data:** Dataset has class imbalance (~35% positive); consider SMOTE or `class_weight='balanced'` for production use

## 🤝 Contributing

Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests
- Share feedback

## 📄 License

This project is open source and available for educational and research purposes.

---

**Last Updated:** May 2026
**Version:** 1.0
**Status:** Complete ✅
