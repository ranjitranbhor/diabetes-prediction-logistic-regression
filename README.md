## 📊 Project Overview

This project implements a binary classification model to predict whether a patient has diabetes based on various health metrics. The analysis includes:

- Exploratory Data Analysis (EDA)
- Data visualization
- Logistic Regression model training
- Model evaluation using confusion matrix and classification metrics

## 🎯 Objective

Build a predictive model for Cedars Sinai to identify patients with Diabetes Mellitus, enabling:
- Early diagnosis and treatment
- Disease prevention through targeted interventions
- Improved patient quality of life
- Strategic healthcare resource allocation

## 📁 Dataset

**Source**: Diabetes Dataset

**Features**:
- `Pregnancies`: Number of times pregnant
- `Glucose`: Plasma glucose concentration (2 hours oral glucose tolerance test)
- `BloodPressure`: Diastolic blood pressure (mm Hg)
- `SkinThickness`: Triceps skin fold thickness (mm)
- `Insulin`: 2-Hour serum insulin (mu U/ml)
- `BMI`: Body mass index (weight in kg/(height in m)²)
- `DiabetesPedigreeFunction`: Diabetes pedigree function
- `Age`: Age (years)
- `Outcome`: Target variable (0 = No Diabetes, 1 = Diabetes)

**Dataset Statistics**:
- Total Observations: 768
- Class Distribution: ~65% No Diabetes, ~35% Diabetes (Class Imbalance Present)

## 🔧 Technologies Used

- **Python 3.x**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **scikit-learn**: Machine learning algorithms and metrics

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/RanjitRanbhor/diabetes-prediction-logistic-regression.git
cd diabetes-prediction-logistic-regression
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebook:
```bash
jupyter notebook notebooks/diabetes_prediction_analysis.ipynb
```

## 🚀 Usage

### Running the Analysis

1. Open the Jupyter notebook in Google Colab or locally
2. Mount Google Drive (if using Colab) or adjust the file path
3. Run all cells sequentially

### Key Steps in the Notebook

1. **Data Loading**: Import dataset from CSV
2. **Data Exploration**: 
   - View first 5 observations
   - Check data types and missing values
   - Visualize missing data with heatmap
3. **Target Variable Analysis**: Visualize class distribution
4. **Data Preparation**: Split data into train/test sets (70/30)
5. **Model Training**: Fit Logistic Regression with liblinear solver
6. **Model Evaluation**: Generate predictions and performance metrics

## 📈 Model Performance

### Results

- **Accuracy**: 76%
- **Sensitivity (Recall for Class 1)**: 54%
- **Specificity**: 88%
- **Precision (Class 1)**: 69%

### Confusion Matrix
```
                Predicted
              No (0)  Yes (1)
Actual No (0)   134      19
       Yes (1)   36      42
```

### Key Findings

1. **Class Imbalance**: Dataset shows significant imbalance (65% vs 35%)
2. **Coefficient Analysis**: 
   - Positive correlation: More pregnancies → Higher diabetes probability
   - Strongest predictor: DiabetesPedigreeFunction (coefficient: 0.777)
   - Negative correlation: Blood Pressure shows slight negative coefficient

3. **Model Limitations**:
   - **Low Sensitivity (54%)**: Misses ~46% of diabetes cases
   - High false negative rate unsuitable for medical diagnosis
   - Good specificity but insufficient for clinical use

## 💡 Insights & Recommendations

### Current Model Assessment
❌ **Not recommended for clinical deployment** due to:
- High false negative rate (46% of diabetes cases missed)
- Potential for undiagnosed patients
- Insufficient sensitivity for healthcare applications

### Improvement Strategies

1. **Address Class Imbalance**:
   - Apply SMOTE (Synthetic Minority Over-sampling Technique)
   - Use class weights in model training
   - Consider ensemble methods

2. **Feature Engineering**:
   - Create interaction features
   - Handle zero values appropriately (medical impossibilities)
   - Normalize/standardize features

3. **Model Enhancements**:
   - Try Random Forest, XGBoost, or Neural Networks
   - Implement hyperparameter tuning
   - Use cross-validation for robust evaluation

4. **Evaluation Metrics**:
   - Prioritize recall/sensitivity for diabetes detection
   - Use ROC-AUC score
   - Consider cost-sensitive learning

## 📊 Visualizations

The project includes:
- Missing data heatmap
- Target variable distribution plot
- Confusion matrix visualization (can be added)

## 🔍 Future Work

- [ ] Implement advanced feature engineering
- [ ] Apply class balancing techniques
- [ ] Experiment with ensemble methods
- [ ] Add ROC curve and precision-recall curves
- [ ] Deploy model as web API
- [ ] Create interactive dashboard

## 👤 Author

**Ranjit Ranbhor**

- GitHub: [@RanjitRanbhor](https://github.com/RanjitRanbhor)

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Dataset: Diabetes Database
- scikit-learn documentation and community

## 📧 Contact

For questions or collaboration opportunities, please open an issue or contact through GitHub.
