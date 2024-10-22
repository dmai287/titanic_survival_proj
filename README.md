# Titanic Survival Prediction

## Project Overview

The Titanic Survival Prediction project aims to predict whether a passenger survived the Titanic disaster based on various features such as passenger class, gender, age, fare, and others. Using the famous Titanic dataset from [Kaggle](https://www.kaggle.com/c/titanic), this project implements various machine learning models to perform binary classification on the dataset.

This project walks through the complete data science workflow, including data exploration, preprocessing, feature engineering, model training, evaluation, and testing.

## Dataset

The dataset used in this project can be found on [Kaggle](https://www.kaggle.com/c/titanic/data). It contains the following files:

- `titanic.csv`: The training dataset, including features like `PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, and `Embarked`.

## Project Structure

```plaintext
titanic-survival-prediction/
│
├── README.md                  # Project overview
├── data/
│   ├── raw/                   # Raw dataset (titanic.csv)
├── notebooks/
│   ├── 01_EDA.ipynb      # EDA and basic data analysis
│   ├── 02_models.ipynb    # Data cleaning and preprocessing
├── requirements.txt            # Dependencies list
├── .gitignore                  # Files to ignore in the repository
└── submission.csv              # Predictions on the test dataset
```
## Project Workflow


### 1. Data Exploration
In the 01_data_exploration.ipynb notebook, we explore the Titanic dataset to understand the features and their relationships. The following steps were performed:

- Visualized distributions of key features like Age, Fare, Pclass, and Sex (`01_EDA.html` and `01_EDA.ipynb`)
- Analyzed survival rates based on gender, passenger class, and other features.
- Checked for missing values and potential outliers.
- The final models and results are in `02_models.html` or `02_models.ipynb`. 

### 2. Data Preprocessing

The `02_models.html` or `02_models.ipynb` notebook deals with cleaning the dataset:

- Handling missing values: Imputed missing values in Age and Embarked columns.
- Encoding categorical features: Converted categorical variables like Sex and Embarked into numerical form using one-hot encoding.
- Feature scaling: Scaled numerical features like Fare using standardization.
- Feature engineering: Created new features such as FamilySize (combining SibSp and Parch) and extracted titles from passenger names.

### 3. Model Training
In the `02_models.ipynb`, I trained several machine learning models using the cleaned dataset:

- Types of model(s) tried: Logistic Regression, kNN, Standard Scaler + Logistic Regression, Standard Scaler + kNN.

### 4. Model Evaluation and Results
- Validation results for your model(s): 
	LogisticRegression	0.798387
	KNN	0.687097
	StandardScaler + LogisticRegression	0.798387
	StandardScaler + KNN	0.801613
- Statement about which model is the best: StandardScaler + KNN gives me the best accuracy.

### 5. Major libraries used:

- pandas: For data manipulation and analysis
- numpy: For numerical computations
- matplotlib and seaborn: For data visualization
- scikit-learn: For machine learning models and preprocessing

## Future Improvements
1. Model Optimization: Use hyperparameter tuning techniques like GridSearchCV or RandomizedSearchCV to further improve model performance.
2. Feature Engineering: Experiment with more advanced feature engineering techniques, such as deriving interaction terms or applying dimensionality reduction.
3. Ensemble Methods: Try combining multiple models (e.g., Random Forest, XGBoost, SVM) using ensemble techniques like stacking or boosting to achieve better results.
4. Deployment: Deploy the trained model using Flask or FastAPI and create a web application where users can input passenger details and get survival predictions.

## Conclusion
The Titanic Survival Prediction project successfully applies machine learning models to predict the survival of passengers based on key features. The project demonstrates the complete data science process, from data exploration and preprocessing to model training and evaluation.
