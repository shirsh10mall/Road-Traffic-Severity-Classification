# Road Traffic Accidents Severity Prediction | Multi-Class Classification

Kaggle Notebook: https://www.kaggle.com/code/shirshmall/road-traffic-accidents-severity-classification

## Road Traffic Accidents Severity Prediction

**Problem Statement:** The goal of this project is to predict the severity of road traffic accidents using machine learning classification algorithms. The target variable, "Accident_severity," is a multi-class variable indicating different levels of accident severity. The objective is to create a predictive model that accurately classifies accident severity and assess the model's performance using the F1-score metric.

**Description:** The dataset for this project was collected from the Addis Ababa Sub-city police departments for a master's research work. It covers road traffic accidents recorded between 2017 and 2020. The dataset, containing 32 features and 12,316 accident instances, has been carefully curated to exclude sensitive information. The primary objective is to preprocess and analyze the data to identify the major causes of accidents using various machine learning classification algorithms.

**Performance Metric:** The model's performance will be evaluated based on the F1-score, a metric that considers both precision and recall, providing a balanced evaluation of classification performance.

**Steps for Model Creation:**

1. **Feature Identification:** The initial step involves identifying the types of features, such as whether they are ordinal, categorical, numerical, or unique. Additionally, a list of features with missing values is compiled, and strategies for handling these gaps are determined.

2. **Exploratory Data Analysis (EDA):** A comprehensive data visualization approach is used to gain insights into the dataset. Various visualizations like cross tabs, bar plots, violin plots, pair plots, joint plots, and density plots are generated to understand variable relationships and data patterns.

3. **Encoding Features:** Categorical and ordinal features are encoded to transform them into a suitable format for machine learning algorithms.

4. **Addressing Imbalance:** The dataset is examined for class imbalance in the target variable. Strategies to tackle this issue are explored to ensure fair model predictions.

5. **Handling Missing Values:** Missing values are imputed using methods like simple imputation with mean or median values. The imputed column distributions are compared with the original using density plots and box plots to validate the imputation.

6. **Model Training:** Multi-class classification models, including XGBoost, RandomForest, DecisionTree, KNeighbors, LogisticRegression, MLPClassifier, and SVC, are trained on the data. However, some models display overfitting due to the class imbalance.

7. **Overfitting Mitigation:** To address overfitting, techniques like Synthetic Minority Over-sampling Technique (SMOTE) and other upsampling methods are applied. Hyperparameter tuning using Optuna and Grid Search is attempted, but overfitting persists.

8. **Model Selection:** Following thorough testing, the Random Forest Classifier is chosen as the final model due to its lower overfitting risk and acceptable F1-score performance.
