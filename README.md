# Home_Credit_Default_Risk

This is a competition from Kaggle, predicting if the client would default or not.

Data source: [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/overview/description).

```
📦Home_Credit_Default_Risk
 ┣ 📂images
 ┃ ┣ ...
 ┣ 📂submissions
 ┃ ┣ ...
 ┣ 📜Part1_Introduction_and_EDA.ipynb
 ┣ 📜Part1_Introduction_and_EDA.pdf
 ┣ 📜Part2_Data_Cleaning_and Feature_Engineering.ipynb
 ┣ 📜Part2_Data_Cleaning_and Feature_Engineering.pdf
 ┣ 📜Part3_Model_Training.ipynb
 ┣ 📜Part3_Model_Training.pdf
 ┣ 📜test_CatBoost.ipynb
 ┣ 📜test_CatBoost.pdf
```

## `Part1_Introduction_and_EDA.ipynb` inlcudes introduction and exploratory data analysis. 

1. Defining utility functions
2. Exploratory Data Analysis (EDA)
2.1 application_train.csv and application_test.csv
2.1.1 Basic Stats
2.1.2 NaN columns and percentages
2.1.3 Distribution of target variable
2.1.4 Phi-K matrix
2.1.5 Correlation matrix of numerical features
2.1.6 Plotting distribution of categorical variables
2.1.7 Plotting distribution of Continuous Variables
2.2 bureau.csv
...
2.3 bureau_balance.csv
...
2.4 previous_application.csv
...
2.5 installments_payments.csv
...
2.6 POS_CASH_balance.csv
...
2.7 credit_card_balance.csv
...
3 Conclusions From EDA

## `Part2_Data_Cleaning_and Feature_Engineering.ipynb` includes data cleaning and feature engineering.

1. Defining Utility Functions and Classes
2. Data Clearning and Feature Engineering
2.1 Preprocessing Tables
2.1.1 bureau_balance.csv and bureau.csv
2.1.2 previous_application.csv
2.1.3 installments_payments.csv
2.1.4 POS_CASH_balance.csv
2.1.5 credit_card_balance.csv
2.1.6 application_train and application_test
2.2 Merging all tables
3. Feature Engineering more
4. Feature selection
4.1 Looking for empty features
4.2 Recursive feature selection using LightGBM
4.3 Saving Processed Data

## `Part3_Model_Training.ipynb` inlcudes data modeling and scores. 
Some models: `Random Forest, LightGBM, XGBoost`, etc. are used to make better prediction.

1. Defining Utility Functions and Classes
2. Modelling
2.1 Random model
2.2 Dominant class model
2.3 Logistic Regression L2 Regularization
2.4 Linear SVM
2.5 Random Forest Classifier
2.6 ExtraTreesClassifier
2.7 XGBoost GPU
2.8 XGBoost GPU on Reduced Features
2.9 LightGBM
2.10 Stacking Classifiers
2.11 Blending of Predictions
3 Results Summarization and Conclusion

## `test_CatBoost.ipynb` includes the training of CatBoostClassifier Model on GPU

1. Test CPU
2. Boosting method
3. Test GPU
4. Grid search
5. Hyperparameter tuning on GPU

## Results Summarization

<img src="./images/Results Summarization.png"/>


## GPU Training:
