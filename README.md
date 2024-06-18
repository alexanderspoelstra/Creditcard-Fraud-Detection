# Creditcard-Fraud-Detection
Fraud detection is considered an archetypal application of machine learning within the banking sector. However, whilst the internet is littered with jupyter notebooks and blog posts tackling the problem as one of imbalanced classification, in real-life applications financial institutions detecting fraud often do not have the luxury of labelled data. The reason for this is partly practical – it can be time consuming even for a domain expert to label a transaction as fraudulent or legitimate – but also of a more fundamental kind – fraudsters are constantly adapting their behaviour so as to avoid detection, meaning that a model trained on yesterday’s fraudulent activities may not be sufficient to detect today’s fraud.

## The Dataset
The dataset is located in the /data/ folder of this repo. It consists of 284,807 transactions, each of these transactions is associated with a euro value ('Amount') and a unique ID ('Transaction_ID'). Each transaction is further characterized by 28 features, labelled X01-X28. No data cleaning is required of this dataset before starting your analysis.

Download the dataset from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Code info

main.py:
 
This file contains the main logic of the project. It loads data from the specified filepath, processes it,
and then fits an Isolation Forest model to detect anomalies. It provides an option for hyperparameter tuning
based on the value of the HYPER_TUNING environment variable.

customIsolationForest.py: 

This file defines the CustomIsolationForest class, which extends the functionality of the Isolation Forest model
from scikit-learn. It allows customization of hyperparameters and provides methods for fitting, predicting, and
getting model parameters.

hyper_parameter_tuning.py: 

This file contains functions for hyperparameter tuning using GridSearchCV. It defines a custom scorer for evaluating 
model performance and tunes the Isolation Forest model based on specified parameter grids.

utils.py: 

This file contains utility functions for loading data, processing data, saving/loading models, and exporting 
results to CSV files.
