# Creditcard-Fraud-Detection
Fraud Detection model based on anonymized credit card transactions. 

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
